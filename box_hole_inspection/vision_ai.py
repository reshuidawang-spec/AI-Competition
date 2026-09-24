"""Vision perception: image -> DetectionResult.

Two interchangeable backends behind ONE interface (callers see only
``detector.detect(image) -> schemas.DetectionResult``):

- GeometricHoleDetector — W1 placeholder: threshold + contour + fitEllipse.
  Works on the sim renders and on real photos with clear dark holes. Lets
  the whole pipeline close in week 1 before any model exists.

- YoloSegDetector — W2: locally trained YOLO-Seg inference (classes
  0=box, 1=hole), hole centre = mask centroid (sub-pixel). Trained on the
  box only, runs offline — no cloud / no LLM anywhere in the loop.

AI 只做感知：网络/几何只回答"哪里有孔"；H01~H04 的最终编号与箱体位姿由
hole_assigner 的 CAD 模式配准决定。这里输出的 id 是占位编号（按图像行、
列排序），不保证与真实孔号一致。
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from box_hole_inspection import schemas

# 几何占位检测器默认参数（对仿真渲染图验证过：4/4 孔、像素误差 <1px）
BOX_BRIGHT_THRESH = 140        # 箱体顶面比台面亮的二值阈值
HOLE_DARK_THRESH = 45          # 孔比台面暗的二值阈值（台面 ~72、孔 ~22）
HOLE_AREA_MIN_PX = 100         # 最小孔轮廓面积
HOLE_AREA_MAX_PX = 50000       # 最大孔轮廓面积
MIN_CONFIDENCE = 0.3           # 低于该置信度的检测丢弃


class GeometricHoleDetector:
    """阈值 + 轮廓 + 椭圆拟合的占位检测器（W1；W2 换 YOLO 后接口不变）。"""

    def __init__(self, part_id: str = "BOX_001",
                 box_thresh: float = BOX_BRIGHT_THRESH,
                 hole_thresh: float = HOLE_DARK_THRESH,
                 hole_area_range: tuple = (HOLE_AREA_MIN_PX, HOLE_AREA_MAX_PX),
                 min_confidence: float = MIN_CONFIDENCE):
        self.part_id = part_id
        self.box_thresh = box_thresh
        self.hole_thresh = hole_thresh
        self.area_min, self.area_max = hole_area_range
        self.min_confidence = min_confidence

    def detect(self, image: np.ndarray,
               timestamp_ms: Optional[float] = None) -> schemas.DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        box_bbox, rz_hint, box_conf = self._detect_box(gray)
        holes = self._detect_holes(gray)

        conf = box_conf if box_bbox else 0.0
        return schemas.DetectionResult(
            part_id=self.part_id,
            confidence=conf,
            timestamp_ms=timestamp_ms if timestamp_ms is not None
            else time.time() * 1000.0,
            box_bbox=box_bbox or [],
            box_rz_hint_deg=rz_hint,
            holes=holes,
        )

    # -- internals -----------------------------------------------------------

    def _detect_box(self, gray: np.ndarray):
        """最亮的大区域 = 箱体顶面；返回 (bbox, rz_hint_deg, confidence)。"""
        _, binary = cv2.threshold(gray, self.box_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0.0
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)          # (center, (w, h), angle)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:        # 保证角度描述长边
            angle += 90.0
        angle %= 180.0
        if angle > 90.0:
            angle -= 180.0
        # 图像 y 轴向下 → 世界系逆时针 = 图像系顺时针 → 取负号。
        # 提示值：精确 rz 由 hole_assigner 的模式配准给出。
        rz_hint = -angle
        conf = float(cv2.contourArea(c) / max(w * h, 1))
        return [x, y, x + w, y + h], rz_hint, min(conf, 1.0)

    def _detect_holes(self, gray: np.ndarray) -> list:
        """最暗的小区域 = 孔；椭圆拟合给亚像素中心。"""
        _, binary = cv2.threshold(gray, self.hole_thresh, 255,
                                  cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.area_min <= area <= self.area_max):
                continue
            if len(c) < 5:
                continue
            (u, v), (ma, MA), _ = cv2.fitEllipse(c)
            ellipse_area = np.pi * (ma / 2.0) * (MA / 2.0)
            if ellipse_area <= 0:
                continue
            # 椭圆度置信：轮廓面积 / 拟合椭圆面积（≈1 表示像椭圆）
            conf = min(area / ellipse_area, 1.0)
            if conf < self.min_confidence:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append(schemas.HoleDetection(
                id="", confidence=conf,
                pixel_u=float(u), pixel_v=float(v),
                pixel_bbox=[x, y, x + w, y + h]))
        # 占位编号：按行、列排序（hole_assigner 之后做真实编号）
        dets.sort(key=lambda d: (round(d.pixel_v), round(d.pixel_u)))
        for i, d in enumerate(dets):
            d.id = f"H{i + 1:02d}"
        return dets


class YoloSegDetector:
    """本地 YOLO-Seg 推理（W2）。类 0=box、1=hole；孔中心 = mask 质心。

    训练流程见 docs/architecture.md 与 PROJECT_MEMORY.md；推理离线运行，
    无任何云端依赖。模型文件默认 data/models/box_yolo_seg.pt。
    """

    def __init__(self, model_path: str, part_id: str = "BOX_001",
                 conf: float = 0.5, device: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "YoloSegDetector 需要 ultralytics: pip install ultralytics") from e
        self.part_id = part_id
        self.conf = conf
        self.model = YOLO(model_path)
        self._device = device

    def detect(self, image: np.ndarray,
               timestamp_ms: Optional[float] = None) -> schemas.DetectionResult:
        result = self.model.predict(image, conf=self.conf, device=self._device,
                                    verbose=False)[0]
        names = {int(k): v for k, v in result.names.items()}
        cls = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
        confs = (result.boxes.conf.cpu().numpy()
                 if result.boxes is not None else [])
        boxes = (result.boxes.xyxy.cpu().numpy()
                 if result.boxes is not None else np.empty((0, 4)))
        masks = (result.masks.data.cpu().numpy()
                 if result.masks is not None else None)

        box_bbox, box_conf = None, 0.0
        holes = []
        for i, c in enumerate(cls):
            name = names.get(int(c), str(int(c)))
            if name in ("box", "0"):
                x0, y0, x1, y1 = boxes[i]
                box_bbox = [float(x0), float(y0), float(x1), float(y1)]
                box_conf = float(confs[i])
            elif name in ("hole", "1"):
                if masks is None:
                    continue
                m = masks[i]
                mm = cv2.moments(m.astype(np.uint8))
                if mm["m00"] < 1e-6:
                    continue
                u, v = mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]
                ys, xs = np.nonzero(m)
                holes.append(schemas.HoleDetection(
                    id="", confidence=float(confs[i]),
                    pixel_u=float(u), pixel_v=float(v),
                    pixel_bbox=[float(xs.min()), float(ys.min()),
                                float(xs.max()), float(ys.max())]))

        holes.sort(key=lambda d: (round(d.pixel_v), round(d.pixel_u)))
        for i, d in enumerate(holes):
            d.id = f"H{i + 1:02d}"

        return schemas.DetectionResult(
            part_id=self.part_id,
            confidence=box_conf,
            timestamp_ms=timestamp_ms if timestamp_ms is not None
            else time.time() * 1000.0,
            box_bbox=box_bbox or [],
            box_rz_hint_deg=None,
            holes=holes,
        )
