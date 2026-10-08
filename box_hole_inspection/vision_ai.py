"""Vision perception: image -> DetectionResult.

Interchangeable backends behind ONE interface (callers see only
``detector.detect(image) -> schemas.DetectionResult``):

- GeometricHoleDetector — W1 placeholder: threshold + contour + fitEllipse.
  Works on the sim renders and on real photos with clear dark holes. Lets
  the whole pipeline close in week 1 before any model exists.

- YoloSegDetector — W2: locally trained YOLO-Seg inference (classes
  0=box, 1=hole), hole centre = mask centroid (sub-pixel). Trained on the
  box only, runs offline — no cloud / no LLM anywhere in the loop.

- EngineBoreDetector — supplied EngineBoreInspection v1 two-model detector;
  preserves its part filter, recovery passes and ellipse quality checks.

AI 只做感知：网络/几何只回答"哪里有孔"；H01~H04 的最终编号与箱体位姿由
hole_assigner 的 CAD 模式配准决定。这里输出的 id 是占位编号（按图像行、
列排序），不保证与真实孔号一致。
"""

from __future__ import annotations

import time
import logging
import math
from pathlib import Path
import re
import tempfile
from typing import Optional

import cv2
import numpy as np

from box_hole_inspection import schemas

log = logging.getLogger(__name__)

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


class EngineBoreDetector:
    """The supplied EngineBoreInspection v1 detector behind DetectionResult.

    It runs the package's two active segmentation models, part filtering,
    recovery passes and ellipse quality checks. Only reliable original-image
    pixel centres are returned. Four reliable bores are required by the
    downstream HoleAssigner; incomplete detections therefore stop the run.
    """

    def __init__(self, bore_model_path: Optional[str] = None,
                 part_model_path: Optional[str] = None,
                 part_id: str = "ENGINE_001"):
        try:
            from box_hole_inspection.engine_vision.detect_core import (
                DEFAULT_MODEL, PART_MODEL, EngineDetector)
        except ImportError as exc:
            raise RuntimeError(
                "EngineBoreDetector 需要 torch、ultralytics 和 Pillow") from exc
        bore_path = Path(bore_model_path or DEFAULT_MODEL)
        part_path = Path(part_model_path or PART_MODEL)
        for path in (bore_path, part_path):
            if not path.is_file():
                raise FileNotFoundError(f"视觉模型不存在: {path}")
        self.part_id = part_id
        self._detector = EngineDetector(model_path=bore_path,
                                        part_model_path=part_path)
        self.last_diagnostics: dict = {}

    def detect(self, image: np.ndarray,
               timestamp_ms: Optional[float] = None) -> schemas.DetectionResult:
        if (image is None or image.ndim != 3 or image.shape[2] != 3
                or image.dtype != np.uint8):
            raise ValueError("EngineBoreDetector expects a uint8 BGR image")
        # The supplied detector accepts a path. PNG preserves the camera pixels
        # exactly; its own rendered image and CSV stay inside the temp directory.
        with tempfile.TemporaryDirectory(prefix="engine_bore_") as tmp:
            image_path = Path(tmp) / "frame.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"cannot write temporary frame: {image_path}")
            report = self._detector.analyze(image_path, output_dir=tmp)

        warnings = list(report.get("warnings", []))
        review_ids = set()
        for warning in warnings:
            match = re.search(r"孔\s*(\d+)\s*(?:分割轮廓不稳定|拟合质量不足)",
                              str(warning))
            if match:
                review_ids.add(int(match.group(1)))

        selected = {int(h["hole_id"]): float(h["confidence"])
                    for h in report.get("selected_bores", [])}
        height, width = image.shape[:2]
        holes = []
        if report.get("part_constraint_applied", False):
            for center in report.get("centers", []):
                number = int(center["hole_id"])
                confidence = selected.get(number, 0.0)
                support = center.get("contour_support")
                edge_verified = center.get("fit_source") == "image_edge_verified"
                if (number in review_ids or confidence < 0.5
                        or (support is not None and float(support) < 0.85
                            and not edge_verified)):
                    continue
                cx, cy = float(center["center_x_px"]), float(center["center_y_px"])
                axis1 = float(center["ellipse_axis1_px"])
                axis2 = float(center["ellipse_axis2_px"])
                angle = float(center["ellipse_angle_deg"])
                if (not all(map(math.isfinite, (cx, cy, axis1, axis2, angle)))
                        or min(axis1, axis2) <= 0
                        or not (0 <= cx < width and 0 <= cy < height)):
                    continue
                corners = cv2.boxPoints(((cx, cy), (axis1, axis2), angle))
                x0, y0 = corners.min(axis=0)
                x1, y1 = corners.max(axis=0)
                holes.append(schemas.HoleDetection(
                    id=f"H{number:02d}", confidence=confidence,
                    pixel_u=cx, pixel_v=cy,
                    pixel_bbox=[float(max(0, x0)), float(max(0, y0)),
                                float(min(width - 1, x1)), float(min(height - 1, y1))]))

        part_proposals = [p for p in report.get("proposals", [])
                          if p.get("class") == "part"]
        best_part = max(part_proposals,
                        key=lambda p: float(p.get("confidence", 0.0)),
                        default=None)
        box_bbox = list(best_part["box_xyxy"]) if best_part else []
        self.last_diagnostics = {
            "selected_count": len(selected),
            "reliable_count": len(holes),
            "part_constraint_applied": bool(report.get("part_constraint_applied")),
            "warnings": warnings,
        }
        if len(holes) != 4:
            log.warning("EngineBoreDetector: %d/4 reliable bores; %s",
                        len(holes), warnings)
        return schemas.DetectionResult(
            part_id=self.part_id,
            confidence=float(report.get("part_max_confidence", 0.0)),
            timestamp_ms=timestamp_ms if timestamp_ms is not None
            else time.time() * 1000.0,
            box_bbox=box_bbox,
            box_rz_hint_deg=None,
            holes=sorted(holes, key=lambda h: h.id),
        )
