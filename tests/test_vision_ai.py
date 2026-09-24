#!/usr/bin/env python3
"""Tests for vision perception backends — including the W1 acceptance
preview: render -> detect -> coordinate chain -> world, error < 2 mm."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.coordinate_transform import CoordinateTransform
from box_hole_inspection.utils import transforms as tf
from box_hole_inspection.vision_ai import GeometricHoleDetector
from interfaces.camera import CameraIntrinsics
from sim.sim_camera import SimCamera
from sim.sim_world import SimWorld, default_box

FX, FY, CX, CY = 1000.0, 1000.0, 640.0, 512.0
W, H = 1280, 1024


def make_intrinsics():
    return CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=W, height=H)


def make_scene(rz=20.0, seed=0):
    world = SimWorld(default_box())
    world.set_box_pose(900.0, 0.0, rz)
    t_w_c = tf.pose_to_T({"x": 900, "y": 0, "z": 480,
                          "rx": 180, "ry": -8, "rz": 0})
    cam = SimCamera(make_intrinsics(), t_w_c, seed=seed)
    return world, cam


def make_chain(world, t_w_c):
    cal = {
        "T_EA_C": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_EB_P": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_BA_BB": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "camera_intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "table_plane_w": {"point": [0, 0, world.box.z_top_mm],
                          "normal": [0, 0, 1]},
        "probe": {"ball_radius_mm": 1.0, "trigger_io": "DI1"},
    }
    return CoordinateTransform(cal, t_ba_ea=np.asarray(t_w_c, dtype=float))


def match_by_nearest_pixel(det: schemas.DetectionResult, gt_pixels: dict):
    """检测孔 → 真值孔的最近邻像素匹配（编号分配属于 hole_assigner）。"""
    out = {}
    for hid, (gu, gv) in gt_pixels.items():
        best, best_d = None, float("inf")
        for h in det.holes:
            d = (h.pixel_u - gu) ** 2 + (h.pixel_v - gv) ** 2
            if d < best_d:
                best, best_d = h, d
        if best is not None:
            out[hid] = best
    return out


class TestGeometricDetection(unittest.TestCase):
    def setUp(self):
        self.detector = GeometricHoleDetector()

    def test_detects_four_holes(self):
        for rz in (0.0, 20.0, -15.0):
            for seed in (0, 1):
                world, cam = make_scene(rz=rz, seed=seed)
                det = self.detector.detect(cam.render(world).image)
                self.assertEqual(len(det.holes), 4,
                                 f"rz={rz} seed={seed}: {len(det.holes)} holes")
                for h in det.holes:
                    self.assertGreaterEqual(h.confidence, 0.3)

    def test_box_detection_matches_render(self):
        world, cam = make_scene()
        render = cam.render(world)
        det = self.detector.detect(render.image)
        x0, y0, x1, y1 = det.box_bbox
        gx0, gy0, gx1, gy1 = render.box_bbox_px
        ix0, iy0 = max(x0, gx0), max(y0, gy0)
        ix1, iy1 = min(x1, gx1), min(y1, gy1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = (x1 - x0) * (y1 - y0) + (gx1 - gx0) * (gy1 - gy0) - inter
        self.assertGreater(inter / union, 0.8)

    def test_rz_hint_near_true(self):
        world, cam = make_scene(rz=20.0)
        det = self.detector.detect(cam.render(world).image)
        self.assertIsNotNone(det.box_rz_hint_deg)
        self.assertLess(abs(det.box_rz_hint_deg - 20.0), 20.0)

    def test_empty_scene_no_crash(self):
        det = self.detector.detect(np.full((H, W, 3), 80, dtype=np.uint8))
        self.assertEqual(len(det.holes), 0)
        self.assertEqual(det.box_bbox, [])
        self.assertEqual(det.confidence, 0.0)


class TestVisionClosedLoop(unittest.TestCase):
    """W1 验收预演：粗定位误差 < 2mm（编号分配用最近邻代替 hole_assigner）。"""

    def test_coarse_localization_within_2mm(self):
        detector = GeometricHoleDetector()
        worst = 0.0
        for rz in (0.0, 20.0, -15.0):
            for seed in (0, 1):
                world, cam = make_scene(rz=rz, seed=seed)
                render = cam.render(world)
                det = detector.detect(render.image)
                self.assertEqual(len(det.holes), 4)
                chain = make_chain(world, cam.T_W_C)
                matched = match_by_nearest_pixel(det, render.hole_pixel_gt)
                centers_w = world.hole_centers_w()
                for hid, h in matched.items():
                    p_w = chain.pixel_to_world(h.pixel_u, h.pixel_v)
                    err = float(np.linalg.norm(p_w - centers_w[hid]))
                    worst = max(worst, err)
        self.assertLess(worst, 2.0)
        # 记录实际最好水平（比赛指标"视觉粗定位误差"的仿真基线）
        print(f"\n[仿真基线] 视觉粗定位最大误差: {worst:.3f} mm")


class TestDetectorInterface(unittest.TestCase):
    """接口契约：任何返回 DetectionResult 的探测器都可接入主流程。"""

    def test_stub_detector_contract(self):
        class StubDetector:
            def detect(self, image, timestamp_ms=None):
                return schemas.DetectionResult(
                    part_id="BOX_001", confidence=0.9, timestamp_ms=1.0,
                    holes=[schemas.HoleDetection(
                        id="H01", confidence=0.8,
                        pixel_u=10.0, pixel_v=20.0)])

        det = StubDetector().detect(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertIsInstance(det, schemas.DetectionResult)
        self.assertEqual(det.holes[0].pixel_u, 10.0)

    @unittest.skipUnless(
        (PROJECT_DIR / "data" / "models" / "box_yolo_seg.pt").exists(),
        "YOLO 权重不存在（W2 数据训练后生成）")
    def test_yolo_detector_on_render(self):
        """权重就绪后：YOLO 后端在仿真渲染图上也能完成检测闭环。"""
        from box_hole_inspection.vision_ai import YoloSegDetector
        det = YoloSegDetector(str(PROJECT_DIR / "data" / "models"
                                  / "box_yolo_seg.pt"))
        world, cam = make_scene()
        result = det.detect(cam.render(world).image)
        self.assertEqual(len(result.holes), 4)


if __name__ == "__main__":
    unittest.main()
