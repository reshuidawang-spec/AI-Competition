#!/usr/bin/env python3
"""Tests for the camera -> world -> robot-B coordinate chain.

Scenario: robot-A base sits at a known pose in the world frame (W ≡ robot-B
base); the camera on A's flange looks straight down at the box top-face plane
(z = 150 mm). We project known world points to pixels with the calibration,
then verify pixel_to_world() recovers them, the BB roundtrip is exact, the
plane-intersection edge cases behave, and 1 px of noise maps to a sane
millimetre scale (this is the W1 "coarse localisation error" sanity bound).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.coordinate_transform import (
    CoordinateTransform,
    load_calibration,
    CalibrationError,
)
from box_hole_inspection.utils import transforms as tf

# 台面/顶面平面
Z_TOP = 150.0
# 相机内参
FX, FY, CX, CY = 1000.0, 1000.0, 640.0, 512.0


def make_calibration(t_ea_c=None, t_ba_bb=None):
    return {
        "T_EA_C": t_ea_c or {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_EB_P": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_BA_BB": t_ba_bb or {"x": 1000, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 90},
        "camera_intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
        "table_plane_w": {"point": [0, 0, Z_TOP], "normal": [0, 0, 1]},
        "probe": {"ball_radius_mm": 1.0, "trigger_io": "DI1"},
    }


# 机器人A固定拍摄位姿（A基座系）：法兰在 (300, 100, 600)，相机光轴竖直向下
T_BA_EA = tf.pose_to_T({"x": 300, "y": 100, "z": 600,
                        "rx": 180, "ry": 0, "rz": 0})


def project_to_pixel(chain, p_w):
    """已知世界点 -> 像素（测试用投影，与 chain 逆运算互证）。"""
    T = tf.invert_T(chain.T_W_C())
    p_c = tf.apply_T(T, p_w)
    assert p_c[2] > 0, "test point must be in front of the camera"
    u = FX * p_c[0] / p_c[2] + CX
    v = FY * p_c[1] / p_c[2] + CY
    return u, v


class TestCoordinateChain(unittest.TestCase):
    def setUp(self):
        self.chain = CoordinateTransform(make_calibration(), t_ba_ea=T_BA_EA)

    def test_pixel_to_world_recovers_table_points(self):
        points = np.array([[950.0, 50.0, Z_TOP],
                           [900.0, 120.0, Z_TOP],
                           [880.0, 30.0, Z_TOP],
                           [970.0, 140.0, Z_TOP]])
        for p_w in points:
            u, v = project_to_pixel(self.chain, p_w)
            back = self.chain.pixel_to_world(u, v)
            self.assertIsNotNone(back)
            np.testing.assert_allclose(back, p_w, atol=1e-6)

    def test_world_bb_roundtrip_identity(self):
        p_w = np.array([950.0, 50.0, Z_TOP])
        np.testing.assert_allclose(self.chain.world_to_bb(p_w), p_w)
        np.testing.assert_allclose(self.chain.bb_to_world(p_w), p_w)

    def test_camera_ray_is_unit(self):
        _, d = self.chain.camera_ray(100.0, 200.0)
        self.assertAlmostEqual(np.linalg.norm(d), 1.0, places=12)

    def test_intersect_plane_parallel_returns_none(self):
        o = np.zeros(3)
        d = np.array([1.0, 0.0, 0.0])  # 与水平面平行
        self.assertIsNone(self.chain.intersect_plane(o, d))

    def test_intersect_plane_behind_camera_returns_none(self):
        o = np.array([0.0, 0.0, 600.0])
        d = np.array([0.0, 0.0, 1.0])  # 向上，背离平面
        self.assertIsNone(self.chain.intersect_plane(o, d))

    def test_intersect_plane_basic_hit(self):
        o = np.array([0.0, 0.0, 600.0])
        d = np.array([0.0, 0.0, -1.0])
        p = self.chain.intersect_plane(o, d)
        np.testing.assert_allclose(p, [0.0, 0.0, Z_TOP], atol=1e-12)

    def test_pixel_noise_scale_is_sub_millimetre(self):
        """相机距顶面 ~450mm、f=1000 → 1px ≈ 0.45mm，粗定位误差尺度合理。"""
        u, v = project_to_pixel(self.chain, np.array([950.0, 50.0, Z_TOP]))
        err = self.chain.pixel_error_to_mm(u, v, du_px=1.0, dv_px=0.0)
        self.assertIsNotNone(err)
        self.assertLess(err, 0.6)
        self.assertGreater(err, 0.3)

    def test_localize_fills_all_frames(self):
        pts = {  # 像素来自真实投影，孔中心应被精确恢复
            "H01": np.array([950.0, 50.0, Z_TOP]),
            "H02": np.array([900.0, 120.0, Z_TOP]),
        }
        det = schemas.DetectionResult(
            part_id="BOX_001", confidence=0.97, timestamp_ms=1.0,
            box_rz_hint_deg=10.0,
            holes=[schemas.HoleDetection(id=k, confidence=0.95,
                                         pixel_u=project_to_pixel(self.chain, p)[0],
                                         pixel_v=project_to_pixel(self.chain, p)[1])
                   for k, p in pts.items()])
        loc = self.chain.localize(det)
        self.assertEqual(len(loc.holes), 2)
        self.assertEqual(loc.pose.rz, 10.0)
        for h in loc.holes:
            np.testing.assert_allclose(h.world_xyz, pts[h.id], atol=1e-6)
            np.testing.assert_allclose(h.bb_xyz, pts[h.id], atol=1e-6)
            self.assertIsNotNone(h.camera_xyz)
            self.assertGreater(h.camera_xyz[2], 0)  # 相机前方


class TestNonTrivialMounting(unittest.TestCase):
    """相机相对法兰有平移和俯仰（更接近真实手眼安装）。"""

    def test_tilted_hand_eye_still_recovers_points(self):
        cal = make_calibration(
            t_ea_c={"x": 40, "y": 0, "z": -60, "rx": 0, "ry": -10, "rz": 0})
        chain = CoordinateTransform(cal, t_ba_ea=T_BA_EA)
        # 选相机正下方附近的点，保证投影在相机前方
        o_w = tf.apply_T(chain.T_W_C(), np.zeros(3))
        points = np.array([[o_w[0] + 40, o_w[1], Z_TOP],
                           [o_w[0] - 30, o_w[1] + 25, Z_TOP]])
        for p_w in points:
            u, v = project_to_pixel(chain, p_w)
            back = chain.pixel_to_world(u, v)
            self.assertIsNotNone(back)
            np.testing.assert_allclose(back, p_w, atol=1e-6)


class TestCalibrationLoading(unittest.TestCase):
    def test_load_calibration_file(self):
        cal_path = PROJECT_DIR / "config" / "calibration.json"
        cal = load_calibration(cal_path)
        self.assertIn("T_EA_C", cal)
        chain = CoordinateTransform(cal, t_ba_ea=T_BA_EA)
        self.assertIsNotNone(chain.T_W_C())

    def test_missing_file_raises(self):
        with self.assertRaises(CalibrationError):
            load_calibration(PROJECT_DIR / "config" / "does_not_exist.json")

    def test_missing_key_raises(self):
        with self.assertRaises(CalibrationError):
            CoordinateTransform({"camera_intrinsics": {"fx": 1, "fy": 1,
                                                       "cx": 0, "cy": 0},
                                 "table_plane_w": {"point": [0, 0, 0],
                                                   "normal": [0, 0, 1]}},
                                t_ba_ea=T_BA_EA)

    def test_missing_robot_a_pose_raises(self):
        with self.assertRaises(CalibrationError):
            CoordinateTransform(make_calibration()).T_W_EA()


if __name__ == "__main__":
    unittest.main()
