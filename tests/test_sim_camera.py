#!/usr/bin/env python3
"""Tests for the simulated pinhole camera — including the KEY closed-loop
check: render -> ground-truth pixel -> coordinate chain -> world point."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection.coordinate_transform import CoordinateTransform
from box_hole_inspection.utils import transforms as tf
from interfaces.camera import CameraIntrinsics
from sim.sim_camera import SimCamera
from sim.sim_robot import SimRobot
from sim.sim_world import SimWorld, default_box

FX, FY, CX, CY = 1000.0, 1000.0, 640.0, 512.0
W, H = 1280, 1024


def make_intrinsics():
    return CameraIntrinsics(fx=FX, fy=FY, cx=CX, cy=CY, width=W, height=H)


def make_scene(box_x=900.0, box_y=0.0, box_rz=20.0,
               cam_ry_deg=-8.0):
    """箱体在 (900,0)，相机在其上方 480mm 俯视（带小俯仰）。"""
    world = SimWorld(default_box())
    world.set_box_pose(box_x, box_y, box_rz)
    t_w_c = tf.pose_to_T({"x": box_x, "y": box_y, "z": 480.0,
                          "rx": 180.0, "ry": cam_ry_deg, "rz": 0.0})
    cam = SimCamera(make_intrinsics(), t_w_c, seed=0)
    return world, cam


def make_chain(world, t_w_c):
    """与渲染同参数的坐标链（T_BA_BB=I, T_EA_C=I, 固定拍摄位姿 T_BA_EA=T_W_C）。"""
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


class TestRendering(unittest.TestCase):
    def setUp(self):
        self.world, self.cam = make_scene()
        self.render = self.cam.render(self.world)

    def test_image_shape_and_type(self):
        img = self.render.image
        self.assertEqual(img.shape, (H, W, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_box_brighter_than_table(self):
        img = self.render.image
        table_px = float(img[100, 100, 0])
        face_center = self.cam.project(
            np.array([900.0, 0.0, self.world.box.z_top_mm]))
        box_px = float(img[int(face_center[1]), int(face_center[0]), 0])
        self.assertGreater(box_px, table_px + 50)

    def test_hole_center_is_dark(self):
        for hid, (u, v) in self.render.hole_pixel_gt.items():
            if not self.render.hole_visible[hid]:
                continue
            self.assertLess(float(self.render.image[int(v), int(u), 0]), 80)

    def test_all_holes_visible_in_clean_view(self):
        self.assertTrue(all(self.render.hole_visible.values()))

    def test_box_bbox_reasonable(self):
        x0, y0, x1, y1 = self.render.box_bbox_px
        self.assertGreater(x1 - x0, 100)
        self.assertGreater(y1 - y0, 30)
        self.assertTrue(0 <= x0 < x1 < W and 0 <= y0 < y1 < H)


class TestClosedLoopRenderToWorld(unittest.TestCase):
    """核心闭环：渲染图像的真值像素 → 坐标链 → 世界点 ≈ 场景真值。"""

    def _assert_roundtrip(self, rz, ry=-8.0, atol_mm=0.1):
        world, cam = make_scene(box_rz=rz, cam_ry_deg=ry)
        chain = make_chain(world, cam.T_W_C)
        render = cam.render(world)
        centers_w = world.hole_centers_w()
        for hid, (u, v) in render.hole_pixel_gt.items():
            if not render.hole_visible[hid]:
                continue
            p_w = chain.pixel_to_world(u, v)
            self.assertIsNotNone(p_w)
            np.testing.assert_allclose(p_w, centers_w[hid], atol=atol_mm)

    def test_roundtrip_straight_down(self):
        self._assert_roundtrip(rz=0.0, ry=0.0)

    def test_roundtrip_tilted_camera(self):
        self._assert_roundtrip(rz=0.0, ry=-8.0)

    def test_roundtrip_rotated_box(self):
        self._assert_roundtrip(rz=20.0, ry=-8.0)

    def test_roundtrip_moved_box(self):
        """实验2场景：箱体被移动+旋转后，链仍自洽（这是仿真的前提）。"""
        world, cam = make_scene(box_x=870.0, box_y=35.0, box_rz=-15.0)
        chain = make_chain(world, cam.T_W_C)
        render = cam.render(world)
        for hid, (u, v) in render.hole_pixel_gt.items():
            p_w = chain.pixel_to_world(u, v)
            np.testing.assert_allclose(p_w, world.hole_centers_w()[hid],
                                       atol=0.1)


class TestMasks(unittest.TestCase):
    def test_hole_masks_match_geometry(self):
        world, cam = make_scene()
        render = cam.render(world)
        for hid, mask in render.masks.items():
            if hid == "box":
                continue
            u, v = render.hole_pixel_gt[hid]
            self.assertTrue(mask[int(v), int(u)], f"{hid} mask miss at centre")
            # 各孔掩膜互不重叠
            for other, m2 in render.masks.items():
                if other not in ("box", hid):
                    self.assertFalse(
                        np.any(mask & m2), f"{hid}/{other} masks overlap")

    def test_box_mask_covers_face_centre(self):
        world, cam = make_scene()
        render = cam.render(world)
        u, v = cam.project(np.array([900.0, 0.0, world.box.z_top_mm]))
        self.assertTrue(render.masks["box"][int(v), int(u)])


class TestNoiseAndDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        world, _ = make_scene()
        cam1 = SimCamera(make_intrinsics(), make_scene()[1].T_W_C, seed=7)
        cam2 = SimCamera(make_intrinsics(), cam1.T_W_C, seed=7)
        np.testing.assert_array_equal(cam1.render(world).image,
                                      cam2.render(world).image)

    def test_different_seed_differs(self):
        world, cam = make_scene()
        cam2 = SimCamera(make_intrinsics(), cam.T_W_C, seed=8)
        d = np.abs(cam.render(world).image.astype(int)
                   - cam2.render(world).image.astype(int))
        self.assertGreater(d.max(), 0)


class TestVisibility(unittest.TestCase):
    def test_camera_below_top_face_sees_nothing(self):
        world = SimWorld(default_box())
        world.set_box_pose(900.0, 0.0, 0.0)
        t_w_c = tf.pose_to_T({"x": 900, "y": 0, "z": 50,
                              "rx": 180, "ry": 0, "rz": 0})
        cam = SimCamera(make_intrinsics(), t_w_c)
        render = cam.render(world)
        self.assertFalse(any(render.hole_visible.values()))
        self.assertEqual(render.hole_pixel_gt, {})


class TestFromRobotA(unittest.TestCase):
    def test_chain_matches_robot_state(self):
        """相机位姿 = 视觉臂基座位姿 @ FK(拍摄关节) @ 手眼。"""
        base = tf.pose_to_T({"x": 1000, "y": 0, "z": 0,
                             "rx": 0, "ry": 0, "rz": 90})
        robot = SimRobot(base_pose_w=base,
                         start_joints_deg=[0, -30, 60, 0, -30, 0])
        t_ea_c = tf.pose_to_T({"x": 0, "y": 0, "z": -50,
                               "rx": 0, "ry": 0, "rz": 0})
        cam = SimCamera.from_robot_a(robot, t_ea_c,
                                     capture_joints_deg=[0, -30, 60, 0, -30, 0])
        expected = tf.compose(tf.compose(base, robot.fk(robot.get_joint_angles())),
                              t_ea_c)
        np.testing.assert_allclose(cam.T_W_C, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
