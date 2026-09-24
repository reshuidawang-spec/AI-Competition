#!/usr/bin/env python3
"""Unit tests for pose/transform utilities (sxyz intrinsic RPY convention)."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection.utils import transforms as tf


class TestElementaryRotations(unittest.TestCase):
    def test_rot_z_90_maps_x_to_y(self):
        v = tf.apply_R(tf.rot_z(np.pi / 2), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-12)

    def test_rot_x_90_maps_y_to_z(self):
        v = tf.apply_R(tf.rot_x(np.pi / 2), np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-12)

    def test_rot_y_90_maps_z_to_x(self):
        v = tf.apply_R(tf.rot_y(np.pi / 2), np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-12)


class TestEulerRoundtrip(unittest.TestCase):
    def test_roundtrip_random_poses(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            angles = rng.uniform(-60.0, 60.0, size=3)  # 避开万向锁附近
            R = tf.euler_to_R(*angles)
            back = tf.R_to_euler(R)
            np.testing.assert_allclose(back, angles, atol=1e-9)

    def test_identity(self):
        np.testing.assert_allclose(tf.euler_to_R(0, 0, 0), np.eye(3), atol=1e-15)


class TestPoseConversion(unittest.TestCase):
    def test_pose_to_T_to_pose_roundtrip(self):
        pose = {"x": 12.3, "y": -45.6, "z": 78.9, "rx": 10.0, "ry": -20.0, "rz": 30.0}
        T = tf.pose_to_T(pose)
        back = tf.T_to_pose(T)
        for k in ("x", "y", "z"):
            self.assertAlmostEqual(back[k], pose[k], places=10)
        for k in ("rx", "ry", "rz"):
            self.assertAlmostEqual(back[k], pose[k], places=8)


class TestTransformAlgebra(unittest.TestCase):
    def test_compose_with_identity(self):
        T = tf.pose_to_T({"x": 1, "y": 2, "z": 3, "rx": 5, "ry": 7, "rz": 11})
        np.testing.assert_allclose(tf.compose(T, np.eye(4)), T, atol=1e-12)
        np.testing.assert_allclose(tf.compose(np.eye(4), T), T, atol=1e-12)

    def test_inverse(self):
        T = tf.pose_to_T({"x": 1, "y": 2, "z": 3, "rx": 5, "ry": 7, "rz": 11})
        np.testing.assert_allclose(tf.compose(T, tf.invert_T(T)),
                                   np.eye(4), atol=1e-12)
        np.testing.assert_allclose(tf.compose(tf.invert_T(T), T),
                                   np.eye(4), atol=1e-12)

    def test_apply_T_single_and_batch(self):
        T = tf.pose_to_T({"x": 10, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 90})
        p1 = tf.apply_T(T, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(p1, [10.0, 1.0, 0.0], atol=1e-12)
        # rz=90°: (0,2,0) -> (-2,0,0)，加平移 (10,0,0) -> (8,0,0)
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        out = tf.apply_T(T, pts)
        np.testing.assert_allclose(out[1], [8.0, 0.0, 0.0], atol=1e-12)

    def test_is_identity(self):
        self.assertTrue(tf.is_identity(np.eye(4)))
        T = tf.pose_to_T({"x": 1, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0})
        self.assertFalse(tf.is_identity(T))


class TestNormalize(unittest.TestCase):
    def test_unit_vector(self):
        v = tf.normalize(np.array([3.0, 0.0, 4.0]))
        np.testing.assert_allclose(v, [0.6, 0.0, 0.8], atol=1e-15)

    def test_zero_stays_zero(self):
        v = tf.normalize(np.zeros(3))
        np.testing.assert_allclose(v, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
