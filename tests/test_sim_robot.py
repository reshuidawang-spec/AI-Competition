#!/usr/bin/env python3
"""Tests for the simulated CR5 robot (real-URDF kinematics + numerical IK)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.utils import transforms as tf
from sim.sim_robot import (
    CR5_JOINTS,
    SINGULAR_COND_THRESHOLD,
    SimRobot,
    _Kinematics,
)


# ---------------------------------------------------------------------------
# 测试内独立实现：基础旋转矩阵与链乘（与 transforms 模块互为独立路径）
# ---------------------------------------------------------------------------

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _urdf_static(xyz_mm, rpy_deg):
    """URDF 固定轴 xyz rpy == 内旋 sxyz: Rz(rz)Ry(ry)Rx(rx)."""
    x, y, z = xyz_mm
    rx, ry, rz = map(math.radians, rpy_deg)
    T = np.eye(4)
    T[:3, :3] = _rz(rz) @ _ry(ry) @ _rx(rx)
    T[:3, 3] = [x, y, z]
    return T


def _manual_fk(q_deg):
    """独立链乘实现（与 sim_robot.fk 互证）。"""
    T = np.eye(4)
    for (_, xyz, rpy, _, _), qi in zip(CR5_JOINTS, q_deg):
        T = T @ _urdf_static(xyz, rpy)
        a = math.radians(qi)
        c, s = math.cos(a), math.sin(a)
        T = T @ np.array([[c, -s, 0, 0], [s, c, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]])
    return T


class TestFK(unittest.TestCase):
    def test_zero_pose_regression_pin(self):
        """q=0 法兰位姿 = 真实 CR5 URDF 链的特征值（2026-09-24 提取）。

        若该值变化，说明运动学规格被改，必须先核对 URDF 来源。
        """
        pose = SimRobot().get_flange_pose()
        self.assertAlmostEqual(pose.x, 0.0, places=6)
        self.assertAlmostEqual(pose.y, -246.0, places=1)
        self.assertAlmostEqual(pose.z, 1047.0, places=1)
        self.assertAlmostEqual(pose.rx, -90.0, places=6)
        self.assertAlmostEqual(pose.ry, 0.0, places=6)
        self.assertAlmostEqual(pose.rz, -180.0, places=6)

    def test_fk_matches_independent_chain(self):
        kin = _Kinematics(CR5_JOINTS)
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.uniform(-90, 90, size=6)
            np.testing.assert_allclose(kin.fk(np.radians(q)),
                                       _manual_fk(q), atol=1e-9)

    def test_reach_height_sanity(self):
        """法兰最高可达 ~1m 级（CR5 量级），防规格整体错位。"""
        pose = SimRobot().get_flange_pose()
        self.assertGreater(pose.z, 900.0)
        self.assertLess(pose.z, 1200.0)


class TestIK(unittest.TestCase):
    def setUp(self):
        self.kin = _Kinematics(CR5_JOINTS)
        self.rng = np.random.default_rng(7)

    def test_roundtrip_30_random_configs(self):
        """FK 生成的目标位姿，IK 必须全部收敛（往返一致性）。"""
        lo, hi = self.kin.lower * 0.8, self.kin.upper * 0.8
        for _ in range(30):
            q = self.rng.uniform(lo, hi)
            T = self.kin.fk(q)
            q2, singular = self.kin.ik(T, rng=self.rng)
            self.assertIsNotNone(q2, "IK 必须收敛于 FK 生成的位姿")
            T2 = self.kin.fk(q2)
            pos_err = np.linalg.norm(T2[:3, 3] - T[:3, 3])
            R_err = T2[:3, :3] @ T[:3, :3].T
            ang = math.acos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            self.assertLess(pos_err, 0.1)
            self.assertLess(math.degrees(ang), 0.2)
            self.assertFalse(singular)

    def test_unreachable_pose_fails(self):
        T = tf.pose_to_T({"x": 2000, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0})
        q, _ = self.kin.ik(T, rng=self.rng)
        self.assertIsNone(q)

    def test_ik_respects_joint_limits(self):
        q0 = np.radians([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 目标在可达域外（2m），即使给可行初值也必须失败而不是越限求解
        T = tf.pose_to_T({"x": 2000, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0})
        q, _ = self.kin.ik(T, seed_rad=q0, rng=self.rng)
        self.assertIsNone(q)


class TestSingularity(unittest.TestCase):
    def setUp(self):
        self.kin = _Kinematics(CR5_JOINTS)

    def test_wrist_singularity_separates(self):
        """q5=0（4/6 轴重合）条件数必须远超阈值；正常构型必须低于阈值。"""
        q_sing = np.radians([10, -30, 40, 20, 0, 30])
        q_norm = np.radians([10, -30, 40, 20, 60, 30])
        self.assertGreater(self.kin.condition_number(q_sing),
                           SINGULAR_COND_THRESHOLD * 100)
        self.assertLess(self.kin.condition_number(q_norm),
                        SINGULAR_COND_THRESHOLD)


class TestSimRobotAdapter(unittest.TestCase):
    def setUp(self):
        self.robot = SimRobot()

    def test_move_j_updates_state_and_log(self):
        self.robot.move_j([10.0, -20.0, 30.0, 40.0, -50.0, 60.0],
                          speed_ratio=0.3)
        self.assertEqual(self.robot.get_joint_angles(),
                         [10.0, -20.0, 30.0, 40.0, -50.0, 60.0])
        self.assertEqual(len(self.robot.log), 1)
        self.assertGreater(self.robot.sim_time_ms, 0)

    def test_move_j_clamps_to_limits(self):
        self.robot.move_j([999.0, 0, 0, 0, 0, 0])
        self.assertLessEqual(self.robot.get_joint_angles()[0], 359.2 + 1e-9)

    def test_move_l_roundtrip(self):
        """move_l 后法兰位姿必须等于目标位姿（6R 臂关节解不唯一，
        只能对比位姿，不能对比关节角）。"""
        q_target = [10.0, -25.0, 40.0, 15.0, -30.0, 60.0]
        T = self.robot.fk(q_target)
        pose = schemas.Pose(**tf.T_to_pose(T))
        self.robot.move_l(pose, speed_ratio=0.3)
        reached = tf.pose_to_T({"x": self.robot.get_flange_pose().x,
                                "y": self.robot.get_flange_pose().y,
                                "z": self.robot.get_flange_pose().z,
                                "rx": self.robot.get_flange_pose().rx,
                                "ry": self.robot.get_flange_pose().ry,
                                "rz": self.robot.get_flange_pose().rz})
        np.testing.assert_allclose(reached[:3, 3], T[:3, 3], atol=1.0)
        R_err = reached[:3, :3] @ T[:3, :3].T
        ang = math.acos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        self.assertLess(math.degrees(ang), 1.0)

    def test_move_l_unreachable_raises(self):
        far = schemas.Pose(x=2000.0, y=0.0, z=0.0)
        with self.assertRaises(RuntimeError):
            self.robot.move_l(far)

    def test_estop_blocks_motion(self):
        self.robot.estop()
        self.assertTrue(self.robot.is_estop())
        with self.assertRaises(RuntimeError):
            self.robot.move_j([0, 0, 0, 0, 0, 0])
        with self.assertRaises(RuntimeError):
            self.robot.move_l(schemas.Pose(x=0, y=0, z=500))

    def test_status(self):
        s = self.robot.status()
        self.assertEqual(s.state, "idle")
        self.assertEqual(len(s.joint_deg), 6)
        self.assertFalse(s.estop)

    def test_check_pose_ok_and_unreachable(self):
        T = self.robot.fk([10.0, -25.0, 40.0, 15.0, -30.0, 60.0])
        pose_ok = schemas.Pose(**tf.T_to_pose(T))
        checks = self.robot.check_pose(pose_ok)
        self.assertTrue(checks.reachable)
        self.assertTrue(checks.singularity_ok)
        self.assertTrue(self.robot.reachable(pose_ok))
        checks_far = self.robot.check_pose(schemas.Pose(x=2000, y=0, z=0))
        self.assertFalse(checks_far.reachable)
        self.assertTrue(any("IK" in m for m in checks_far.messages))

    def test_base_pose_in_world(self):
        base = tf.pose_to_T({"x": 1000, "y": 0, "z": 0,
                             "rx": 0, "ry": 0, "rz": 90})
        robot = SimRobot(base_pose_w=base,
                         start_joints_deg=[10.0, -25.0, 40.0, 15.0, -30.0, 60.0])
        T_expected = tf.compose(base, robot.fk(robot.get_joint_angles()))
        pose_w = robot.flange_pose_w()
        np.testing.assert_allclose(
            [pose_w.x, pose_w.y, pose_w.z], T_expected[:3, 3], atol=1e-9)
        # 自身基座系位姿不受 base_pose_w 影响
        own = robot.get_flange_pose()
        self.assertNotAlmostEqual(own.x, pose_w.x)


if __name__ == "__main__":
    unittest.main()
