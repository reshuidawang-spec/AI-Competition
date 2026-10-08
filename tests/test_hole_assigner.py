#!/usr/bin/env python3
"""Tests for hole ID assignment / box pose via CAD pattern matching."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.hole_assigner import (
    HoleAssigner,
    HoleAssignmentError,
    umeyama_2d,
)

SPEC = {"part_id": "BOX_001", "box_dims_mm": [165, 55, 110],
        "holes": [{"id": "H01", "center_o_mm": [-45.0, -12.0]},
                  {"id": "H02", "center_o_mm": [45.0, -12.0]},
                  {"id": "H03", "center_o_mm": [-45.0, 12.0]},
                  {"id": "H04", "center_o_mm": [45.0, 12.0]}]}


def make_localization(world_pts, ids=None):
    ids = ids or [f"H{i:02d}" for i in range(1, len(world_pts) + 1)]
    return schemas.LocalizationResult(
        part_id="BOX_001", confidence=0.9, timestamp_ms=1.0,
        pose=schemas.Pose(z=110.0),
        holes=[schemas.LocalizedHole(
            id=hid, confidence=0.95, pixel_u=0.0, pixel_v=0.0,
            world_xyz=[float(v) for v in p], bb_xyz=[float(v) for v in p])
            for hid, p in zip(ids, world_pts)])


def true_world_points(x=540.0, y=30.0, rz=-10.0):
    a = math.radians(rz)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    out = []
    for c in [h["center_o_mm"] for h in SPEC["holes"]]:
        p = R @ np.array(c) + np.array([x, y])
        out.append([p[0], p[1], 110.0])
    return out


class TestUmeyama(unittest.TestCase):
    def test_exact_recovery(self):
        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        a = math.radians(30.0)
        R = np.array([[math.cos(a), -math.sin(a)],
                      [math.sin(a), math.cos(a)]])
        dst = src @ R.T + np.array([10.0, -5.0])
        theta, tx, ty, rms = umeyama_2d(src, dst)
        self.assertAlmostEqual(theta, 30.0, places=6)
        self.assertAlmostEqual(tx, 10.0, places=6)
        self.assertAlmostEqual(ty, -5.0, places=6)
        self.assertAlmostEqual(rms, 0.0, places=9)


class TestAssign(unittest.TestCase):
    def setUp(self):
        self.assigner = HoleAssigner(SPEC)

    def test_recovers_pose_and_ids_with_noise(self):
        rng = np.random.default_rng(0)
        pts = np.array(true_world_points(540.0, 30.0, -10.0))
        pts[:, :2] += rng.normal(0, 0.3, (len(pts), 2))
        loc = make_localization(pts)
        out = self.assigner.assign(loc)
        self.assertEqual([h.id for h in out.holes],
                         ["H01", "H02", "H03", "H04"])
        self.assertAlmostEqual(out.pose.x, 540.0, delta=0.5)
        self.assertAlmostEqual(out.pose.y, 30.0, delta=0.5)
        self.assertAlmostEqual(out.pose.rz, -10.0, delta=1.0)
        self.assertLess(out.coarse_error_est_mm, 0.5)

    def test_input_order_irrelevant(self):
        pts = true_world_points(500.0, 0.0, 10.0)
        shuffled = [pts[3], pts[0], pts[2], pts[1]]
        out = self.assigner.assign(make_localization(shuffled))
        # 输入顺序打乱，输出编号必须恢复
        self.assertEqual([h.id for h in out.holes],
                         ["H01", "H02", "H03", "H04"])

    def test_symmetric_pattern_tiebreak_prefers_rz_range(self):
        """对称孔模式：rz=-10 与 rz=170 两个等 RMS 解，平局裁决必须取 -10。"""
        pts = true_world_points(540.0, 30.0, -10.0)
        out = self.assigner.assign(make_localization(pts))
        self.assertAlmostEqual(out.pose.rz, -10.0, places=3)
        self.assertEqual([h.id for h in out.holes],
                         ["H01", "H02", "H03", "H04"])

    def test_count_mismatch_raises(self):
        pts = true_world_points()[:3]
        with self.assertRaises(HoleAssignmentError):
            self.assigner.assign(make_localization(pts))

    def test_wrong_part_pattern_is_rejected(self):
        loc = make_localization(true_world_points())
        loc.part_id = "ENGINE_001"
        with self.assertRaisesRegex(HoleAssignmentError, "工件 ID 不匹配"):
            self.assigner.assign(loc)

    def test_residuals_attached(self):
        pts = true_world_points()
        out = self.assigner.assign(make_localization(pts))
        for h in out.holes:
            self.assertIsNotNone(h.fit_residual_mm)
            self.assertLess(h.fit_residual_mm, 1e-3)


if __name__ == "__main__":
    unittest.main()
