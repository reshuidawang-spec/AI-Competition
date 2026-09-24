#!/usr/bin/env python3
"""Tests for path planning: measurement waypoints, sequence, and checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.path_planner import PathPlanner
from box_hole_inspection.task_generator import TaskGenerator
from sim.sim_robot import SimRobot
from sim.sim_world import SimWorld, default_box, part_spec_from_box

SAFETY = {"stop_overtravel_reserve_mm": 2.0, "probe_step_mm": 0.5,
          "contact_speed_ratio": 0.05, "transfer_speed_ratio": 0.3}

CX, CY, CZ = 500.0, 0.0, 110.0          # 机器人B基座原点可达场景


def make_task(hole_id="H01", diameter=20.0, center_bb=None):
    spec = {"part_id": "BOX_001", "box_dims_mm": [165, 55, 110],
            "holes": [{"id": hole_id, "center_o_mm": [0.0, 0.0],
                       "diameter_mm": diameter, "tol_diameter_mm": 0.05,
                       "tol_position_mm": 0.10, "tol_perp_deg": 0.10}]}
    loc = schemas.LocalizationResult(
        part_id="BOX_001", timestamp_ms=1.0,
        holes=[schemas.LocalizedHole(
            id=hole_id, confidence=0.95, pixel_u=0.0, pixel_v=0.0,
            bb_xyz=center_bb or [CX, CY, CZ])])
    tl = TaskGenerator(spec, safety=SAFETY).generate(loc)
    return tl.tasks[0]


class TestSingleHolePlan(unittest.TestCase):
    def setUp(self):
        self.planner = PathPlanner(SimRobot(), safety=SAFETY)
        self.task = make_task()
        self.wps, self.checks = self.planner.plan_hole(self.task)

    def test_waypoint_count(self):
        # 1 safe + 2截面×(1 insert + 8 search + 1 center) + 1 safe = 22
        self.assertEqual(len(self.wps), 22)

    def test_stage_pattern(self):
        stages = [w.stage for w in self.wps]
        self.assertEqual(stages[0], "approach")
        self.assertEqual(stages[-1], "leave")
        self.assertEqual(stages.count("measure"), 20)

    def test_safe_pose_geometry(self):
        safe = self.wps[0].pose_bb
        self.assertAlmostEqual(safe.x, CX)
        self.assertAlmostEqual(safe.z, CZ + 40.0)      # safe_z_margin
        self.assertEqual(safe.rx, 180.0)               # 测针竖直向下

    def test_search_geometry(self):
        """搜索位姿在孔心；search_target 在名义触点外 overshoot 处。

        截面1 深度 z=110-6=104；搜索终点 = 孔心 + e(θ)·(R−r+3)=12mm。
        """
        w = next(w for w in self.wps if w.name == "H01_s0_a0")
        self.assertAlmostEqual(w.pose_bb.x, CX, places=6)      # 起点=孔心
        self.assertAlmostEqual(w.pose_bb.z, 104.0, places=6)
        tx, ty, tz = w.search_target_bb
        self.assertAlmostEqual(tx, CX + 12.0, places=6)
        self.assertAlmostEqual(ty, CY, places=6)
        self.assertAlmostEqual(tz, 104.0, places=6)
        w90 = next(w for w in self.wps if w.name == "H01_s0_a90")
        tx90, ty90, _ = w90.search_target_bb
        self.assertAlmostEqual(tx90, CX, places=6)
        self.assertAlmostEqual(ty90, CY + 12.0, places=6)

    def test_normal_waypoints_have_no_search_target(self):
        insert = next(w for w in self.wps if w.name == "H01_s0_insert")
        self.assertIsNone(insert.search_target_bb)

    def test_speed_ratios(self):
        self.assertAlmostEqual(self.wps[0].speed_ratio, 0.3)      # 转移
        insert = next(w for w in self.wps if w.name == "H01_s0_insert")
        self.assertAlmostEqual(insert.speed_ratio, 0.15)          # 孔内

    def test_checks_pass_in_reachable_scene(self):
        self.assertTrue(self.checks.reachable)
        self.assertTrue(self.checks.singularity_ok)
        self.assertEqual(self.checks.messages, [])


class TestFullPlan(unittest.TestCase):
    def setUp(self):
        world = SimWorld(default_box())
        world.set_box_pose(500.0, 0.0, 10.0)
        spec = part_spec_from_box(world.box)
        gen = TaskGenerator(spec, safety=SAFETY)
        loc = schemas.LocalizationResult(
            part_id="BOX_001", timestamp_ms=1.0,
            holes=[schemas.LocalizedHole(
                id=hid, confidence=0.95, pixel_u=0.0, pixel_v=0.0,
                bb_xyz=[float(v) for v in p])
                for hid, p in world.hole_centers_w().items()])
        self.tl = gen.generate(loc)
        self.planner = PathPlanner(SimRobot(), safety=SAFETY)

    def test_plan_all_holes(self):
        plan = self.planner.plan(self.tl)
        self.assertEqual(len(plan.sequence), 4)
        self.assertEqual(set(plan.sequence), {"H01", "H02", "H03", "H04"})
        self.assertEqual(len(plan.waypoints), 4 * 22)
        self.assertTrue(plan.checks.reachable)
        self.assertFalse(plan.checks.messages)

    def test_explicit_sequence_respected(self):
        plan = self.planner.plan(self.tl, sequence=["H04", "H03", "H02", "H01"])
        self.assertEqual(plan.sequence, ["H04", "H03", "H02", "H01"])
        self.assertEqual(plan.waypoints[0].hole_id, "H04")

    def test_missing_task_flagged(self):
        plan = self.planner.plan(self.tl, sequence=["H99"])
        self.assertEqual(len(plan.waypoints), 0)
        self.assertTrue(any("H99" in m for m in plan.checks.messages))


class TestUnreachableHole(unittest.TestCase):
    def test_far_hole_flagged(self):
        task = make_task(center_bb=[950.0, 0.0, 110.0])
        planner = PathPlanner(SimRobot(), safety=SAFETY)
        wps, checks = planner.plan_hole(task)
        self.assertFalse(checks.reachable)
        self.assertTrue(any("IK" in m for m in checks.messages))
        self.assertEqual(len(wps), 22)   # 位姿仍生成（供诊断）


if __name__ == "__main__":
    unittest.main()
