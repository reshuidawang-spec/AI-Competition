#!/usr/bin/env python3
"""Tests for sim_probe + probe_measurement: guarded contact measurement."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.path_planner import PathPlanner
from box_hole_inspection.probe_measurement import ProbeMeasurement
from box_hole_inspection.task_generator import TaskGenerator
from sim.sim_probe import SimProbe
from sim.sim_robot import SimRobot
from sim.sim_world import SimWorld, default_box, part_spec_from_box

SAFETY = {"stop_overtravel_reserve_mm": 2.0, "probe_step_mm": 0.5,
          "contact_speed_ratio": 0.05, "transfer_speed_ratio": 0.3}


def build_cell(box_x=500.0, box_y=0.0, box_rz=10.0, hole_diameter=20.0):
    """完整测量单元：世界 + 机器人B + 测头 + 任务 + 规划。"""
    world = SimWorld(default_box())
    world.set_box_pose(box_x, box_y, box_rz)
    if hole_diameter != 20.0:
        for h in world.box.holes:
            h.diameter_mm = hole_diameter
    robot = SimRobot()
    probe = SimProbe(world, ball_radius_mm=1.0, seed=42)
    spec = part_spec_from_box(world.box)
    gen = TaskGenerator(spec, safety=SAFETY)
    loc = schemas.LocalizationResult(
        part_id="BOX_001", timestamp_ms=1.0,
        holes=[schemas.LocalizedHole(
            id=hid, confidence=0.95, pixel_u=0.0, pixel_v=0.0,
            bb_xyz=[float(v) for v in p])
            for hid, p in world.hole_centers_w().items()])
    tl = gen.generate(loc)
    plan = PathPlanner(robot, safety=SAFETY).plan(tl)
    return world, robot, probe, tl, plan


class TestSimProbe(unittest.TestCase):
    def setUp(self):
        self.world, self.robot, self.probe, self.tl, self.plan = build_cell()

    def test_no_trigger_at_hole_center(self):
        c = self.world.hole_centers_w()["H01"]
        info = self.probe.evaluate(np.array(c))
        self.assertFalse(info.triggered)
        self.assertAlmostEqual(info.radial_clearance_mm, 9.0, places=6)
        # R=10, r_ball=1 → 球面距壁面 10-1=9

    def test_trigger_at_wall(self):
        c = self.world.hole_centers_w()["H01"]
        ball = np.array([c[0] + 9.0, c[1], c[2]])     # 球面正好触壁
        info = self.probe.evaluate(ball)
        self.assertTrue(info.triggered)
        self.assertEqual(info.hole_id, "H01")

    def test_no_trigger_above_hole(self):
        c = self.world.hole_centers_w()["H01"]
        info = self.probe.evaluate(np.array([c[0] + 9.0, c[1], c[2] + 100.0]))
        self.assertFalse(info.triggered)              # 球在孔外高度

    def test_overtravel_sampled_in_range(self):
        for _ in range(200):
            v = self.probe.sample_overtravel()
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.5)


class TestProbeMeasurement(unittest.TestCase):
    def setUp(self):
        self.world, self.robot, self.probe, self.tl, self.plan = build_cell()
        self.pm = ProbeMeasurement(self.robot, self.probe, ball_radius_mm=1.0)

    def test_measure_all_holes_all_trigger(self):
        """四个孔全部 16/16 触发（2截面×8向），接触点落在孔壁上。"""
        by_hole = {}
        for hid in self.plan.sequence:
            wps = [w for w in self.plan.waypoints if w.hole_id == hid]
            task = next(t for t in self.tl.tasks if t.hole_id == hid)
            sections = self.pm.measure_hole(wps, task)
            by_hole[hid] = sections
        for hid, sections in by_hole.items():
            self.assertEqual(len(sections), 2)
            for sec in sections:
                self.assertEqual(sec.n_triggered, 8, f"{hid} 截面触发数")
                self.assertEqual(sec.misses, [])
        # 接触点几何验证：到真实孔轴距离 ≈ R（孔壁在 R 处）
        centers = self.world.hole_centers_w()
        for hid, sections in by_hole.items():
            c = centers[hid]
            for sec in sections:
                for ev in sec.contacts:
                    rho = np.hypot(ev.contact_point_bb[0] - c[0],
                                   ev.contact_point_bb[1] - c[1])
                    self.assertAlmostEqual(rho, 10.0, delta=0.2)

    def test_contact_accuracy(self):
        """触发球心 + r_ball 的接触点应贴近真实孔壁（R=10）。"""
        hid = "H01"
        wps = [w for w in self.plan.waypoints if w.hole_id == hid]
        task = next(t for t in self.tl.tasks if t.hole_id == hid)
        sections = self.pm.measure_hole(wps, task)
        c = self.world.hole_centers_w()["H01"]
        errs = []
        for sec in sections:
            for ev in sec.contacts:
                rho = np.hypot(ev.contact_point_bb[0] - c[0],
                               ev.contact_point_bb[1] - c[1])
                errs.append(abs(rho - 10.0))
        self.assertLess(max(errs), 0.1)    # 搜索步长 0.5mm 内触发，误差 < 0.5 步

    def test_angles_parsed_from_names(self):
        hid = "H01"
        wps = [w for w in self.plan.waypoints if w.hole_id == hid]
        task = next(t for t in self.tl.tasks if t.hole_id == hid)
        sections = self.pm.measure_hole(wps, task)
        angles = [ev.angle_deg for sec in sections for ev in sec.contacts]
        self.assertEqual(len(angles), 16)
        self.assertAlmostEqual(angles[0], 0.0)
        self.assertAlmostEqual(angles[1], 45.0)


class TestProbeMeasurementRoughCenter(unittest.TestCase):
    def test_vision_error_absorbed_by_overshoot(self):
        """粗定位注入 ±2mm 误差：overshoot 3mm 仍保证全部触发。"""
        world, robot, probe, tl, plan = build_cell()
        # 给任务里的粗定位注入误差（模拟视觉误差）
        for t in tl.tasks:
            t.rough_center_bb[0] += 2.0
            t.rough_center_bb[1] -= 2.0
        plan = PathPlanner(robot, safety=SAFETY).plan(tl)
        pm = ProbeMeasurement(robot, probe, ball_radius_mm=1.0)
        total_triggered = 0
        for hid in plan.sequence:
            wps = [w for w in plan.waypoints if w.hole_id == hid]
            task = next(t for t in tl.tasks if t.hole_id == hid)
            for sec in pm.measure_hole(wps, task):
                total_triggered += sec.n_triggered
        self.assertEqual(total_triggered, 4 * 16)


if __name__ == "__main__":
    unittest.main()
