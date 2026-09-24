#!/usr/bin/env python3
"""Roundtrip tests for all JSON message schemas."""

import json
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas as S


def _full_detection():
    return S.DetectionResult(
        part_id="BOX_001", confidence=0.98, timestamp_ms=1727000000000,
        box_bbox=[120, 80, 980, 720], box_rz_hint_deg=12.3,
        holes=[S.HoleDetection(id="H01", confidence=0.97,
                               pixel_u=512.3, pixel_v=384.1,
                               pixel_bbox=[508, 380, 517, 388]),
               S.HoleDetection(id="H02", confidence=0.95,
                               pixel_u=640.0, pixel_v=400.0)])


def _full_localization():
    return S.LocalizationResult(
        part_id="BOX_001", confidence=0.98, timestamp_ms=1,
        pose=S.Pose(x=0.52, y=0.31, z=150.0, rz=12.3),
        holes=[S.LocalizedHole(id="H01", confidence=0.97,
                               pixel_u=512.3, pixel_v=384.1,
                               camera_xyz=[0.1, 0.02, 0.45],
                               world_xyz=[0.52, 0.31, 150.0],
                               bb_xyz=[0.52, 0.31, 150.0],
                               fit_residual_mm=0.6)],
        coarse_error_est_mm=2.0)


def _full_task_list():
    return S.TaskList(
        part_id="BOX_001", timestamp_ms=1,
        tasks=[S.HoleTask(
            hole_id="H01",
            nominal=S.NominalHole(id="H01", diameter_mm=20.0,
                                  center_o=[50.0, 40.0],
                                  tol_diameter_mm=0.033,
                                  tol_position_mm=0.05),
            rough_center_bb=[0.52, 0.31, 150.0],
            measure=S.MeasureConfig(
                branch="z", standard_pose="bore_z_verified",
                sections=[S.MeasureSection(z_offset_mm=0.0, n_points=8,
                                           angles_deg=[0, 45, 90, 135,
                                                       180, 225, 270, 315]),
                          S.MeasureSection(z_offset_mm=-8.0, n_points=8,
                                           angles_deg=[0, 45, 90, 135,
                                                       180, 225, 270, 315])],
                search_overshoot_mm=3.0, stop_overtravel_reserve_mm=2.0,
                probe_step_mm=0.5, contact_speed_ratio=0.05,
                setup_id="box_hole_cell_2026", trigger_io="DI1"))])


def _full_motion_plan():
    return S.MotionPlan(
        sequence=["H01", "H02"], order_method="nearest_neighbor",
        waypoints=[S.Waypoint(hole_id="H01", stage="approach",
                              name="H01_safe",
                              pose_bb=S.Pose(x=0.52, y=0.31, z=350.0),
                              joint_deg=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                              speed_ratio=0.3, blend_mm=10.0),
                   S.Waypoint(hole_id="H01", stage="measure",
                              name="H01_meas_z0",
                              pose_bb=S.Pose(x=0.52, y=0.31, z=150.0),
                              search_target_bb=[0.53, 0.31, 150.0],
                              speed_ratio=0.05)],
        checks=S.PlanChecks(reachable=True, joint_limit_ok=True,
                            singularity_ok=True, collision_ok=True,
                            messages=["ok"]))


def _full_contact_points():
    return S.ContactPoints(
        hole_id="H01", section_z_mm=0.0, n_planned=8, n_triggered=8,
        contacts=[S.ContactEvent(idx=0, angle_deg=0.0,
                                 touch_tcp_bb=[0.5325, 0.3201, 150.0],
                                 contact_point_bb=[0.5325, 0.3181, 150.0])],
        misses=[{"idx": 7, "angle_deg": 315.0, "reason": "no trigger"}])


def _full_fit_result():
    return S.FitResult(
        hole_id="H01",
        sections=[S.SectionFit(z_mm=0.0, center_bb=[0.531, 0.320, 150.0],
                               radius_mm=10.012, residual_rms_um=2.1,
                               n_points=8)],
        center_bb=[0.531, 0.320, 150.0], diameter_mm=20.024,
        axis=S.AxisFit(point=[0.531, 0.320, 146.0],
                       direction=[0.0002, -0.0003, 0.99999]))


def _full_eval_result():
    return S.EvalResult(
        holes=[S.HoleEval(id="H01", diameter_mm=20.024, dev_mm=0.024,
                          tol_mm=0.033, pass_=True, center_dev_mm=0.012,
                          position_tol_mm=0.05, position_ok=True,
                          perp_dev_deg=0.05, perp_tol_deg=0.10, perp_ok=True),
               S.HoleEval(id="H02", diameter_mm=20.010, dev_mm=0.010,
                          tol_mm=0.033, pass_=True)],
        hole_pairs=[S.HolePairEval(ids=["H01", "H02"], dist_mm=80.012,
                                   nominal_mm=80.0, dev_mm=0.012,
                                   tol_mm=0.05, pass_=True)],
        summary={"n_total": 4, "n_pass": 4, "n_fail": 0})


class TestSchemasRoundtrip(unittest.TestCase):
    def _assert_roundtrip(self, obj, cls):
        d = S.to_dict(obj)
        # JSON 序列化必须无异常（消息最终要落盘/传输）
        json.dumps(d)
        back = S.from_dict(cls, d)
        self.assertEqual(back, obj)

    def test_pose(self):
        self._assert_roundtrip(S.Pose(x=1, y=2, z=3, rx=4, ry=5, rz=6), S.Pose)

    def test_detection_result(self):
        self._assert_roundtrip(_full_detection(), S.DetectionResult)

    def test_localization_result(self):
        self._assert_roundtrip(_full_localization(), S.LocalizationResult)

    def test_task_list(self):
        self._assert_roundtrip(_full_task_list(), S.TaskList)

    def test_motion_plan(self):
        self._assert_roundtrip(_full_motion_plan(), S.MotionPlan)

    def test_contact_points(self):
        self._assert_roundtrip(_full_contact_points(), S.ContactPoints)

    def test_fit_result(self):
        self._assert_roundtrip(_full_fit_result(), S.FitResult)

    def test_eval_result(self):
        self._assert_roundtrip(_full_eval_result(), S.EvalResult)

    def test_run_report(self):
        r = S.RunReport(run_id="run_001", timestamp_ms=1.0, part_id="BOX_001",
                        metrics={"coarse_error_mm": 1.8, "n_trigger": 8},
                        eval=_full_eval_result())
        self._assert_roundtrip(r, S.RunReport)

    def test_nominal_hole_null_diameter(self):
        h = S.NominalHole(id="H01", diameter_mm=None, center_o=[None, None])
        d = S.to_dict(h)
        self.assertIsNone(d["diameter_mm"])
        self.assertEqual(S.from_dict(S.NominalHole, d), h)


class TestJsonKeyConvention(unittest.TestCase):
    def test_pass_serialised_as_pass(self):
        e = S.HoleEval(id="H01", pass_=True)
        d = S.to_dict(e)
        self.assertIn("pass", d)
        self.assertNotIn("pass_", d)
        back = S.from_dict(S.HoleEval, d)
        self.assertTrue(back.pass_)

    def test_from_dict_missing_fields_use_defaults(self):
        back = S.from_dict(S.HoleEval, {"id": "H01"})
        self.assertEqual(back.diameter_mm, 0.0)
        self.assertTrue(back.pass_)


if __name__ == "__main__":
    unittest.main()
