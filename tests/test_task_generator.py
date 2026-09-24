#!/usr/bin/env python3
"""Tests for task_generator: strategy selection and TaskList generation."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.task_generator import (
    TaskGenerator,
    load_part_spec,
)
from sim.sim_world import SimWorld, default_box, part_spec_from_box

SAFETY = {"stop_overtravel_reserve_mm": 2.0, "probe_step_mm": 0.5,
          "contact_speed_ratio": 0.05, "trigger_io": "DI1"}


def make_localization(holes, part_id="BOX_001"):
    return schemas.LocalizationResult(
        part_id=part_id, confidence=0.9, timestamp_ms=1.0,
        holes=[schemas.LocalizedHole(
            id=hid, confidence=0.95, pixel_u=u, pixel_v=v,
            bb_xyz=xyz) for hid, u, v, xyz in holes])


def spec_with_diameters(diams):
    return {"part_id": "BOX_001", "box_dims_mm": [165, 55, 110],
            "holes": [{"id": f"H{i+1:02d}",
                       "center_o_mm": [10.0 * i, 20.0],
                       "diameter_mm": d,
                       "tol_diameter_mm": 0.05,
                       "tol_position_mm": 0.10,
                       "tol_perp_deg": 0.10}
                      for i, d in enumerate(diams)]}


class TestBranchSelection(unittest.TestCase):
    def setUp(self):
        self.gen = TaskGenerator(spec_with_diameters([20.0, 80.0, 65.0, 64.9]),
                                 safety=SAFETY)

    def test_small_hole_uses_axial(self):
        self.assertEqual(self.gen.pick_branch(20.0), "z")
        self.assertEqual(self.gen.pick_branch(64.9), "z")

    def test_large_hole_uses_cross(self):
        self.assertEqual(self.gen.pick_branch(80.0), "cross")
        self.assertEqual(self.gen.pick_branch(65.0), "cross")

    def test_unknown_diameter_returns_none(self):
        self.assertIsNone(self.gen.pick_branch(None))
        self.assertIsNone(self.gen.pick_branch(0.0))

    def test_sections_per_branch(self):
        axial = self.gen.make_sections("z")
        self.assertEqual(len(axial), 2)
        self.assertEqual(len(axial[0].angles_deg), 8)
        np.testing.assert_allclose(axial[0].angles_deg,
                                   np.linspace(0, 360, 8, endpoint=False))
        cross = self.gen.make_sections("cross")
        self.assertEqual(len(cross[0].angles_deg), 4)
        self.assertEqual(cross[0].angles_deg, [0.0, 90.0, 180.0, 270.0])

    def test_section_depth_offsets(self):
        axial = self.gen.make_sections("z")
        self.assertEqual([s.z_offset_mm for s in axial], [-6.0, -18.0])


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.spec = spec_with_diameters([20.0, 80.0, 20.0, None])
        self.gen = TaskGenerator(self.spec, safety=SAFETY)
        self.loc = make_localization([
            ("H01", 100.0, 200.0, [900.0, 0.0, 110.0]),
            ("H02", 300.0, 200.0, [930.0, 5.0, 110.0]),
            ("H03", 500.0, 200.0, [960.0, -5.0, 110.0]),
            ("H04", 700.0, 200.0, [990.0, 8.0, 110.0]),
        ])

    def test_generates_tasks_with_strategy(self):
        tl = self.gen.generate(self.loc)
        self.assertEqual(len(tl.tasks), 3)          # H04 直径未知被跳过
        by_id = {t.hole_id: t for t in tl.tasks}
        self.assertEqual(by_id["H01"].measure.branch, "z")
        self.assertEqual(len(by_id["H01"].measure.sections[0].angles_deg), 8)
        self.assertEqual(by_id["H02"].measure.branch, "cross")
        self.assertEqual(len(by_id["H02"].measure.sections[0].angles_deg), 4)

    def test_skipped_recorded_explicitly(self):
        tl = self.gen.generate(self.loc)
        self.assertEqual(len(tl.skipped), 1)
        self.assertEqual(tl.skipped[0]["id"], "H04")
        self.assertIn("diameter", tl.skipped[0]["reason"])

    def test_rough_center_and_tolerances_carried(self):
        tl = self.gen.generate(self.loc)
        t = next(t for t in tl.tasks if t.hole_id == "H01")
        np.testing.assert_allclose(t.rough_center_bb, [900.0, 0.0, 110.0])
        self.assertEqual(t.nominal.diameter_mm, 20.0)
        self.assertEqual(t.nominal.tol_diameter_mm, 0.05)
        self.assertEqual(t.nominal.tol_perp_deg, 0.10)

    def test_safety_params_from_config(self):
        tl = self.gen.generate(self.loc)
        m = tl.tasks[0].measure
        self.assertEqual(m.stop_overtravel_reserve_mm, 2.0)
        self.assertEqual(m.probe_step_mm, 0.5)
        self.assertEqual(m.contact_speed_ratio, 0.05)
        self.assertEqual(m.trigger_io, "DI1")
        self.assertEqual(m.search_overshoot_mm, 3.0)
        self.assertEqual(m.setup_id, "box_hole_cell_2026")

    def test_unmatched_hole_skipped(self):
        loc = make_localization([("HXX", 0.0, 0.0, [0.0, 0.0, 0.0])])
        tl = self.gen.generate(loc)
        self.assertEqual(len(tl.tasks), 0)
        self.assertEqual(tl.skipped[0]["id"], "HXX")
        self.assertIn("no nominal", tl.skipped[0]["reason"])

    def test_missing_bb_skipped(self):
        loc = schemas.LocalizationResult(
            part_id="BOX_001", timestamp_ms=1.0,
            holes=[schemas.LocalizedHole(id="H01", confidence=0.9,
                                         pixel_u=1.0, pixel_v=1.0,
                                         bb_xyz=None)])
        tl = self.gen.generate(loc)
        self.assertEqual(len(tl.tasks), 0)
        self.assertIn("no bb", tl.skipped[0]["reason"])

    def test_tasklist_json_roundtrip(self):
        import json
        tl = self.gen.generate(self.loc)
        json.dumps(schemas.to_dict(tl))
        back = schemas.from_dict(schemas.TaskList, schemas.to_dict(tl))
        self.assertEqual(back, tl)


class TestPartSpecIntegration(unittest.TestCase):
    def test_sim_box_to_part_spec(self):
        box = default_box()
        spec = part_spec_from_box(box)
        gen = TaskGenerator(spec, safety=SAFETY)
        loc = make_localization(
            [(h.id, float(i) * 100, 100.0,
              [900.0 + i * 30, 0.0, 110.0]) for i, h in enumerate(box.holes)])
        tl = gen.generate(loc)
        self.assertEqual(len(tl.tasks), 4)
        self.assertTrue(all(t.measure.branch == "z" for t in tl.tasks))

    def test_load_project_part_yaml(self):
        """真实配置（孔径全 null）→ 所有孔跳过并给出明确原因。"""
        spec = load_part_spec(PROJECT_DIR / "config" / "part_box001.yaml")
        gen = TaskGenerator(spec, safety=SAFETY)
        loc = make_localization(
            [("H01", 0.0, 0.0, [0.0, 0.0, 0.0])])
        tl = gen.generate(loc)
        self.assertEqual(len(tl.tasks), 0)
        self.assertEqual(len(tl.skipped), 1)
        self.assertIn("diameter", tl.skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()
