#!/usr/bin/env python3
"""Tests for the simulated workcell scene (ground-truth geometry)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from sim.sim_world import (
    DEFAULT_HOLE_CENTERS_O,
    DEFAULT_HOLE_DIAMETER_MM,
    SimBox,
    SimHole,
    SimWorld,
    box_from_part_yaml,
    default_box,
)


class TestSimWorldGeometry(unittest.TestCase):
    def test_default_box_has_four_holes(self):
        box = default_box()
        self.assertEqual([h.id for h in box.holes],
                         ["H01", "H02", "H03", "H04"])
        self.assertEqual(box.z_top_mm, 110.0)

    def test_hole_centers_rotation(self):
        """rz=90°: 箱体系 (-45,-12) -> 世界 (100+12, 50-45, 110) = (112, 5, 110)."""
        world = SimWorld(default_box())
        world.set_box_pose(100.0, 50.0, 90.0)
        c = world.hole_centers_w()["H01"]
        np.testing.assert_allclose(c, [112.0, 5.0, 110.0], atol=1e-9)

    def test_hole_centers_translation_only(self):
        world = SimWorld(default_box())
        world.set_box_pose(200.0, -30.0, 0.0)
        c = world.hole_centers_w()["H02"]
        np.testing.assert_allclose(c, [245.0, -42.0, 110.0], atol=1e-9)

    def test_all_holes_on_top_plane(self):
        world = SimWorld(default_box())
        world.set_box_pose(100.0, 50.0, 33.0)
        for c in world.hole_centers_w().values():
            self.assertAlmostEqual(c[2], 110.0)

    def test_top_face_corners(self):
        world = SimWorld(default_box())
        world.set_box_pose(100.0, 50.0, 0.0)
        corners = world.top_face_corners_w()
        np.testing.assert_allclose(
            sorted(map(tuple, corners)),
            [(17.5, 22.5, 110.0), (17.5, 77.5, 110.0),
             (182.5, 22.5, 110.0), (182.5, 77.5, 110.0)],
            atol=1e-9)


class TestBoxFromPartYaml(unittest.TestCase):
    def test_loads_project_config_with_fallbacks(self):
        """真实孔径/孔位为 null 时用仿真默认值补齐。"""
        box = box_from_part_yaml(PROJECT_DIR / "config" / "part_box001.yaml")
        self.assertEqual(box.part_id, "BOX_001")
        self.assertEqual(box.dims_mm, (165.0, 55.0, 110.0))
        self.assertEqual(len(box.holes), 4)
        for h in box.holes:
            self.assertEqual(h.diameter_mm, DEFAULT_HOLE_DIAMETER_MM)
            self.assertEqual(h.center_o, DEFAULT_HOLE_CENTERS_O[h.id])

    def test_real_values_override_fallbacks(self):
        """真实值填入后仿真自动跟随（不需要改代码）。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "part.yaml"
            p.write_text(
                "part_id: BOX_001\n"
                "box_dims_mm: [200, 60, 120]\n"
                "holes:\n"
                "  - {id: H01, center_o_mm: [10, 20], diameter_mm: 30.0, "
                "tol_diameter_mm: 0.05, tol_position_mm: 0.1, tol_perp_deg: 0.1}\n",
                encoding="utf-8")
            box = box_from_part_yaml(p)
            self.assertEqual(box.dims_mm, (200.0, 60.0, 120.0))
            self.assertEqual(box.holes[0].center_o, (10.0, 20.0))
            self.assertEqual(box.holes[0].diameter_mm, 30.0)


if __name__ == "__main__":
    unittest.main()
