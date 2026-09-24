#!/usr/bin/env python3
"""Tests for circle / axis least-squares fitting."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection.geometry_fitting import (
    fit_axis,
    fit_circle_geometric,
    fit_circle_kasa,
    fit_circle_section,
)


def make_noisy_circle(cx, cy, r, n=8, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
    return pts + rng.normal(0, sigma, pts.shape)


class TestCircleFitting(unittest.TestCase):
    def test_kasa_exact_circle(self):
        pts = np.stack([10.0 * np.cos(np.linspace(0, 2 * np.pi, 8,
                                                  endpoint=False)),
                        10.0 * np.sin(np.linspace(0, 2 * np.pi, 8,
                                                  endpoint=False))], axis=1)
        cx, cy, r = fit_circle_kasa(pts)
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)
        self.assertAlmostEqual(r, 10.0, places=6)

    def test_geometric_recovers_noisy_circle(self):
        """σ=10μm 噪声下：半径与圆心误差 << 噪声量级。"""
        pts = make_noisy_circle(3.0, -4.0, 10.0, n=8, sigma=0.01)
        cx, cy, r, rms = fit_circle_geometric(pts)
        self.assertAlmostEqual(cx, 3.0, delta=0.02)
        self.assertAlmostEqual(cy, -4.0, delta=0.02)
        self.assertAlmostEqual(r, 10.0, delta=0.02)
        self.assertLess(rms, 0.03)

    def test_geometric_on_partial_arc_stays_near_true(self):
        """半圆弧 + 噪声：几何拟合必须收敛到真值附近（不炸飞）。"""
        rng = np.random.default_rng(0)
        theta = np.linspace(0, np.pi, 6)         # 半圆弧
        pts = np.stack([10.0 * np.cos(theta), 10.0 * np.sin(theta)], axis=1)
        pts += rng.normal(0, 0.05, pts.shape)
        cx, cy, r, rms = fit_circle_geometric(pts)
        self.assertLess(np.hypot(cx, cy) + abs(r - 10.0), 0.1)
        self.assertLess(rms, 0.1)

    def test_section_fit_uses_mean_z(self):
        pts = np.stack([10.0 * np.cos(np.linspace(0, 2 * np.pi, 8,
                                                  endpoint=False)),
                        10.0 * np.sin(np.linspace(0, 2 * np.pi, 8,
                                                  endpoint=False)),
                        np.full(8, 104.0)], axis=1)
        center, r, rms = fit_circle_section(pts)
        self.assertAlmostEqual(center[2], 104.0)
        self.assertAlmostEqual(r, 10.0, places=6)
        self.assertLess(rms, 1e-6)

    def test_degenerate_inputs(self):
        cx, cy, r = fit_circle_kasa(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual((cx, cy, r), (0.0, 0.0, 0.0))
        center, r, rms = fit_circle_section(np.zeros((2, 3)))
        self.assertIsNone(center)


class TestAxisFitting(unittest.TestCase):
    def test_vertical_axis(self):
        pts = np.array([[5.0, -2.0, 100.0], [5.0, -2.0, 110.0]])
        point, direction = fit_axis(pts)
        np.testing.assert_allclose(point, [5.0, -2.0, 105.0])
        np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)

    def test_tilted_axis_flipped_up(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, -2.0]])  # 主轴向 -z
        point, direction = fit_axis(pts)
        self.assertGreater(direction[2], 0.0)                # 约定向上

    def test_needs_two_sections(self):
        with self.assertRaises(ValueError):
            fit_axis(np.array([[0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
