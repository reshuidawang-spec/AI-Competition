#!/usr/bin/env python3
"""Tests for hole measurement order planning (nearest neighbour / 2-opt)."""

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection.sequence_planner import (
    order_nearest_neighbor,
    order_tsp_2opt,
)


def length(order, points, start=(0.0, 0.0)):
    total, prev = 0.0, list(start)
    for hid in order:
        p = points[hid]
        total += ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
        prev = [p[0], p[1]]
    return total


class TestNearestNeighbor(unittest.TestCase):
    def test_greedy_chain(self):
        points = {"A": [100.0, 0.0, 110.0],
                  "B": [10.0, 0.0, 110.0],
                  "C": [50.0, 0.0, 110.0]}
        self.assertEqual(order_nearest_neighbor(points, start=[0, 0]),
                         ["B", "C", "A"])

    def test_empty(self):
        self.assertEqual(order_nearest_neighbor({}), [])

    def test_single(self):
        self.assertEqual(order_nearest_neighbor({"A": [5.0, 5.0, 0.0]}), ["A"])


class TestTwoOpt(unittest.TestCase):
    def test_improves_crossed_path(self):
        # 暴力搜索找到的真实反例：NN 431.1 -> 2-opt 373.0（2-opt 翻转修复
        # 交叉路径）。
        points = {"P0": [-83.2, 66.5, 0.0],
                  "P1": [57.4, -52.1, 0.0],
                  "P2": [75.3, -88.3, 0.0],
                  "P3": [-32.8, -69.9, 0.0]}
        nn = order_nearest_neighbor(points, start=[0, 0])
        opt = order_tsp_2opt(points, start=[0, 0])
        self.assertEqual(nn, ["P3", "P1", "P2", "P0"])
        self.assertLess(length(opt, points), length(nn, points))
        self.assertEqual(opt, ["P1", "P2", "P3", "P0"])

    def test_small_inputs_unchanged(self):
        points = {"A": [1.0, 0.0, 0.0], "B": [2.0, 0.0, 0.0]}
        self.assertEqual(order_tsp_2opt(points), ["A", "B"])


if __name__ == "__main__":
    unittest.main()
