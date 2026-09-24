#!/usr/bin/env python3
"""Tests for the synthetic YOLO-Seg dataset generator."""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))   # 沿用姊妹项目惯例

import numpy as np

from gen_synthetic_dataset import main, mask_to_yolo_lines


class TestMaskToYoloLines(unittest.TestCase):
    def test_bbox_and_polygon_normalised(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:60] = True
        line = mask_to_yolo_lines(mask, 1, 100, 100)
        parts = line.split()
        self.assertEqual(parts[0], "1")
        cx, cy, bw, bh = map(float, parts[1:5])
        self.assertAlmostEqual(cx, 0.45, places=3)
        self.assertAlmostEqual(cy, 0.30, places=3)
        self.assertAlmostEqual(bw, 0.30, places=3)
        self.assertAlmostEqual(bh, 0.20, places=3)
        # 多边形点：偶数个、数量>=3、坐标在 [0,1]
        pts = list(map(float, parts[5:]))
        self.assertGreaterEqual(len(pts), 6)
        self.assertEqual(len(pts) % 2, 0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in pts))

    def test_empty_mask_gives_empty(self):
        self.assertEqual(mask_to_yolo_lines(np.zeros((10, 10), dtype=bool),
                                            0, 10, 10), "")


class TestGenerator(unittest.TestCase):
    def test_small_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--out", f"{tmp}/ds", "--n-train", "3",
                       "--n-val", "2", "--seed", "0"])
            self.assertEqual(rc, 0)
            out = Path(tmp) / "ds"
            for split, expect in (("train", 3), ("val", 2)):
                imgs = sorted((out / "images" / split).glob("*.png"))
                lbls = sorted((out / "labels" / split).glob("*.txt"))
                self.assertEqual(len(imgs), expect, split)
                self.assertEqual(len(lbls), expect, split)
                # 每张图必须恰好 5 个实例：1 box + 4 hole
                for lbl in lbls:
                    lines = lbl.read_text(encoding="utf-8").strip().splitlines()
                    self.assertEqual(len(lines), 5, lbl.name)
                    self.assertEqual([l.split()[0] for l in lines].count("0"), 1)
                    self.assertEqual([l.split()[0] for l in lines].count("1"), 4)
            self.assertTrue((out / "dataset.yaml").exists())

    def test_dataset_yaml_references_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--out", f"{tmp}/ds", "--n-train", "1", "--n-val", "1",
                  "--seed", "0"])
            text = (Path(tmp) / "ds" / "dataset.yaml").read_text(
                encoding="utf-8")
            self.assertIn("train: images/train", text)
            self.assertIn("val: images/val", text)
            self.assertIn("1: hole", text)


if __name__ == "__main__":
    unittest.main()
