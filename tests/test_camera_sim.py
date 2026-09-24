#!/usr/bin/env python3
"""Tests for the simulated file-replay camera adapter."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from interfaces.adapters.sim_camera import SimImageFileCamera


class TestSimImageFileCamera(unittest.TestCase):
    def setUp(self):
        import cv2
        self._tmp = tempfile.TemporaryDirectory()
        self.img_dir = Path(self._tmp.name)
        for i in range(2):
            img = np.zeros((64, 80, 3), dtype=np.uint8)
            img[:, :, 2] = 100 + i * 50
            cv2.imwrite(str(self.img_dir / f"frame_{i:03d}.png"), img)
        self.config = {
            "image_dir": str(self.img_dir),
            "loop": True,
            "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 40.0, "cy": 32.0,
                           "width": 80, "height": 64},
        }

    def tearDown(self):
        self._tmp.cleanup()

    def test_open_grab_loop(self):
        cam = SimImageFileCamera()
        self.assertFalse(cam.is_open())
        cam.open(self.config)
        self.assertTrue(cam.is_open())
        f0 = cam.grab()
        self.assertEqual(f0.image.shape, (64, 80, 3))
        self.assertEqual(f0.intrinsics.fx, 800.0)
        self.assertGreater(f0.timestamp_ms, 0)
        f1 = cam.grab()
        self.assertNotEqual(f1.image[0, 0, 2], f0.image[0, 0, 2])
        # loop 回绕到第一帧
        f2 = cam.grab()
        self.assertEqual(f2.image[0, 0, 2], f0.image[0, 0, 2])
        cam.close()
        self.assertFalse(cam.is_open())

    def test_no_loop_raises_after_exhaustion(self):
        cam = SimImageFileCamera()
        cam.open({**self.config, "loop": False})
        cam.grab()
        cam.grab()
        with self.assertRaises(RuntimeError):
            cam.grab()

    def test_missing_dir_raises(self):
        cam = SimImageFileCamera()
        with self.assertRaises(FileNotFoundError):
            cam.open({"image_dir": str(self.img_dir / "nope"),
                      "intrinsics": {}})

    def test_empty_dir_raises(self):
        empty = Path(tempfile.mkdtemp())
        cam = SimImageFileCamera()
        with self.assertRaises(FileNotFoundError):
            cam.open({"image_dir": str(empty), "intrinsics": {}})


if __name__ == "__main__":
    unittest.main()
