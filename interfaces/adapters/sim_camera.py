"""Simulated camera: replays image files from a directory.

Used in W1/W2 before (and alongside) the real camera: feed it rendered or
captured sample images of the box top face. Timestamps come from file mtime.

Config keys (see config/camera.yaml):
    image_dir:  directory to replay
    patterns:   glob patterns, default ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    loop:       wrap around at the end of the directory (default True)
    intrinsics: dict matching CameraIntrinsics fields
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from interfaces.camera import CameraBase, CameraFrame, CameraIntrinsics


class SimImageFileCamera(CameraBase):
    def __init__(self):
        self._files: list = []
        self._idx = 0
        self._loop = True
        self._intr: CameraIntrinsics | None = None
        self._open = False

    # -- CameraBase --------------------------------------------------------

    def open(self, config: dict) -> None:
        image_dir = Path(config["image_dir"])
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        patterns = config.get("patterns",
                              ["*.png", "*.jpg", "*.jpeg", "*.bmp"])
        seen = set()
        files = []
        for pat in patterns:
            for p in image_dir.glob(pat):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        files.sort()
        if not files:
            raise FileNotFoundError(
                f"no images matching {patterns} in {image_dir}")
        self._files = files
        self._idx = 0
        self._loop = bool(config.get("loop", True))
        self._intr = CameraIntrinsics(**config["intrinsics"])
        self._open = True

    def close(self) -> None:
        self._open = False

    def grab(self) -> CameraFrame:
        if not self._open:
            raise RuntimeError("camera not open")
        if self._idx >= len(self._files):
            if not self._loop:
                raise RuntimeError(
                    "no more frames; open with loop=True to wrap around")
            self._idx = 0
        path = self._files[self._idx]
        self._idx += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        return CameraFrame(image=image,
                           intrinsics=self._intr,
                           timestamp_ms=os.path.getmtime(path) * 1000.0)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intr

    def is_open(self) -> bool:
        return self._open
