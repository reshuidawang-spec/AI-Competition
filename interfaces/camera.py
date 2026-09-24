"""Camera abstraction layer.

Real hardware plugs in ONLY here. Algorithm modules depend on CameraBase,
never on a concrete camera.

Intrinsics do NOT come from the driver: they are measured by chessboard
calibration (docs/calibration_plan.md) and stored in config/calibration.json.
The coordinate chain (coordinate_transform.py) expects UNDISTORTED pixels;
adapters should apply cv2.undistort with the calibration distortion vector
before handing frames to detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics + image size (undistorted convention)."""
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 640.0
    cy: float = 512.0
    width: int = 1280
    height: int = 1024
    dist: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class CameraFrame:
    """One captured frame."""
    image: np.ndarray          # BGR uint8
    intrinsics: CameraIntrinsics
    timestamp_ms: float


class CameraBase(ABC):
    """Abstract camera. Adapters: SimImageFileCamera, RealCameraOpenCV."""

    @abstractmethod
    def open(self, config: dict) -> None:
        """Open with an adapter-specific config dict (see config/camera.yaml)."""

    @abstractmethod
    def close(self) -> None:
        """Release the device."""

    @abstractmethod
    def grab(self) -> CameraFrame:
        """Capture one frame (blocking)."""

    @property
    @abstractmethod
    def intrinsics(self) -> CameraIntrinsics:
        """Intrinsics of the current stream (from calibration, not driver)."""

    @abstractmethod
    def is_open(self) -> bool:
        """True while the device is open."""
