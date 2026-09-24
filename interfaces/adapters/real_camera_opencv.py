"""Real camera adapter for the ASSUMED camera: USB3 industrial camera with
UVC support (streams as a standard OpenCV VideoCapture device).

Assumption is explicit (2026-09-23): the actual industrial camera model is
not yet fixed. If the purchased camera is GigE or requires a vendor SDK
(e.g. Hikvision MVS, Basler pylon, Daheng Galaxy), write a SIBLING adapter
implementing CameraBase and replace ONLY open()/close()/grab(); all callers
and algorithms stay untouched.

Intrinsics come from config/calibration.json (chessboard calibration,
docs/calibration_plan.md) — never from the driver. The coordinate chain
expects undistorted pixels: undistortion is applied here with the
calibration distortion vector.

Config keys (see config/camera.yaml):
    device_index:  int or device path string passed to cv2.VideoCapture
    width, height, fps: requested stream format
    exposure_auto: bool (best-effort; UVC exposure control is vendor-specific)
    exposure_ms:   float requested exposure (best-effort)
    intrinsics:    dict matching CameraIntrinsics fields (from calibration)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from interfaces.camera import CameraBase, CameraFrame, CameraIntrinsics

log = logging.getLogger(__name__)


class RealCameraOpenCV(CameraBase):
    def __init__(self):
        self._cap = None
        self._intr: CameraIntrinsics | None = None
        self._open = False

    # -- CameraBase --------------------------------------------------------

    def open(self, config: dict) -> None:
        import cv2  # lazy: W1 pure-math tests must not need cv2

        device = config.get("device_index", 0)
        if isinstance(device, str):
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(int(device))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"cannot open camera device {device!r} — check connection, "
                "permissions (video group), or that the camera supports UVC")

        for prop, value in (("width", config.get("width")),
                            ("height", config.get("height")),
                            ("fps", config.get("fps")),
                            ("exposure_auto", config.get("exposure_auto")),
                            ("exposure_ms", config.get("exposure_ms"))):
            if value is None:
                continue
            ok = {
                "width": lambda: self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, value),
                "height": lambda: self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value),
                "fps": lambda: self._cap.set(cv2.CAP_PROP_FPS, value),
                "exposure_auto": lambda: self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                                       0.0 if not value else 3.0),
                "exposure_ms": lambda: self._cap.set(cv2.CAP_PROP_EXPOSURE, value),
            }[prop]()
            if not ok:
                log.warning("camera property %s=%r not accepted (UVC best-effort)",
                            prop, value)

        self._intr = CameraIntrinsics(**config["intrinsics"])
        self._open = True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._open = False

    def grab(self) -> CameraFrame:
        if not self._open:
            raise RuntimeError("camera not open")
        ret, image = self._cap.read()
        if not ret:
            raise RuntimeError("camera grab failed — device unplugged?")
        if self._intr is not None and any(self._intr.dist):
            import cv2
            k = np.array([[self._intr.fx, 0.0, self._intr.cx],
                          [0.0, self._intr.fy, self._intr.cy],
                          [0.0, 0.0, 1.0]])
            image = cv2.undistort(image, k, np.array(self._intr.dist))
        return CameraFrame(image=image,
                           intrinsics=self._intr,
                           timestamp_ms=time.time() * 1000.0)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intr

    def is_open(self) -> bool:
        return self._open and self._cap is not None and self._cap.isOpened()
