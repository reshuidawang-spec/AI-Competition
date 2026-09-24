"""Camera -> World -> Robot-B coordinate chain (pure math, no hardware).

The single view -> execution chain of the system::

    p_BB = T_BB_BA · T_BA_EA(θ_A) · T_EA_C · λ·K⁻¹·[u, v, 1]ᵀ

where W ≡ BB (robot-B base frame is the world frame, agreed 2026-09-23):

- ``T_EA_C``   hand-eye calibration, fixed (AX=XB, docs/calibration_plan.md).
- ``T_BA_EA``  robot-A forward kinematics; supplied as a fixed capture pose
               in Phase 1 (robot A always shoots from the same pose), or by a
               provider callable when A moves.
- ``λ``        ray-plane intersection with the known box top-face plane
               (Phase 1: the box sits on the worktable, top face at z_top).

Input pixels must be UNDISTORTED: the camera adapter applies lens-distortion
correction before detection; this module stays pure.

All inputs come from config/calibration.json (loaded with
:func:`load_calibration`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import schemas
from .utils import transforms as tf


class CalibrationError(Exception):
    """Raised when calibration.json is missing or malformed."""


def load_calibration(path) -> dict:
    """Load calibration.json (extra keys such as _comment are tolerated)."""
    path = Path(path)
    if not path.exists():
        raise CalibrationError(f"calibration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pose_required(cal: dict, key: str) -> np.ndarray:
    if key not in cal:
        raise CalibrationError(f"calibration.json missing '{key}'")
    return tf.pose_to_T(cal[key])


class CoordinateTransform:
    """The coordinate chain. Pure numpy; fully unit-testable."""

    def __init__(self, calibration: dict,
                 t_ba_ea: Optional[np.ndarray] = None,
                 t_ba_ea_provider: Optional[Callable[[], np.ndarray]] = None):
        """
        Args:
            calibration: dict from load_calibration().
            t_ba_ea: fixed robot-A flange pose in robot-A base frame (mm, 4x4).
                     Phase 1 uses the fixed capture pose.
            t_ba_ea_provider: callable returning the current T_BA_EA; used
                     once robot A starts moving. Fixed pose wins if both set.
        """
        self.T_EA_C = _pose_required(calibration, "T_EA_C")
        self.T_BA_BB = _pose_required(calibration, "T_BA_BB")   # A base -> B base
        self.T_BB_BA = tf.invert_T(self.T_BA_BB)                # B base -> A base

        k = calibration.get("camera_intrinsics") or {}
        for key in ("fx", "fy", "cx", "cy"):
            if key not in k:
                raise CalibrationError(f"camera_intrinsics missing '{key}'")
        self.K = np.array([[k["fx"], 0.0, k["cx"]],
                           [0.0, k["fy"], k["cy"]],
                           [0.0, 0.0, 1.0]], dtype=float)
        self.Kinv = np.linalg.inv(self.K)

        plane = calibration.get("table_plane_w") or {}
        if "point" not in plane or "normal" not in plane:
            raise CalibrationError("calibration.json missing 'table_plane_w'")
        self.plane_normal = tf.normalize(np.asarray(plane["normal"], dtype=float))
        self.plane_point = np.asarray(plane["point"], dtype=float)

        self._t_ba_ea = np.asarray(t_ba_ea, dtype=float) if t_ba_ea is not None else None
        self._provider = t_ba_ea_provider

    # -- robot A flange ----------------------------------------------------

    def T_BA_EA(self) -> np.ndarray:
        """Robot-A flange pose in robot-A base frame (mm)."""
        if self._t_ba_ea is not None:
            return self._t_ba_ea
        if self._provider is not None:
            T = self._provider()
            if T is None:
                raise CalibrationError("robot-A FK provider returned None")
            return np.asarray(T, dtype=float)
        raise CalibrationError(
            "no robot-A pose: pass t_ba_ea or t_ba_ea_provider")

    def T_W_EA(self) -> np.ndarray:
        """Robot-A flange pose in world (= robot-B base) frame."""
        return tf.compose(self.T_BB_BA, self.T_BA_EA())

    def T_W_C(self) -> np.ndarray:
        """Camera pose in world frame."""
        return tf.compose(self.T_W_EA(), self.T_EA_C)

    # -- rays --------------------------------------------------------------

    def camera_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> unit direction in the camera frame; origin is C's origin."""
        d = self.Kinv @ np.array([u, v, 1.0], dtype=float)
        return np.zeros(3), tf.normalize(d)

    def ray_in_world(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> (origin, unit direction) of the ray in world frame."""
        o_c, d_c = self.camera_ray(u, v)
        T = self.T_W_C()
        return tf.apply_T(T, o_c), tf.apply_R(T[:3, :3], d_c)

    # -- intersection ------------------------------------------------------

    def intersect_plane(self, o: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
        """Ray-plane intersection; None if parallel or the plane is behind."""
        denom = float(self.plane_normal @ np.asarray(d, dtype=float))
        if abs(denom) < 1e-12:
            return None
        lam = float(self.plane_normal @ (self.plane_point - np.asarray(o, dtype=float))) / denom
        if lam < 0.0:
            return None
        return np.asarray(o, dtype=float) + lam * np.asarray(d, dtype=float)

    # -- chain -------------------------------------------------------------

    def pixel_to_world(self, u: float, v: float) -> Optional[np.ndarray]:
        """Pixel -> 3D point in world frame (None if no plane hit)."""
        o, d = self.ray_in_world(u, v)
        return self.intersect_plane(o, d)

    def world_to_bb(self, p_w) -> np.ndarray:
        """World -> robot-B base. Identity while W ≡ BB; kept explicit."""
        return np.asarray(p_w, dtype=float).copy()

    def bb_to_world(self, p_bb) -> np.ndarray:
        """Robot-B base -> world. Identity while W ≡ BB."""
        return np.asarray(p_bb, dtype=float).copy()

    def pixel_to_bb(self, u: float, v: float) -> Optional[np.ndarray]:
        """Pixel -> robot-B base frame (None if no plane hit)."""
        p_w = self.pixel_to_world(u, v)
        return None if p_w is None else self.world_to_bb(p_w)

    def pixel_error_to_mm(self, u: float, v: float,
                          du_px: float = 1.0, dv_px: float = 0.0) -> Optional[float]:
        """World displacement caused by pixel noise (du, dv); mm per pixel."""
        p0 = self.pixel_to_world(u, v)
        p1 = self.pixel_to_world(u + du_px, v + dv_px)
        if p0 is None or p1 is None:
            return None
        return float(np.linalg.norm(p1 - p0))

    # -- localisation ------------------------------------------------------

    def localize(self, detection: schemas.DetectionResult) -> schemas.LocalizationResult:
        """DetectionResult -> LocalizationResult (fills world/bb/camera coords).

        The box pose is only initialised from the mask rz hint and the hole
        centroid; the precise (x, y, rz) pose and hole-ID assignment are the
        job of hole_assigner (CAD pattern fit), which runs after this module.
        """
        T_inv = tf.invert_T(self.T_W_C())
        holes = []
        for h in detection.holes:
            p_w = self.pixel_to_world(h.pixel_u, h.pixel_v)
            if p_w is None:
                continue
            p_c = tf.apply_T(T_inv, p_w)
            holes.append(schemas.LocalizedHole(
                id=h.id,
                confidence=h.confidence,
                pixel_u=h.pixel_u,
                pixel_v=h.pixel_v,
                camera_xyz=[float(x) for x in p_c],
                world_xyz=[float(x) for x in p_w],
                bb_xyz=[float(x) for x in self.world_to_bb(p_w)],
            ))

        pose = schemas.Pose()
        if holes:
            c = np.mean([h.world_xyz for h in holes], axis=0)
            pose.x, pose.y, pose.z = float(c[0]), float(c[1]), float(c[2])
        pose.rz = detection.box_rz_hint_deg or 0.0

        coarse_err = None
        if holes:
            coarse_err = self.pixel_error_to_mm(holes[0].pixel_u, holes[0].pixel_v)

        return schemas.LocalizationResult(
            part_id=detection.part_id,
            confidence=detection.confidence,
            timestamp_ms=detection.timestamp_ms,
            pose=pose,
            holes=holes,
            coarse_error_est_mm=coarse_err,
        )
