"""Pose / transform utilities for the box-hole inspection project.

Project-wide conventions:

- Lengths are millimetres; angles are DEGREES at the JSON/YAML boundary and
  radians inside numpy math.
- A transform ``T_B_A`` maps points from frame A to frame B::

      p_B = T_B_A @ p_A      (homogeneous 4x4, p = [x, y, z, 1])

- Orientation at the JSON boundary is roll/pitch/yaw with the intrinsic
  convention ``sxyz``::

      R = Rz(rz) @ Ry(ry) @ Rx(rx)

  This is THIS package's convention. The mapping between these angles and the
  Dobot controller's ``rx/ry/rz`` is a KNOWN OPEN ISSUE in the sibling project
  ``cr5_rmp60_measurement`` (see its ``validate_cr5_pose_convention.py``).
  Real-robot adapters must validate that mapping before trusting orientations;
  flange POSITIONS reported by the controller are treated as reliable.

No scipy dependency; pure numpy.
"""

from __future__ import annotations

import math

import numpy as np

_IDENTITY = np.eye(4)


def rot_x(a: float) -> np.ndarray:
    """Rotation matrix around the x axis, angle in radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    """Rotation matrix around the y axis, angle in radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    """Rotation matrix around the z axis, angle in radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def euler_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Roll/pitch/yaw (degrees, intrinsic sxyz) to rotation matrix.

    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def R_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix to roll/pitch/yaw (degrees, intrinsic sxyz).

    Inverse of :func:`euler_to_R`; returns rx, ry, rz. Unstable near the
    gimbal singularity (ry = ±90°).
    """
    R = np.asarray(R, dtype=float)
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
    rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def pose_to_T(pose: dict) -> np.ndarray:
    """Pose dict {x,y,z,rx,ry,rz} (mm, deg) to a 4x4 homogeneous transform."""
    R = euler_to_R(pose["rx"], pose["ry"], pose["rz"])
    return make_T(R, np.array([pose["x"], pose["y"], pose["z"]], dtype=float))


def T_to_pose(T: np.ndarray) -> dict:
    """4x4 homogeneous transform to a pose dict {x,y,z,rx,ry,rz} (mm, deg)."""
    T = np.asarray(T, dtype=float)
    rx, ry, rz = R_to_euler(T[:3, :3])
    return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
            "rx": rx, "ry": ry, "rz": rz}


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def compose(T_outer: np.ndarray, T_inner: np.ndarray) -> np.ndarray:
    """Compose transforms: T_B_C = compose(T_B_A, T_A_C)."""
    return np.asarray(T_outer, dtype=float) @ np.asarray(T_inner, dtype=float)


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ T[:3, 3]
    return Tinv


def apply_T(T: np.ndarray, p) -> np.ndarray:
    """Apply T to point(s): accepts (3,) or (N, 3), returns the same shape."""
    T = np.asarray(T, dtype=float)
    arr = np.asarray(p, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    if arr.ndim == 1:
        return R @ arr + t
    if arr.ndim == 2:
        return arr @ R.T + t
    raise ValueError("apply_T expects (3,) or (N, 3)")


def apply_R(R: np.ndarray, v) -> np.ndarray:
    """Apply a 3x3 rotation to vector(s)."""
    R = np.asarray(R, dtype=float)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        return R @ arr
    if arr.ndim == 2:
        return arr @ R.T
    raise ValueError("apply_R expects (3,) or (N, 3)")


def normalize(v) -> np.ndarray:
    """Normalize a 3-vector; returns zeros unchanged."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return v.copy()
    return v / n


def is_identity(T: np.ndarray, tol: float = 1e-9) -> bool:
    """True if T is an identity transform within tolerance."""
    return bool(np.allclose(T, _IDENTITY, atol=tol))
