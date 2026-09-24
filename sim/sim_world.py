"""Simulated workcell scene: table + box with holes (ground truth).

The scene is the GROUND TRUTH against which the whole vision -> coordinate
chain is verified in simulation. Layout (all in world frame W ≡ robot-B base):

- Table top plane at z = 0.
- Box sits on the table; holes are on the TOP face (z = box height),
  consistent with the 2026-09-23 hardware decision.
- Phase 1 box degrees of freedom: x, y translation + rz rotation.

Hole pattern / diameters come from config/part_box001.yaml when available
(real values are currently null); missing values fall back to
DEFAULT_HOLE_* so the sim stays self-contained until real data arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from box_hole_inspection.utils import transforms as tf

DEFAULT_BOX_DIMS_MM = (165.0, 55.0, 110.0)   # (长, 宽, 高)
DEFAULT_HOLE_DIAMETER_MM = 20.0               # 真实孔径待定时的仿真默认值
DEFAULT_HOLE_CENTERS_O = {                    # 箱体系 (x, y) 默认孔模式
    # y=±12（而非 ±10）：两排孔保留间隙，实例掩膜不重叠（训练标签干净）
    "H01": (-45.0, -12.0),
    "H02": (45.0, -12.0),
    "H03": (-45.0, 12.0),
    "H04": (45.0, 12.0),
}


@dataclass
class SimHole:
    """One hole in the simulated box (box-frame centre + diameter)."""
    id: str = ""
    center_o: tuple = (0.0, 0.0)              # 箱体系 (x, y) mm
    diameter_mm: float = DEFAULT_HOLE_DIAMETER_MM
    depth_mm: float = 30.0                    # 渲染孔内深度感用


@dataclass
class SimBox:
    """The simulated box: dims + pose (x, y, rz) in world frame."""
    part_id: str = "BOX_001"
    dims_mm: tuple = DEFAULT_BOX_DIMS_MM
    pose: tuple = (0.0, 0.0, 0.0)             # (x_mm, y_mm, rz_deg)
    holes: list = field(default_factory=list)

    @property
    def length_mm(self) -> float:
        return float(self.dims_mm[0])

    @property
    def width_mm(self) -> float:
        return float(self.dims_mm[1])

    @property
    def z_top_mm(self) -> float:
        """Top-face height in world frame (box sits on the z=0 table)."""
        return float(self.dims_mm[2])


class SimWorld:
    """Workcell scene. Phase 1: table plane + one box."""

    def __init__(self, box: Optional[SimBox] = None):
        self.box = box if box is not None else default_box()

    # -- ground-truth geometry ----------------------------------------------

    def set_box_pose(self, x_mm: float, y_mm: float, rz_deg: float) -> None:
        self.box.pose = (float(x_mm), float(y_mm), float(rz_deg))

    def hole_centers_w(self) -> dict:
        """Ground-truth hole centres in world frame: {id: np.ndarray(3)}."""
        x, y, rz = self.box.pose
        R = tf.euler_to_R(0.0, 0.0, rz)
        out = {}
        for h in self.box.holes:
            p_o = np.array([h.center_o[0], h.center_o[1], 0.0])
            p_w = np.array([x, y, self.box.z_top_mm]) + R @ p_o
            out[h.id] = p_w
        return out

    def top_face_corners_w(self) -> np.ndarray:
        """Top-face corner points (4, 3) in world frame."""
        x, y, rz = self.box.pose
        R = tf.euler_to_R(0.0, 0.0, rz)
        l2, w2 = self.box.length_mm / 2.0, self.box.width_mm / 2.0
        corners_o = np.array([[-l2, -w2, 0.0], [l2, -w2, 0.0],
                              [l2, w2, 0.0], [-l2, w2, 0.0]])
        return np.array([x, y, self.box.z_top_mm]) + (R @ corners_o.T).T


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def default_box() -> SimBox:
    holes = [SimHole(id=hid, center_o=c, diameter_mm=DEFAULT_HOLE_DIAMETER_MM)
             for hid, c in DEFAULT_HOLE_CENTERS_O.items()]
    return SimBox(holes=holes)


def box_from_part_yaml(path, fallback_diameter_mm: float = DEFAULT_HOLE_DIAMETER_MM,
                       fallback_centers_o: Optional[dict] = None) -> SimBox:
    """Build the sim box from config/part_box001.yaml.

    Null diameters / centres are filled with fallbacks (real values are
    pending); when the real values are filled in the yaml, the sim scene
    tracks them automatically.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if fallback_centers_o is None:
        fallback_centers_o = DEFAULT_HOLE_CENTERS_O
    holes = []
    for i, h in enumerate(spec.get("holes", [])):
        hid = h["id"]
        center = h.get("center_o_mm")
        if center is None or center[0] is None or center[1] is None:
            center = fallback_centers_o.get(hid, (0.0, 0.0))
        diam = h.get("diameter_mm")
        if diam is None:
            diam = fallback_diameter_mm
        holes.append(SimHole(id=hid, center_o=(float(center[0]), float(center[1])),
                             diameter_mm=float(diam)))
    dims = spec.get("box_dims_mm") or DEFAULT_BOX_DIMS_MM
    return SimBox(part_id=spec.get("part_id", "BOX_001"),
                  dims_mm=tuple(float(d) for d in dims), holes=holes)
