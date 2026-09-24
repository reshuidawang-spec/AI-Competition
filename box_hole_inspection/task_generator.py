"""task_generator: 视觉粗定位 -> 每孔测量任务（策略生成，纯逻辑，无硬件）。

按孔径分档选择测针策略（2026-09-23 硬件决策）：

- diameter >= branch_threshold (65mm): cross —— 十字四水平支触内壁，
  每截面 4 向接触（环规已验证流程）；
- diameter <  65mm: z —— 轴向针侧触，每截面 8 向接触（W3 需新验证）。

测量截面深度默认避开孔口倒角（-6/-18mm 相对顶面），孔径、截面、角度等
均可被 part spec 覆盖。安全参数（停止过冲预留、分段探测步长、探测速度）
来自 config/system.yaml，与真机约定一致。

生成的 TaskList 不包含运动位姿——那是 path_planner 的职责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from box_hole_inspection import schemas

BRANCH_THRESHOLD_DIAMETER_MM = 65.0
BRANCH_CROSS = "cross"                 # 十字四水平支（大孔）
BRANCH_AXIAL = "z"                     # z 轴向针侧触（小孔）
DEFAULT_SECTION_Z_OFFSETS_MM = (-6.0, -18.0)
N_POINTS_AXIAL = 8
N_POINTS_CROSS = 4


def load_part_spec(path) -> dict:
    """Load config/part_box001.yaml (tolerates any extra keys)."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TaskGenerator:
    """LocalizationResult + part spec + safety params -> TaskList."""

    def __init__(self, part_spec: dict, safety: Optional[dict] = None,
                 branch_threshold_diameter_mm: float = BRANCH_THRESHOLD_DIAMETER_MM,
                 section_z_offsets_mm: tuple = DEFAULT_SECTION_Z_OFFSETS_MM,
                 n_points_axial: int = N_POINTS_AXIAL,
                 n_points_cross: int = N_POINTS_CROSS,
                 search_overshoot_mm: float = 3.0,
                 setup_id: str = "box_hole_cell_2026",
                 standard_pose_axial: str = "bore_z_sim",
                 standard_pose_cross: str = "bore_cross_sim"):
        self.part_spec = part_spec
        self.safety = safety or {}
        self.branch_threshold = branch_threshold_diameter_mm
        self.section_z_offsets = list(section_z_offsets_mm)
        self.n_points_axial = n_points_axial
        self.n_points_cross = n_points_cross
        self.search_overshoot_mm = search_overshoot_mm
        self.setup_id = setup_id
        self.standard_pose_axial = standard_pose_axial
        self.standard_pose_cross = standard_pose_cross
        self._nominal_by_id = {
            h["id"]: h for h in part_spec.get("holes", [])}

    # -- strategy -----------------------------------------------------------

    def pick_branch(self, diameter_mm: Optional[float]) -> Optional[str]:
        """孔径 -> 测针策略；直径未知返回 None（无法规划）。"""
        if diameter_mm is None or diameter_mm <= 0:
            return None
        return (BRANCH_CROSS if diameter_mm >= self.branch_threshold
                else BRANCH_AXIAL)

    def make_sections(self, branch: str) -> list:
        """按分支生成测量截面（轴向 8 向、十字 4 向）。"""
        n = self.n_points_axial if branch == BRANCH_AXIAL else self.n_points_cross
        if branch == BRANCH_AXIAL:
            angles = np.linspace(0.0, 360.0, n, endpoint=False)
        else:
            angles = np.array([0.0, 90.0, 180.0, 270.0])  # 十字一次腕姿四向
        return [schemas.MeasureSection(z_offset_mm=z, n_points=int(n),
                                       angles_deg=[float(a) for a in angles])
                for z in self.section_z_offsets]

    # -- generation -----------------------------------------------------------

    def generate(self, localization: schemas.LocalizationResult,
                 timestamp_ms: Optional[float] = None) -> schemas.TaskList:
        tasks, skipped = [], []
        for h in localization.holes:
            nominal = self._nominal_by_id.get(h.id)
            if nominal is None:
                skipped.append({"id": h.id, "reason": "no nominal data"})
                continue
            if h.bb_xyz is None:
                skipped.append({"id": h.id, "reason": "no bb coordinates"})
                continue

            diameter = nominal.get("diameter_mm")
            branch = self.pick_branch(diameter)
            if branch is None:
                skipped.append({"id": h.id,
                                "reason": "nominal diameter unknown"})
                continue

            center_o = nominal.get("center_o_mm")
            if not center_o or center_o[0] is None:
                center_o = [None, None]
            tasks.append(schemas.HoleTask(
                hole_id=h.id,
                nominal=schemas.NominalHole(
                    id=h.id,
                    diameter_mm=float(diameter),
                    center_o=[float(v) if v is not None else None
                              for v in center_o],
                    tol_diameter_mm=float(nominal.get("tol_diameter_mm",
                                                     0.05)),
                    tol_position_mm=float(nominal.get("tol_position_mm",
                                                     0.10)),
                    tol_perp_deg=float(nominal.get("tol_perp_deg", 0.10)),
                ),
                rough_center_bb=[float(v) for v in h.bb_xyz],
                measure=schemas.MeasureConfig(
                    branch=branch,
                    standard_pose=(self.standard_pose_axial
                                   if branch == BRANCH_AXIAL
                                   else self.standard_pose_cross),
                    sections=self.make_sections(branch),
                    search_overshoot_mm=self.search_overshoot_mm,
                    stop_overtravel_reserve_mm=float(
                        self.safety.get("stop_overtravel_reserve_mm", 2.0)),
                    probe_step_mm=float(self.safety.get("probe_step_mm", 0.5)),
                    contact_speed_ratio=float(
                        self.safety.get("contact_speed_ratio", 0.05)),
                    setup_id=self.setup_id,
                    trigger_io=self.safety.get("trigger_io", "DI1"),
                ),
            ))

        return schemas.TaskList(
            part_id=localization.part_id,
            timestamp_ms=(timestamp_ms if timestamp_ms is not None
                          else localization.timestamp_ms),
            tasks=tasks,
            skipped=skipped,
        )
