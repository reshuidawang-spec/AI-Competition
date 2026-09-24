"""Simulated touch probe: geometric trigger (ball surface touches bore wall).

Trigger semantics mirror the real RMP60/DI1 chain:

    ball 表面接触孔壁 -> 触发（DI1=1）；球离开壁面 -> 释放。

Trigger check needs the CURRENT ball-centre position, supplied by the caller
(probe_measurement reads the robot TCP each step). The bore is modelled as a
vertical cylinder of radius R over the hole depth.

Stop overtravel is sampled from a truncated normal per trigger: parameters
are the realistic order of magnitude recorded on the real robot
(0.04~0.47 mm with 0.5 mm segmented probing; older single-segment runs
reached 1.4~2.4 mm). The sim is deliberately CONSERVATIVE (mean 0.30,
max 1.5) so plans that pass here keep the real reserve semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from interfaces.probe import ProbeBase
from sim.sim_world import SimWorld


@dataclass
class TriggerInfo:
    """One trigger evaluation result."""
    triggered: bool = False
    hole_id: Optional[str] = None
    radial_clearance_mm: float = 0.0     # 球面到壁面的距离（负=已过触发）


class SimProbe(ProbeBase):
    def __init__(self, world: SimWorld, ball_radius_mm: float = 1.0,
                 overtravel_mean_mm: float = 0.30,
                 overtravel_std_mm: float = 0.15,
                 overtravel_max_mm: float = 1.5,
                 trigger_noise_sigma_mm: float = 0.04,
                 seed: int = 0):
        self.world = world
        self.ball_r = ball_radius_mm
        self.overtravel_mean = overtravel_mean_mm
        self.overtravel_std = overtravel_std_mm
        self.overtravel_max = overtravel_max_mm
        self.trigger_noise_sigma = trigger_noise_sigma_mm
        # 孔口截面邻域半径余量：只有球心投影落在 [0, R+margin] 内的孔
        # 才参与评估（避免把远方的孔当成无限圆柱误触发）
        self.wall_margin_mm = ball_radius_mm + overtravel_max_mm + 1.0
        self.rng = np.random.default_rng(seed)
        self._io = "DI1"
        self.last_trigger: Optional[TriggerInfo] = None

    # -- ProbeBase -----------------------------------------------------------

    def configure(self, trigger_io: str = "DI1") -> None:
        self._io = trigger_io

    def read_trigger(self) -> bool:
        """最近一次 update_ball() 评估的触发状态。"""
        return self.last_trigger is not None and self.last_trigger.triggered

    def wait_trigger(self, timeout_s: float) -> bool:
        return self.read_trigger()          # 仿真同步：状态即时可得

    def wait_released(self, timeout_s: float) -> bool:
        return not self.read_trigger()

    # -- geometry -------------------------------------------------------------

    def evaluate(self, ball_center: np.ndarray) -> TriggerInfo:
        """球心位置 -> 触发评估（对每个孔检查圆柱壁接触）。

        只有球心水平投影落在孔口截面邻域（rho <= R + wall_margin）且
        高度在孔深范围内的孔才参与评估——搜索时只会命中当前孔。
        """
        info = TriggerInfo(triggered=False, radial_clearance_mm=float("inf"))
        ball = np.asarray(ball_center, dtype=float)
        for hole in self.world.box.holes:
            c = self.world.hole_centers_w()[hole.id]
            rho = float(np.hypot(ball[0] - c[0], ball[1] - c[1]))
            z_top, depth = self.world.box.z_top_mm, hole.depth_mm
            if not (z_top - depth <= ball[2] <= z_top):
                continue                      # 球不在孔深范围内（孔外/已穿过）
            R = hole.diameter_mm / 2.0
            if rho > R + self.wall_margin_mm:
                continue                      # 不在此孔口截面邻域
            # 球面到壁面的间隙 = 孔半径 − 球心径向距离 − 球半径；
            # 间隙 <= 0 即球面触壁（触发）
            clearance = R - rho - self.ball_r
            if clearance < info.radial_clearance_mm:
                info.radial_clearance_mm = clearance
                info.hole_id = hole.id
                if clearance <= 0.0:
                    info.triggered = True
        return info

    def update_ball(self, ball_center: np.ndarray) -> TriggerInfo:
        """测量循环每步调用：更新球心并返回触发评估。"""
        self.last_trigger = self.evaluate(np.asarray(ball_center, float))
        return self.last_trigger

    def clearance(self, ball_center: np.ndarray) -> float:
        """球面到最近孔壁的间隙（最小者）；无有效孔返回 +inf。

        与 evaluate 同样的邻域/深度筛选。
        """
        return self.evaluate(np.asarray(ball_center, dtype=float)).radial_clearance_mm

    def bisect_trigger(self, ball_a: np.ndarray, ball_b: np.ndarray,
                       noise: float, iters: int = 40) -> np.ndarray:
        """在 [ball_a, ball_b] 段上二分求触发球心（触发重复性噪声计入）。

        真机是运动中 1ms 轮询 + 立即停止，分段步长不应量化触发位置——
        二分把触发点恢复到几何精度。
        """
        ball_a = np.asarray(ball_a, dtype=float)
        ball_b = np.asarray(ball_b, dtype=float)
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            ball = ball_a + mid * (ball_b - ball_a)
            if self.clearance(ball) <= noise:
                hi = mid
            else:
                lo = mid
        return ball_a + hi * (ball_b - ball_a)

    def sample_overtravel(self) -> float:
        """触发后的停止过冲采样（截断正态，>=0）。"""
        v = self.rng.normal(self.overtravel_mean, self.overtravel_std)
        return float(np.clip(v, 0.0, self.overtravel_max))

    def sample_trigger_noise(self) -> float:
        """触发重复性噪声（真机实测 σ≈0.04mm）：预行程/lobing 等效。"""
        return float(self.rng.normal(0.0, self.trigger_noise_sigma))
