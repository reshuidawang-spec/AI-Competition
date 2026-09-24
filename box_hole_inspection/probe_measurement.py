"""probe_measurement: 执行有保护的接触测量（仿真与真机同一驱动逻辑）。

对 MotionPlan 中带 search_target_bb 的位姿执行"从孔心向外触壁"的有保护
步进搜索：

    1. 到位 search 起点（孔心，截面深度）
    2. 沿搜索方向按 probe_step_mm 步进（真机: MovL + DI1 轮询；仿真: 几何触发）
    3. 触发即停（真机 Stop 后记录真实过冲；仿真按分布采样过冲模拟停止）
    4. 接触点 = 触发时球心 + ball_r · ê（ê=搜索径向方向，竖直孔壁侧触几何）
    5. 回退到截面孔心，继续下一角度

MISS（搜索走完未触发）显式记录，不静默跳过。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from box_hole_inspection import schemas
from box_hole_inspection.utils import transforms as tf

SEARCH_STEP_MM = 0.5            # 分段探测步长（真机已验证约定）


class ProbeMeasurement:
    def __init__(self, robot, probe, ball_radius_mm: float = 1.0,
                 search_step_mm: float = SEARCH_STEP_MM,
                 tool_t_eb_p: Optional[np.ndarray] = None,
                 simulate_overtravel: bool = True):
        self.robot = robot
        self.probe = probe
        self.ball_r = ball_radius_mm
        self.step_mm = search_step_mm
        self.T_EB_P = (np.eye(4) if tool_t_eb_p is None
                       else np.asarray(tool_t_eb_p, dtype=float))
        self.simulate_overtravel = simulate_overtravel

    # -- helpers ---------------------------------------------------------------

    def tcp_pose(self) -> schemas.Pose:
        return self.robot.get_tcp_pose()

    def ball_center(self) -> np.ndarray:
        """TCP -> 测球球心（世界/机器人B基座系）。"""
        pose = self.tcp_pose()
        T = tf.pose_to_T({"x": pose.x, "y": pose.y, "z": pose.z,
                          "rx": pose.rx, "ry": pose.ry, "rz": pose.rz})
        return tf.apply_T(T, self.T_EB_P[:3, 3])

    # -- guarded search ----------------------------------------------------------

    def search_once(self, wp: schemas.Waypoint,
                    contact_speed: float) -> Optional[schemas.ContactEvent]:
        """一次有保护搜索；返回触发事件（含接触点），MISS 返回 None。

        调用前机器人已位于 wp.pose_bb（搜索起点=孔心）。机器人是唯一
        状态源：每步真实 move_l，随后读取 TCP 评估触发（真机分段探测
        同此语义）。
        """
        target = np.array(wp.search_target_bb, dtype=float)
        start = np.array([wp.pose_bb.x, wp.pose_bb.y, wp.pose_bb.z])
        direction = target - start
        dist = float(np.linalg.norm(direction))
        if dist < 1e-9:
            return None
        direction /= dist

        noise = 0.0
        if hasattr(self.probe, "sample_trigger_noise"):
            noise = self.probe.sample_trigger_noise()   # 触发重复性噪声

        travelled = 0.0
        prev_ball = self.ball_center().copy()
        while travelled < dist - 1e-9:
            step = min(self.step_mm, dist - travelled)
            travelled += step
            self.robot.move_l(self._pose_at(wp, start + direction * travelled),
                              speed_ratio=contact_speed)
            cur_ball = self.ball_center()
            info = self.probe.update_ball(cur_ball)
            if info.triggered:
                # 二分定位触发点（真机是运动中 1ms 轮询+立即停止，
                # 分段步长不应量化触发位置）
                trigger_ball = self.probe.bisect_trigger(
                    prev_ball, cur_ball, noise)
                # 接触点 = 触发球心 + 球半径 · 搜索方向（径向向外）
                contact = trigger_ball + direction * self.ball_r
                overtravel = 0.0
                if self.simulate_overtravel and hasattr(self.probe,
                                                        "sample_overtravel"):
                    overtravel = self.probe.sample_overtravel()
                if overtravel > 0:
                    # 模拟停止过冲：沿搜索方向多走 overtravel 再停
                    self.robot.move_l(
                        self._pose_at(wp, trigger_ball
                                      + direction * overtravel),
                        speed_ratio=contact_speed)
                    self.probe.update_ball(self.ball_center())
                self.robot.stop()
                return schemas.ContactEvent(
                    idx=0, angle_deg=0.0,
                    touch_tcp_bb=[float(v) for v in trigger_ball],
                    contact_point_bb=[float(v) for v in contact])
            prev_ball = cur_ball.copy()
        return None

    @staticmethod
    def _pose_at(wp: schemas.Waypoint, xyz: np.ndarray) -> schemas.Pose:
        """保持位姿姿态不变，仅换位置。"""
        return schemas.Pose(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                            rx=wp.pose_bb.rx, ry=wp.pose_bb.ry,
                            rz=wp.pose_bb.rz)

    # -- per-hole execution -------------------------------------------------------

    def measure_hole(self, waypoints: list, task: schemas.HoleTask,
                     contact_speed: Optional[float] = None) -> list:
        """执行一个孔的全部位姿；返回按截面分组的 ContactPoints 列表。"""
        if contact_speed is None:
            contact_speed = task.measure.contact_speed_ratio
        sections: dict = {}      # section_idx -> ContactPoints(构建中)
        order: list = []

        def get_section(idx: int, z_mm: float) -> schemas.ContactPoints:
            if idx not in sections:
                sections[idx] = schemas.ContactPoints(
                    hole_id=task.hole_id, section_z_mm=z_mm)
                order.append(idx)
            return sections[idx]

        for wp in waypoints:
            self.robot.move_l(wp.pose_bb, speed_ratio=wp.speed_ratio)
            if wp.search_target_bb is None:
                continue
            sec = get_section(wp.section_idx, wp.pose_bb.z)
            sec.n_planned += 1
            event = self.search_once(wp, contact_speed)
            if event is not None:
                event.idx = sec.n_triggered
                event.angle_deg = self._angle_of(wp)
                sec.contacts.append(event)
                sec.n_triggered += 1
            else:
                sec.misses.append({"angle_deg": self._angle_of(wp),
                                   "reason": "no trigger"})
            # 回退到截面孔心（搜索起点），准备下一角度
            self.robot.move_l(wp.pose_bb, speed_ratio=wp.speed_ratio)
        return [sections[i] for i in order]

    @staticmethod
    def _angle_of(wp: schemas.Waypoint) -> float:
        """从位姿名解析角度（'H01_s0_a45' -> 45.0）；解析失败返回 NaN。"""
        try:
            return float(wp.name.rsplit("_a", 1)[1])
        except (ValueError, IndexError):
            return float("nan")
