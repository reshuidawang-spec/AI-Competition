"""path_planner: 每孔测量任务 -> 机器人B运动位姿（MotionPlan）。

第一阶段（孔轴竖直、测针从孔内向外触壁）位姿几何（环规流程同此）：

    safe      : 孔中心正上方 safe_z = z_top + safe_z_margin
    insert    : 测球中心降到截面深度 z_top + z_offset（孔心处）
    search θ  : 位姿=孔心（同截面深度），search_target = 孔心 +
                e(θ)·(R_hole − r_ball + search_overshoot)。由
                probe_measurement 从位姿朝 search_target 做有保护的
                步进搜索（触发即停），规划器不负责搜索段本身
    center    : 回到截面孔心（换角度/换截面的过渡点）

预期触发点在 R_true − r_ball；overshoot 覆盖粗定位误差（实测约
0.3~2mm，默认 3mm）与孔径公差。每个位姿都经 robot.check_pose 检查
（可达/限位/奇异），全部通过才整体放行；碰撞检查在障碍模块接入后
补充（当前恒为 True）。

仿真简化说明：十字分支（大孔）在仿真里沿用同一"点测头 + 搜索半径"模型，
真实分支 offset 由真机适配器/标定注册表负责——仿真只验证规划管线。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from box_hole_inspection import schemas
from box_hole_inspection.sequence_planner import order_tsp_2opt

DEFAULT_SAFE_Z_MARGIN_MM = 40.0
DEFAULT_PROBE_BALL_RADIUS_MM = 1.0     # 红宝石球 φ2（真实测针）
DEFAULT_IN_HOLE_SPEED_RATIO = 0.15


class PathPlanner:
    """TaskList -> MotionPlan（机器人B基座系位姿 + 顺序 + 检查）。"""

    def __init__(self, robot, safety: Optional[dict] = None,
                 probe_ball_radius_mm: float = DEFAULT_PROBE_BALL_RADIUS_MM,
                 safe_z_margin_mm: float = DEFAULT_SAFE_Z_MARGIN_MM,
                 in_hole_speed_ratio: float = DEFAULT_IN_HOLE_SPEED_RATIO):
        self.robot = robot
        self.safety = safety or {}
        self.ball_r = probe_ball_radius_mm
        self.safe_z_margin = safe_z_margin_mm
        self.transfer_speed = float(self.safety.get("transfer_speed_ratio", 0.3))
        self.contact_speed = float(self.safety.get("contact_speed_ratio", 0.05))
        self.in_hole_speed = in_hole_speed_ratio

    # -- geometry -----------------------------------------------------------

    @staticmethod
    def _orientation() -> tuple:
        """测针竖直向下（第一阶段统一姿态）。"""
        return 180.0, 0.0, 0.0

    def safe_pose(self, cx: float, cy: float, z_top: float) -> schemas.Pose:
        rx, ry, rz = self._orientation()
        return schemas.Pose(x=cx, y=cy, z=z_top + self.safe_z_margin,
                            rx=rx, ry=ry, rz=rz)

    def search_target(self, cx: float, cy: float, z: float, angle_deg: float,
                      radius_mm: float, overshoot_mm: float) -> list:
        """搜索终点：孔心 + e(θ)·(R_hole − r_ball + overshoot)。

        有保护搜索从孔心出发向外触壁（环规流程同此）：预期触发点在
        R_true − r_ball，overshoot 覆盖粗定位误差与孔径公差。
        """
        a = math.radians(angle_deg)
        d = radius_mm - self.ball_r + overshoot_mm
        return [cx + d * math.cos(a), cy + d * math.sin(a), z]

    # -- planning -----------------------------------------------------------

    def plan_hole(self, task: schemas.HoleTask):
        """单孔位姿序列 + 检查结果。"""
        m = task.measure
        cx, cy, cz = task.rough_center_bb
        z_top = cz                        # 粗定位 z 即顶面高度
        R = task.nominal.diameter_mm / 2.0
        waypoints = []
        checks = schemas.PlanChecks()

        def add(stage, name, pose, speed, search_target=None, section=-1):
            wp = schemas.Waypoint(hole_id=task.hole_id, stage=stage,
                                  name=name, pose_bb=pose,
                                  search_target_bb=search_target,
                                  section_idx=section,
                                  speed_ratio=speed)
            c = self.robot.check_pose(pose)
            if not c.reachable:
                checks.reachable = False
                checks.messages.append(f"{name}: IK failed")
            if not c.singularity_ok:
                checks.singularity_ok = False
                checks.messages.append(f"{name}: near-singular")
            checks.collision_ok = checks.collision_ok and c.collision_ok
            waypoints.append(wp)

        add("approach", f"{task.hole_id}_safe",
            self.safe_pose(cx, cy, z_top), self.transfer_speed)

        for si, sec in enumerate(m.sections):
            z_sec = z_top + sec.z_offset_mm
            add("measure", f"{task.hole_id}_s{si}_insert",
                schemas.Pose(x=cx, y=cy, z=z_sec,
                             rx=180.0, ry=0.0, rz=0.0),
                self.in_hole_speed, section=si)
            for ang in sec.angles_deg:
                add("measure", f"{task.hole_id}_s{si}_a{ang:g}",
                    schemas.Pose(x=cx, y=cy, z=z_sec,
                                 rx=180.0, ry=0.0, rz=0.0),
                    self.in_hole_speed,
                    search_target=self.search_target(
                        cx, cy, z_sec, ang, R, m.search_overshoot_mm),
                    section=si)
            add("measure", f"{task.hole_id}_s{si}_center",
                schemas.Pose(x=cx, y=cy, z=z_sec,
                             rx=180.0, ry=0.0, rz=0.0),
                self.in_hole_speed, section=si)

        add("leave", f"{task.hole_id}_safe",
            self.safe_pose(cx, cy, z_top), self.transfer_speed)
        return waypoints, checks

    def plan(self, task_list: schemas.TaskList,
             sequence: Optional[list] = None,
             start_xy: Optional[list] = None) -> schemas.MotionPlan:
        if sequence is None:
            points = {t.hole_id: t.rough_center_bb for t in task_list.tasks}
            sequence = order_tsp_2opt(points, start=start_xy)
        waypoints: list = []
        checks = schemas.PlanChecks()
        for hid in sequence:
            task = next((t for t in task_list.tasks if t.hole_id == hid), None)
            if task is None:
                checks.messages.append(f"{hid}: no task (skipped by generator)")
                continue
            wps, c = self.plan_hole(task)
            waypoints.extend(wps)
            checks.reachable = checks.reachable and c.reachable
            checks.singularity_ok = checks.singularity_ok and c.singularity_ok
            checks.collision_ok = checks.collision_ok and c.collision_ok
            checks.messages.extend(c.messages)
        return schemas.MotionPlan(
            sequence=sequence,
            order_method="nearest_neighbor+2opt",
            waypoints=waypoints,
            checks=checks,
        )
