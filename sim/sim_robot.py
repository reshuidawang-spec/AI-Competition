"""Simulated CR5 6-DOF robot: real URDF kinematics + numerical IK.

Kinematics extracted from the REAL CR5 URDF (2026-09-24):

    dobot_ws/install/dobot_rviz/share/dobot_rviz/urdf/cr5_robot.urdf

Static link origins are in METRES there; converted to mm here. Joint limits
are the URDF <limit> values (joint1/2/4/5/6 ±6.27 rad, joint3 ±2.86 rad).
Every joint rotates about its local z axis.

FK is exact (product of static origin transforms and joint rotations).
IK is numerical damped least squares with random restarts — good enough to
validate the PLANNING PIPELINE in simulation. The REAL robot uses driver IK
/ MoveIt preflight; this module is never the authority for real reachability.

URDF rpy (fixed-axis xyz) maps 1:1 onto the project sxyz intrinsic
convention: Rz(rz) @ Ry(ry) @ Rx(rx) — see utils/transforms.py.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from box_hole_inspection import schemas
from box_hole_inspection.utils import transforms as tf
from interfaces.robot import RobotBase

# ---------------------------------------------------------------------------
# Real CR5 kinematics (URDF source above)
# ---------------------------------------------------------------------------

# (name, origin_xyz_mm, origin_rpy_deg, lower_deg, upper_deg)
CR5_JOINTS = [
    ("joint1", (0.0, 0.0, 147.0), (0.0, 0.0, 0.0), -359.2, 359.2),
    ("joint2", (0.0, 0.0, 0.0), (90.0, 90.0, 0.0), -359.2, 359.2),
    ("joint3", (-427.0, 0.0, 0.0), (0.0, 0.0, 0.0), -163.9, 163.9),
    ("joint4", (-357.0, 0.0, 141.0), (0.0, 0.0, -90.0), -359.2, 359.2),
    ("joint5", (0.0, -116.0, 0.0), (90.0, 0.0, 0.0), -359.2, 359.2),
    ("joint6", (0.0, 105.0, 0.0), (-90.0, 0.0, 0.0), -359.2, 359.2),
]

# 条件数阈值（Jacobian 位置列换算为米后计算）：随机正常构型 cond 中位数约 20、
# p90 约 97，真奇异构型（腕轴重合/臂伸直）cond > 1e16，近奇异上限约 2500。
SINGULAR_COND_THRESHOLD = 2000.0
DEFAULT_MAX_JOINT_VEL_DEG_S = 180.0  # 仿真运动时间估算用


class _Kinematics:
    """FK / Jacobian / numerical IK for the CR5 chain (radians internally)."""

    def __init__(self, joints):
        self.statics = [
            tf.pose_to_T({"x": x, "y": y, "z": z,
                          "rx": rx, "ry": ry, "rz": rz})
            for _, (x, y, z), (rx, ry, rz), _, _ in joints]
        self.lower = np.radians(np.array([j[3] for j in joints]))
        self.upper = np.radians(np.array([j[4] for j in joints]))

    # -- FK ---------------------------------------------------------------

    def fk(self, q_rad: np.ndarray) -> np.ndarray:
        """Joint angles (rad) -> 4x4 flange pose in the robot base frame."""
        T = np.eye(4)
        for T_s, qi in zip(self.statics, q_rad):
            T = T @ T_s @ make_rotz(qi)
        return T

    def axis_origins(self, q_rad: np.ndarray):
        """Per-joint axis (unit) and origin, both in the base frame."""
        axes, origins = [], []
        T = np.eye(4)
        for T_s, qi in zip(self.statics, q_rad):
            T = T @ T_s                       # joint i frame
            z = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
            axes.append(z)
            origins.append(T[:3, 3].copy())
            T = T @ make_rotz(qi)             # advance to next link
        return np.array(axes), np.array(origins)

    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """Geometric 6x6 Jacobian (linear + angular) at the flange."""
        T = self.fk(q_rad)
        p_ee = T[:3, 3]
        axes, origins = self.axis_origins(q_rad)
        J = np.zeros((6, 6))
        for i in range(6):
            J[:3, i] = np.cross(axes[i], p_ee - origins[i])
            J[3:, i] = axes[i]
        return J

    def condition_number(self, q_rad: np.ndarray) -> float:
        """Jacobian condition number.

        Linear columns are converted to metres first so the linear (m) and
        angular (rad) blocks are commensurable; otherwise the mm-scale
        linear block makes every configuration look ill-conditioned.
        """
        J = self.jacobian(q_rad).copy()
        J[:3] /= 1000.0
        s = np.linalg.svd(J, compute_uv=False)
        if s[-1] < 1e-15:
            return float("inf")
        return float(s[0] / s[-1])

    # -- IK ---------------------------------------------------------------

    def ik(self, T_target: np.ndarray, seed_rad: Optional[np.ndarray] = None,
           n_restarts: int = 12, pos_tol_mm: float = 0.1,
           ori_tol_deg: float = 0.2, max_iter: int = 250,
           rng: Optional[np.random.Generator] = None
           ) -> tuple[Optional[np.ndarray], bool]:
        """Damped least squares IK.

        Returns (q_rad | None, singular_flag). q in joint limits or None if
        no seed converged; singular_flag refers to the returned solution.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        ori_tol = math.radians(ori_tol_deg)

        seeds = []
        if seed_rad is not None:
            seeds.append(np.clip(np.asarray(seed_rad, float),
                                 self.lower, self.upper))
        seeds.extend(rng.uniform(self.lower, self.upper,
                                 size=(n_restarts, 6)))

        for q0 in seeds:
            q = q0.copy()
            for _ in range(max_iter):
                T_cur = self.fk(q)
                e = np.zeros(6)
                e[:3] = T_target[:3, 3] - T_cur[:3, 3]
                R_err = T_target[:3, :3] @ T_cur[:3, :3].T
                tr = float(np.trace(R_err))
                ang = math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0)))
                if ang > 1e-9:
                    k = 1.0 / (2.0 * math.sin(ang))
                    axis = k * np.array([R_err[2, 1] - R_err[1, 2],
                                         R_err[0, 2] - R_err[2, 0],
                                         R_err[1, 0] - R_err[0, 1]])
                    e[3:] = axis * ang
                J = self.jacobian(q)
                lam = 0.1 + 0.5 * float(np.linalg.norm(e))  # 自适应阻尼
                A = J @ J.T + lam * lam * np.eye(6)
                dq = J.T @ np.linalg.solve(A, e)
                dq = np.clip(dq, -0.5, 0.5)                # 单步限幅
                q = np.clip(q + dq, self.lower, self.upper)

                err = float(np.linalg.norm(e[:3]))
                if err < pos_tol_mm and ang < ori_tol:
                    singular = (self.condition_number(q)
                                > SINGULAR_COND_THRESHOLD)
                    return q, singular

        # 严格语义：任何种子未收敛即返回 None（不可达/未解出），
        # 不返回"最佳努力解"——否则可达性检查会把错误位姿当成可达。
        return None, False


def make_rotz(a: float) -> np.ndarray:
    """4x4 rotation about the local z axis (radians)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0, 0.0],
                     [s, c, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# RobotBase adapter
# ---------------------------------------------------------------------------

class SimRobot(RobotBase):
    """Kinematic simulation of a CR5. Instant motion (teleport) + time
    estimate; the stop chain / dynamics are NOT simulated — only geometry."""

    def __init__(self, name: str = "sim_cr5",
                 base_pose_w: Optional[np.ndarray] = None,
                 start_joints_deg: Optional[list] = None,
                 max_joint_vel_deg_s: float = DEFAULT_MAX_JOINT_VEL_DEG_S):
        self.name = name
        self.kin = _Kinematics(CR5_JOINTS)
        self.base_pose_w = (np.eye(4) if base_pose_w is None
                            else np.asarray(base_pose_w, dtype=float))
        q0 = ([0.0] * 6 if start_joints_deg is None
              else list(start_joints_deg))
        if len(q0) != 6:
            raise ValueError("start_joints_deg must have 6 values")
        self._q = np.clip(np.array(q0, dtype=float),
                          np.degrees(self.kin.lower),
                          np.degrees(self.kin.upper))
        self.max_joint_vel_deg_s = max_joint_vel_deg_s
        self._speed_ratio = 1.0
        self._estop = False
        self.log: list = []          # (sim_time_ms, event, payload)
        self.sim_time_ms = 0.0

    # -- kinematics helpers ------------------------------------------------

    def fk(self, joint_deg) -> np.ndarray:
        """Flange 4x4 in the robot's own base frame."""
        return self.kin.fk(np.radians(np.asarray(joint_deg, dtype=float)))

    def flange_pose(self) -> schemas.Pose:
        return schemas.Pose(**tf.T_to_pose(self.fk(self._q)))

    def flange_pose_w(self) -> schemas.Pose:
        """Flange pose in the world frame (W ≡ robot-B base)."""
        T_w = tf.compose(self.base_pose_w, self.fk(self._q))
        return schemas.Pose(**tf.T_to_pose(T_w))

    def check_pose(self, pose: schemas.Pose) -> schemas.PlanChecks:
        """Full feasibility check of a pose (no motion)."""
        T = tf.pose_to_T({"x": pose.x, "y": pose.y, "z": pose.z,
                          "rx": pose.rx, "ry": pose.ry, "rz": pose.rz})
        q, singular = self.kin.ik(T)
        checks = schemas.PlanChecks(
            reachable=q is not None,
            joint_limit_ok=True,            # IK 内部已夹紧限位
            singularity_ok=not singular,
            collision_ok=True,              # sim_world 模块接入后覆盖
        )
        if q is None:
            checks.messages.append("IK failed within joint limits")
        if singular:
            checks.messages.append(
                f"near-singular configuration (cond > "
                f"{SINGULAR_COND_THRESHOLD:.0f})")
        return checks

    # -- RobotBase ----------------------------------------------------------

    def get_joint_angles(self) -> list:
        return [float(x) for x in self._q]

    def get_flange_pose(self) -> schemas.Pose:
        return self.flange_pose()

    def _guard(self):
        if self._estop:
            raise RuntimeError(f"{self.name}: emergency stop active")

    def move_j(self, joint_deg: list, speed_ratio: float = 0.3) -> None:
        self._guard()
        target = np.clip(np.asarray(joint_deg, dtype=float),
                         np.degrees(self.kin.lower),
                         np.degrees(self.kin.upper))
        dt_s = (float(np.max(np.abs(target - self._q)))
                / max(self.max_joint_vel_deg_s * speed_ratio, 1e-9))
        self._q = target
        self.sim_time_ms += dt_s * 1000.0
        self.log.append((self.sim_time_ms, "move_j",
                         [float(x) for x in target]))

    def move_l(self, pose: schemas.Pose, speed_ratio: float = 0.3) -> None:
        self._guard()
        T = tf.pose_to_T({"x": pose.x, "y": pose.y, "z": pose.z,
                          "rx": pose.rx, "ry": pose.ry, "rz": pose.rz})
        q, _ = self.kin.ik(T, seed_rad=np.radians(self._q))
        if q is None:
            raise RuntimeError(f"{self.name}: move_l IK failed for {pose}")
        self.move_j(np.degrees(q), speed_ratio=speed_ratio)

    def stop(self) -> None:
        self.log.append((self.sim_time_ms, "stop", None))

    def set_speed_ratio(self, ratio: float) -> None:
        self._speed_ratio = float(ratio)

    def reachable(self, pose: schemas.Pose) -> bool:
        return self.check_pose(pose).reachable

    def estop(self) -> None:
        self._estop = True
        self.log.append((self.sim_time_ms, "estop", None))

    def is_estop(self) -> bool:
        return self._estop

    def status(self) -> schemas.RobotStatus:
        return schemas.RobotStatus(
            state="error" if self._estop else "idle",
            joint_deg=self.get_joint_angles(),
            tcp_pose=self.flange_pose(),
            estop=self._estop,
        )
