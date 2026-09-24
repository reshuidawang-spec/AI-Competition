"""Robot abstraction layer (6-DOF arm).

Both robots (A: vision arm 192.168.1.7, B: measurement arm 192.168.5.1) use
the same interface; only the adapter instance differs.

REAL ADAPTER REPLACEMENT POINTS (interfaces/adapters/real_robot_ros2.py, to be
implemented on the validated driver conventions of cr5_rmp60_measurement):

- get_joint_angles()  -> rclpy client of /dobot_bringup_ros2/srv/GetAngle,
                          or the /joint_states_robot feedback topic.
- get_flange_pose()   -> /dobot_bringup_ros2/srv/GetPose (+ FeedInfo freshness
                          check, as the sibling project does).
- move_j()/move_l()   -> /dobot_bringup_ros2/srv/MovJ / MovL. NOTE: the
                          Dobot rx/ry/rz convention must be validated before
                          trusting orientations (known open issue).
- stop()              -> /dobot_bringup_ros2/srv/Stop (software stop chain:
                          leave stop_overtravel_reserve_mm clearance).
- reachable()         -> /dobot_bringup_ros2/srv/InverseKin (srv-based) or
                          MoveIt /compute_ik preflight.
- estop()/is_estop()  -> EmergencyStop service / driver state.

Robot A (192.168.1.7) uses the same adapter with driver node
robot_node_name2 (config/system.yaml).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from box_hole_inspection import schemas


class RobotBase(ABC):
    """Abstract 6-DOF robot. Sim and real adapters implement the same
    interface so callers never branch on hardware."""

    @abstractmethod
    def get_joint_angles(self) -> list:
        """Current joint angles in degrees."""

    @abstractmethod
    def get_flange_pose(self) -> schemas.Pose:
        """Current flange pose in the robot's own base frame (mm, deg)."""

    @abstractmethod
    def move_j(self, joint_deg: list, speed_ratio: float = 0.3) -> None:
        """Joint-space move to target joint angles."""

    @abstractmethod
    def move_l(self, pose: schemas.Pose, speed_ratio: float = 0.3) -> None:
        """Cartesian move to a pose in the robot's own base frame."""

    @abstractmethod
    def stop(self) -> None:
        """Software stop (with the real driver: Stop service)."""

    @abstractmethod
    def set_speed_ratio(self, ratio: float) -> None:
        """Set global speed factor (real driver: SpeedFactor service)."""

    @abstractmethod
    def reachable(self, pose: schemas.Pose) -> bool:
        """IK feasibility check for a pose (no motion)."""

    @abstractmethod
    def estop(self) -> None:
        """Emergency stop."""

    @abstractmethod
    def is_estop(self) -> bool:
        """True while emergency stop is active."""

    @abstractmethod
    def status(self) -> schemas.RobotStatus:
        """Status snapshot for the GUI / safety monitor."""
