"""Touch-probe abstraction (RMP60 -> controller DI1).

REAL ADAPTER REPLACEMENT POINTS (interfaces/adapters/real_probe_ros2.py,
following the validated cr5_rmp60_measurement conventions):

- read_trigger()   -> /dobot_bringup_ros2/srv/DI request; trigger = DI1 == 1.
- wait_trigger()   -> poll DI at ~1 ms cadence while the robot moves along a
                      segmented probe path (probe_step_mm); on trigger the
                      caller stops the robot immediately (software stop chain,
                      so keep stop_overtravel_reserve_mm clearance).
- wait_released()  -> poll until DI1 == 0 after retract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProbeBase(ABC):
    """Trigger semantics shared by sim and real probes."""

    @abstractmethod
    def configure(self, trigger_io: str = "DI1") -> None:
        """Select the trigger input channel."""

    @abstractmethod
    def read_trigger(self) -> bool:
        """Instantaneous trigger state."""

    @abstractmethod
    def wait_trigger(self, timeout_s: float) -> bool:
        """Block until trigger or timeout; returns True if triggered."""

    @abstractmethod
    def wait_released(self, timeout_s: float) -> bool:
        """Block until the trigger input is released; True if released."""
