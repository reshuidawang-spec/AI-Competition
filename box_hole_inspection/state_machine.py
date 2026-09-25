"""state_machine: 表驱动流程状态机（最小通用实现）。

状态 = 名称 -> handler；handler 返回 (next_state, message)。历史记录供
GUI 显示"当前阶段"与复盘。终止状态：COMPLETE / ERROR / ABORT。

真实系统扩展（安全监控线程注入 ESTOP 事件、可恢复错误重试点）由
main_controller 的 handler 实现；本模块只提供驱动骨架。
"""

from __future__ import annotations

from typing import Callable

TERMINAL_STATES = {"COMPLETE", "ERROR", "ABORT"}


class StateMachine:
    def __init__(self, initial: str = "IDLE"):
        self.states: dict = {}
        self.initial = initial
        self.current = initial
        self.history: list = []       # [(state, message), ...]

    def add_state(self, name: str,
                  handler: Callable[[], tuple]) -> None:
        self.states[name] = handler

    def run(self, max_steps: int = 500) -> str:
        """驱动状态机直到终止状态；返回最终状态名。"""
        self.current = self.initial
        steps = 0
        while self.current not in TERMINAL_STATES:
            handler = self.states.get(self.current)
            if handler is None:
                raise RuntimeError(f"no handler for state {self.current}")
            nxt, msg = handler()
            self.history.append((self.current, msg))
            self.current = nxt
            steps += 1
            if steps > max_steps:
                raise RuntimeError(
                    f"state machine exceeded {max_steps} steps at "
                    f"{self.current}")
        return self.current

    def trace(self) -> str:
        """一行一状态的可读轨迹。"""
        return "\n".join(f"{i:3d} {s:20s} {m}" for i, (s, m)
                         in enumerate(self.history))
