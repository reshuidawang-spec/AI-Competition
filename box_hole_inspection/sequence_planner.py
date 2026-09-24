"""sequence_planner: 多孔检测顺序规划（纯几何，第一阶段用最近邻/2-opt）。

输入孔中心（机器人B基座系），输出检测顺序。第一阶段只保证：
- 总转移路径短（最近邻起步 + 2-opt 局部优化）；
- 顺序确定、可复现（固定输入 → 固定输出）。

更复杂的 TSP 求解器（LKH 等）不属于第一阶段范围。
"""

from __future__ import annotations

from typing import Optional


def order_nearest_neighbor(points: dict, start: Optional[list] = None) -> list:
    """最近邻链：从 start 出发，每次去最近的未访问孔。

    Args:
        points: {id: [x, y, z]}（只用 x/y 平面距离）。
        start: 起始点 [x, y]（默认原点）。
    """
    if not points:
        return []
    ids = list(points.keys())
    current = [0.0, 0.0] if start is None else [float(start[0]), float(start[1])]
    order = []
    remaining = set(ids)
    while remaining:
        nxt = min(remaining, key=lambda i:
                  (points[i][0] - current[0]) ** 2 + (points[i][1] - current[1]) ** 2)
        order.append(nxt)
        remaining.discard(nxt)
        current = [points[nxt][0], points[nxt][1]]
    return order


def _path_length(order: list, points: dict, start) -> float:
    total = 0.0
    prev = start
    for hid in order:
        p = points[hid]
        total += ((p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2) ** 0.5
        prev = [p[0], p[1]]
    return total


def order_tsp_2opt(points: dict, start: Optional[list] = None,
                   max_passes: int = 20) -> list:
    """最近邻初解 + 2-opt 局部优化（不回到起点，从 start 出发的开放路径）。"""
    order = order_nearest_neighbor(points, start=start)
    if len(order) < 3:
        return order
    s = [0.0, 0.0] if start is None else [float(start[0]), float(start[1])]
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        best = _path_length(order, points, s)
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                cand = order[:i] + order[i:j][::-1] + order[j:]
                if _path_length(cand, points, s) < best - 1e-9:
                    order, best, improved = cand, _path_length(cand, points, s), True
    return order
