"""geometry_fitting: 圆 / 轴线最小二乘拟合（纯数学，无硬件依赖）。

- fit_circle_kasa: 代数法线性初值（min Σ(‖p−c‖²−r²)²）
- fit_circle_geometric: Gauss-Newton 几何精修（min Σ(‖p−c‖−r)²），输出 RMS
- fit_circle_section: 截面接触点（同高度）→ 2D 圆拟合
- fit_axis: 两个及以上截面圆心 → 孔轴（PCA 直线拟合）

第一阶段孔轴竖直：截面点落在同一 z 平面，用 xy 直接拟合；轴线方向
向上归一化（与顶面法向 [0,0,1] 夹角即垂直度偏差）。
"""

from __future__ import annotations

import numpy as np


def fit_circle_kasa(points_xy: np.ndarray) -> tuple:
    """Kåsa 代数圆拟合：返回 (cx, cy, r)。退化（点数<3）返回 (0,0,0)。"""
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    # min Σ(‖p‖² − 2p·c + k)²，未知量 (cx, cy, k)，k = ‖c‖² − r²
    A = np.column_stack([-2.0 * pts[:, 0], -2.0 * pts[:, 1],
                         np.ones(len(pts))])
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = x
    r2 = cx * cx + cy * cy - k
    if r2 <= 0:
        return 0.0, 0.0, 0.0
    return float(cx), float(cy), float(np.sqrt(r2))


def fit_circle_geometric(points_xy: np.ndarray, init=None,
                         max_iter: int = 100,
                         tol: float = 1e-9) -> tuple:
    """几何圆拟合（Gauss-Newton）：返回 (cx, cy, r, rms_mm)。

    init 缺省用 Kåsa 结果；rms 为 Σ(‖p−c‖−r)² 的均方根。
    """
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0, float("inf")
    if init is None:
        cx, cy, r = fit_circle_kasa(pts)
    else:
        cx, cy, r = init
    for _ in range(max_iter):
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        d[d < 1e-12] = 1e-12
        res = d - r
        J = np.column_stack([-(pts[:, 0] - cx) / d,
                             -(pts[:, 1] - cy) / d,
                             -np.ones(len(pts))])
        # Gauss-Newton 步长：δ = −J⁺·res（缺负号会在噪声下发散）
        step, *_ = np.linalg.lstsq(J, res, rcond=None)
        cx -= step[0]; cy -= step[1]; r -= step[2]
        if np.linalg.norm(step) < tol:
            break
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    rms = float(np.sqrt(np.mean((d - r) ** 2)))
    return float(cx), float(cy), float(r), rms


def fit_circle_section(points_3d: np.ndarray) -> tuple:
    """截面接触点（N,3，同 z 平面）→ (center_3d, radius, rms_mm)。"""
    pts = np.asarray(points_3d, dtype=float)
    if len(pts) < 3:
        return None, 0.0, float("inf")
    cx, cy, r, rms = fit_circle_geometric(pts[:, :2])
    center = np.array([cx, cy, float(np.mean(pts[:, 2]))])
    return center, r, rms


def fit_axis(section_centers: np.ndarray) -> tuple:
    """截面圆心 (M,3) → 孔轴 (point, direction)。

    point = 质心；direction = PCA 主轴（SVD 最大奇异向量），z 分量为负
    时翻转（孔轴约定向上）。
    """
    pts = np.asarray(section_centers, dtype=float)
    if len(pts) < 2:
        raise ValueError("fit_axis 至少需要两个截面圆心")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    direction = vh[0]
    if direction[2] < 0:
        direction = -direction
    n = np.linalg.norm(direction)
    if n > 1e-12:
        direction = direction / n
    return centroid, direction
