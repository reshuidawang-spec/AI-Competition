"""hole_assigner: 检测孔 -> CAD 名义孔的编号分配与箱体位姿估计。

第一阶段箱体只有 (x, y, rz) 自由度：检测到的孔点（世界系 xy）与 CAD
名义孔模式（箱体系 xy）之间是平面刚体变换。4 个孔只有 24 种对应排列，
逐一用 Umeyama SVD 拟合刚体变换取最小残差——精确、无初值、无迭代。

输出：
- 每个检测孔的正确编号（H01..H04，与名义孔对应）
- 箱体位姿 (x, y, rz)（z = 顶面平面，由坐标链提供）
- 每孔配准残差（= 视觉粗定位误差的逐孔估计）

AI 只做感知原则的落地：网络不学"谁是 H01"，编号由几何配准决定，
箱体移动/旋转后编号依然正确。
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from box_hole_inspection import schemas


class HoleAssignmentError(Exception):
    """孔数与名义模式不匹配等无法配准的情况。"""


def umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple:
    """2D 刚体变换拟合（Umeyama/SVD）：返回 (theta_deg, tx, ty, rms)。"""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s = src - mu_s
    d = dst - mu_d
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    t = mu_d - R @ mu_s
    res = (R @ s.T).T - d
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    return theta, float(t[0]), float(t[1]), rms


class HoleAssigner:
    """检测孔集 -> 名义孔模式配准。

    rz_range: 平局裁决区间（度）。孔模式有 180° 旋转对称性（如四角对称
    孔）时，正确解与翻转解 RMS 完全相等——此时优先取 rz 落在该区间内
    的解（对应箱体长边与世界系夹角 ±90° 内的摆放约定）。模式不对称时
    严格最小 RMS 自然消歧，此裁决不会触发。
    """

    def __init__(self, part_spec: dict, rz_range: tuple = (-90.0, 90.0)):
        nominal = []
        for h in part_spec.get("holes", []):
            c = h.get("center_o_mm")
            if c and c[0] is not None and c[1] is not None:
                nominal.append((h["id"], np.array([c[0], c[1]], dtype=float)))
        if len(nominal) < 2:
            raise HoleAssignmentError("part spec 至少需要两个已知孔位")
        self.nominal_ids = [i for i, _ in nominal]
        self.nominal_pts = np.array([p for _, p in nominal])
        self.rz_lo, self.rz_hi = rz_range

    @staticmethod
    def _normalize_rz(theta: float) -> float:
        rz = theta % 360.0
        if rz > 180.0:
            rz -= 360.0
        return rz

    def assign(self, localization: schemas.LocalizationResult,
               ) -> schemas.LocalizationResult:
        """配准并修正孔编号与箱体位姿；返回新的 LocalizationResult。"""
        holes = [h for h in localization.holes if h.world_xyz is not None]
        if len(holes) != len(self.nominal_pts):
            raise HoleAssignmentError(
                f"检测到 {len(holes)} 个孔，名义模式 {len(self.nominal_pts)} 个")
        detected = np.array([[h.world_xyz[0], h.world_xyz[1]]
                             for h in holes])

        best_rms = float("inf")
        candidates = []
        for perm in itertools.permutations(range(len(holes))):
            theta, tx, ty, rms = umeyama_2d(detected[list(perm)],
                                            self.nominal_pts)
            if rms < best_rms - 1e-9:
                best_rms, candidates = rms, []
            if abs(rms - best_rms) <= 1e-9:
                candidates.append((theta, tx, ty, perm))
        # 平局裁决：优先 rz 在约定区间内的解
        chosen = min(candidates, key=lambda c: (
            0.0 if self.rz_lo <= self._normalize_rz(c[0]) <= self.rz_hi
            else float("inf"),))
        theta, tx, ty, perm = chosen
        # 拟合方向: nominal = R(θ)·detected + t
        # 世界位姿取逆映射: detected = R(−θ)·nominal + (−R(−θ)·t)
        theta_w = -theta
        rz = self._normalize_rz(theta_w)
        a = math.radians(theta_w)
        R_w = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]])
        t_w = -R_w @ np.array([tx, ty])
        tx_w, ty_w = float(t_w[0]), float(t_w[1])

        # 名义编号 -> 检测孔索引
        matched = {self.nominal_ids[k]: holes[i]
                   for k, i in enumerate(perm)}
        out_holes = []
        for hid, h in matched.items():
            # 单个残差：检测点经拟合变换后与名义点的距离
            a = math.radians(theta)
            R = np.array([[math.cos(a), -math.sin(a)],
                          [math.sin(a), math.cos(a)]])
            p_fit = R @ np.array([h.world_xyz[0], h.world_xyz[1]]) \
                + np.array([tx, ty])
            k = self.nominal_ids.index(hid)
            res = float(np.linalg.norm(p_fit - self.nominal_pts[k]))
            out_holes.append(schemas.LocalizedHole(
                id=hid, confidence=h.confidence,
                pixel_u=h.pixel_u, pixel_v=h.pixel_v,
                camera_xyz=h.camera_xyz,
                world_xyz=h.world_xyz, bb_xyz=h.bb_xyz,
                fit_residual_mm=res))

        pose = schemas.Pose(x=tx_w, y=ty_w,
                            z=localization.pose.z, rz=rz)
        return schemas.LocalizationResult(
            part_id=localization.part_id,
            confidence=localization.confidence,
            timestamp_ms=localization.timestamp_ms,
            pose=pose,
            holes=sorted(out_holes, key=lambda h: h.id),
            coarse_error_est_mm=best_rms,
        )
