"""tolerance_evaluator: FitResult -> EvalResult（公差评价，纯逻辑）。

第一阶段评价项（与比赛演示对应）：
- 孔径：测量直径 vs 名义直径 + 公差
- 垂直度：孔轴 vs 顶面法向（箱体平放时 = [0,0,1]）
- 孔距：任意两孔中心距 vs 名义孔距（箱体系内，与箱体位姿无关）
- 位置度/孔心偏差：需要名义孔心在世界系的参考——依赖箱体位姿
  （hole_assigner 或仿真真值）。box_pose 未提供时该项为 None。

明确不做：圆柱度、轮廓度、MMC/LMC 高级公差（第一阶段范围外）。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from box_hole_inspection import schemas


class ToleranceEvaluator:
    def __init__(self, part_spec: dict,
                 box_pose: Optional[schemas.Pose] = None,
                 top_face_normal=(0.0, 0.0, 1.0)):
        self._nominal_by_id = {h["id"]: h for h in part_spec.get("holes", [])}
        self.box_pose = box_pose
        self.top_normal = np.asarray(top_face_normal, dtype=float)
        self.top_normal = self.top_normal / np.linalg.norm(self.top_normal)

    # -- helpers ---------------------------------------------------------------

    def _nominal_center_w(self, hole_id: str) -> Optional[np.ndarray]:
        """名义孔心在世界系（需箱体位姿）；无位姿返回 None。"""
        h = self._nominal_by_id.get(hole_id)
        if h is None or self.box_pose is None:
            return None
        c = h.get("center_o_mm")
        if not c or c[0] is None or c[1] is None:
            return None
        rz = math.radians(self.box_pose.rz)
        cos, sin = math.cos(rz), math.sin(rz)
        return np.array([
            self.box_pose.x + cos * c[0] - sin * c[1],
            self.box_pose.y + sin * c[0] + cos * c[1],
            self.box_pose.z,
        ])

    # -- per hole -----------------------------------------------------------------

    def evaluate_hole(self, fit: schemas.FitResult) -> schemas.HoleEval:
        h = self._nominal_by_id.get(fit.hole_id, {})
        d_nom = h.get("diameter_mm")
        tol_d = float(h.get("tol_diameter_mm", 0.05))
        tol_p = float(h.get("tol_position_mm", 0.10))
        tol_perp = float(h.get("tol_perp_deg", 0.10))

        ev = schemas.HoleEval(id=fit.hole_id)
        if d_nom is not None:
            ev.diameter_mm = fit.diameter_mm
            ev.dev_mm = fit.diameter_mm - float(d_nom)
            ev.tol_mm = tol_d
            ev.pass_ = abs(ev.dev_mm) <= tol_d

        # 垂直度：孔轴与顶面法向夹角
        d = np.asarray(fit.axis.direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-12)
        ev.perp_dev_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, float(np.dot(d, self.top_normal))))))
        ev.perp_tol_deg = tol_perp
        ev.perp_ok = ev.perp_dev_deg <= tol_perp

        # 位置度：测量孔心 vs 名义孔心（世界系）
        c_nom = self._nominal_center_w(fit.hole_id)
        if c_nom is not None:
            ev.center_dev_mm = float(
                np.linalg.norm(np.asarray(fit.center_bb) - c_nom))
            ev.position_tol_mm = tol_p
            ev.position_ok = ev.center_dev_mm <= tol_p
        return ev

    # -- whole part -----------------------------------------------------------------

    def evaluate(self, fits: dict) -> schemas.EvalResult:
        """{hole_id: FitResult} -> EvalResult（含孔距评价）。"""
        holes = [self.evaluate_hole(f) for f in fits.values()]
        pairs = []
        ids = sorted(fits.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                hi, hj = self._nominal_by_id.get(ids[i], {}), \
                    self._nominal_by_id.get(ids[j], {})
                ci, cj = hi.get("center_o_mm"), hj.get("center_o_mm")
                if (not ci or not cj or ci[0] is None or cj[0] is None):
                    continue
                nominal = float(np.hypot(ci[0] - cj[0], ci[1] - cj[1]))
                measured = float(np.linalg.norm(
                    np.asarray(fits[ids[i]].center_bb)
                    - np.asarray(fits[ids[j]].center_bb)))
                dev = measured - nominal
                tol_d = float(hi.get("tol_position_mm", 0.10))
                pairs.append(schemas.HolePairEval(
                    ids=[ids[i], ids[j]], dist_mm=measured,
                    nominal_mm=nominal, dev_mm=dev, tol_mm=tol_d,
                    pass_=abs(dev) <= tol_d))

        n_pass = sum(1 for h in holes if h.pass_)
        return schemas.EvalResult(
            holes=holes, hole_pairs=pairs,
            summary={"n_total": len(holes), "n_pass": n_pass,
                     "n_fail": len(holes) - n_pass})
