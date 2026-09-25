#!/usr/bin/env python3
"""第一周最小闭环一键脚本：渲染 -> 检测 -> 坐标 -> 任务 -> 规划 -> 测量
-> 拟合 -> 评价 -> 报告（全部仿真）。

用法（实验1：位置A / 实验2：移动箱体后重新检测，均不改任何示教）：

    python3 scripts/run_sim_closed_loop.py --experiment 1
    python3 scripts/run_sim_closed_loop.py --experiment 2
    python3 scripts/run_sim_closed_loop.py --box-x 520 --box-y -25 --box-rz -15

报告写入 data/runs/YYYY-MM-DD/<session_id>/report.json，指标与比赛
记录表对应（docs/experiment_plan.md）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection import schemas
from box_hole_inspection.main_controller import build_sim_controller

# 实验场景（比赛演示实验1/实验2）
EXPERIMENTS = {
    1: {"box_x": 500.0, "box_y": 0.0, "box_rz": 10.0},
    2: {"box_x": 540.0, "box_y": 30.0, "box_rz": -10.0},   # ΔX+40 ΔY+30 ΔRz-20
}

# 仿真评价公差（真实值按 part_box001.yaml；演示用放宽值）
TOLERANCES = {"tol_diameter_mm": 0.15, "tol_position_mm": 0.40,
              "tol_perp_deg": 0.50}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=int, choices=(1, 2),
                   help="比赛演示场景（1=位置A，2=移动箱体）")
    p.add_argument("--box-x", type=float, default=None)
    p.add_argument("--box-y", type=float, default=None)
    p.add_argument("--box-rz", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="报告输出目录（默认自动）")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.experiment is not None:
        pose = EXPERIMENTS[args.experiment]
        box_x = pose["box_x"]; box_y = pose["box_y"]; box_rz = pose["box_rz"]
    else:
        box_x = args.box_x if args.box_x is not None else 500.0
        box_y = args.box_y if args.box_y is not None else 0.0
        box_rz = args.box_rz if args.box_rz is not None else 10.0

    controller, world = build_sim_controller(
        box_pose=(box_x, box_y, box_rz), seed=args.seed,
        tolerances=TOLERANCES)

    report = controller.run()

    out_dir = Path(args.out) if args.out else (
        PROJECT_DIR / "data" / "runs" / datetime.now().strftime("%Y-%m-%d")
        / f"sim_closed_loop_{int(report.timestamp_ms)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(schemas.to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "state_trace.txt").write_text(
        controller.sm.trace(), encoding="utf-8")

    if not args.quiet:
        m = report.metrics
        print(f"箱体位姿: x={box_x:.0f} y={box_y:.0f} rz={box_rz:.0f}°")
        print(f"状态机终点: {controller.sm.current}")
        print(f"孔检测率   : {m['ai_hole_detect_rate']*100:.0f}% "
              f"(置信度均值 {m['ai_confidence_mean']:.2f})")
        print(f"粗定位误差 : {m['coarse_localization_error_mm']:.3f} mm "
              f"(验收线 2.0)")
        print(f"触发成功率 : {m['probe_trigger_success_rate']*100:.0f}%")
        print(f"总耗时     : {m['total_time_s']:.1f} s "
              f"(机器人B运动 {m['robot_b_motion_time_s']:.1f} s)")
        for h in report.eval.holes:
            print(f"  {h.id}: D={h.diameter_mm:.3f} (dev {h.dev_mm:+.3f} "
                  f"tol ±{h.tol_mm}) 垂直度 {h.perp_dev_deg:.3f}° "
                  f"{'PASS' if h.pass_ and h.perp_ok else 'FAIL'}")
        for pr in report.eval.hole_pairs:
            print(f"  {pr.ids[0]}-{pr.ids[1]}: dist={pr.dist_mm:.3f} "
                  f"(nom {pr.nominal_mm:.1f}, dev {pr.dev_mm:+.3f}) "
                  f"{'PASS' if pr.pass_ else 'FAIL'}")
        print(f"报告: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
