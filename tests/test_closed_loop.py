#!/usr/bin/env python3
"""W1 端到端验收：完整闭环（渲染→检测→坐标→任务→规划→测量→拟合→评价
→报告）一键跑通，含实验1/实验2（移动箱体不重新示教）。"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from box_hole_inspection.main_controller import build_sim_controller

TOLERANCES = {"tol_diameter_mm": 0.15, "tol_position_mm": 0.40,
              "tol_perp_deg": 0.50}


class TestSimClosedLoop(unittest.TestCase):
    def _run(self, box_pose, seed=0):
        controller, world = build_sim_controller(
            box_pose=box_pose, seed=seed, tolerances=TOLERANCES)
        report = controller.run()
        return controller, world, report

    def test_experiment_1_full_loop(self):
        controller, world, report = self._run((500.0, 0.0, 10.0))
        self.assertEqual(controller.sm.current, "COMPLETE")
        # 状态顺序符合设计
        states = [s for s, _ in controller.sm.history]
        self.assertEqual(states[0], "IDLE")
        self.assertIn("AI_DETECTION", states)
        self.assertIn("PATH_PLANNING", states)
        self.assertIn("PROBE_MEASURING", states)
        self.assertIn("REPORT_GENERATION", states)
        # 指标：验收线
        m = report.metrics
        self.assertEqual(m["ai_hole_detect_rate"], 1.0)
        self.assertEqual(m["probe_trigger_success_rate"], 1.0)
        self.assertLess(m["coarse_localization_error_mm"], 2.0)
        self.assertEqual(m["task_success_rate"], 1.0)
        # 评价：4 孔全 PASS
        self.assertEqual(len(report.eval.holes), 4)
        self.assertTrue(all(h.pass_ for h in report.eval.holes))
        self.assertTrue(all(h.perp_ok for h in report.eval.holes))
        # 孔径精度（仿真基线：触发噪声 σ=0.04mm 级）
        for h in report.eval.holes:
            self.assertLess(abs(h.dev_mm), 0.15)

    def test_experiment_2_box_moved_no_reteach(self):
        """实验2：移动+旋转箱体，同一控制器/同一代码路径重新检测。"""
        report1 = self._run((500.0, 0.0, 10.0))[2]
        report2 = self._run((540.0, 30.0, -10.0))[2]
        for r in (report1, report2):
            self.assertEqual(r.metrics["probe_trigger_success_rate"], 1.0)
            self.assertLess(r.metrics["coarse_localization_error_mm"], 2.0)
            self.assertTrue(all(h.pass_ for h in r.eval.holes))

    def test_hole_distance_evaluation(self):
        """孔距评价与真值一致（H01-H02 名义 90mm）。"""
        _, _, report = self._run((500.0, 0.0, 10.0))
        pair = next(p for p in report.eval.hole_pairs
                    if p.ids == ["H01", "H02"])
        self.assertAlmostEqual(pair.nominal_mm, 90.0, places=1)
        self.assertAlmostEqual(pair.dist_mm, 90.0, delta=0.3)
        self.assertTrue(pair.pass_)

    def test_deterministic_given_seed(self):
        r1 = self._run((500.0, 0.0, 10.0), seed=7)[2]
        r2 = self._run((500.0, 0.0, 10.0), seed=7)[2]
        self.assertEqual(r1.eval, r2.eval)
        self.assertEqual(r1.metrics["coarse_localization_error_mm"],
                         r2.metrics["coarse_localization_error_mm"])


class TestRunScriptCLI(unittest.TestCase):
    def test_script_runs_experiment_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run(
                [sys.executable, "scripts/run_sim_closed_loop.py",
                 "--experiment", "1", "--seed", "1",
                 "--out", f"{tmp}/runs"],
                cwd=PROJECT_DIR, capture_output=True, text=True,
                timeout=300)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            report = json.loads(
                (Path(tmp) / "runs" / "report.json").read_text())
            self.assertEqual(report["metrics"]["probe_trigger_success_rate"],
                             1.0)
            self.assertTrue(
                (Path(tmp) / "runs" / "state_trace.txt").exists())


if __name__ == "__main__":
    unittest.main()
