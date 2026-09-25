"""main_controller: 全流程编排（状态机驱动）+ 仿真单元工厂。

状态流程（2026-09-23 定稿）：

    IDLE -> VISION_POSITIONING -> IMAGE_CAPTURE -> AI_DETECTION
    -> COORDINATE_TRANSFORM -> TASK_GENERATION -> PATH_PLANNING
    -> [ROBOT_MOVING -> PROBE_MEASURING] × 每孔
    -> GEOMETRY_FITTING -> TOLERANCE_EVALUATION -> REPORT_GENERATION
    -> COMPLETE         （任何失败 -> ERROR）

build_sim_controller() 构造完整仿真单元（世界/相机/双臂/测头/检测器/
坐标链），供一键脚本与端到端测试共用。仿真标定自洽（T_BA_BB 固定、
手眼恒等、平面=箱体顶面），真实标定值在 config/calibration.json 中
替换后走同一路径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from box_hole_inspection import schemas
from box_hole_inspection.coordinate_transform import CoordinateTransform
from box_hole_inspection.geometry_fitting import fit_axis, fit_circle_section
from box_hole_inspection.hole_assigner import HoleAssigner, HoleAssignmentError
from box_hole_inspection.path_planner import PathPlanner
from box_hole_inspection.probe_measurement import ProbeMeasurement
from box_hole_inspection.state_machine import StateMachine
from box_hole_inspection.task_generator import TaskGenerator
from box_hole_inspection.tolerance_evaluator import ToleranceEvaluator
from box_hole_inspection.utils import transforms as tf
from box_hole_inspection.vision_ai import GeometricHoleDetector
from interfaces.camera import CameraIntrinsics
from sim.sim_camera import SimCamera
from sim.sim_probe import SimProbe
from sim.sim_robot import SimRobot
from sim.sim_world import SimWorld, default_box, part_spec_from_box


@dataclass
class RunContext:
    """一次检测运行的全部中间产物（复盘/GUI 用）。"""
    image: Optional[np.ndarray] = None
    detection: Optional[schemas.DetectionResult] = None
    localization: Optional[schemas.LocalizationResult] = None
    task_list: Optional[schemas.TaskList] = None
    plan: Optional[schemas.MotionPlan] = None
    contacts: dict = field(default_factory=dict)     # hole_id -> [ContactPoints]
    fits: dict = field(default_factory=dict)         # hole_id -> FitResult
    eval: Optional[schemas.EvalResult] = None
    report: Optional[schemas.RunReport] = None
    error: Optional[str] = None


def build_sim_controller(box_pose=(500.0, 0.0, 10.0), seed: int = 0,
                         safety: Optional[dict] = None,
                         tolerances: Optional[dict] = None,
                         robot_a_base_pose: Optional[dict] = None):
    """构造完整仿真单元：返回 (controller, world)。

    视觉臂 A 基座默认在世界系 (1000, 0, 0, rz=90°)（与测量臂 B 不重叠）；
    拍摄位姿由 IK 自动求解到箱体正上方 480mm 俯视。
    """
    safety = safety or {"stop_overtravel_reserve_mm": 2.0,
                        "probe_step_mm": 0.5,
                        "contact_speed_ratio": 0.05,
                        "transfer_speed_ratio": 0.3,
                        "safe_z_margin_mm": 40.0}

    world = SimWorld(default_box())
    world.set_box_pose(*box_pose)

    t_ba_bb = tf.pose_to_T(robot_a_base_pose or
                           {"x": 1000, "y": 0, "z": 0,
                            "rx": 0, "ry": 0, "rz": 90})
    robot_a = SimRobot(name="robot_a", base_pose_w=t_ba_bb)
    robot_b = SimRobot(name="robot_b")
    probe = SimProbe(world, ball_radius_mm=1.0, seed=seed)

    # 视觉臂拍摄位姿：相机在箱体正上方 480mm 俯视（IK 自动求关节角）
    intrinsics = CameraIntrinsics(fx=1000, fy=1000, cx=640, cy=512,
                                  width=1280, height=1024)
    t_w_c_target = tf.pose_to_T({"x": box_pose[0], "y": box_pose[1],
                                 "z": 480.0, "rx": 180, "ry": -8, "rz": 0})
    t_ea_c = np.eye(4)          # 仿真手眼恒等；真实值来自标定
    t_a_target = tf.compose(tf.invert_T(t_ba_bb), t_w_c_target)
    pose_a = schemas.Pose(**tf.T_to_pose(t_a_target))
    if not robot_a.reachable(pose_a):
        raise RuntimeError("机器人A拍摄位姿不可达，请调整布局参数")
    robot_a.move_l(pose_a)
    camera = SimCamera.from_robot_a(robot_a, t_ea_c,
                                    capture_joints_deg=robot_a.get_joint_angles(),
                                    intrinsics=intrinsics, seed=seed)

    calibration = {
        "T_EA_C": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_EB_P": {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0},
        "T_BA_BB": tf.T_to_pose(t_ba_bb),   # 与视觉臂 base_pose_w 一致
        "camera_intrinsics": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 512},
        "table_plane_w": {"point": [0, 0, world.box.z_top_mm],
                          "normal": [0, 0, 1]},
        "probe": {"ball_radius_mm": 1.0, "trigger_io": "DI1"},
    }
    # 坐标链的相机位姿 = T_BA_BB @ FK(θ_A) @ T_EA_C，与 SimCamera 的
    # base_pose_w @ FK @ T_EA_C 完全一致（闭环测试交叉验证）
    chain = CoordinateTransform(
        calibration,
        t_ba_ea_provider=lambda: robot_a.fk(robot_a.get_joint_angles()))

    part_spec = part_spec_from_box(world.box, tolerances=tolerances)
    detector = GeometricHoleDetector()
    assigner = HoleAssigner(part_spec)
    generator = TaskGenerator(part_spec, safety=safety)
    planner = PathPlanner(robot_b, safety=safety)
    measurer = ProbeMeasurement(robot_b, probe, ball_radius_mm=1.0)
    evaluator = ToleranceEvaluator(
        part_spec,
        box_pose=schemas.Pose(x=box_pose[0], y=box_pose[1],
                              z=world.box.z_top_mm, rz=box_pose[2]))
    controller = MainController(
        world=world, camera=camera, robot_a=robot_a, robot_b=robot_b,
        probe=probe, detector=detector, chain=chain, assigner=assigner,
        generator=generator, planner=planner, measurer=measurer,
        evaluator=evaluator, start_xy=[0.0, 0.0])
    return controller, world


class MainController:
    """状态机编排器：持有全部模块，暴露 run() -> RunReport。"""

    def __init__(self, world, camera, robot_a, robot_b, probe, detector,
                 chain, assigner, generator, planner, measurer, evaluator,
                 start_xy=None, part_id="BOX_001"):
        self.world = world
        self.camera = camera
        self.robot_a = robot_a
        self.robot_b = robot_b
        self.probe = probe
        self.detector = detector
        self.chain = chain
        self.assigner = assigner
        self.generator = generator
        self.planner = planner
        self.measurer = measurer
        self.evaluator = evaluator
        self.start_xy = start_xy
        self.part_id = part_id
        self.ctx = RunContext()
        self._hole_iter = None
        self._t_start_ms = time.time() * 1000.0
        self.sm = self._build_state_machine()

    # -- state handlers ------------------------------------------------------

    def _h_vision_positioning(self):
        # 视觉臂已在构造时到达拍摄位姿（仿真）；真机在此执行运动+确认
        return "IMAGE_CAPTURE", "robot A at capture pose"

    def _h_image_capture(self):
        render = self.camera.render(self.world)
        self.ctx.image = render.image
        self._render = render
        return "AI_DETECTION", "frame captured"

    def _h_ai_detection(self):
        self.ctx.detection = self.detector.detect(self.ctx.image)
        n = len(self.ctx.detection.holes)
        return "COORDINATE_TRANSFORM", f"detected {n} holes"

    def _h_coordinate_transform(self):
        loc = self.chain.localize(self.ctx.detection)
        try:
            self.ctx.localization = self.assigner.assign(loc)
        except HoleAssignmentError as e:
            self.ctx.error = f"hole assignment failed: {e}"
            return "ERROR", self.ctx.error
        rms = self.ctx.localization.coarse_error_est_mm
        return "TASK_GENERATION", f"localized, pattern-fit RMS {rms:.3f} mm"

    def _h_task_generation(self):
        self.ctx.task_list = self.generator.generate(self.ctx.localization)
        if not self.ctx.task_list.tasks:
            self.ctx.error = "no measurement tasks generated: " + \
                "; ".join(f"{s['id']}:{s['reason']}"
                          for s in self.ctx.task_list.skipped)
            return "ERROR", self.ctx.error
        return "PATH_PLANNING", f"{len(self.ctx.task_list.tasks)} tasks"

    def _h_path_planning(self):
        self.ctx.plan = self.planner.plan(self.ctx.task_list,
                                          start_xy=self.start_xy)
        c = self.ctx.plan.checks
        if not (c.reachable and c.singularity_ok):
            self.ctx.error = "plan checks failed: " + "; ".join(c.messages)
            return "ERROR", self.ctx.error
        self._hole_iter = iter(self.ctx.plan.sequence)
        return "ROBOT_MOVING", "plan OK: " + " -> ".join(self.ctx.plan.sequence)

    def _h_robot_moving(self):
        hid = next(self._hole_iter, None)
        if hid is None:
            return "GEOMETRY_FITTING", "all holes measured"
        self._current_hole = hid
        task = next(t for t in self.ctx.task_list.tasks if t.hole_id == hid)
        wps = [w for w in self.ctx.plan.waypoints if w.hole_id == hid]
        self.ctx.contacts[hid] = self.measurer.measure_hole(wps, task)
        return "PROBE_MEASURING", f"{hid} motion+probe done"

    def _h_probe_measuring(self):
        secs = self.ctx.contacts[self._current_hole]
        n = sum(s.n_triggered for s in secs)
        planned = sum(s.n_planned for s in secs)
        return "ROBOT_MOVING", (f"{self._current_hole}: "
                                f"{n}/{planned} triggered")

    def _h_geometry_fitting(self):
        for hid, sections in self.ctx.contacts.items():
            sec_fits = []
            for sec in sections:
                pts = np.array([e.contact_point_bb for e in sec.contacts])
                center, r, rms = fit_circle_section(pts)
                if center is None:
                    self.ctx.error = f"{hid}: 拟合点不足"
                    return "ERROR", self.ctx.error
                sec_fits.append(schemas.SectionFit(
                    z_mm=sec.section_z_mm,
                    center_bb=[float(v) for v in center],
                    radius_mm=r, residual_rms_um=rms * 1000.0,
                    n_points=len(pts)))
            centers = np.array([s.center_bb for s in sec_fits])
            if len(centers) >= 2:
                point, direction = fit_axis(centers)
            else:
                point, direction = np.asarray(sec_fits[0].center_bb), \
                    np.array([0.0, 0.0, 1.0])
            d_mean = float(np.mean([2.0 * s.radius_mm for s in sec_fits]))
            self.ctx.fits[hid] = schemas.FitResult(
                hole_id=hid, sections=sec_fits,
                center_bb=[float(v) for v in point],
                diameter_mm=d_mean,
                axis=schemas.AxisFit(
                    point=[float(v) for v in point],
                    direction=[float(v) for v in direction]))
        return "TOLERANCE_EVALUATION", f"{len(self.ctx.fits)} holes fitted"

    def _h_tolerance_evaluation(self):
        self.ctx.eval = self.evaluator.evaluate(self.ctx.fits)
        return "REPORT_GENERATION", "evaluated"

    def _h_report_generation(self):
        centers_w = self.world.hole_centers_w()
        coarse = []
        if self.ctx.localization is not None:
            for h in self.ctx.localization.holes:
                if h.id in centers_w:
                    coarse.append(float(np.linalg.norm(
                        np.asarray(h.world_xyz) - centers_w[h.id])))
        n_plan = sum(s.n_planned for secs in self.ctx.contacts.values()
                     for s in secs)
        n_trig = sum(s.n_triggered for secs in self.ctx.contacts.values()
                     for s in secs)
        n_holes_det = (len(self.ctx.detection.holes)
                       if self.ctx.detection else 0)
        n_holes_true = len(centers_w)
        metrics = {
            "ai_hole_detect_rate": (n_holes_det / max(n_holes_true, 1)),
            "ai_confidence_mean": (float(np.mean(
                [h.confidence for h in self.ctx.detection.holes]))
                if self.ctx.detection and self.ctx.detection.holes else 0.0),
            "coarse_localization_error_mm": (float(max(coarse))
                                             if coarse else None),
            "probe_trigger_success_rate": n_trig / max(n_plan, 1),
            "task_success_rate": 1.0 if n_trig == n_plan else
            n_trig / max(n_plan, 1),
            "total_time_s": (time.time() * 1000.0 - self._t_start_ms)
            / 1000.0,
            "robot_b_motion_time_s": self.robot_b.sim_time_ms / 1000.0,
        }
        self.ctx.report = schemas.RunReport(
            run_id=f"run_{int(self._t_start_ms)}",
            timestamp_ms=self._t_start_ms,
            part_id=self.part_id,
            metrics=metrics,
            eval=self.ctx.eval,
        )
        return "COMPLETE", "report ready"

    def _h_error(self):
        return "ERROR", self.ctx.error or "unknown error"

    # -- build & run -----------------------------------------------------------

    def _build_state_machine(self) -> StateMachine:
        sm = StateMachine(initial="IDLE")
        sm.add_state("IDLE", lambda: ("VISION_POSITIONING", "start"))
        for name, handler in [
            ("VISION_POSITIONING", self._h_vision_positioning),
            ("IMAGE_CAPTURE", self._h_image_capture),
            ("AI_DETECTION", self._h_ai_detection),
            ("COORDINATE_TRANSFORM", self._h_coordinate_transform),
            ("TASK_GENERATION", self._h_task_generation),
            ("PATH_PLANNING", self._h_path_planning),
            ("ROBOT_MOVING", self._h_robot_moving),
            ("PROBE_MEASURING", self._h_probe_measuring),
            ("GEOMETRY_FITTING", self._h_geometry_fitting),
            ("TOLERANCE_EVALUATION", self._h_tolerance_evaluation),
            ("REPORT_GENERATION", self._h_report_generation),
            ("ERROR", self._h_error),
        ]:
            sm.add_state(name, handler)
        return sm

    def run(self) -> schemas.RunReport:
        final = self.sm.run()
        if final != "COMPLETE":
            raise RuntimeError(
                f"run failed at {self.sm.current}: "
                f"{self.ctx.error or 'see state machine trace'}\n"
                f"{self.sm.trace()}")
        return self.ctx.report
