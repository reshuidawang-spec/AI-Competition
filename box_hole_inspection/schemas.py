"""JSON message schemas shared by all modules.

Module boundary contract: modules exchange ONLY these dataclasses, serialised
with :func:`to_dict` / :func:`from_dict` for JSON logs and inter-module
messages. This keeps every module independently unit-testable.

Conventions:
- Lengths in mm, angles in degrees, timestamps in epoch milliseconds.
- 3D coordinates are ``[x, y, z]`` lists (JSON-friendly); algorithms convert
  to numpy internally.
- Field names that would collide with Python keywords end with ``_``
  (e.g. ``pass_``); the trailing underscore is stripped in JSON keys, so the
  wire format uses ``"pass"``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, MISSING
from typing import Any, List, Optional, Union, get_args, get_origin, get_type_hints

__all__ = [
    "Pose",
    "HoleDetection", "DetectionResult",
    "LocalizedHole", "LocalizationResult",
    "NominalHole", "MeasureSection", "MeasureConfig", "HoleTask", "TaskList",
    "Waypoint", "PlanChecks", "MotionPlan",
    "ContactEvent", "ContactPoints",
    "SectionFit", "AxisFit", "FitResult",
    "HoleEval", "HolePairEval", "EvalResult",
    "RobotStatus", "RunReport",
    "to_dict", "from_dict",
]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _json_name(name: str) -> str:
    """Strip a single trailing underscore (keyword-safe fields) in JSON."""
    return name[:-1] if name.endswith("_") and not name.endswith("__") else name


def to_dict(obj: Any) -> Any:
    """Convert a dataclass tree to plain JSON-ready dicts/lists."""
    if dataclasses.is_dataclass(obj):
        return {_json_name(f.name): to_dict(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _convert(val: Any, typ: Any) -> Any:
    """Convert a plain JSON value to the field's declared type."""
    if val is None:
        return None
    origin = get_origin(typ)
    if origin is Union:  # includes Optional[...]
        args = [a for a in get_args(typ) if a is not type(None)]
        if args and dataclasses.is_dataclass(args[0]):
            return from_dict(args[0], val)
        return val
    if origin in (list, List):
        args = get_args(typ)
        if args and dataclasses.is_dataclass(args[0]):
            return [from_dict(args[0], x) for x in val]
        return val
    if dataclasses.is_dataclass(typ):
        return from_dict(typ, val)
    return val


def from_dict(cls, d: dict) -> Any:
    """Build a dataclass (tree) from a plain dict."""
    if d is None:
        return None
    # get_type_hints resolves postponed string annotations (PEP 563) into
    # real types, so nested dataclasses and Optional/List[...] convert.
    hints = get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        name, json_name = f.name, _json_name(f.name)
        if json_name in d:
            kwargs[name] = _convert(d[json_name], hints.get(name, f.type))
        elif f.default is not MISSING:
            kwargs[name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            kwargs[name] = f.default_factory()
        else:
            kwargs[name] = None
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

@dataclass
class Pose:
    """Position (mm) + roll/pitch/yaw (deg, intrinsic sxyz convention)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


# ---------------------------------------------------------------------------
# Vision -> Localization
# ---------------------------------------------------------------------------

@dataclass
class HoleDetection:
    """One hole as detected in the image (undistorted pixel coordinates)."""
    id: str = ""
    confidence: float = 0.0
    pixel_u: float = 0.0
    pixel_v: float = 0.0
    pixel_bbox: List[float] = field(default_factory=list)  # [x0, y0, x1, y1]


@dataclass
class DetectionResult:
    """vision_ai module output."""
    part_id: str = ""
    confidence: float = 0.0
    timestamp_ms: float = 0.0
    box_bbox: List[float] = field(default_factory=list)    # [x0, y0, x1, y1]
    box_rz_hint_deg: Optional[float] = None                # mask 主轴角度初值
    holes: List[HoleDetection] = field(default_factory=list)


@dataclass
class LocalizedHole:
    """One hole after the coordinate chain (camera / world / robot-B frames)."""
    id: str = ""
    confidence: float = 0.0
    pixel_u: float = 0.0
    pixel_v: float = 0.0
    camera_xyz: Optional[List[float]] = None   # in camera frame C
    world_xyz: Optional[List[float]] = None    # in W (W ≡ BB)
    bb_xyz: Optional[List[float]] = None       # in robot-B base frame
    fit_residual_mm: Optional[float] = None    # hole_assigner 模式配准残差


@dataclass
class LocalizationResult:
    """coordinate_transform module output (coarse visual localisation)."""
    part_id: str = ""
    confidence: float = 0.0
    timestamp_ms: float = 0.0
    pose: Pose = field(default_factory=Pose)   # 箱体位姿 in W（第一阶段 x/y/rz）
    holes: List[LocalizedHole] = field(default_factory=list)
    coarse_error_est_mm: Optional[float] = None  # 1px 噪声在台面平面的误差尺度


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

@dataclass
class NominalHole:
    """CAD nominal hole data from part_box001.yaml."""
    id: str = ""
    diameter_mm: Optional[float] = None        # 待定时为 null
    center_o: List[float] = field(default_factory=lambda: [None, None])  # 箱体系 [x, y]
    tol_diameter_mm: float = 0.05
    tol_position_mm: float = 0.10
    tol_perp_deg: float = 0.10


@dataclass
class MeasureSection:
    """One circular cross-section of contact points."""
    z_offset_mm: float = 0.0                   # 相对孔口/基准深度的下探量
    n_points: int = 8
    angles_deg: List[float] = field(default_factory=list)


@dataclass
class MeasureConfig:
    """Probing strategy for one hole (inherits real-robot verified conventions)."""
    branch: str = "z"                          # z 轴向针侧触 / x_pos..y_neg 十字水平支
    standard_pose: str = ""                    # measurement_poses.yaml 验证姿态名
    sections: List[MeasureSection] = field(default_factory=list)
    search_overshoot_mm: float = 3.0           # 覆盖视觉粗定位误差
    stop_overtravel_reserve_mm: float = 2.0    # 软件停止链实测过冲必须预留的间隙
    probe_step_mm: float = 0.5                 # 分段探测步长
    contact_speed_ratio: float = 0.05
    setup_id: str = ""                         # 注册面门禁语义
    trigger_io: str = "DI1"


@dataclass
class HoleTask:
    """Measurement task for one hole."""
    hole_id: str = ""
    nominal: NominalHole = field(default_factory=NominalHole)
    rough_center_bb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    measure: MeasureConfig = field(default_factory=MeasureConfig)


@dataclass
class TaskList:
    """task_generator module output."""
    part_id: str = ""
    timestamp_ms: float = 0.0
    tasks: List[HoleTask] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Path planning
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """One robot motion waypoint (robot-B base frame)."""
    hole_id: str = ""
    stage: str = ""                            # approach | measure | safe | leave
    name: str = ""
    pose_bb: Pose = field(default_factory=Pose)
    joint_deg: Optional[List[float]] = None    # IK 解（若可解）
    speed_ratio: float = 0.3
    blend_mm: float = 10.0


@dataclass
class PlanChecks:
    """Safety/feasibility checks applied to a motion plan."""
    reachable: bool = True
    joint_limit_ok: bool = True
    singularity_ok: bool = True
    collision_ok: bool = True
    messages: List[str] = field(default_factory=list)


@dataclass
class MotionPlan:
    """path_planner module output."""
    sequence: List[str] = field(default_factory=list)   # 孔检测顺序
    order_method: str = ""
    waypoints: List[Waypoint] = field(default_factory=list)
    checks: PlanChecks = field(default_factory=PlanChecks)


# ---------------------------------------------------------------------------
# Probing / fitting / evaluation
# ---------------------------------------------------------------------------

@dataclass
class ContactEvent:
    """One probe trigger event (ball-radius compensated contact point)."""
    idx: int = 0
    angle_deg: float = 0.0
    touch_tcp_bb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    contact_point_bb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ContactPoints:
    """probe_measurement module output for one section."""
    hole_id: str = ""
    section_z_mm: float = 0.0
    n_planned: int = 0
    n_triggered: int = 0
    contacts: List[ContactEvent] = field(default_factory=list)
    misses: List[dict] = field(default_factory=list)


@dataclass
class SectionFit:
    """Least-squares circle fit of one cross-section."""
    z_mm: float = 0.0
    center_bb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius_mm: float = 0.0
    residual_rms_um: float = 0.0
    n_points: int = 0


@dataclass
class AxisFit:
    """Hole axis from two or more section centres."""
    point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    direction: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])


@dataclass
class FitResult:
    """geometry_fitting module output for one hole."""
    hole_id: str = ""
    sections: List[SectionFit] = field(default_factory=list)
    center_bb: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    diameter_mm: float = 0.0
    axis: AxisFit = field(default_factory=AxisFit)


@dataclass
class HoleEval:
    """Tolerance evaluation for one hole."""
    id: str = ""
    diameter_mm: float = 0.0
    dev_mm: float = 0.0
    tol_mm: float = 0.0
    pass_: bool = True                          # JSON key: "pass"
    center_dev_mm: Optional[float] = None
    position_tol_mm: Optional[float] = None
    position_ok: Optional[bool] = None
    perp_dev_deg: Optional[float] = None
    perp_tol_deg: Optional[float] = None
    perp_ok: Optional[bool] = None


@dataclass
class HolePairEval:
    """Hole-to-hole distance evaluation."""
    ids: List[str] = field(default_factory=list)
    dist_mm: float = 0.0
    nominal_mm: float = 0.0
    dev_mm: float = 0.0
    tol_mm: float = 0.0
    pass_: bool = True                          # JSON key: "pass"


@dataclass
class EvalResult:
    """tolerance_evaluator module output."""
    holes: List[HoleEval] = field(default_factory=list)
    hole_pairs: List[HolePairEval] = field(default_factory=list)
    summary: dict = field(default_factory=lambda: {
        "n_total": 0, "n_pass": 0, "n_fail": 0})


# ---------------------------------------------------------------------------
# Status / report
# ---------------------------------------------------------------------------

@dataclass
class RobotStatus:
    """Robot interface status snapshot."""
    state: str = "unknown"                     # moving | idle | error
    joint_deg: Optional[List[float]] = None
    tcp_pose: Optional[Pose] = None
    estop: bool = False
    error_code: int = 0


@dataclass
class RunReport:
    """report_generator module output (competition metrics included)."""
    run_id: str = ""
    timestamp_ms: float = 0.0
    part_id: str = ""
    metrics: dict = field(default_factory=dict)
    eval: EvalResult = field(default_factory=EvalResult)
