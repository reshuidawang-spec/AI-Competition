"""box_hole_inspection — 基于AI视觉感知的双机械臂箱体孔系自主检测系统.

Competition project built on the validated CR5 + RMP60 five-branch probe
measurement stack (sibling project cr5_rmp60_measurement, read-only reuse).

Module map (each module is independently unit-testable, modules exchange
only schemas.py messages):

    vision_ai -> coordinate_transform -> task_generator -> path_planner
              -> probe_measurement -> geometry_fitting
              -> tolerance_evaluator -> report_generator
    main_controller / state_machine drive the whole loop.
"""

__version__ = "0.1.0"
