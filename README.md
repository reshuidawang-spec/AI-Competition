# box_hole_inspection — 基于AI视觉感知的双机械臂箱体孔系自主检测系统

"人工智能+"创新比赛项目。视觉臂（CR5 @ 192.168.1.7 + 工业相机）AI 识别箱体孔并粗定位，测量臂（CR5 @ 192.168.5.1 + RMP60 五向测针）接触式精密测量，输出孔径/孔距/位置度/垂直度评价与检测报告。

本系统建立在姊妹项目 `cr5_rmp60_measurement` 已验证的真机测量体系之上（只读复用其测头几何注册表、安全门禁约定与停止过冲经验值），**不修改**该冻结项目。

## 设计原则

- AI 只做感知与检测区域识别，不预测高精度尺寸；高精度数据全部来自接触测头。
- 所有真实机器人动作先支持 `simulation/dry_run`；真实模式三重门禁（`mode=real` + 注册面 approved + 姿态模板 verified）。
- 模块间只传 `box_hole_inspection/schemas.py` 定义的 JSON 消息，每个模块独立可测。
- 硬件全部藏在 `interfaces/` 抽象层后：仿真适配器为默认，真机适配器只替换标注过的函数。
- 每个运动任务必须含：安全高度、速度限制、工作空间限制、急停、过冲间隙预留。

## 目录

| 目录 | 内容 |
|---|---|
| `box_hole_inspection/` | 算法包（schemas / 坐标链 / 后续任务、规划、测量、拟合、评价模块） |
| `interfaces/` | 硬件抽象层（CameraBase / RobotBase / ProbeBase + sim/real 适配器） |
| `config/` | system.yaml / camera.yaml / calibration.json / part_box001.yaml |
| `sim/` | 仿真世界/机器人/相机/测头（W1 后续模块） |
| `scripts/` | 闭环入口与演示脚本（W1 后续模块） |
| `tests/` | unittest 单测，每模块一个文件 |
| `docs/` | architecture / calibration_plan / experiment_plan |
| `data/runs/` | 检测运行输出（YYYY-MM-DD/<object>/<session_id>/） |

## 快速开始

```bash
cd ~/dobot_ws/src/box_hole_inspection
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## 当前状态（2026-09-25：W1 仿真最小闭环完成）

- [x] 全部模块：schemas / 坐标链 / 相机接口 / sim_robot(真实CR5 URDF) /
      sim_world / sim_camera / vision_ai(几何+YOLO骨架) / 合成数据生成器 /
      task_generator / path+sequence_planner / sim_probe / probe_measurement /
      geometry_fitting / tolerance_evaluator / hole_assigner / 状态机 / 主控
- [x] 一键闭环：`python3 scripts/run_sim_closed_loop.py --experiment 1|2`
- [x] 146 unittest 全绿；实验1 粗定位 0.316mm / 实验2（移动箱体）0.314mm
- [ ] W2：真实相机+标定、箱体照片标注+YOLO训练、视觉精度实验、A臂真机只读检查

## 重要约定

- 坐标：长度 mm、角度 deg（JSON 边界）/ rad（numpy 内部）；W ≡ 机器人B基座。
- 姿态：本包 RPY 约定为内旋 sxyz；**Dobot 控制器 rx/ry/rz 的映射是已知未决问题**（见姊妹项目 validate_cr5_pose_convention.py），真机适配器上线前必须验证。
- 安全：软件停止链实测过冲 0.04~1.7mm，路径规划必须预留 `stop_overtravel_reserve_mm` 间隙。
- 文档维护：每次里程碑同步更新 PROJECT_MEMORY.md / ROADMAP.md / COMMANDS.md。
