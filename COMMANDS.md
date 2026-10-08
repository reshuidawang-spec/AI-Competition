# 人工智能比赛常用命令

工程总览见 `README.md`；路线见 `ROADMAP.md`；事实记录见 `PROJECT_MEMORY.md`。新增脚本或命令后必须同步更新本文件。

## 0. 单元测试（每个模块的验收入口）

```bash
cd <仓库克隆目录>
./.venv/bin/python scripts/verify_portable.py
```

## 0A. 一键仿真闭环（W1 验收入口）

```bash
./.venv/bin/python scripts/run_sim_closed_loop.py --experiment 1     # 位置A
./.venv/bin/python scripts/run_sim_closed_loop.py --experiment 2     # 移动箱体，不重新示教
./.venv/bin/python scripts/run_sim_closed_loop.py --box-x 520 --box-y -25 --box-rz -15
```

报告输出 `data/runs/YYYY-MM-DD/sim_closed_loop_*/report.json`。

## 0B. 合成训练数据生成（YOLO 预训练）

```bash
python3 scripts/gen_synthetic_dataset.py --out data/synthetic_box --n-train 300 --n-val 60
yolo segment train model=yolo11n-seg.pt data=data/synthetic_box/dataset.yaml epochs=100 imgsz=1280 batch=8
```

## 0C. 已训练双模型的照片检测（只输出像素坐标，不控制机器人）

正式模型、28 对照片和标注都已随仓库发布。先按 `README.md` 运行
`python3.10 scripts/bootstrap.py`；Windows 使用 `py -3.10 scripts/bootstrap.py`
及 `.venv\Scripts\python.exe`。依赖与验收记录见 `docs/engine_bore_acceptance.md`。

```bash
./.venv/bin/python scripts/run_engine_vision.py --images data/images/engine_bore_seed
./.venv/bin/python scripts/run_engine_vision.py --images /path/to/photo.jpg --out /tmp/engine_vision.json
```

结果写入 `data/runs/YYYY-MM-DD/engine_vision_detections.json`。新照片属于
`ENGINE_001` 的 4 个直列缸孔，不能使用旧 `BOX_001` 的 2×2 占位孔型。
只有 4 个可靠圆心、正确 CAD 孔型和相机/双机标定均齐备后，才能进入三维定位。

## 1. 安全门禁约定（继承 cr5_rmp60_measurement）

- 默认 `mode: sim`：任何脚本不加 `--execute` 不得产生真实机器人运动。
- 真实模式三重门禁：`mode=real` + 当前孔 `setup/face approved` + 姿态模板 `verified`，缺一拒绝。
- 真机触碰前必须 DI1 预检；触碰中 1ms 监测；触发即 Stop；回退后确认 DI1 释放。
- 软件停止链：规划必须预留 `stop_overtravel_reserve_mm`（当前 2.0mm）。

## 2. 数据输出约定

真实运行输出到 `data/runs/YYYY-MM-DD/<object>/<session_id>/`；不要把不同装夹状态的数据混在同一文件。

## 3. 双机地址

| 角色 | IP | 驱动节点 |
|---|---|---|
| 视觉臂 A | 192.168.1.7 | robot_node_name2 |
| 测量臂 B | 192.168.5.1 | dobot_bringup_ros2 |

环境加载（仅配置有本机 CR5/ROS 2 的真机终端；克隆后的仿真和离线视觉无需执行）：

```bash
source ~/教程/CR5_真实机械臂连接配置包/env/cr5_env.sh
```

## 3A. RViz/MoveIt 3D 仿真（2026-09-25 冒烟测试通过）

注意：以下是原设备工作站的示例，依赖另一个未随本仓库发布的
`cr5_rmp60_measurement` 项目及 ROS 2；`cr5_rmp60_measurement` 不是 colcon 包，
在该工作站用 launch 文件路径启动：

```bash
# CR5 + RMP60 五向测针模型可视化
ros2 launch /home/zhu/dobot_ws/src/cr5_rmp60_measurement/launch/view_cr5_rmp60_model.launch.py

# MoveIt 交互演示（拖拽 + 规划，tip=rmp60_tip）
ros2 launch /home/zhu/dobot_ws/src/cr5_rmp60_measurement/launch/cr5_rmp60_moveit_demo.launch.py

# 不带测头的原始 CR5 稳定 demo
ros2 launch my_cr5_control cr5_moveit_stable_demo.launch.py
```

用途：W3 真机预检（MoveIt /compute_ik + 碰撞检查）与演示可视化。

## 4. 相机

假定 USB3 UVC 兼容工业相机（OpenCV 直采）；实际采购 GigE/厂商 SDK 相机时新增同接口适配器，算法层不变。内参来自 `config/calibration.json`（棋盘格标定），不来自驱动。
