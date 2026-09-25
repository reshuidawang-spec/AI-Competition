# box_hole_inspection 常用命令

工程总览见 `README.md`；路线见 `ROADMAP.md`；事实记录见 `PROJECT_MEMORY.md`。新增脚本或命令后必须同步更新本文件。

## 0. 单元测试（每个模块的验收入口）

```bash
cd ~/dobot_ws/src/box_hole_inspection
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## 0A. 一键仿真闭环（W1 验收入口）

```bash
python3 scripts/run_sim_closed_loop.py --experiment 1     # 位置A
python3 scripts/run_sim_closed_loop.py --experiment 2     # 移动箱体，不重新示教
python3 scripts/run_sim_closed_loop.py --box-x 520 --box-y -25 --box-rz -15
```

报告输出 `data/runs/YYYY-MM-DD/sim_closed_loop_*/report.json`。

## 0B. 合成训练数据生成（YOLO 预训练）

```bash
python3 scripts/gen_synthetic_dataset.py --out data/synthetic_box --n-train 300 --n-val 60
yolo segment train model=yolo11n-seg.pt data=data/synthetic_box/dataset.yaml epochs=100 imgsz=1280 batch=8
```

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

环境加载（真机终端）：

```bash
source ~/教程/CR5_真实机械臂连接配置包/env/cr5_env.sh
```

## 4. 相机

假定 USB3 UVC 兼容工业相机（OpenCV 直采）；实际采购 GigE/厂商 SDK 相机时新增同接口适配器，算法层不变。内参来自 `config/calibration.json`（棋盘格标定），不来自驱动。
