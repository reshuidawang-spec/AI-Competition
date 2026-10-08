# 人工智能比赛：双机械臂箱体孔系检测工程

这是“人工智能+”创新比赛的可复用工程。仓库内包含仿真闭环、已训练的缸孔视觉模型、28 组照片及 LabelMe 标注、资源校验清单和依赖安装脚本。克隆后无需原始安装包，即可运行仿真和照片推理。

视觉臂识别工件孔位，测量臂以 RMP60 测针取得精密尺寸。当前可直接复现的是**仿真闭环和离线照片检测**；真机部署还需要本机 ROS 2、CR5 驱动、相机标定、工件 CAD 参数和现场安全验证。`ENGINE_001` 的模型给出原图像素圆心，不能直接作为机器人三维运动目标。

## 环境与安装

已在 Ubuntu、Python 3.10 上验证。需要网络连接以从 PyPI 和 PyTorch 官方 CPU 源下载 Python 库；模型和照片已随仓库提供。推荐至少 4 GB 可用磁盘空间。其他操作系统的 Python 安装入口见下表，真机相关依赖按具体硬件另行配置。

| 系统 | 安装与验证命令 | 虚拟环境 Python |
|---|---|---|
| Ubuntu / Linux | `python3.10 scripts/bootstrap.py` | `./.venv/bin/python` |
| Windows | `py -3.10 scripts/bootstrap.py` | `.venv\Scripts\python.exe` |

```bash
git clone https://github.com/reshuidawang-spec/AI-Competition.git
cd AI-Competition
python3.10 scripts/bootstrap.py
```

安装脚本创建项目内的 `.venv`，安装固定版本依赖，核验随包资源 SHA256，运行单元测试，并用已训练权重完成一次照片推理。Ubuntu 如缺少 `ensurepip`，脚本会尝试 `virtualenv`；若两者都没有，先安装 `python3.10-venv` 和 `python3-pip`。只需仿真时可运行 `python3.10 scripts/bootstrap.py --core-only`，省去视觉推理依赖。安装后可随时运行 `./.venv/bin/python scripts/verify_portable.py` 复查；Windows 请换用上表的 Python 路径。

GitHub 将中文仓库名自动转换为 `-`，所以网址采用 `AI-Competition`；工程名称为“人工智能比赛”。

## 运行

```bash
# 已训练模型检测随包照片；结果写入 data/runs/日期/engine_vision_detections.json
./.venv/bin/python scripts/run_engine_vision.py --images data/images/engine_bore_seed

# 单张新照片；只输出像素圆心和质量诊断，不控制机械臂
./.venv/bin/python scripts/run_engine_vision.py --images /path/to/photo.jpg --out /tmp/vision.json

# 两组仿真闭环实验
./.venv/bin/python scripts/run_sim_closed_loop.py --experiment 1
./.venv/bin/python scripts/run_sim_closed_loop.py --experiment 2
```

随包照片可能用于模型训练或选模，不能视为独立测试集。在这 28 张照片上，23 张给出 4 个可靠圆心，5 张需要复核；详细条件和结果见 [视觉验收记录](docs/engine_bore_acceptance.md)。

## 目录与工程边界

| 路径 | 内容 |
|---|---|
| `box_hole_inspection/` | 消息结构、视觉检测、配准、规划、测量和评价算法 |
| `interfaces/`、`sim/` | 硬件接口与默认仿真实现 |
| `config/` | 系统配置、占位标定与工件名义参数 |
| `data/models/engine_bore_v1/` | 正式 bore/part 训练权重和模型登记表 |
| `data/images/engine_bore_seed/` | 28 对 JPG 照片和 LabelMe 标注 |
| `data/portable_manifest.json` | 已发布资源的大小与 SHA256 校验值 |
| `docs/engine_bore_eval/` | 验收结果、示例叠图 |
| `scripts/`、`tests/` | 安装、自检、实验入口与单元测试 |

原始 3.1 GB Windows 安装包及重复权重不在仓库中；运行上述流程所需的正式权重、照片和标注均已包含。运行结果保存在忽略提交的 `data/runs/`。旧 `BOX_001` 仿真孔型为 2×2，新缸体 `ENGINE_001` 为直列四孔，代码会拒绝混用。工件 CAD 数值和实物标定尚未完成。真机规划见 [路线图](ROADMAP.md)，命令索引见 [COMMANDS](COMMANDS.md)。

## 许可

仓库自有代码、随包模型权重及照片按 [MIT 许可证](LICENSE) 发布。安装脚本下载的第三方库遵循各自许可证。
