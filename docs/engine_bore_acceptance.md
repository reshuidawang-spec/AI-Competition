# EngineBoreInspection 1.0.0 视觉模型验收（2026-10-08）

## 输入与落位

用户提供的 `EngineBoreInspection-1.0.0.zip` 内为 Windows 安装器。
安装器内的照片、标注、模型和检测核心经选择性提取；没有运行安装程序。
安装器 SHA256 与包内 `SHA256.txt` 一致：
`b63b828ba187bf2f6da168a114eded0076f1830277a09e5daf7137543538dcae`。

| 资产 | 本地位置 | 数量 |
|---|---|---:|
| 正式 bore/part 权重（另外两份重复权重未发布） | `data/models/engine_bore_v1/` | 2 |
| JPG/LabelMe JSON 配对照片与标注 | `data/images/engine_bore_seed/` | 28 对 |
| 检测核心（调整为工程内相对导入与模型路径） | `box_hole_inspection/engine_vision/` | 4 模块 |
| 工程接口 | `box_hole_inspection/vision_ai.py` 的 `EngineBoreDetector` | 1 类 |

正式权重是 `bore_best.pt`（类别 `part`、`bore`）和 `part_best.pt`
（类别 `part`）。`engine_part_bore_100.pt` 与 `bore_best.pt` 的 SHA256 相同；
`engine_part_only_fixed.pt` 与 `part_best.pt` 的 SHA256 相同。
正式权重的 SHA256 分别为
`cc73481cf22dda50b610d075ea809907ae8b3bfa8fd696e8bd7c40dd3d61081a` 和
`dd362b1dffb614a82af5f1c1f6023d97f630022ade451d06c24bd95336b6a26f`。
28 份 JSON 均为 1 个 `part`、4 个 `bore` 多边形，且与对应照片尺寸一致。

## 验收结果

使用 CPU、PyTorch 2.9.1、Ultralytics 8.4.173。先直接运行两个模型，
再运行安装包的正式检测核心及拟合质量检查。比较圆心时，以 LabelMe 孔
多边形质心为参考，采用最小总距离的 4 孔匹配。

| 检查 | 结果 |
|---|---:|
| 直接分割检出 4 个置信度 ≥0.5 的 bore | 26/28 张 |
| 正式检测得到 4 个可靠圆心 | 23/28 张 |
| 可靠样本的圆心与标注质心误差中位数 | 4.05 px |
| 可靠样本的圆心与标注质心误差最大值 | 12.87 px |

需人工复核的 5 张文件名前缀：`14a87586`、`45752d38`、`a6aaaeb4`、
`e4900e8b`、`e76fe6f5`。前四张有 3 个可靠圆心，最后一张有 2 个。
`e4900e8b` 在直接分割时虽然有 4 个候选，但一个轮廓质心偏差约 697 px；
正式拟合质量检查拒绝了该圆心。`c38e5c35` 则由旋转补检恢复为 4 孔。

逐图结果和示例叠图随仓库保存在 `docs/engine_bore_eval/`；原运行输出在
`data/runs/2026-10-08/`。这些照片来自安装包，可能参与了
训练或选模；上表仅证明本地模型、标注和接口可以复现，不能推断新拍摄条件下的
准确率。LabelMe 质心也不是经测量仪器确认的真实孔中心。

## 工程边界

`EngineBoreDetector` 返回 `ENGINE_001` 的原图像素圆心，孔号是图像顺序的
占位编号。叠图可见 4 个缸孔沿长轴排列；旧 `BOX_001` 仿真是 2×2 占位孔型。
因此增设 `config/part_engine001.yaml`（CAD 数值均为 null），并让
`HoleAssigner` 在工件 ID 不一致时拒绝配准。当前相机内参、畸变矫正、
手眼、双机基座标定，以及新工件 CAD 孔距、孔径和顶面高度仍缺失；
尚不能把这些像素圆心转成可执行的三维测量点。

下一次验收应使用未参与训练或选模的新照片，覆盖箱体平移、旋转、反光和
拍摄距离变化，并对照已知台面坐标测量粗定位误差。
