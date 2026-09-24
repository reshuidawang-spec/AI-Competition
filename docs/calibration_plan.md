# 标定计划

所有标定结果写入 `config/calibration.json`，并把 `source` 改为对应的标定记录文件。真实值未测前保持占位，**禁止用占位值做真机结论**。

## 1. 相机内参（棋盘格，张正友）

- 打印/购买 9×7 角点棋盘格，贴于平整板面。
- OpenCV `calibrateCamera` 多姿态采集（≥15 张，覆盖图像四角与倾斜）。
- 输出：fx/fy/cx/cy/dist → `camera_intrinsics`。真机适配器按 dist 做 undistort。

## 2. 手眼标定 T_EA_C（eye-in-hand，AX=XB）

- 标定板固定于工作台（不动）；视觉臂 A 以 ≥10 个姿态拍摄标定板。
- 每姿态记录 FK_A(θ_A)（法兰位姿）与标定板位姿（solvePnP）。
- 解 AX=XB（OpenCV `calibrateHandEye`，Tsai-Lenz 或 Park 法）→ T_EA_C。
- 验证：任一姿态下重投影误差 < 1px；再用已知孔位做一次端到端验证。

## 3. 测头 TCP T_EB_P

- **权威来源**：只读复用 `cr5_rmp60_measurement/config/calibration_registry.json`（五分支球心 offset 注册表）。
- 本项目 `calibration.json` 的 T_EB_P 是"当前生效 TCP"的镜像副本，复制时必须带来源记录与哈希。

## 4. 双机基座标定 T_BA_BB

- 同一标定尖点/标准球心：B 臂用测针采球心 C_W；A 臂（装尖针或借助标定板）测得同一点在 BA 下的坐标。
- T_BA_BB = 最小二乘刚体变换（≥3 个非共线点更稳）。
- 验证：两臂各自"指"同一点，残差 < 0.5mm（该误差只影响粗定位，不影响 B 臂自洽精度）。

## 5. 顶面平面 table_plane_w（z_top）

- B 臂竖直测针在工作台顶面采 3 点以上（或直接测箱体顶面），拟合平面。
- z_top = 台面高 + 箱高；换箱体或换装夹后必须重测。

## 6. 视觉端到端验证（标定闭环验收）

已知孔位箱体放工作台 → 拍照 → 坐标链输出世界坐标 → 与真值对比，粗定位误差应 <2mm（注入噪声仿真已达标的前提下）。不合格先查：内参、手眼、平面高度，依次隔离。
