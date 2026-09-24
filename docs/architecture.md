# 系统总体架构（2026-09-23 定稿）

## 1. 模块图

```
┌──────────────────────────────────────────────────────────────────┐
│             main_controller（状态机 + 双机编排）                    │
├──────────────┬───────────────────────────┬───────────────────────┤
│ vision_ai    │ coord_transform / task_   │ probe_measurement      │
│ +hole_assign │ generator / path_planner  │ geometry_fitting /     │
│              │ (sequence_planner)        │ tolerance_eval / report│
└──────┬───────┴──────────────┬────────────┴──────────┬────────────┘
       │ schemas.py（JSON 消息边界，全部模块可单测）       │
┌──────▼──────────────────────▼───────────────────────▼────────────┐
│                   硬件抽象层 HAL（interfaces/）                    │
│  RobotBase │ CameraBase │ ProbeBase（=RMP60+DI1 触发语义）        │
│  ┌────────────────────────────┐  ┌───────────────────────────────┐│
│  │ SimAdapter（默认）          │  │ RealAdapter（rclpy，W3）       ││
│  └────────────────────────────┘  │  复用已验证调用约定:            ││
│                                  │  MovJ/MovL/Stop/GetPose/DI/    ││
│                                  │  SpeedFactor/FeedInfo 等       ││
│                                  └───────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
        ↑ 只读复用: calibration_registry.json / cross_probe_geometry.yaml
        ↑ / measurement_poses 约定 / MoveIt 预检 / 停止过冲经验值
        └──── cr5_rmp60_measurement（冻结，不修改）
```

数据主链：图像 → DetectionResult → LocalizationResult → TaskList → MotionPlan → 运动+触发 → ContactPoints → FitResult → EvalResult → RunReport。

## 2. 坐标系与转换链

| 符号 | 定义 |
|---|---|
| W | 统一世界系，**W ≡ BB**（测量臂基座） |
| BA/EA | 视觉臂 A 基座/法兰；FK_A(θ_A) |
| C | 相机（eye-in-hand，OpenCV 约定） |
| O | 箱体（顶面中心，CAD 名义孔已知） |
| BB/EB | 测量臂 B 基座/法兰；FK_B(θ_B) |
| P | 测头 TCP（测球球心），工具标定 T_EB_P |

```
手眼:     T_W_EA = T_BB_BA · FK_A(θ_A)      （T_BA_BB 双机基座标定，固定）
相机:     T_W_C  = T_W_EA · T_EA_C
像素→射线: d_C = K⁻¹·[u,v,1]ᵀ（单位化）；d_W = R_W_C·d_C
平面求交:  λ = (z_top − o_W,z)/d_W,z ;  p_W = o_W + λ·d_W
执行:     p_BB = p_W（W≡BB）；θ_B = IK_B(T_BB_P_target)
```

孔径测量：轴向针侧触（小孔）或十字水平支（大孔），每截面 6~8 点 × 2 截面 → 圆/圆柱最小二乘拟合 → 轴 v=(O2−O1)/‖·‖；垂直度 = arccos(|v·n_top|)。

## 3. 状态机

IDLE → VISION_POSITIONING → IMAGE_CAPTURE → AI_DETECTION → COORDINATE_TRANSFORM → TASK_GENERATION → PATH_PLANNING → [ROBOT_MOVING → PROBE_MEASURING]×N → GEOMETRY_FITTING → TOLERANCE_EVALUATION → REPORT_GENERATION → COMPLETE；任意状态可入 ERROR（可恢复→RECOVERY 重试点；不可恢复→ABORT→IDLE）；SAFETY_MONITOR 后台线程常驻（急停/限位/速度/碰撞/双机互斥）。

真实模式三重门禁：`mode=real` + 注册面 approved + 姿态模板 verified。

## 4. 比赛实验（实验1/实验2）

实验1：箱体位置A，AI 识别 H01~H04，测量臂自主完成四孔测量。
实验2：人为移动箱体（ΔX 30~50mm、ΔY 20~50mm、Rz 10°~20°），不重新示教，重新拍照→识别→规划→测量。指标：识别成功率、置信度、粗定位误差、寻位成功率、触发成功率、孔径/圆心/孔距误差、重复性标准差、完整任务成功率、单箱检测总时间；参考值用数显千分尺/CMM。

## 5. 第一周最小闭环（目标A）

仿真渲染图像(4孔) → 检测(几何占位) → 坐标链 → 任务生成(H01) → 路径规划 → 仿真运动 → 与仿真真值对比。验收：一键跑通；注入 ±2mm 噪声下粗定位误差 <2mm；位姿 100% 可达无越限；全模块 unittest。
