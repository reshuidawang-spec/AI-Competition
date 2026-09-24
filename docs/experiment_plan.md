# 比赛演示实验计划

## 实验1：固定位置自主检测

箱体放位置A → 点击开始 → A臂拍照 → AI识别箱体与 H01~H04 → 粗定位 → B臂自主完成四孔测量 → 报告输出。

## 实验2：移动箱体，不重新示教

人为移动箱体：ΔX 30~50mm、ΔY 20~50mm、Rz 10°~20° → 再次点击开始 → 系统重新拍照/识别/计算/规划/测量。证明 AI 视觉使系统摆脱固定工位与固定示教点。

## 记录指标（每次运行落盘到 RunReport.metrics）

| 指标 | 说明 |
|---|---|
| ai_hole_detect_rate | AI 孔识别成功率 |
| ai_confidence_mean | 平均识别置信度 |
| coarse_localization_error_mm | 视觉粗定位误差（对照真值） |
| robot_seek_success_rate | 自主寻位成功率 |
| probe_trigger_success_rate | 测头触发成功率 |
| diameter_error_mm | 孔径测量误差（对照参考值） |
| center_error_mm | 孔中心测量误差 |
| hole_distance_error_mm | 孔距误差 |
| repeatability_std_mm | 重复测量标准差（同孔 N≥3 次） |
| task_success_rate | 完整任务成功率 |
| total_time_s | 单个箱体总检测时间 |

参考值：数显千分尺 / 卡尺实测（或 CMM 若可用），记录设备与测量者。

## 演示脚本

`scripts/demo_experiment.py`（W4）：参数化 `--experiment 1|2`，一键执行并生成报告；演示现场先跑 dry-run 预演，再真实执行。
