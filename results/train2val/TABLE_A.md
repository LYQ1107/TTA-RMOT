# 表 A：Train 标签拟合，尚未进行学生 Val 推理

Train教师判断 31,200；POS(p≥0.95) 1,765 (5.657051%)。
历史Naive为656/8500；用户给出的657/8500来自PSR自身缓存，分母与manifest不可混用。

| Arm | POS选中率 | NEG选中率 | 帧内pair agreement | KD BCE | source-dev CE(final) | 窗口平均曝光 | GPUh(进程body) |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2V_D1_s17 | 0.089518 | 0.004552 | 0.589389 | 0.306831 | 0.894119 | 4.601538 | 1.177736 |
| T2V_D1_s23 | 0.086686 | 0.004179 | 0.572088 | 0.308130 | 0.901015 | 4.601538 | 1.188113 |
| T2V_D1_s42 | 0.076487 | 0.004043 | 0.577855 | 0.311043 | 0.898237 | 4.601538 | 1.177743 |
| T2V_D3_s17 | 0.240793 | 0.038424 | 0.649366 | 0.309570 | 0.675946 | 4.601538 | 1.575634 |

固定final-step，source-best仅日志、不用于Val模型选择。各seed目标标签完全相同。
POS按p≥.95；NEG统计桶沿D-FIT原定义为所有有效p<.95（含不确定soft标签），并不把UNKNOWN变成硬负例；优化始终用原始soft概率。
所有值仅Train教师拟合/源dev诊断，不是目标GT准确率或Val HOTA；不因表A好坏更换方法、阈值或checkpoint。
8973×4/7800≈4.60次窗口曝光（实际窗口数与平均值见JSON）；原Val2125窗口约16.89次，训练预算不随窗口数增加。

同一批Train判断的冻结源模型初始拟合（CPU缓存复用，不是新增GPU实验）：
```json
{
  "teacher_POS_selected_rate": 0.10198300283286119,
  "teacher_NEG_selected_rate": 0.06940716833701376,
  "mean_student_prob_on_POS": 0.09996140194661939,
  "mean_student_prob_on_NEG": 0.07243176781853007,
  "pair_agreement": 0.5213379469434832,
  "pair_agreement_pairs": 867,
  "teacher_POS": 1765,
  "teacher_NEG": 29435,
  "POS_threshold": 0.95,
  "target_GT_open_attempts": 0,
  "scope": "teacher training anchor fit, not GT accuracy"
}
```
