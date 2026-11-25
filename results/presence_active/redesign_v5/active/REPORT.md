# v5 B/C：主动教师与快慢适应

阶段A构造已在自身源域通过保留门槛，不等于跨域方法有效。完整A报告位于上级REPORT.md；旧实验主报告不改写。

本目录独立记录原B/C方法。首批为C1/C3、rho=.3、首视频可恢复pilot；真实成本估算完成前不展开108项。不根据目标GT选择参数或试验臂。

尚未产生完整B/C成绩；缺失结果为NOT_RUN/PILOT_ONLY，不是0 HOTA。教师、LoRA、预算与GT隔离的工程测试不能替代正式科研评价。

## 真实pilot最新状态

BOTH_REAL_UPDATES_MEASURED。尚无完整B/C HOTA。

| pair | videos | teacher_calls | new_forward_calls | optimizer_updates | gradient_min | gradient_max | LoRA_B_norm | teacher_forward_seconds | LoRA_seconds | rescore_seconds | final_positive_pairs | confident_never_asked | GT_open_attempts | full_RMOT_evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 3 | 162 | 162 | 350 | 0.243896 | 27.045340 | 1.984835 | 5.688805 | 940.427426 | 103.193622 | 24 | 170 | 0 | False |
| C3 | 1 | 43 | 43 | 200 | 0.154162 | 2.747926 | 1.450984 | 2.166746 | 424.603327 | 43.560146 | 76 | 115 | 0 | False |

高置信不等于正确，未问过即高置信单列；GT未进入本诊断，不据此调整阈值。C1缺少有效互斥列表样本时应跳过，不能强制伪正例制造梯度；完整矩阵成本门槛仍须真实C1/C3更新计时，未通过不展开108项。每次pilot快照、日志和stream checkpoint SHA独立保留；后续按视频顺序恢复，不重做健康首视频。
