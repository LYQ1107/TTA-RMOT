# Source-Anchored MAP：两段真实pilot结果

NUMERICS_AND_EVIDENCE_GATE_VERIFIED_WITH_REAL_MAP_UPDATES。

37项定向与既有回归、compileall和GT-free静态证据重放通过。原版全部结果保留；新版本从官方checkpoint及fresh LoRA/DS开始，在完全相同的两个首视频、rho=.3和四轮预算上运行。不启动完整矩阵、不读目标GT、不做新的HOTA评测；数值稳定不是语义恢复证据。

| pair | variant | video | logical_teacher_calls | new_forward_calls | cache_hits | teacher_forward_seconds | teacher_equivalent_forward_seconds | teacher_peak_allocated_bytes | optimizer_updates | LoRA_seconds | rescore_seconds | final_slope | final_temperature | final_bias | max_slope | max_temperature | max_abs_bias | MAP_identity_updates | MAP_presence_updates | final_confident_positive | final_confident_never_asked | prediction_rows | GT_open_attempts | full_RMOT_HOTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | LEGACY | dancetrack0004 | 56 | 56 | 0 | 2.414452 | 2.414452 | 5456981504 | 0 | 0.000000 | 0.000000 | -0.000021 | 2.353945 | -20.444820 | 0.668017 | 2.353945 | 20.444820 | 0 | 0 | 0 | 18 | 4371 | 0 | NOT_EVALUATED |
| C1 | MAPv1 | dancetrack0004 | 56 | 31 | 25 | 1.240312 | 2.461009 | 4676065280 | 200 | 129.588372 | 43.105102 | 0.473038 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 2 | 0 | 5 | 144 | 2296 | 0 | NOT_EVALUATED |
| C3 | LEGACY | v_00HRwkvvjtQ_c001 | 43 | 43 | 0 | 2.166746 | 2.166746 | 5828211712 | 200 | 424.603327 | 43.560146 | 430.290965 | 4480260.353007 | 20.565322 | 584.543860 | 4480260.353883 | 20.565322 | 0 | 0 | 76 | 115 | 49951 | 0 | NOT_EVALUATED |
| C3 | MAPv1 | v_00HRwkvvjtQ_c001 | 43 | 28 | 15 | 1.373147 | 2.597348 | 5804913664 | 200 | 421.034086 | 42.199032 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0 | 0 | 0 | 33 | 29918 | 0 | NOT_EVALUATED |

## 1. 每次校准调用

| variant | pair | round | position | status | slope | temperature | bias | identity_status | supported_positive | supported_negative | presence_status | confirmed_positive | confirmed_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LEGACY | C1 | 1 | BEFORE_SLOW | FAST_UPDATED | 0.668017 | 1.000000 | 0.000000 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C1 | 2 | BEFORE_SLOW | FAST_UPDATED | 0.493460 | 2.353945 | -20.444819 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C1 | 3 | BEFORE_SLOW | FAST_UPDATED | 0.000030 | 2.353945 | -20.444820 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C1 | 4 | BEFORE_SLOW | FAST_UPDATED | -0.000021 | 2.353945 | -20.444820 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| MAPv1 | C1 | 1 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 10 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C1 | 1 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 9 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C1 | 2 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 20 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 8 |
| MAPv1 | C1 | 2 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 20 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 8 |
| MAPv1 | C1 | 3 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 28 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 8 |
| MAPv1 | C1 | 3 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 28 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 8 |
| MAPv1 | C1 | 4 | BEFORE_SLOW | SOURCE_MAP_UPDATED | 0.419462 | 1.000000 | 0.000000 | MAP_UPDATED | 3 | 32 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 9 |
| MAPv1 | C1 | 4 | AFTER_SLOW | SOURCE_MAP_UPDATED | 0.473038 | 1.000000 | 0.000000 | MAP_UPDATED | 5 | 32 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 9 |
| LEGACY | C3 | 1 | BEFORE_SLOW | FAST_UPDATED | 2.186018 | 1.000000 | 0.000000 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 1 | AFTER_SLOW | FAST_UPDATED | 52.161855 | 1.000000 | 0.000000 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 2 | BEFORE_SLOW | FAST_UPDATED | 4.299067 | 4480260.353883 | 20.564616 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 2 | AFTER_SLOW | FAST_UPDATED | 245.302938 | 4480260.353647 | 20.564806 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 3 | BEFORE_SLOW | FAST_UPDATED | 584.543860 | 4480260.353330 | 20.565062 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 3 | AFTER_SLOW | FAST_UPDATED | 584.543860 | 4480260.353330 | 20.565062 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 4 | BEFORE_SLOW | FAST_UPDATED | 430.290965 | 4480260.353007 | 20.565322 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| LEGACY | C3 | 4 | AFTER_SLOW | FAST_UPDATED | 430.290965 | 4480260.353007 | 20.565322 | LEGACY_UNRESTRICTED | None | None | LEGACY_UNRESTRICTED | None | None |
| MAPv1 | C3 | 1 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 1 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 2 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 2 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 3 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 3 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 4 | BEFORE_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |
| MAPv1 | C3 | 4 | AFTER_SLOW | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 1.000000 | 1.000000 | 0.000000 | SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES | 0 | 0 | SKIP_INSUFFICIENT_CONFIRMED_CLASSES | 0 | 0 |

两类教师支持不足时明确SKIP，不复制未问伪标签作为fast监督；presence矛盾记UNKNOWN。原源头、冻结教师/三视图、DS、预算、关键帧、LoRA和slow伪标签逻辑不变。固定单位高斯源锚以及正b/T参数化见PREREG.md，未按GT搜参。

默认单类/分离数据下的极端参数与[Heinze & Schemper](https://onlinelibrary.wiley.com/doi/10.1002/sim.1047)讨论的非有限MLE问题一致，是机制解释而非唯一因果证明。固定源先验借鉴[Gelman et al.](https://doi.org/10.1214/08-AOAS191)的正则化原则，不复现其Cauchy默认先验或声称RMOT保证。

## 2. 成本与覆盖

```json
{
  "new_teacher_calls": 59,
  "logical_calls": 99,
  "cache_hits": 40,
  "new_teacher_forward_seconds": 2.6134593300521374,
  "teacher_equivalent_forward_seconds": 5.058356592734343,
  "LoRA_seconds": 550.6224582330324,
  "rescore_seconds": 85.30413341312669,
  "workers": [
    {
      "path": "/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_v1/C1/FULL/rho_0.3/worker_attempts/1793638889277350326.json",
      "status": "RETURNED_SUCCESSFULLY",
      "phase": "pilot",
      "pid": 2161603,
      "physical_gpu": "1",
      "gpu_uuid": "GPU-f35654b3-bbf5-7303-cfcb-9bd5d3e16e69",
      "wall_time_is_not_utilization_integral": true,
      "wall_seconds": 207.29914348199964,
      "parent_peak_allocated_bytes": 4223456768,
      "teacher_peak_is_reported_in_its_separate_receipt": true
    },
    {
      "path": "/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_v1/C3/FULL/rho_0.3/worker_attempts/1793638889271774667.json",
      "status": "RETURNED_SUCCESSFULLY",
      "phase": "pilot",
      "pid": 2161604,
      "physical_gpu": "4",
      "gpu_uuid": "GPU-44f62835-4095-23ec-d3dd-5af16a00705c",
      "wall_time_is_not_utilization_integral": true,
      "wall_seconds": 499.1620398859959,
      "parent_peak_allocated_bytes": 4572250624,
      "teacher_peak_is_reported_in_its_separate_receipt": true
    }
  ],
  "meaning": "shared-device elapsed phase timing, not utilization integral; teacher cache is not a zero-cost independent system",
  "measured_GPU_phase_hours": 0.1773722363822809,
  "assigned_worker_wall_hours": 0.196239217602221
}
```

每个目标只有1视频，不是25/45视频全量。教师实际新增forward与缓存命中/独立等价forward分开，前端、源训练和v5 Stage A既有成本不假定为零。原默认矩阵330.345 GPU小时是旧方法小pilot保守外推，不是本轮已花费，也不是新MAP版已验证成本。

## 3. 尚未解决的问题

- 没有目标GT质量评估，不能报告伪标签准确率、语义恢复或HOTA提升；输出行数不是正确目标数。
- SlowAdapter原规则仍可使用未问先验伪标签、非互斥场景也可只有负例；这里只隔离修正fast，未声称消除自确认偏差。
- 两次候选No推断整句不存在、视图相关性和DS可辨识性属于原机制风险，未用本轮GT调解。
- 保留完整视频统计/轨迹均值，是video-transductive pilot，不冒充严格逐帧因果TTA。
- 完成此小pilot后停止，不启动完整方法矩阵或额外源训练；后续正式验证需用户新指令。

原始逐问题输出/轮记录、预测mask、可恢复stream checkpoint与SHA见artifacts.json及pilot_map_first_video_logs.tar.gz；大文件留data1，通过原data2链接访问。
