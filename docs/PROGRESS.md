# QCS-TTA progress

更新时间：2026-10-20；状态：本轮工程与匹配实验已完成，科研结论为未观察到 QCS-TTA 收益。

- 项目身份已确认：`/data2/user/LocateMOT`；既有 GroundJEV dirty changes 未覆盖。
- 官方 TempRMOT source commit `6a65640d849fdee4a32bb055945ee34c3b0edeb1`、V1/V2 checkpoint 和本地 RoBERTa 均可追溯加载；source 参数冻结，missing/unexpected keys 为空。
- `/data2` 上 V1/V2 数据和 runtime/evaluation 分离 manifest 已审计；V2 runtime 457,503 行且无 GT 字段。
- 已完成 V1 0012/0014 的 56 episode、5,320 帧完整匹配对照，官方 TrackEval 返回 0；结果见 `docs/FINAL_REPORT.md` 和 `results/summary.csv`。
- V2 0005 一个完整 297 帧 episode 也已完成，作为补充一致性检查，不冒充四视频完整结果。
- 已实现并实际调用 strict causal runner、isolated frozen probe、IoU alignment、set-valued evidence、semantic state、external residual/HEAD257、B0/B2/B4/B5/O1/A1 入口和官方 evaluator。
- 18 个研究 sanity tests 和 compileall 全部通过；V1 runner 未读取 GT/evaluation manifest，预测目录封存后才由 evaluator materialize GT。
- AerialMind 完整 release 已下载到 `/data2` 并通过 93 视频/24,617 expression 审计；官方 30-video test manifest 已生成，runtime/evaluation 各 5,125,243 行。
- AerialMind 两条完整 episode 已在 GPU 2 实测；relative-rank 只恢复候选可见性（HOTA 3.0240，IDF1 0），QCS 在该域仍 0 次更新。63-video train split 已审计但没有伪造 source-free 训练结果。
- GPU 选择遵循显存足够后进程少优先；本轮物理 GPU 2，所有正式 GPU run metadata 均记录 UUID、显存峰值和协议 hash。

当前边界：AerialMind 全 30-video 模型推理需要 5,125,243 个 frame-query，已生成可复现 manifest 但没有把未完成的全量推理冒充成 benchmark；要获得可用 source checkpoint 仍需在独立的 63-video train split 上进行目标域监督训练。Refer-BDD/Refer-Dance 仍缺少本轮所需的合法接口。
