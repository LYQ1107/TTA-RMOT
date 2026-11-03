# TTA-RMOT 最终执行目标

2026-11-03，用户明确要求由代理设立最终 goal 并自行完成。
goal 状态：ACTIVE，线程 `01a154f2-1527-7b80-a8d7-3679ea95c32b`，无另设 token 预算。

## 最终交付

完成“存在性—身份解耦 + 主动冻结 MLLM 校验 + 快慢双速适应”的可信科研验收，
不是只交设计、单元测试、首视频 pilot 或 loss 曲线。

1. 保留全部历史实验，复用已完成的九格 Stage A，不重跑健康完成单元。
2. 完成当前 MAP v1 的 C1/C3、ρ=0.3 全视频 FULL 与四类关键对照共10项，
   Dance 全25视频/425表达，SportsMOT 全45视频/466表达；FULL 从现有流检查点恢复。
3. 依照原预注册推进三目标六方向的固定预算、对照及消融108项矩阵，不删去未跑项目来假称完成。
   原40 GPU小时关卡保留：当前10项估11.030 GPU小时，完整矩阵估292.886；
   本阶段完成后先报告实测与估算，未获扩大范围决定不擅自启动剩余完整矩阵。
4. 预测先封存，独立官方 TrackEval 后报告全部指标、视频聚类 bootstrap、Query-Empty/Present、
   行为、可靠性、真实恢复与误检及计算成本。SportsMOT 主结果用v4全部466，敏感性用相同v4的原418 ID。
5. 代码、真实正负结果、必要日志、失败与修复、资产SHA与可恢复路径持续推送 `LYQ1107/TTA-RMOT`。
6. 最后据真实结果回答方法是否有效、增益来自哪里、局限与是否值得继续。
   不预先保证正增益，不以工程运行正常或微小点增益宣布科研成功。

## 执行边界

- 唯一工作区 `/data2/user/FlexHook-TTA`，不得混入 LocateMOT/GroundJEV。
- 最多2张经准入的共享GPU，显存足够时优先进程少的卡，不停止或抢占他人任务。
- 原教师、源checkpoint、固定MOT候选、MAP v1参数和目标GT隔离不变；不按目标指标调参。
- 大文件继续使用已授权data1路径和data2链接，不删除旧数据。
- 单阻塞协调器等待；失败定位具体原因，最小修复后只恢复未完成单元。
- 正在运行和等待不是阻塞；10项完成也不等于完整最终目标完成。

## 当前恢复入口

- 清单：`results/presence_active/redesign_v5/execution_map_v1_full.json`
- 固定队列：`results/presence_active/redesign_v5/active_map_v1_full/registered_jobs.json`
- 报告：`results/presence_active/redesign_v5/active_map_v1_full/REPORT.md`
- 协调器：`results/runs/presence_active_20261102/redesign_v5/active_map_full_launchers/coordinator_1793694616111172762`
- 已启动C1/C3健康流保持运行，不因设置goal重启或重置。

2026-11-03外部SIGTERM后恢复：旧协调器真实返回143且所有本任务子进程已退出，C3从stream_033（next_video34）恢复，34个已完成视频不重跑。
当前durable观察入口为
`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_full_launchers/durable_1793710403540105887`。
原始启动信息保留作历史；恢复证据与成本界限见本阶段interruption_C3_FULL_20261103.json。
观察会话丢失时，先确认该manager真实状态，再运行同环境Python的
`scripts/active_durable.py --attach <上述durable目录>`；不要另开第二条队列。

随后C3 FULL已完整封存并官方评测HOTA43.914。两个YT对照在首视频暴露共享缓存反序领取死锁，已CPU复现修复，仅停止确认死锁的两个teacher并恢复两项。
当前最新durable目录为
`/data2/user/FlexHook-TTA/results/runs/presence_active_20261102/redesign_v5/active_map_full_launchers/durable_1793718645470682622`，
manager启动PID72357；旧4050136已有exit1完成收据，不是当前运行入口。
恢复依据见deadlock_C3_YESTRACK_20261103.json，8项完整GPU结果不重跑，实际idle worker时间保留。
