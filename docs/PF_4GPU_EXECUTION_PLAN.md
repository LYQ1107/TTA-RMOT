# Positive-Free 正式实验执行计划（2026-10-25）

项目仅为 `LYQ1107/TTA-RMOT`，执行目录 `/data2/user/FlexHook-TTA`。
用户本轮已授权完整目标 TTA 和官方评测；最新指令明确允许显存足够时共享 GPU，优先四张进程少的卡。
不授权源端重训、额外教师或终止其他任务。
从核实后的 `a0cb27ee338edaaf0362fb89316b22f1775c9aab` 创建
`research/v2dance-positivefree-four-gpu-full-20261025`，保留原分支、历史输出与权重。

简短计划：

1. 局部缓存冻结 backbone 多尺度输出和每查询文本编码；原 C-Hook/PCD 解码不变。
   FP32，有界 LRU，只取当前/过去帧；先真实候选集前向和梯度等价验证，单独提交。
   正式 runner 上限 16 GiB/worker（初版等价回归用 4 GiB），以容纳最长 2,402 帧视频的约 15.2 GiB 冻结特征，
   避免 17 条表达之间的顺序扫描 LRU 抖动；共享时至少 32 GiB 剩余显存，单进程 PyTorch 分配器上限 24 GiB，
   保留 8 GiB 余量。此上限不是跨用户显存预留，别人后续分配仍可能引发资源竞争。
2. 固定 25 视频/425 episode，以 tracker rows 作 LPT 工作量分片；最多四张物理 GPU。
   最新授权下先按进程数少排序，再按剩余显存多排序；无足够显存时等待，不停止别人进程。
   已完成 shard0 的 6 视频/102 episode 不重跑；其余 19 视频重新 LPT 为四个完整视频组。
   原 manifest 保留，新 active_manifest 固定后供恢复使用，不能切断 episode 或搬移已开始的分片。
3. 每 episode 原子封存预测和候选分数，写完成收据；失败重置源状态后重跑该 episode，保留失败日志。
4. 顺序运行 ZERO_UPDATE、RELATIVE_ONLY、PF_RANK、PF_RANK_GUARD、PF_IDENTITY、PF_HYBRID。
   Guard 先验证从 PF_RANK 分数派生的等价性，再省去重复 GPU 前向。
5. 每方法封存全部预测、核对完整覆盖和 MOT 原始行后，独立官方 TrackEval 才读取目标 GT。
   发现旧 Frozen 仅 210,393 行：vendor 独占 range 少采每轨迹末观测，遗漏 692 行且存在正分候选。
   用户已明确选择补齐全部 211,085 行、重建匹配基线。旧 8.206 仅作历史采样口径对照。
   原 Frozen 文件和 SHA 不变；同一源模型生成缺失观测的补充 logits，所有方法共享相同补充。
   完整 ZERO_UPDATE 对照完整冻结分数的选择结果，六方法以新实测完整 Frozen/ZERO_UPDATE 为基线。
6. 记录召回恢复/丢失、FP、排序、梯度、回滚、速度显存；开发性目标域结果，无论正负均完整报告并推送。

默认配置：rank=4、lr=5e-5、update_interval=4、min_history=3、min_separation=0.6、seed=0。
不启用未验证的 BF16，不修改 MOT、表达、阈值或历史研究日志。
