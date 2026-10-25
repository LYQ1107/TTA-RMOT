# Positive-Free 正式实验执行计划（2026-10-25）

项目仅为 `LYQ1107/TTA-RMOT`，执行目录 `/data2/user/FlexHook-TTA`。
用户本轮已授权完整目标 TTA 和官方评测；不授权源端重训、额外教师、终止其他任务或擅自共享 GPU。
从核实后的 `a0cb27ee338edaaf0362fb89316b22f1775c9aab` 创建
`research/v2dance-positivefree-four-gpu-full-20261025`，保留原分支、历史输出与权重。

简短计划：

1. 局部缓存冻结 backbone 多尺度输出和每查询文本编码；原 C-Hook/PCD 解码不变。
   FP32，最多 4 GiB/worker 的有界 LRU，只取当前/过去帧；先真实候选集前向和梯度等价验证，单独提交。
2. 固定 25 视频/425 episode，以 tracker rows 作 LPT 工作量分片；最多四张无计算进程且资源充足的 GPU。
   不足四张就用 1–3 张运行全部逻辑分片；当前初查仅 GPU5 真正空闲。
3. 每 episode 原子封存预测和候选分数，写完成收据；失败重置源状态后重跑该 episode，保留失败日志。
4. 顺序运行 ZERO_UPDATE、RELATIVE_ONLY、PF_RANK、PF_RANK_GUARD、PF_IDENTITY、PF_HYBRID。
   Guard 先验证从 PF_RANK 分数派生的等价性，再省去重复 GPU 前向。
5. 每方法封存全部预测、核对完整覆盖和 MOT 原始行后，独立官方 TrackEval 才读取目标 GT。
   ZERO_UPDATE 先对齐 Frozen 8.206/3.1857/21.138/4.436，未对齐先诊断，不报告虚假收益。
6. 记录召回恢复/丢失、FP、排序、梯度、回滚、速度显存；开发性目标域结果，无论正负均完整报告并推送。

默认配置：rank=4、lr=5e-5、update_interval=4、min_history=3、min_separation=0.6、seed=0。
不启用未验证的 BF16，不修改 MOT、表达、阈值或历史研究日志。
