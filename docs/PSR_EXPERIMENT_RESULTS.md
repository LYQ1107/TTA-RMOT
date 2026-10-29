# PSR 实验状态

2026-10-29，用户允许的25Val同集适应，非Train→Val/非盲测。

已完成匹配预算教师缓存及真实学生smoke；Naive/PSR已在独立目录完成固定8973步训练。
两组均保持24576参数、seed17、lr5e-5和源val CE选checkpoint，不读取目标GT。

完整RMOT新指标：**尚未产生**。Frozen8.2180和Source-only7.4352为已核验可复用参考，
旧Naive8.0327使用另一采样协议，仅保留历史，不能填入本轮新Naive的空位。

主实验训练耗时：Naive主体2554.08秒，PSR主体2391.76秒（含完整合法源覆盖和源val选择；完整进程时间另列）。
两组分别使用8500直接软标签；PSR使用254条不同的一帧弱插值；这些都不是GT正确性统计。
匹配Mean Teacher已通过6步真实smoke，目标外部MLLM监督为0。
现有四卡队列：主实验独立完整推理，与Uniform/Confidence训练并行；随后Discovery/Geometry/普通Temporal/匹配Mean Teacher。
全部预测封存后，再生成双向数量对照并独立执行官方TrackEval与GT归因。
所有GT派生正确率、覆盖、传播错误和HOTA增益均以实际封存后计算为准，不使用loss或高分标签数代替。
