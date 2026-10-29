# PSR 实验状态

2026-10-29，用户允许的25Val同集适应，非Train→Val/非盲测。

已完成匹配预算教师缓存及真实学生smoke；Naive/PSR已在独立目录完成固定8973步训练。
两组均保持24576参数、seed17、lr5e-5和源val CE选checkpoint，不读取目标GT。

完整RMOT新指标：**尚未产生**。Frozen8.2180和Source-only7.4352为已核验可复用参考，
旧Naive8.0327使用另一采样协议，仅保留历史，不能填入本轮新Naive的空位。

主实验训练耗时：Naive主体2554.08秒，PSR主体2391.76秒（含完整合法源覆盖和源val选择；完整进程时间另列）。
两组分别使用8500直接软标签；PSR使用254条不同的一帧弱插值；这些都不是GT正确性统计。
匹配Mean Teacher已通过6步真实smoke，目标外部MLLM监督为0。
Naive/PSR主实验独立完整推理已分别封存425表达，实际进程1851.47/2021.72秒，参数更新0、目标GT0。
两组都由源val选中8973步，源CE分别0.88351128/0.88294703；不是按Dance指标选择。
Uniform/Discovery/Confidence/Geometry现已各完成8973步，8500直接标签均实际使用，目标GT0，已进入或完成完整推理；6/8学生训练已完成。
Geometry全部263条独立弱扩展均实际访问。剩余普通Temporal与匹配Mean Teacher由原队列及自动释放卡接力执行。
Uniform、Discovery的完整425预测现也已封存，合计4/8学生完整预测。
普通Temporal已在原GPU1开始固定8973步；Discovery进程退出后，自动Mean Teacher接力已在GPU6入场。
Geometry在GPU4继续完整推理；Confidence退出后，GPU5上的同名MeanTeacher排队项只在CPU锁等待，实际训练仍在GPU6，不重复分配CUDA。
Confidence、Geometry完整425预测现也已封存（6/8完整学生预测）。前六组的12个双向数量对照已在CPU预生成，各425表达、目标GT0；
这只是GT-free部分预计算，不写最终GT评测授权seal，不把派生对照当作无需父模型的独立系统。
匹配MeanTeacher现已完成8973步/8973次EMA更新，17源视频及425目标表达完整覆盖，实际未标注输入8500、外部MLLM标签0、目标GT0（7/8训练完成）。
原GPU5排队副本在训练锁等待后复用完成训练；完整推理实际由GPU5 PID3400322执行，GPU6的同名推理副本在CPU锁等待，无双份GPU计算。
普通Temporal随后完成8973步及全部270条独立弱标签覆盖；8/8训练全部完成，各17源视频/425目标表达/24576可训练参数/目标GT0。
普通Temporal实际推理在GPU1 PID3405790，MeanTeacher在GPU5；完整推理尚余这两组，其余6/8完整425预测已封存。
Discovery/Geometry首次因原GPU2/0显存被外部新进程占用而入场失败；失败时尚未开始优化。
保留失败日志与原16+4GiB标准，修复子队列排除健康GPU1/5后，在新审计GPU6/4只恢复两项失败任务；最多同时四卡。
后续普通Temporal/匹配Mean Teacher仍由原健康队列执行，不重复已完成训练或主实验完整推理。
全部预测封存后，再生成双向数量对照并独立执行官方TrackEval与GT归因。
所有GT派生正确率、覆盖、传播错误和HOTA增益均以实际封存后计算为准，不使用loss或高分标签数代替。
