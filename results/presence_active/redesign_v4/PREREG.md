# v4：源校准的多伯努利数量分支

2026-11-02；LYQ1107/TTA-RMOT，method/presence-active-20261102。用户明确要求继续尝试另一种构造。v1/v2/v3全部结果、门槛和日志保留。

## 学习与假设

- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) 用低维温度校准概率；正温度不改变单个候选的分类边界。这里采用两个正斜率，分别校准正/负margin，不复制论文的实验结论。
- [Reuter et al., The Labeled Multi-Bernoulli Filter](https://ba-ngu.vo-au.com/vo/RVVD_LMB_TSP14.pdf) 给出多伯努利集合及数量分布的建模。这里仅借用集合概率的数量聚合，不复现该跟踪器、不改变固定MOT。
- v3用帧级count CE训练PCD残差，仍在KITTI/Dance大量删掉真目标。新假设：先用候选级真实标签校准冻结分数，再对全部候选集合边际化，比重学整帧数量更能保留源模型并提供可用的不确定性。此假设未验证，不预先宣称成功。

## 一次固定构造与拟合

1. 每个官方模型仅拟合自己源Train缓存的(margin, candidate label)。Sports用expression_id修正标签v2。PCD/句子/图像不重导，也不读取其他域的数值标签。
2. 两个参数 alpha_pos / alpha_neg >0，l_i=alpha_sign(m_i)*m_i，p_i=sigmoid(l_i)。没有偏置、句子/video参数或目标域统计；变换严格单调，单候选的m>0判断不变。
3. 固定alpha范围[0.05,20]。在完整源Train候选上，分别精确拟合两组的一维凸候选BCE/NLL，scipy bounded标量优化，xatol=1e-6、maxiter=128。源验证仅报告NLL/Brier，不选参数；不按源或跨域HOTA选拟合结果。固定seed0；无需GPU、没有SGD/LoRA更新。
4. 条件独立伯努利近似：P(K=k)为prod_i[(1-p_i)+p_i*z]中z^k的系数，用FP64动态规划计算。P(存在)=1-prod_i(1-p_i)，E[K]=sum_i p_i。允许0..n、零候选时P(K=0)=1。不存在必须有正例的约束。
5. 原始输出k=argmax P(K=k)，并列取最小k，再按原封存margin及track_id规则取top-k。**不会把k强行抬到Native数量，也不会在失败时回退Native。** 单候选符号保持不代表集合数量保持；因此仍必须真实跑HOTA。平滑只保留原±round(fps/2)帧号窗口对照，不用于过关或选头。
6. alpha=1的未校准分布只作源概率诊断；不加入另一套HOTA候选供挑选。报告校准前后候选NLL/Brier和数量变化，所有正式结果均来自上述唯一源Train最优解。

## 验证与执行

- 先提交并push本预注册，再拟合或评测。定向测试精确枚举小集合/空集/>10/归一化/期望/排列及padding/符号排序保持/严格加载/GT隔离。
- 先完成三套源拟合，封存并推送参数和选择清单，然后GT-free解码三个完整同域集合，预测SHA封存后独立官方TrackEval；Sports v4全466及相同内容原418ID敏感性。
- **原门槛不变：三套PRES_TOPK自身源HOTA均>=完整Native−1。** 不能用原生不变的候选分类符号替代此验收。失败如实归因，不修改阈值或改用平滑过关。
- 通过三域门槛才跑全部九格A并发布，再接入B/C实际C1/C3 pilot及40 GPU小时成本关卡。未通过则封存本轮，不制造跨域/TTA结果。
- 新目录results/presence_active/redesign_v4及既有data1缓存根redesign_v4。复用全部原候选/封存分数；本头拟合与解码仅CPU。未来GPU仍最多2张、显存合适且优先进程少、不停止其他任务。长任务单阻塞协调器等待。

局限：候选依赖性、源GT候选→目标tracker支持偏移仍可能破坏校准；noisy-or会在大量弱候选时高估存在性。两参数不产生新视觉语言语义，源通过也不能证明跨域恢复。所有源概率诊断/同域检查不是盲测证据。
