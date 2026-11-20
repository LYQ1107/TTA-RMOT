# Frozen MLLM → FlexHook：普通软概率蒸馏预注册

2026-10-28；TTA-RMOT，独立分支 `research/v2dance-mllm-naive-distill-20261028`，
从核验的公开P1 `7026d82cbd9ccde2465f98d6c24c48e2fccd1e34` 创建。

## 当前用户明确修订的范围

原最终方向是40Train→25Val，但指定40Train预测MOT仍缺。用户明确要求“40Train没有的话先用val吧”。
当前执行 **25Val同集、无标签、离线软蒸馏开发诊断**，不是Train→Val泛化，不是Source-Free Online TTA。
25Val已多次开发，不能称盲测；保留原40Train目标的资产缺口，不用GT导出轨迹或更换检测器。
用户重新允许显存足够时共享GPU，优先进程少的卡，最多4物理卡，不停止其他任务。

## 固定假设和数据

检验冻结T0原整句教师是否能让24576参数PCD LoRA学会更好的query-conditioned匹配，
而不是先堆叠分属性、验证器或Positive-Support Recovery。P1已经指出Evidence无独立贡献。
复用未改写的P1 A：25视频/425表达、10/30/50/70/90%活动帧、每表达每锚点≤4固定候选，8381个真实配对。
直接导出P1原始BF16 Yes/No logits与原`p_yes`，不重新调用MLLM、不按新GT筛选；所有输入/输出hash追溯原封存。
缓存逐候选identity键显式包含模型revision、prompt、完整图像/ROI、bbox、视频/表达/帧/预测ID。
低置信度但有效的原始soft概率保留；P1阈值UNKNOWN不被偷换成NEG；真正无效响应不参与KD。
未采样人物没有目标损失；同帧各人物独立sigmoid，不能用人物间softmax。

源域继续使用合法17视频KITTI-V2 Train全部8896表达与77轨迹覆盖锚点，共8973源窗口，
源验证独立4视频×16固定表达。目标训练GT读取0；只有封存完整学生预测后才独立评测GT。
Landlock输入白名单复用现有隔离模块，允许源真实标签而不允许目标`labels.json`/`gt.txt`/表达标签文件。

## 固定优化与选择

V2初始化SHA `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。
保留PCD最后两层Q/V LoRA rank4、零输出初始化、24576参数；seed17、AdamW lr5e-5/wd1e-4、clip1、FP16 AMP，
每步一个源窗口＋一个目标已标注窗口，单卡训练，不写未验证的DDP。
源多层CE保持原实现；目标只用最后一层`Yes−No` margin和原始soft BCE。
lambda_KD候选先固定为1.0（两个域分别按有效候选均值归一化）；正式目标训练前必须在合法源开发样本上验证有限梯度与尺度，
只允许源校验发现实际数值错误时修复并重登记，不能根据Dance GT改lambda。
每周期覆盖完整8973源窗口和全部2125目标窗口，8973次优化；目标较短调度显式循环，记录唯一覆盖与重复exposure，不虚增覆盖。
每250步保存完整恢复状态并只按源val CE选checkpoint，包含step0基线；不得依据Dance Val指标选epoch或延长训练。

E0复核匹配完整211085行MOT、425表达的Frozen HOTA8.2180。
E1优先逐项核验历史SOURCE_REHEARSAL是否完全同初始化/源计划/参数/优化器/步数/源选择；完全相同时可复用，不重训健康完成的结果。
E2为独立MLLM_DISTILL；历史Mean Teacher只作为注明不同target采样/损失和额外教师监督的参考，不自动当公平教师对照。
先P1缓存无step接口验证，再真实Val教师标签短训练smoke，再完整E2。每步记录源/KD loss、各自梯度范数/夹角、参数变化与覆盖。
学习失败应诊断源/KD冲突、拟合与容量，不通过目标GT重新调阈值。

## 完整评测和停止

同一25视频/425表达/固定MOT，FP32最终层margin>0，完整TrackEval HOTA/DetA/AssA/IDF1/IDSW。
复用既有query-batched评测时，每个checkpoint进行真实逐query等价性检查；近0用原独立query路径。
Count-Matched双向报告：Frozen按学生逐帧数量取top-k；学生按Frozen逐帧数量取top-k，固定ID破同分。
均为依赖已知数量的POST-HOC诊断，构造不读GT，不是新部署阈值。
候选AP/AUROC/R@1/3/5、query-present/官方-empty、恢复/丢失GT、新增/删除FP行及匹配重分配分开。
评估教师锚点与未直接标注帧，区分教师未询问/MOT缺失/教师错答；单人多框不当独立GT。
25视频cluster bootstrap2000次seed0，不把帧/表达当独立样本。先seed17；可信正向结果才按预定seed0/42复核。
成功须完整RMOT正向变化、正确排序/覆盖改善、数量匹配不能解释全部、访问审计通过；不能靠loss下降宣布成功。
只有普通蒸馏经严格归因成立，才做独立E4/E5/E6常规对照和后续机制设计；否则先完成失败诊断，不启动大规模Ours。

所有P1/旧UDA/TTA结果只读保留。新代码/配置/小型记录逐阶段公开，大型模型/缓存保留data2和可恢复索引。
