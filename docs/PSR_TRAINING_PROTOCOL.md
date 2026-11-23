# PSR v1 训练预注册（2026-10-29）

用户本轮明确允许25Val fallback及显存足够时共享最多4卡。正式40Train目录缺失，作者公开声明NeuralSORT/NKF未发布；
保留正式资产审计，当前所有新方法只标Val同集无标签适应，不称Train→Val或盲测。

每方法相同425表达×5锚点×2轨迹×2端点=8500次逻辑教师判断（真实预处理已证实各策略均8500），
四采样缓存Naive/Uniform/Confidence/PSR；后续PSR发现、几何扩展、Full三个变体共享PSR原始缓存，
Naive与普通Temporal共享Naive缓存。共享cache命中不抹去各方法所需独立判断预算及输入成本。

固定V2初始化SHA71dbd691…613，最后两层PCD Q/V rank4 LoRA共24576参数，seed17。
固定源17视频8896表达+77覆盖锚点=8973窗口；目标每策略2125窗口，循环覆盖，所有方法8973步。
AdamW lr5e-5/wd1e-4，clip1，FP16 AMP；每步源与目标各1窗口，direct两帧合并为同一均值目标。
source/direct/expanded损失权重1/1/0.25，按各自有效候选数归一化，不让扩展条数重复放大损失。
逐步记录三loss/三有效数/梯度范数及两两余弦/直接和扩展POS-NEG-UNKNOWN；固定每250步按源val CE选择含step0，绝不按Dance HOTA选。

先真实4候选教师smoke→封存完整4策略教师输出→Naive与PSR各6步及真实恢复→主实验E2/E3。
主实验健康完成后再做Uniform/Confidence/Discovery-only/Geometry/NaiveTemporal。原Frozen8.218和Source-only7.4352可在匹配协议审计通过后复用，旧Naive8.0327不是本轮对照。
另补做 `MEAN_TEACHER_MATCHED`，不把旧Mean Teacher同集结果冒充匹配对照。
它仅读取Naive的相同未标注输入清单（2125双帧窗口），不打开Qwen软标签；相同源计划、8973步、24576参数、优化器及源val选择。
沿用既有Mean Teacher常规设置：EMA衰减0.99，0.05特征dropout强视图，0.2 SmoothL1一致性与0.2置信伪标签BCE，
source/EMA/student一致性阈值0.9、容差0.1、无正例时不造硬负例。
这项常规对照的外部语义教师调用/标签预算为0，不属于8500次MLLM预算的采样消融；不能将未标注输入数写成教师标签数。
首次真实interface与6步训练通过时尚未读取本轮目标评测指标；其后保持同一配置完成8973步与封存后官方评价。
完整推理25视频/425表达，固定官方MOT、框和ID不动，FP32 margin>0；query-batch真实逐表达等价性检查保持。
封存后独立GT评测TrackEval HOTA/DetA/AssA/IDF1/IDSW与presence/empty/候选AP/AUROC/R@K/恢复FP/新增覆盖。
所有新方法做双向Count-Matched，仅离线诊断。25完整视频配对bootstrap2000次seed0。
若PSR同时超过Frozen与匹配Naive的HOTA点值，按用户6.4要求用预先固定的训练seed0/42确认，加上17共三seed，
不因区间跨0或语义归因不支持就免除该确认，也不根据指标改数值超参。
确认时教师样本/软标签/调用预算不变，仅训练随机seed不同；每个seed独立run根目录，Naive/PSR同seed配对。
Frozen无可训练参数且固定确定性输出，复用同一425结果，不将复制三份冒充三次独立观测。

2026-10-30 条件已触发：首轮完整官方HOTA Frozen=8.2180、Naive=7.4499、PSR=8.4204。
执行 `scripts/run_psr_confirmations.py`：seed0/42各Naive与PSR，共4个学生；所有数值配置和缓存保持不变。
外层统一注册最多4张物理卡并固定lane，内部调用既有单卡训练及425表达推理；原seed17和历史结果不覆盖。
Uniform=8.4347、匹配MeanTeacher=8.6095均高于PSR，因此不能据此宣称PSR独立方法成立。

若PSR只超过Naive或无独立收益，保留负结果，按原提示分解采样/教师/拟合/扩展/最终选择，仅允许有证据的单变量下一步。
不能以测试通过/梯度非零结束研究，也不延长训练到出现更好数字。完整日志/恢复checkpoint留data2，小型记录逐阶段GitHub。

## 2026-10-30 最终执行核对

首轮8学生、确认seed0/42各2学生和POST_HOC端点权重控制1学生均实际完成8973步及25视频425表达推理。
所有checkpoint仍只按源val CE选择；同一教师cache/8500调用预算、采样seed17、相同优化器/容量未因Dance指标改变。
端点权重控制使用263几何机会的526原始直接端点项，额外0.25独立均值，无新弱标签；它是后验机制诊断，不冒充预注册主方法。
三seed确认/父权重控制均已完成；保留全部原入场失败、锁等待和恢复记录，未停止外部进程、未重跑健康完整episode。
官方26+8+8+5=47套（含复用基线/后验数量控制）与所有归因已封存，当前停止，无待执行第二域或附加长训。
