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
完整推理25视频/425表达，固定官方MOT、框和ID不动，FP32 margin>0；query-batch真实逐表达等价性检查保持。
封存后独立GT评测TrackEval HOTA/DetA/AssA/IDF1/IDSW与presence/empty/候选AP/AUROC/R@K/恢复FP/新增覆盖。
所有新方法做双向Count-Matched，仅离线诊断。25完整视频配对bootstrap2000次seed0。
若PSR同时超过Frozen与Naive且有正向HOTA/语义覆盖证据，再用预先固定的训练seed0/42确认，不根据指标改数值超参。

若PSR只超过Naive或无独立收益，保留负结果，按原提示分解采样/教师/拟合/扩展/最终选择，仅允许有证据的单变量下一步。
不能以测试通过/梯度非零结束研究，也不延长训练到出现更好数字。完整日志/恢复checkpoint留data2，小型记录逐阶段GitHub。
