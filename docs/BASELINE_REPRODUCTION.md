# Baseline reproduction boundary

本项目把官方 TempRMOT 当作冻结 source，不重新训练 detector、backbone、text encoder、tracker 或 QIM。源权重加载要求 `missing_keys=[]`、`unexpected_keys=[]`（仅允许官方 RoBERTa `position_ids` buffer 兼容处理），并报告源码 commit、权重 SHA 和参数冻结数。

当前方法名与实验含义：

| 名称 | 本项目实现 |
|---|---|
| FROZEN / B0 | 原始 referring 概率与原始框/ID，外部不更新 |
| ENTROPY_HEAD / B1 | 候选级 Bernoulli entropy 的 residual adapter 移植对照 |
| EMA_ST / B2 | 高置信候选的因果概率 EMA，不使用 query logic/state |
| HEAD257 | 外部复制 `Linear(256,1)` 的容量对照 |
| PROBE_CACHE / B4 | 相同 frozen probe/evidence 的有界、无梯度后续校准 |
| PROBE_ST / B5 | 相同 probe 预算下的普通 q0 高置信自训练 |
| QCS_TTA / O1 | 等价/互斥 query evidence、集合式监督和 semantic state |
| NO_STATE / A1 | 相同证据和 adapter，使用 fixed frame-unit TTL |

默认方法使用相同 source 框、物理 ID、面积/objectness/referring 阈值；AerialMind 的显式 `relative_margin` 结构对照则取消 absolute referring gate，按当前帧相对 logit 保留候选集合，并在配置中单独标注。在线 runner 在当前帧提交预测后才执行 probe、observe 和 optimizer step；更新最多从下一帧生效。评测器独立于 runtime manifest，只在预测文件封存后打开 evaluation manifest。

V1 完整匹配 run 实际覆盖 0012/0014 的 56 个 expression episodes、5,320 帧；V2 另外保留了 0005 的单个完整 297 帧 episode。`EMA_ST` 在本项目中是概率 EMA 对照，和完整论文方法/同一 residual adapter 的 B2 并不等价，结果中按此名称解释。

这不是各论文原始代码的完整复现，而是明确标注的 RMOT 移植对照；参考位置记录在 `docs/REFERENCE_IMPLEMENTATION_MAP.md`。
