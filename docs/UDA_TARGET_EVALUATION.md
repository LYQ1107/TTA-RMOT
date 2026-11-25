# 完整目标评测与复现

协议与全部数值见`UDA_FINAL_SCIENTIFIC_REPORT.md`，机器表在`results/uda/*.csv`。
预测：`scripts/evaluate_uda_full.py`；逐视频FP32 query-parity、近零回退，0optimizer/EMA。
封存：`scripts/finalize_uda_evaluation.py`，每组25/425、固定3588445候选-query行。
官方评测：`scripts/evaluate_flexhook_dance_tta.py`；使用固定FlexHook TrackEval fork，HOTA/CLEAR/Identity，非分片均值。
同一gt_template、视频编号及原始1-based帧，无偏移重写；mark非0及class1过滤与既有独立协议审计一致。
Count-Matched构造：`scripts/diag_uda_countmatched.py`，不读取GT；随后独立官方评测。
离线归因：`scripts/analyze_uda_full.py`复用旧归因器；伪标签质量另由`analyze_uda_pseudo_offline.py`读取GT。
训练与推理从未打开目标GT。这里只在完整预测封存之后附加GT并评测，不把离线分组回流模型。

所有输出保持官方MOT框/ID/置信度原行不变；候选末观测692行已补齐，不再与旧8.206候选覆盖混用。
每个完成episode有prediction/margin SHA256和完整帧receipt；大分数流与权重保留data2/server，索引在full_recovery_index.json。
