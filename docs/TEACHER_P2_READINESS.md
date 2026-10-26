# P2 readiness：证据驱动，不自动开训

本轮P1判决：`SIMPLE_TEACHER_ONLY_NO_EVIDENCE_CONTRIBUTION`。任何教师Gate A通过=True；
Evidence Gate C通过=False。正式P2当前不可执行，不启动学生训练。

完整Gate逐项证据见`results/teacher_probe/final_scientific_verdict.json`、bootstrap及P1报告。
如果Gate A失败/支持不足，保留负结论，停止无依据的蒸馏；不能延长LoRA训练替代伪标签有效性证据。
若仅简单教师通过、B失败，只能保留简单蒸馏作为将来明确授权的效率基线，不宣称Evidence机制成功。

## 即使未来有足够证据，仍须先解决

1. 40个Train视频匹配官方冻结DETR+NeuralSORT的合法预测MOT；不能用GT框/ID，也不能静默换跟踪器。
2. GT无关的预注册随机30适应/10确认研究划分（不是官方Train/Test）；先核实确认10视频有合法指代GT。
3. 原25Val已经多轮诊断，只能作为开发性官方基准，不能再叫盲测。
4. 相同教师、候选、源真实监督、学生参数和训练预算的S0 Source/S1整句/S2置信加权/S3证据蒸馏。
5. C-Hook/PCD保留；若新增属性头，应配参数量匹配对照；不把容量或更大MLLM资源误当方法贡献。
6. 只有同预算优于现成S1/S2，并经完整TrackEval、presence、ranking、Count-Matched及独立视频确认后才讨论贡献。

这些是条件性研究准备，不是已执行的P2实验，也不是本轮新增训练授权。当前遵守STOP_AFTER_P1。
