# V2 → Refer-Dance Offline UDA 进度

日期：2026-10-26（Asia/Shanghai）。项目仅为TTA-RMOT，工作区`/data2/user/FlexHook-TTA`。
起点`5db7dadd50cd3a8d0e3c18aae18f4e4f7cab0ab4`，新分支`research/flexhook-v2-dance-uda-20261026`。
此前六组TTA及可恢复性诊断不覆盖、不重新训练。

## 已实际核验

- V2 checkpoint重新计算SHA256一致：`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`。
- 官方Dance源码Train40/Val25，不能按`training/image_02`目录名称推断划分。
- 已有展开图像仅25个Val视频，但原始11GB Refer-Dance.zip中含Train图像；优先从既有合法包仅提取40Train图像，数据保留data2，不复制Val。
- 25Val官方DETR+NeuralSORT输出已在；40Train同算法候选尚未确认，不能用GT导出或用Val替代。
- 首次GPU审计：5号零进程；3/4/6号各一进程、约76GiB空闲。此快照不是显存预留，启动前重新审计；不停止他人进程。

## 执行计划与权限边界

1. 资产审计、GT-free manifest、同配置Train MOT来源；先解决数据关键路径。
2. 独立`flex_uda`实现，复用原FlexHook decode/LoRA；源CE贯穿训练，模型跨episode持续更新。
3. 四组独立单卡真实smoke，通过后才运行预注册的全覆盖训练。
4. 仅用源域开发集选择模型；封存四组预测后完整25/425官方评测及数量匹配/查询存在性/排序归因。
5. 阶段提交代码、必要测试和实际记录到GitHub，不把框架完成或pilot当任务完成。

没有独立可靠目标presence证据时保留UNKNOWN并报告零有效监督；不伪造正例、不把UNKNOWN判负。
正式Train→Val UDA目前未启动：40Train同配置MOT缺失。

## 最新用户授权：先用25 Val看效果

用户2026-10-26追加授权“先用25个Val视频看看效果，先不用考虑train”。
据此增加独立阶段`TRANSDUCTIVE_VAL_DIAGNOSTIC`：25Val的无标签图像/425输入表达/固定官方MOT用于离线适应，
再在同一25/425上封存预测、隔离GT评测。不是未见目标视频泛化，也不是原Train→Val UDA。
训练进程不能打开目标标注，不能用Val GT选参数；不借40/385比例产生presence先验。
四组源监督/EMA/合法关系与弱MIL/组合对照采用同一预注册短预算、rank4、seed17；先真实20step smoke。
该中间阶段不将缺失Train候选替换成GT或其它前端，也不据诊断结果无限调参。

## 真机smoke与基线门禁已完成

- UDA必要测试23通过。v1烟雾实验虽产生有限梯度，但原冻结C-Hook跨表达regular在N=1是NaN常量，
  独立完整loss审计撤销四组v1通过状态，旧日志/权重完整保留在`results/runs/uda_val_smoke_20261026_v1`。
- 修复为仅训练授权LoRA的有效源CE，不使用不适用、无梯度的NaN常量；逐步显式检查源/目标实际目标有限。
  v2四组各20步完成，所有source/target loss有限，adapter非零更新，三个目标方法实际无标签loss梯度非零，
  冻结源参数未变、目标GT读取0；独立审计`results/uda/smoke_v2/independent_smoke_audit.json`。
- v2在最后写receipt时出现缩进错误，20步checkpoint与日志有效；修复后从20/20恢复仅完成收尾，未重跑任何更新。
- 从既有完整Frozen候选分数CPU重放全部425表达，425个预测SHA256逐一完全一致；独立官方TrackEval
  HOTA8.2180/DetA3.1937/AssA21.1470/IDF1 4.4414。不是重新测量GPU基线速度。
- 下一步四个独立worker各一个8973步完整覆盖周期，源17视频/8896表达，目标25视频/425表达/1275固定位置窗口。
  这不是只适应25帧，也不是40Train→25Val实验。原40Train官方同配置MOT仍缺失，暂不扩大任务。

## 完整覆盖执行中的AMP修复（保留原始失败）

四组首次覆盖运行在GPU5/4/6/3。SOURCE_REHEARSAL于2666步触发非有限缩放梯度；
真实出错源窗口诊断确认scale131072溢出、scale65536/32768及FP32梯度有限，source loss约2.47并未发散。
Teacher/Weak随后也在不同实际窗口触发同类错误；不打断仍健康的Combined。
恢复只修改数值执行：降低scale，恢复该步RNG/记忆/计数，重试**同一**窗口，不执行optimizer/EMA、不推进样本。
checkpoint之后的已运行但未保存后缀日志/伪证据归档保存，从最近有效checkpoint恢复该段。
每个恢复launcher使用独立记录目录，避免覆盖仍在等待健康worker的原launcher状态。
这是正常AMP溢出的可恢复处理，不改lr、rank、seed、源/目标loss权重、目标阈值或源CE模型选择。
恢复回归测试26通过；诊断记录见`results/uda/assets/amp_real_source_diagnostic.json`。
