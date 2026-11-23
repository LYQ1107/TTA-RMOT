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

## A完成，先评测而不空等

SOURCE_REHEARSAL已完成全部8973更新，17源视频/全部8896原表达及77补充窗口，3次同样本AMP重试且0跳过。
source-val-only选择7750步CE0.9631536889；最终步没有因目标指标而被替换。
完整训练记录与checkpoint索引已归档`results/uda/full/SOURCE_REHEARSAL`。
实际GPU3三帧×17表达批推理parity：最大margin误差约1.15e-5、0选框差异，开始完整25/425评测。
B/C/D仍各自恢复原覆盖周期，未增加方法、轮数或目标GT监督。
新旧PF+UDA核心集成回归52通过，原六组结果不修改。

## 四组训练全部通过、首两组完整官方评测（非最终四组报告）

全四组8973×4=35892次实际optimizer更新，有限目标/梯度独立审计PASS，跳样本0、每组AMP同样本重试3次、
源参数冻结不变、目标GT模型访问0。B/C/D各覆盖25目标视频/425表达；A不使用目标loss。
source完整视频/表达覆盖实际393个不同anchor（含四观测）；target75个anchor/1275query-window，不能误称逐帧全量优化。

完整25/425官方结果目前：Frozen8.2180；Source Rehearsal7.4352（−0.7828）；Mean Teacher8.2494（+0.0314）。
数量匹配Frozen分别7.4207、8.2479；Teacher相对自己的数量匹配控制只+0.0015。
这是依赖适应数量的POST-HOC诊断，不是可部署方法，也不是因果贡献百分比。
Source Rehearsal虽AssA28.1110更高，却DetRe2.3022、HOTA更低；源CE改善不代表跨域referring改善。
这些官方评测只读各自已封存的425条预测，并在四组训练全部结束、checkpoint选择固定后执行。
Weak/Combined还在推理，不据首两组目标指标改任何模型、门限、参数或剩余推理。
独立首批原始官方表在`results/uda/interim_2_methods`，最终四组结果另行生成。

## 最终状态：25Val四组、全部评测与离线归因完成

前述“执行中/两组完成”是原始阶段记录，不再代表当前状态。
四组均完成8973更新、0跳样本、3次同样本AMP重试；全部冻结源参数未变，模型目标GT打开记录为空。
四组完整推理和四组Count-Matched均为25视频/425表达，主推理每组3588445候选-query观测、433636活动帧-query。
425个独立episode全部封存，MOT几何和ID不变；FP32推理0optimizer/EMA更新，官方HOTA/CLEAR/Identity完整汇总。

| 方法 | HOTA | ΔFrozen | Count-Matched HOTA | 适应−Count-Matched |
|---|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 7.4352 | -0.7828 | 7.4207 | +0.0145 |
| MEAN_TEACHER_UDA | 8.2494 | +0.0314 | 8.2479 | +0.0015 |
| WEAK_MIL_UDA | 6.9281 | -1.2899 | 6.9005 | +0.0276 |
| COMBINED_UDA | 7.7293 | -0.4887 | 7.7218 | +0.0075 |

25视频聚类bootstrap、真实目标恢复/新增误检、40Present/385Empty、正确候选排序、文本互换/视觉偏置和伪标签质量均已离线完成。
Teacher正伪标签仅84/958暴露对应GT，其中862个伪正暴露来自Query-Empty；Repeated exposure不是独立样本。
当前没有可信的语义恢复结论，不根据这些GT结果改模型或启动第五组。

## 原始goal与当前授权阶段的逐项验收

| 原始要求 | 当前证据 | 判定 |
|---|---|---|
| 项目/独立分支、保留旧结果 | 本库origin、当前research分支、原六组目录与旧报告保留 | 已完成 |
| 完整源17Train监督与独立源开发、V2权重 | assets审计、固定训练plan、严格加载SHA、64窗口source-val CE | 资产与本轮覆盖已完成；不是官方骨干全训复现 |
| 40Train与25Val不交叉、完整40Train图像/表达 | `assets/dance_manifest_audit.json`、`train_image_preparation.json` | 资产划分已完成 |
| 40Train同算法同配置预测MOT及Train→Val四组UDA | `assets/official_tracker_release.json`只有25Val公开候选；本轮target plan明确25Val | **未完成**；用户最新授权先不考虑Train，不可用本轮替代 |
| GT隔离、LoRA/EMA/合法关系/UNKNOWN语义 | 训练/推理receipts、finite audit、模型实现与真实smoke | 本轮实现和执行完成；目标presence及静态时间证据覆盖0，不宣称有效目标MIL |
| 四组真实GPU、源域选模型、完整目标官方评测 | `full/full_training_finite_audit.json`、每组425 receipts、官方summary/detailed | 25Val同集诊断完成，不是独立未见视频泛化 |
| Frozen匹配、Count-Matched、presence/排序/成本/视频不确定性 | `summary.csv`、`full/summary.json`、text/pseudo诊断JSON与逐表达/视频CSV | 已完成 |
| GitHub代码、实际指标、完整日志和可恢复路径 | `results/uda/full/`与四个425预测压缩包、恢复索引、训练源码版本 | 已提交；源码本地/公开commit可能不同但逐次树一致 |

原始正式40Train→25Val目标没有被缩小后宣布成功。当前阶段按用户要求完成后停止，
后续是否补同配置Train MOT并执行正式UDA需要用户明确恢复Train阶段；不自行重训MOT、加入教师或延长训练。
完整报告：`UDA_FINAL_SCIENTIFIC_REPORT.md`。原始完成状态分别为
`results/runs/uda_val_full_20261026_v1/full_suite_completion.json`和
`results/runs/uda_val_evaluation_20261026_v1/evaluation_suite_completion.json`；首次失败launcher不是最终状态。
收尾复核52项核心测试通过；94份完整文本日志已归档并逐文件复核，位于
`results/uda/full/logs/runtime_logs.tar.gz`，没有GT、图像、权重或大容量margin流。
实际训练/推理使用的六个本地源码版本均包含在`full/source_publication_map.json`的同树公开提交映射中。
