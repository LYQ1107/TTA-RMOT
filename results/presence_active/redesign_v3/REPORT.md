# PresenceHead v3：授权结构/损失改造的真实结果

同域门槛失败，B/C 未启动。

## 1. 验收

这是各官方模型在自身源域的验证，不是V2→Dance跨域TTA成绩。原门槛：三个自身源 PRES_TOPK HOTA 均不得低于完整 Native 超过 1。只按自身源验证新目标函数选择 checkpoint，不按 HOTA 回选旧头或 epoch。

| target | model | native_HOTA | HOTA | delta_HOTA | passed |
| --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | 42.752000 | 39.130000 | -3.622000 | False |
| Dance | Dance-best | 32.139000 | 25.942000 | -6.197000 | False |
| SportsMOT | LaMOT-best | 69.924000 | 70.360000 | 0.436000 | True |

与完整保留的旧设计对照（仅比较，不按 HOTA 重新选择）：

| source | native_HOTA | original_default_HOTA | original_source_lr1e5_HOTA | residual_v3_HOTA | v3_delta_vs_native |
| --- | --- | --- | --- | --- | --- |
| KITTI-V2 | 42.752000 | 40.599000 | 40.188000 | 39.130000 | -3.622000 |
| Dance | 32.139000 | 21.605000 | 25.671000 | 25.942000 | -6.197000 |
| SportsMOT | 69.924000 | 65.180000 | 63.937000 | 70.360000 | 0.436000 |

## 2. 结构、监督与选择

本轮只执行 [PREREG.md](PREREG.md) 固定的一种结构：小型共享 PCD/几何集合网络为冻结排序的前缀能量学习残差，联合数量 softmax 覆盖 0..n；输出用联合 MAP，不改身份排序。原生 margin 提供零初始化；CE(count)+0.1 KL(native||head)，固定 lr1e-4、seed0、batch256、max20epoch/patience3。未训练 FlexHook、未加入教师。独立句子向量/ID 捷径移除，但 PCD 本身仍是查询条件特征。

三个全量源验证集的 epoch0 原生数量等价检查均必须 0 差异；这只是初始化对照，不是增益。短前缀能量用 FP64 累加以避免近零正增量被大前缀舍入掉，网络与缓存输入仍 FP32，旧 backbone/官方分数不改。Sports 源监督使用已核对全部表达身份的修正标签 v2。按原任务允许，自身源官方验证/测试用于早停，因此同域验收不是独立盲测；跨域 GT 不选训练 checkpoint。

| source | model | epochs | selected_epoch | native_prior_loss | source_loss | changed_source_units | validation_units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | 4 | 1 | 4.173164 | 1.243340 | 144603 | 457503 |
| Dance | Dance-best | 6 | 3 | 3.200518 | 2.324244 | 31607 | 433636 |
| SportsMOT | LaMOT-best | 7 | 4 | 4.537001 | 2.080514 | 54692 | 266786 |

存在性概率是联合分布的 P(k>0)，阈值0.5与联合MAP的 k>0 可能不一致；两者混淆会夸大召回。本表 decoded_* 指实际输出，head_* 为概率分支诊断。无固定10上限，但不保证召回或排序被修复。

## 3. 完整已计算结果

| target | model | method | native_HOTA | HOTA | delta_HOTA | DetA | AssA | IDF1 | IDSW | prediction_rows | query_empty_prediction_rows | Present_HOTA | count_MAE_head | decoded_presence_recall | changed_count_units_vs_native | added_prediction_rows_vs_native | removed_prediction_rows_vs_native | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | PRES_TOPK | 42.752000 | 39.130000 | -3.622000 | 28.394000 | 54.073000 | 43.673000 | 2537.000000 | 482899.000000 | 0.000000 | 39.129525 | 0.782004 | 0.726315 | 144603.000000 | 3503.000000 | 296481.000000 | 0.000000 |
| KITTI-V2 | V2-best | PRES_TOPK_SMOOTH | 42.752000 | 39.491000 | -3.261000 | 28.748000 | 54.403000 | 44.035000 | 2592.000000 | 468777.000000 | 0.000000 | 39.490914 | 0.742579 | 0.736950 | 163095.000000 | 7139.000000 | 314239.000000 | 0.000000 |
| Dance | Dance-best | PRES_TOPK | 32.139000 | 25.942000 | -6.197000 | 22.303000 | 30.222000 | 23.690000 | 1014.000000 | 133810.000000 | 39084.000000 | 27.951917 | 0.469687 | 0.374316 | 31607.000000 | 91.000000 | 121191.000000 | 0.000000 |
| Dance | Dance-best | PRES_TOPK_SMOOTH | 32.139000 | 25.949000 | -6.190000 | 22.067000 | 30.562000 | 23.637000 | 982.000000 | 130119.000000 | 36791.000000 | 27.846580 | 0.467265 | 0.379646 | 33023.000000 | 412.000000 | 125203.000000 | 0.000000 |
| SportsMOT | LaMOT-best | PRES_TOPK | 69.924000 | 70.360000 | 0.436000 | 72.025000 | 68.758000 | 72.188000 | 4343.000000 | 1532188.000000 | 0.000000 | 70.360168 | 0.904144 | 0.926835 | 54692.000000 | 24565.000000 | 52795.000000 | 0.000000 |
| SportsMOT | LaMOT-best | PRES_TOPK_SMOOTH | 69.924000 | 70.440000 | 0.516000 | 72.197000 | 68.751000 | 72.266000 | 4852.000000 | 1525824.000000 | 0.000000 | 70.440149 | 0.827442 | 0.929987 | 81301.000000 | 36219.000000 | 70813.000000 | 0.000000 |

### 3.1 数量决策与概率分支的区别

| source | MAP_presence_recall | native_presence_recall | MAP_presence_FPR | native_presence_FPR | MAP_count_MAE | native_count_MAE | max_MAP_k | above_ten_MAP_units | MAP_vs_probability_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | 0.726315 | 0.872746 | 0.231149 | 0.374954 | 0.782004 | 0.985386 | 14 | 437 | 15682 |
| Dance | 0.374316 | 0.470665 | 0.018280 | 0.055885 | 0.469687 | 0.547706 | 20 | 2555 | 4675 |
| SportsMOT | 0.926835 | 0.910980 | 0.172134 | 0.119864 | 0.904144 | 1.060921 | 21 | 34041 | 1316 |

身份排序完全不变，所以所有 HOTA 变化来自数量选择及其造成的关联/匹配变化，不能称为跨模态排序恢复。若 MAE 下降但 HOTA 不保留，说明该平均误差不足以代表关键真目标的召回与关联；概率阈值准确率同样不能替代实际 MAP 输出。大于10的输出说明容量上限确已去除，不意味着集合外漏检候选被补回。源GT候选→验证tracker候选的支持偏移仍在，这里只能作为与结果一致的机制解释，不据此断言唯一失败原因；不再按这些验证结果改本轮阈值或损失。

### 3.2 同域结果的后验描述性归因（不用于选择）

- KITTI-V2：HOTA相对Native -3.622；实际输出存在性召回 0.872746→0.726315，数量MAE 0.985386→0.782004。
- Dance：HOTA相对Native -6.197；实际输出存在性召回 0.470665→0.374316，数量MAE 0.547706→0.469687。
- SportsMOT：HOTA相对Native +0.436；实际输出存在性召回 0.910980→0.926835，数量MAE 1.060921→0.904144。

| target | delta_HOTA | CI95 | clusters |
| --- | --- | --- | --- |
| KITTI-V2 | -3.622198 | None | 4 |
| Dance | -6.196497 | [-13.033849, -1.003771] | 25 |
| SportsMOT | 0.436406 | [0.232211, 0.650346] | 45 |

这些区间按25/45视频聚类2000次seed0；KITTI四视频仅点估计。所有表达/帧按视频一起重采样，且自身源验证参与早停，因此不把区间当成独立盲测的显著性证明。门槛仍只执行预注册点值，不用区间翻案。Sports即使同域改善也不是跨域或完整TTA成功，当前没有新头的跨域/教师实验结论。

### 3.3 Sports418敏感性

| target | model | method | HOTA | DetA | AssA | IDF1 | IDSW | Dets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SportsMOT | LaMOT-best | PRES_TOPK | 70.650000 | 72.552000 | 68.823000 | 72.464000 | 4325.000000 | 1518271.000000 |
| SportsMOT | LaMOT-best | PRES_TOPK_SMOOTH | 70.717000 | 72.716000 | 68.798000 | 72.537000 | 4835.000000 | 1512379.000000 |

本轮只完成三套同域和各自平滑对照；六个跨域新头结果未计算，不把旧头成绩当新头成绩，也不把 NOT_RUN 写为零。

## 4. 成本与 GT 审计

| source | model | GPU_hours | peak_allocated_GiB | physical_gpu | updates | parameters | first_gradient_norm | target_GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KITTI-V2 | V2-best | 0.373983 | 0.057831 | 4.000000 | 17176 | 69505 | 0.231378 | 0 |
| Dance | Dance-best | 0.200733 | 0.111450 | 6.000000 | 3852 | 69505 | 0.252906 | 0 |
| SportsMOT | LaMOT-best | 0.098227 | 0.067138 | 6.000000 | 833 | 69505 | 0.421970 | 0 |

新增源头训练/验证计时合计 0.672943 GPU进程小时（含epoch0）；共享卡 elapsed，不是利用率积分，不含部分进程启动/CPU单元枚举开销；协调器整段 wall time 另存 costs.json。独立 CPU 解码/官方 TrackEval 时间见 costs.json；backbone 特征导出未重做。新增教师调用=0、目标 LoRA 更新=0（本头阶段）。训练允许自身源数值标签；预测生成用 runtime_guard 禁止 raw GT/labels.npy/GT_counts.npy，预测先 SHA 封存，再由独立进程读取官方 GT。Sports 主评测 v4 全466序列，另存相同 v4 内容限制原418ID的敏感性。

## 5. 保存、局限与后续条件

新结果物理目录 /data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v3，通过原 results/runs/presence_active_20261102 链接访问。checkpoint/预测路径 SHA 见 artifacts.json、selection.json；大文件不入 Git。原默认 A、lr1e-5、Sports 标签修复结果及主 REPORT.md 均不覆写。该头保留冻结模型先验是明确设计，不能把接近原生称为新语义能力。即使通过同域，也须单独检查跨域是否有效；不能以 source loss、有限梯度或测试通过宣称全方法成功。

本次固定改造仍有同域失败：KITTI-V2, Dance。遵守预注册，未启动 B/C、未追加第二个新结构或目标阈值搜索；完整方法仍未完成。

完整新阶段日志、源选择曲线、官方详细表在 logs.tar.gz；清单及 SHA 见 logs.json。历史 experiment_logs.tar.gz 不改。模型/缓存与预测均不打包。

最终完整性检查见 validation.json：22项定向测试通过；六套预测覆盖各自全部视频/表达、每一行均与原MOT候选原文一致，checkpoint/预测封存一致，生成进程GT尝试0；原主报告/vendor/教师后端未改，本任务训练/推理/教师worker剩余0。完整性通过不改变科研门槛失败的结论。
