# PSR 完整首轮结果（2026-10-30）

范围：用户明确授权的25Val/425表达同集无标签开发研究；不是40Train→Val、未见视频泛化、盲测或Source-Free Online TTA。
8学生均完成8973步和25/425独立推理，26组官方TrackEval、双向数量控制、离线归因及归档已完成。
相同V2初始化、24576参数/rank4/lr5e-5、源val CE选择，首轮seed17，测试FP32 margin>0、官方Dance DETR+NeuralSORT框/ID不变。

| 方法 | HOTA | ΔFrozen | DetA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 8.2180 | 0 | 3.1937 | 21.1470 | 4.4414 | 269 |
| Source-only Rehearsal | 7.4352 | −0.7828 | 1.9669 | 28.1110 | 2.8658 | 119 |
| Naive paired | 7.4499 | −0.7681 | 3.7469 | 14.8340 | 4.6833 | 283 |
| PSR | 8.4204 | +0.2024 | 4.3726 | 16.2450 | 5.3330 | 338 |
| Uniform paired | 8.4347 | +0.2167 | 4.4324 | 16.0750 | 5.4020 | 349 |
| Confidence paired | 6.7173 | −1.5007 | 3.5247 | 12.8370 | 4.4218 | 334 |
| PSR Discovery | 7.8127 | −0.4053 | 3.9812 | 15.3590 | 4.9255 | 311 |
| PSR Geometry | 8.4218 | +0.2038 | 4.3769 | 16.2350 | 5.3363 | 340 |
| Naive Temporal | 7.9524 | −0.2656 | 4.0769 | 15.5360 | 5.0264 | 314 |
| Matched Mean Teacher | 8.6095 | +0.3915 | 3.8136 | 19.4380 | 5.1370 | 361 |

完整[实测表](../results/psr/full/TABLES.md)、[summary](../results/psr/full/summary.json)、
[425表达明细/完整预测归档/日志](../results/psr/full/artifact_index.json)。
旧Naive8.0327属于另一8381采样协议，旧Frozen8.206缺692末帧观测，均保留但不混入本表。

## 数量、语义与关联

PSR同数量Frozen=8.1951，PSR@Frozen数量=8.0336。前者残差+0.2253，反向低于Frozen0.1844，不能证明稳定的数量无关排序改善。
PSR−Naive=+0.9705，而Frozen@PSR数量−Frozen@Naive数量=8.1951−7.2621=+0.9330。
这不是因果贡献百分比；Count-Matched不能独立部署，其成本包含父模型。

PSR全候选AUROC0.650204→0.756113、AP0.093446→约0.1618；Naive也有约0.7554/0.1623。
对所有GT的one-to-one R@1/3/5：PSR约0.1552/0.4429/0.6750，Frozen约0.1568/0.4371/0.6781，并非一致改善。
未显式监督窗口之外也有全局AP/AUROC提高，但同视频轨迹历史可能已出现这些图像，不称跨视频泛化。
同帧正确文本top1命中（至少一个正确候选）只提高0.001125，视频聚类95%区间[−0.01877,+0.02104]；
正确文本相对文本互换优势变化−0.001070、区间跨0，不能用全局AUROC替代语言条件理解改善。
PSR AssA21.147→16.245、IDSW269→338，没有关联收益。
MT的+0.3915中同数量Frozen已有+0.3819，也不能作为语义方法上界。

## Query-Present / Empty 和恢复

40Present/385Empty是官方GT离线episode分组，不进入采样/教师/训练/阈值，不等于现实语义穷尽标签。
官方Present HOTA：Frozen9.7456、Naive7.6350、PSR8.8043、MT10.5431；Empty无GT，HOTA记N/A。
PSR整体微增同时，Present HOTA下降，Empty FP87371→20059。
离线IoU≥0.5：PSR恢复8773、丢失8025、净+748 GT帧观测；新增FP行18327、移除85724、FP净−67389。
旧行匹配重分配与新增行误检单独列出；这些不等于官方CLEAR/HOTA匹配计数。

25视频：PSR相对Frozen14升/8降/3不变，官方充分统计量聚类bootstrap2000次95%区间[−6.3931,+5.9808]。
PSR相对Naive+0.9705区间[+0.5259,+1.3056]，对Uniform−0.0144区间跨0。
未把几十万帧当作独立样本，训练seed不确定性另行报告。

## 成本、条件确认与恢复

PSR训练0.6645 GPUh+完整推理0.5616 GPUh=1.2261 GPUh，另继承8500教师判断（等价前向201.93秒）、Frozen分数和公共前端成本。
Naive为0.7096+0.5143 GPUh。共享GPU的进程时间不是独占吞吐率，FPS仅固定MOT之后的第二阶段筛选。
MT排队副本GPU调用0，其CPU锁等待没有重复记GPU成本；原入场失败保留并只修复失败项。

因PSR首轮同时超过Frozen/Naive，按预注册补seed0/42各Naive+PSR，仅学生seed改变，缓存/预算/超参/源val选择不变。
另补POST_HOC端点正例权重对照：263几何机会对应526已有端点项，额外0.25均值损失、新增弱标签0。
确认和权重控制尚在执行，不能把单seed点值称为可复现科研成功。

主运行：`/data2/user/FlexHook-TTA/results/runs/psr_val_20261029_v1`。
确认：`results/runs/psr_confirm_20261030_v1`；控制：`results/runs/psr_parent_weight_20261030_v1`。
所有训练目标GT打开尝试0、底座未变，optimizer/scaler/RNG恢复实测。权重/图像/完整raw教师缓存不上传，路径/SHA随收据公开。
