# FlexHook V2→Refer-Dance：25Val同集适应完整诊断

日期：2026-10-26。全部25视频、425表达；目标适应和评测是同一25Val。
**TRANSDUCTIVE_VAL_DIAGNOSTIC，不是40Train→25Val UDA、不是独立测试泛化、不是Source-Free Online TTA。**

本轮完成用户“先25Val看效果”的阶段。原40Train图像/表达已齐，但同配置Train MOT仍缺，原正式Train→Val目标尚未完成。
没有用目标GT选择阈值、checkpoint或训练参数；源开发集固定64窗口CE选模型，全部方法包括step0。

## 完整官方结果

| 方法 | HOTA | ΔFrozen | DetA | AssA | DetRe | IDF1 | IDSW | 输出行 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 8.2180 | +0.0000 | 3.1937 | 21.1470 | 4.5441 | 4.4414 | 269 | 99704 |
| SOURCE_REHEARSAL | 7.4352 | -0.7828 | 1.9669 | 28.1110 | 2.3022 | 2.8658 | 119 | 41211 |
| MEAN_TEACHER_UDA | 8.2494 | +0.0314 | 3.3240 | 20.4740 | 4.7318 | 4.5823 | 300 | 100268 |
| WEAK_MIL_UDA | 6.9281 | -1.2899 | 1.4777 | 32.4910 | 1.6297 | 2.2292 | 56 | 25389 |
| COMBINED_UDA | 7.7293 | -0.4887 | 2.4336 | 24.5510 | 3.1200 | 3.4910 | 175 | 66694 |

Frozen仍为完整211085条候选的8.2180；旧8.2060遗漏692末观测，仅保留历史，不混用协议。
SOURCE_REHEARSAL不使用目标loss；B与A源loss相同。C/D另外加入source-presence项，其相对A差异不能全部归于目标约束；本轮未增加source-presence-only消融。
固定前端是Dance专用官方DETR+NeuralSORT；本研究适应第二阶段，不宣称整个检测/跟踪系统无目标域监督。

## 数量匹配与语言选择

| 方法 | 原模型AUROC→适应AUROC | 恢复GT / 丢失GT | 新增未匹配预测行 | FP净变化 | Count-Matched HOTA | 适应−Count-Matched |
|---|---:|---:|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 0.650204 → 0.641311 | 6 / 5094 | 158 | -53405 | 7.4207 | +0.0145 |
| MEAN_TEACHER_UDA | 0.650204 → 0.583331 | 1164 / 711 | 10902 | +111 | 8.2479 | +0.0015 |
| WEAK_MIL_UDA | 0.650204 → 0.492772 | 0 / 6618 | 212 | -67697 | 6.9005 | +0.0276 |
| COMBINED_UDA | 0.650204 → 0.622742 | 5 / 3250 | 87 | -29765 | 7.7218 | +0.0075 |

Count-Matched按实际适应模型逐帧输出数量k，在Frozen中取top-k，不使用GT构造预测。
这是依赖适应结果的POST-HOC DIAGNOSTIC，不可独立部署；不能把HOTA差值比例称为因果贡献百分比。
GT恢复/FP行是离线IoU≥0.5最大匹配诊断；新增未匹配预测行与已有行匹配重分配造成的FP净变化不同，官方CLEAR/HOTA仍为主。
Recall@1/3/5见`results/uda/candidate_ranking.csv`；不是只看loss或非零梯度就称语义恢复。

## Query-Present / Query-Empty

官方协议40条有GT、385条空GT；仅用于离线分组，未作为训练先验。全部425条均进入主结果。

| 方法 | Present召回 | Present FP | Empty FP | Empty非空输出率 | 存在性AUROC / AP |
|---|---:|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 2.4696% | 338 | 35624 | 28.8312% | 0.681818 / 0.169157 |
| MEAN_TEACHER_UDA | 5.0767% | 2203 | 87275 | 42.8571% | 0.639221 / 0.158306 |
| WEAK_MIL_UDA | 1.7498% | 162 | 21508 | 22.0779% | 0.574156 / 0.133621 |
| COMBINED_UDA | 3.3368% | 1240 | 58362 | 33.2468% | 0.668117 / 0.166926 |

存在性统计固定使用完整episode的平均逐帧max margin，属于离线诊断，不能冒充因果在线presence规则；没有拟合GT阈值。
同一统计量的Frozen AUROC/AP为0.674545/0.167241。
历史Frozen 0.725779/0.199373使用整段max margin，不与平均逐帧max混比。
Query-Empty无GT，单独HOTA不用于声称改善；重点检查FP与非空率。

## 视频级不确定性

成对bootstrap以25个完整视频为聚类，2000次、bootstrap seed0，使用官方TP加权HOTA充分统计；
不把425表达或几十万相关帧当独立样本。该区间仍是同Val描述性不确定性，不是独立盲测显著性或训练seed不确定性。

| 比较 | ΔHOTA | 25视频95%区间 | 提升/下降/持平视频 |
|---|---:|---:|---:|
| SOURCE_REHEARSAL_vs_FROZEN | -0.782871 | [-3.287616, +0.026029] | 1 / 18 / 6 |
| MEAN_TEACHER_UDA_vs_FROZEN | +0.031417 | [-0.205174, +0.416379] | 12 / 7 / 6 |
| WEAK_MIL_UDA_vs_FROZEN | -1.289911 | [-4.125853, -0.290254] | 1 / 18 / 6 |
| COMBINED_UDA_vs_FROZEN | -0.488729 | [-1.896566, +0.059725] | 0 / 19 / 6 |
| MEAN_TEACHER_UDA_vs_SOURCE_REHEARSAL | +0.814288 | [-0.002211, +3.612036] | 18 / 1 / 6 |
| WEAK_MIL_UDA_vs_SOURCE_REHEARSAL | -0.507040 | [-1.684956, -0.189345] | 1 / 15 / 9 |
| COMBINED_UDA_vs_SOURCE_REHEARSAL | +0.294142 | [-0.050709, +1.558034] | 11 / 5 / 9 |
| COMBINED_UDA_vs_MEAN_TEACHER_UDA | -0.520146 | [-2.154552, +0.012444] | 1 / 18 / 6 |
| COMBINED_UDA_vs_WEAK_MIL_UDA | +0.801182 | [+0.253409, +2.396603] | 15 / 1 / 9 |
| SOURCE_REHEARSAL_vs_SOURCE_REHEARSAL_COUNT_MATCHED_FROZEN | +0.014441 | [-0.011435, +0.028944] | 3 / 5 / 17 |
| MEAN_TEACHER_UDA_vs_MEAN_TEACHER_UDA_COUNT_MATCHED_FROZEN | +0.001580 | [-0.010013, +0.019832] | 5 / 5 / 15 |
| WEAK_MIL_UDA_vs_WEAK_MIL_UDA_COUNT_MATCHED_FROZEN | +0.027612 | [-0.003864, +0.044700] | 4 / 1 / 20 |
| COMBINED_UDA_vs_COMBINED_UDA_COUNT_MATCHED_FROZEN | +0.007503 | [-0.011204, +0.027003] | 6 / 1 / 18 |

逐视频及移除一个视频后的影响见`results/uda/full/summary.json`；HOTA非线性，视频影响不能机械相加。

## 成本与模型选择

| 方法 | source选中step | source-val CE | 更新数 | 训练/推理 GPU-process-h | 完成进程训练峰值GiB | 推理query-frame FPS |
|---|---:|---:|---:|---:|---:|---:|
| SOURCE_REHEARSAL | 7750 | 0.96315369 | 8973 | 0.2553 / 0.8271 | 8.064 | 145.631 |
| MEAN_TEACHER_UDA | 8973 | 0.93873986 | 8973 | 0.8076 / 0.4968 | 7.937 | 242.455 |
| WEAK_MIL_UDA | 8973 | 0.97250435 | 8973 | 0.6010 / 1.0378 | 7.282 | 116.070 |
| COMBINED_UDA | 8000 | 0.95992477 | 8973 | 1.2001 / 0.4922 | 8.230 | 244.736 |

每组8973更新、24576个LoRA参数，rank4/最后两层PCD Q/V/lr5e-5/seed17，完整源17视频与目标25视频查询覆盖。
覆盖指视频/表达/源轨迹，不是逐帧优化：源393个不同anchor；目标每视频3个预定anchor，共75个不同anchor、1275个query窗口，重复配对源周期；每个anchor还使用过去4次轨迹观测。完整评测覆盖全部活动帧。
实际训练scheduler wall 4736.92s；推理scheduler跨度 5533.45s；
重叠流水线实际wall 7319.48s。训练与先行评测交错，不能将二者wall简单相加。
GPU-process-hours是共享卡worker elapsed的和，不是显卡利用率仪表测量；恢复期曾短暂同卡，不冒充独占四卡。
训练峰值指成功/恢复进程记录的allocated峰值；早期失败进程未单独记录的峰值不伪造。失败尾段额外耗时按最后日志恢复记录保留下界。
Frozen本轮是CPU精确重放历史GPU预测，不能把6秒CPU重放当独立GPU RMOT系统速度；Count-Matched总成本包含父方法。

## 机制与失败检查

四组均真实训练、更新LoRA，非骨干完整重训。A是源监督；B为EMA/强弱特征一致性和保守伪标签；
C为源presence及合法文本等价/互斥关系；D再组合源锚点与满足语义条件的短时记忆。
**C/D的目标MIL/presence没有独立正负证据，目标PRESENT/ABSENT覆盖均为0；UNKNOWN跳过，不能宣称已经复现有效目标MIL。**
含dancing+服饰的整句为MIXED，不能硬把整句语言匹配跨所有ID帧固定。纯身份一致性不是服饰匹配真值。
没有外部人物语义教师，也没有原版QCA复现。源码参考固定版本见`UDA_REFERENCE_IMPLEMENTATION_MAP.md`。
原C-Hook的N=1跨表达NaN常量不进入LoRA目标；原v1错误smoke已撤销。AMP溢出采用同样本RNG恢复和scale回退，未跳样本。
全部完成前不以目标GT审查伪标签；完成后的独立结果见`results/uda/full/pseudo_label_quality_offline.json`。

## 伪标签的独立离线质量

| 方法 | POS对应GT / POS总暴露 | POS精度 | Query-Empty中的POS | NEG误删GT暴露 | UNKNOWN中的GT暴露 |
|---|---:|---:|---:|---:|---:|
| MEAN_TEACHER_UDA | 84 / 958 | 8.7683% | 862 | 42 | 4140 |
| COMBINED_UDA | 84 / 878 | 9.5672% | 786 | 42 | 4140 |

这是全部历史训练证据的重复暴露，不是独立人物数或独立样本；候选按IoU≥0.5对应当前表达GT，非一对一跟踪precision。
Source/EMA一致和同一物理ID稳定并没有保证语言正确；本次POS主要落在空查询，不能把这种共识当作独立语义教师。
UNKNOWN中确有正确候选，但没有足够无标签依据把它们转为正例；继续保持UNKNOWN，不根据这份GT审计补标签。

## 冻结与适应模型的文本敏感性

| 方法 | 正确查询Top1命中 | 随机候选Top1 | 视觉偏置Top1 | 互换文本Top1 | 混合候选帧AUROC | 混合帧视觉AUROC |
|---|---:|---:|---:|---:|---:|---:|
| FROZEN | 0.614788 | 0.578299 | 0.576131 | 0.573785 | 0.559705 | 0.504819 |
| SOURCE_REHEARSAL | 0.610638 | 0.578299 | 0.577654 | 0.575540 | 0.553064 | 0.503855 |
| MEAN_TEACHER_UDA | 0.611928 | 0.578299 | 0.577279 | 0.575700 | 0.549962 | 0.504467 |
| WEAK_MIL_UDA | 0.594744 | 0.578299 | 0.580116 | 0.577719 | 0.509430 | 0.487124 |
| COMBINED_UDA | 0.609748 | 0.578299 | 0.577490 | 0.575070 | 0.552767 | 0.506916 |

Top1是至少命中一个正确候选的any-hit，不等于全部正确目标Recall@1；纯视觉控制为同一候选跨17真实查询的均值，互换文本在相同候选库中比较。
完整分析含42657个有GT帧-查询、其中26352个正负候选混合组，以及GT不重叠文本互换控制；不把这些相关组作为独立统计样本。
Teacher相对Frozen的混合帧(查询AUROC−视觉AUROC)变化为-0.009390，25视频描述性95%区间[-0.018439, -0.001956]。
所有适应模型的正确查询Top1和候选Recall@1/3/5点估计均低于Frozen；没有观察到一致的语言排序恢复。
本轮没有重做内部特征探针。此前冻结PCD诊断只证明有限可读出信息，不能把带GT探针当作无标签可恢复性或本轮TTA收益。

## 科研判断与停止边界

工程方面：四组各8973次真实有限更新、完整25/425只读推理、四组数量匹配和官方评测均完成；旧六组结果和失败现场保留。
科研方面：Mean Teacher相对Frozen仅+0.0314 HOTA，相对数量匹配仅+0.0015；视频区间包含0，尚无可靠正向语义结论。
Source Rehearsal和Weak的AssA提高伴随大幅召回损失；Combined没有超过Frozen或较好单模块，不能作为成功最终方法。
主因证据更支持查询存在性/校准与查询条件排序联合失效，而非单纯身份不稳定；目标伪正例不可靠，合法presence监督与静态语义时间监督覆盖为0。
这不证明所有UDA或所有服饰语义不可恢复：本轮只是一轮固定查询/轨迹覆盖、稀疏anchor预算，C/D还有source-presence混杂，且没有有效目标MIL正负监督。
目前没有足够证据支持直接加长同一EMA/MIL训练；如未来继续，需要先明确独立于源模型错误共识的语义依据，并另获授权，而不是依据Val GT调阈值或叠加loss。
不根据此Val结果再改阈值、轮数或挑checkpoint。本轮结束后不自动开始第五个完整方法、外部教师或40Train训练。
原始40Train→25Val正式UDA仍未完成：缺少同算法/同配置40Train预测MOT，且当前用户要求先不考虑Train。本阶段完成不等于原始完整goal完成。
完整准备/smoke/失败与恢复/资源调度/推理/官方评测文本日志归档于`results/uda/full/logs/runtime_logs.tar.gz`，逐文件SHA索引同目录。
原始checkpoint/分数/全部恢复路径见`results/uda/full/full_recovery_index.json`；完整425预测小压缩包不含GT/图像/模型权重。
