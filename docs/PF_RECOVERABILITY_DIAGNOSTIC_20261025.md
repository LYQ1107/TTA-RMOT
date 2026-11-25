# V2→Dance快速可恢复性诊断

仅TTA-RMOT；起点为公开提交`29f265cf6b2c8d82948043534e0044b1341d5899`。
本轮不重跑已完成六方法、不更新checkpoint、不调在线阈值、不引入外部教师。
所有新结果位于`results/diagnostics/pf_recoverability_20261025/`，与原六组结果分开。

状态：**全部诊断完成，已停止；没有启动第七种完整TTA。**
最直接结论是Identity的HOTA收益可由它改变的输出数量配合Frozen旧排序几乎全部重现，
不能归因于新学到的跨模态排序。原模型仍有有限查询/人物外观信息，但尚未找到独立可靠的无标签语义监督。

## Phase0：官方协议核查通过

原始`Refer-Dance.zip`中的852个表达/GT/索引文件与当前资产CRC及长度一致。
逐项核对原始`labels.json`、表达目标ID、GT行、全部25视频425表达seqmap及历史GT软链接。
212,541个GT观测的frame/ID完全一致，框仅有归一化量化误差；图像/GT/MOT为1-based，无帧偏移。
全部GT行mark=1/class=1，不是过滤误删；40个Query-Present/385个Query-Empty属于原发布协议。
这里的Empty指**官方评测标注为空**，不是额外人工核验所有画面后得到的绝对语义不存在标签。
官方来源：[iKUN](https://github.com/dyhBUPT/iKUN)及
[固定版本FlexHook数据说明](https://github.com/buptLwz/FlexHook/blob/bd1acc38634b28525d54dc6e0fcb38335f0029f9/datasets/DATASET.md)。

新审计器第一版误把`labels.json`的归一化左上角xywh当作中心xywh，造成40个假报警；
已修正解析并保留首次诊断/日志。没有修改原GT、原预测或历史指标。这不是数据协议错误。

## Phase1：Count-Matched Frozen（仅后验诊断）

每帧读取封存Identity输出数量k，再用Frozen原分数取top-k；预测构建进程禁止打开目标标注，
不做GPU前向，不优化阈值。完整425表达重新评测的仅是这个新诊断对照，不是重跑六方法。

| 方法 | HOTA | DetA | AssA | IDF1 | 预测行 |
|---|---:|---:|---:|---:|---:|
| Frozen | 8.2180 | 3.1937 | 21.1470 | 4.4414 | 99,704 |
| Count-Matched Frozen | 8.8206 | 4.0143 | 19.3820 | 5.4592 | 114,648 |
| Identity | 8.8179 | 4.0132 | 19.3760 | 5.4476 | 114,648 |

Count-Matched相对Frozen为+0.6026，Identity为+0.5999；同样k配合旧排序可重现约100.45%的
点估计增益。Identity对Count-Matched仅-0.0027，25视频配对95%区间[-0.027423,+0.012497]。
这是描述性重现比，**不是因果贡献百分比**；k来自实际Identity运行，不等于某个独立Frozen阈值。
不能把此方法冒充无需适应即可部署的系统，也不据此调Dance在线阈值。

两者选中行有114,350/114,648相同（99.7401%），仅296个活跃帧发生差异，
各交换298条观测；每帧数量始终完全相同。这进一步限制了将Identity收益解释成新语义排序的说法。

离线IoU≥0.5最大基数匹配：Count-Matched净恢复3,231个GT观测、净增11,713个FP；
Identity为+3,223/+11,721。恢复/新增行/匹配重分配后的FP净变化在JSON中分开保留。

| 完整425表达 | Frozen | Count-Matched | Identity |
|---|---:|---:|---:|
| DetRe | 4.5441 | 5.9371 | 5.9354 |
| 官方CLEAR TP | 10,328 | 13,552 | 13,543 |
| 官方CLEAR FP | 89,376 | 101,096 | 101,105 |
| 离线独立逐帧匹配TP | 10,337 | 13,568 | 13,560 |
| 离线独立逐帧匹配FP | 89,367 | 101,080 | 101,088 |
| 离线净GT恢复，相对Frozen | — | +3,231 | +3,223 |
| 离线FP净增，相对Frozen | — | +11,713 | +11,721 |

CLEAR包含官方时间一致性分配；逐帧最大基数匹配用于追踪行级变化，两者不混算。
Count-Matched新选中FP行12,428、移除FP行726；两者之差11,702不等于FP净增11,713，
另有原有行的匹配状态重分配。不能把全部FP净变化叫新增错误预测行。
Query-Empty FP从87,371增至97,792（与Identity相同），产生非空预测的Empty episode从154/385
变为164/385；数量扩张没有解决空查询。Count-Matched的Present GT recall为0.063837，Identity为0.063799。

数量匹配对Frozen的HOTA差95%视频区间[-0.080111,+2.421641]仍包含0。
冻结排序AUROC保持0.650204；Identity为0.642678。以上不是稳定语义改进证据。

## Phase2：全量冻结分数含有信号，但不能证明强表征恢复

25视频425表达的3,588,445个候选—表达分数全部分析；Source候选AUROC精确复现0.650203661。

固定预设统计量的episode存在性：

| 分数统计量 | AUROC | AUPRC |
|---|---:|---:|
| mean margin | 0.660260 | 0.162801 |
| p90 margin | 0.671688 | 0.168148 |
| max margin | 0.725779 | 0.199373 |
| mean frame-max margin | 0.674545 | 0.167241 |
| Frozen正分比例（固定原0门限） | 0.695065 | 0.170642 |
| 视频视觉分数均值对照 | 0.529805 | 0.099884 |
| 其它24视频估计的query分数偏置 | 0.556039 | 0.125905 |

随机AUPRC基准40/425=0.094118。全片max的25视频聚类95%区间：AUROC[0.626505,0.821203]，
AUPRC[0.143753,0.362548]。这些是封存整片离线统计，未选最终在线阈值；max不是因果可部署presence规则。
全片统计间比较也不能变成GT驱动的聚合器/门限选择。每视频、每表达（含无正例NA）均已输出。

同帧文本互换：正确query Top1命中0.614788，其它真实query平均0.573785，差0.041003，
25视频区间[0.014186,0.065544]。同时保留GT重叠/不重叠/非空不重叠控制，不把其它表达都假定为负。
在26,352个正负混合候选帧，正确文本AUROC仅0.559705，视觉query平均偏置0.504819，
互换文本0.500284。正确文本相对视觉差值区间[0.001207,0.109149]，相对互换区间[-0.000322,0.120828]。
随机候选Top1期望0.578299，避免用多目标稠密帧夸大命中率。

更严格的非空且GT候选不相交文本互换，使用**同一23,008个合格帧**：正确文本Top1为0.321366，
互换文本0.236996，随机候选0.267062；正确减互换为+0.084370，25视频区间[0.045686,0.145339]。
不能用全体帧的正确Top1=0.614788与该困难子集的互换0.236996直接相减。
2/2,000个重采样没有合格困难帧，记为不可计算；有效重采样1,998个，不填0。

视频内源分数方差分解：candidate共同项14.4121%、query边际项65.3135%、交互项20.2744%。
边际query项本身也可能包含场景—查询存在性信息，不能全部叫错误偏置；交互方差也不等于语义信息。
双中心交互候选AUROC0.516358，说明仅删除边际项不能直接宣称恢复语义。
“视觉偏置”对照是17真实query分数的行均值，不冒充另外训练的纯视觉网络。

## Phase3：冻结内部特征，真实提取与诊断已完成

前两阶段说明输出数量能重现Identity收益、source有弱/中等查询信号，但**logit不能确定内部PCD/视觉
信息是否仍可解码**。现有全量封存只有margin，没有可复用PCD数组，故允许进行条件小规模提取。
在读取新特征的GT之前固定：每视频active-MOT帧索引10%/50%/90%处各一帧，25×3=75帧，
全部17真实表达，另加固定person通用PCD以避免服饰文字直接注入探针答案。
先1视频1帧功能检查，再提取完整75帧；严格冻结V2、原框/ID/因果历史，0更新。

提取当前人物ROI、上下半身ROI、浅层视觉、全图背景、geometry、17-query PCD、generic-person PCD
与语言特征。不计算未对齐视觉与RoBERTa空间的跨模态余弦来假装检索。
先做排除同视频的无训练服饰检索，GT只定义评测集合/相关性，并与随机gallery、背景、geometry对照。
固定计划SHA256为`0e52fa3ca971e7a5f6a7f770188184de962c0d3b53565ed50deb9c80002e51cc`。
真实GPU4通过显存/进程审计后共享使用，未停止他人进程，仅用这一张卡。先做1帧/all17 smoke，
再提取全75帧并复用首帧；覆盖643条MOT观测、全部25视频/17类表达。
V2严格加载，person token5970；原参数版本不变、0可训练参数/0更新/0目标标注打开尝试，
抽样源margin与封存结果最大误差0、符号差异0。不是重新运行六方法。

### 无训练、排除同视频的检索

603条GT匹配且有服饰表达标注的观测用于评测相关性。相似度不使用GT、不拟合权重；
gallery排除整个同视频，至少共享一条实际官方服饰表达为相关，距离并列平均计分。
GT只定义离线相关性与已标注gallery，不作为部署时可获得的gallery标签。

| 冻结特征 | 同服饰表达Top1 | 相对随机差的25视频95%区间 |
|---|---:|---:|
| 随机跨视频gallery | 0.327537 | — |
| 当前人物视觉ROI | 0.427861 | [0.028444,0.171049] |
| 上下半身视觉ROI | 0.417910 | [0.016231,0.164977] |
| 浅层人物ROI | 0.452736 | [0.003169,0.255636] |
| generic-person PCD | 0.509121 | [0.084046,0.291589] |
| 全图背景 | 0.385430 | [-0.044111,0.172089] |
| geometry | 0.359867 | [-0.009458,0.072184] |

PCD减背景+0.123691，但区间[-0.035723,0.302250]跨0，因此单靠此检索不能排除场景/舞者混淆。
仅5/17类表达在抽样中跨至少两个视频，另12类只出现于一个视频；不能宣称所有细粒度服饰均可泛化。
检索区间是固定gallery条件下对query所属视频聚类重采样，不是对gallery重新抽样，更不是603个独立人。

### 必要的GT监督诊断线性探针（绝非TTA）

由于上述背景混淆未排除，执行预先规定的C=1、balanced logistic regression、seed0、
liblinear/max_iter1000、5折GroupKFold(video)。每折StandardScaler仅拟合训练视频；
四种表征共20次CPU拟合，训练/测试视频无交集，没有调参或保存/导出权重。
首次缺少sklearn，在拟合前失败；保留原始traceback，单独安装固定版本CPU依赖，不替换原推理环境。
成功运行133.06秒，所有拟合10–15次迭代收敛，无ConvergenceWarning。

| 原始OOF分数/对照 | pooled AUROC | pooled AUPRC | 同帧同query混合候选macro AUROC |
|---|---:|---:|---:|
| 原Frozen head，同一固定样本 | 0.616582 | 0.085485 | **0.616484** |
| Query-conditioned PCD线性探针 | 0.802146 | 0.241785 | **0.586402** |
| 仅文本探针 | 0.744914 | 0.327054 | 0.500805（数值并列误差） |
| 背景+文本探针 | 0.752288 | 0.413481 | 0.499195（数值并列误差） |
| geometry+文本探针 | 0.749398 | 0.349791 | 0.487031 |

这里10931个candidate-query样本、608个正例；同帧候选鉴别仅有**69个正负混合frame-query组，来自11视频**。
不能把69组或10931行当独立样本。文本/背景同帧同query数学上为常量；原始读出差异仅
8.88e-16/1.78e-15，造成严格排序对并列打破的轻微偏差，真实候选鉴别为chance。
保留原OOF数据，没有按GT修正读出。完整逐组分数及此数值说明见`features/probe_qualification.json`。

PCD相对背景+文本的同帧AUROC差+0.087208，视频区间[0.028432,0.166747]，
相对geometry+文本+0.099371，[0.041904,0.188195]：支持内部存在少量人物/表达相关可读出信息。
但它**没有超过原Frozen head**：同一69组差-0.030081，区间[-0.089035,0.026710]。
pooled AUROC0.802不是新TTA成绩，包含表达/场景先验及不同fold读出的跨fold校准影响；
仅文本/背景的高pooled AP也说明不能仅凭总体AUC宣称服饰语义恢复。
所有区间对25视频重采样、条件于已拟合OOF读出，不是每次bootstrap重训的独立确认试验。

## 五项最终判断与停止决定

1. **Identity的+0.5999有多少由数量解释？** 其逐帧k配合Frozen原排序重现+0.6026，
   描述性增益比100.45%，选中行99.7401%一致。没有可分辨的新增语义排序收益；不是因果百分比，
   更不是已发现一个无需Identity即可部署的门限。
2. **原模型有presence信号吗？** 有，但有限。预设整片max AUROC0.725779/AP0.199373，
   优于随机AP0.094118和视频视觉偏置。尚不是可靠的在线presence判断，不能据此用GT选择聚合器/阈值。
3. **内部是否仍有服饰语义？** 不能说完全丢失。无训练跨视频检索及同帧PCD探针均支持有限的
   外观/服饰相关信息，但背景混淆、稀有类跨视频支持不足；探针未超过原head，未证明存在强而可无监督恢复的细粒度语义。
4. **是否值得继续大规模Source-Free TTA？** 当前证据不足以重启完整Rank/Identity/Hybrid或新增十万步LoRA。
   主要可复现效应是输出数量/校准；空查询误检和弱排序共同限制。最多保留小规模、固定预算的因果presence机制假设，
   必须先找到新证据再申请正式实验，不能将本GT诊断数据当新的盲测。
5. **最有证据的独立无标签学习信号是什么？** 本轮**没有确认独立可靠的语义正负监督**。
   跨query反事实响应和冻结人物特征的因果历史稳定性可用于一致性检查、UNKNOWN/跳过更新，
   但稳定响应仍可能稳定地错，物理同ID也不是语言匹配标签。它们不是已验证的语义学习方向。
   不把top1、强制存在一个目标或过去硬负标签循环当新监督；GT可读出不等于无标签可识别。

因此保留诊断与有限研究问题，**本轮到此停止，不启动任何新的完整TTA、源训练或阈值搜索**。

## 成本、复查与恢复

- Count-Matched预测构建CPU14.27秒、0新GPU前向；依赖已有Identity数量，其端到端成本不能忽略原Identity运行。
- 全量score分析一次约34.53秒，另一次CPU复算增加困难互换的同子集控制；没有重复全量GPU推理。
- GPU smoke5.93秒，75帧提取57.63秒、峰值Torch allocated2,029,819,904bytes（1.89GiB）；
  两次进程wall合计63.56秒，约0.01765 GPU-process-hours，不是NVML积分利用率。仅物理GPU4。
- GT诊断探针CPU133.06秒；0源模型/LoRA更新，20个诊断拟合对象未保存。
- 67项测试通过（11.62秒）；compileall与两个原运行脚本bash -n通过。
  一个anyio pytest重写提示，不影响结果。原六组公开结果相对起点git diff为空。

主要代码：`scripts/diag_pf_protocol.py`、`diag_pf_countmatched.py`、`diag_pf_query_information.py`、
`diag_pf_frozen_features.py`；`seal_pf_recoverability.py`仅整理已完成结果，不拟合/推理。

按独立进程复现（不要为复查重新运行六方法）：

```bash
cd /data2/user/FlexHook-TTA
export PYTHONPATH=/data2/user/TTA-RMOT-assets/flexhook/diagnostic-pydeps:/data2/user/FlexHook-TTA/compat:/data2/user/FlexHook-TTA:/data2/user/TTA-RMOT-assets/flexhook/pydeps
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
PY=/home/szgx/anaconda3/envs/dfot/bin/python
"$PY" scripts/diag_pf_protocol.py
"$PY" scripts/diag_pf_countmatched.py --stage predict
"$PY" scripts/diag_pf_countmatched.py --stage evaluate
"$PY" scripts/diag_pf_query_information.py
# 现有75个cache有效时，只做离线评价；不必再启动GPU提取。
"$PY" scripts/diag_pf_frozen_features.py --stage evaluate
# probe仅在明确需要重新做GT诊断时运行，绝不接入在线TTA。
# "$PY" scripts/diag_pf_frozen_features.py --stage probe
"$PY" scripts/seal_pf_recoverability.py
```

恢复根目录：`/data2/user/FlexHook-TTA/results/diagnostics/pf_recoverability_20261025/`。
`countmatched/COUNT_MATCHED_FROZEN/`保留425个原子完成收据、预测、源margin链接，
`features/cache/`保留75包冻结数组及逐包来源校验（不提交大数组）。
`countmatched/official/`为完整官方TrackEval；`query_information/`含全部425表达/25视频/17表达类型表，
`features/probe_oof_scores.csv`保留全部OOF分数，`probe_mixed_frame_groups.csv`保留69组同帧对照。
`delivery_manifest.json`逐项记录覆盖、原文件不变与压缩包内容。
`archives/countmatched_425_predictions.tar.gz`包含全部425预测（无GT/图片/margin/权重）；
`archives/diagnostic_complete_logs.tar.gz`包含20份完整文本日志，包括初次失败与严格加载日志，逐文件核验。
打包器自身日志`seal_stdout.txt`单独保留，避免在运行中把自己打入日志包。
静态科研图：`figures/recoverability_summary.png`和PDF，已实际渲染检查。

实际GPU特征源码local`c6250299e5e8ca2550a6032f05204779aece8d99`对应
[公开相同tree c8b679ed](https://github.com/LYQ1107/TTA-RMOT/commit/c8b679ed70bdd8ae1380c392bc458847775092a0)。
CPU探针源码local`3bb8b03450a80d99255e83f7d3aafcd09d159ee7`对应
[公开相同tree 07098b09](https://github.com/LYQ1107/TTA-RMOT/commit/07098b09266adec88c2a37ef1c8ea83300d79c30)。
Git对象发布保留原研究分支与历史，不force-push；最终报告、分数、代码与日志使用同分支后续提交发布。
