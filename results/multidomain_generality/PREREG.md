# 任务：三数据集多域普遍性验证——“排序能迁移、数量不能”是否普遍成立（不训练、不调用 MLLM）

## 0. 背景与要回答的问题
仓库 LYQ1107/TTA-RMOT，从提交 321ec548 出发，新建分支 research/multidomain-generality-20261101。
已知现象（KITTI-V2→Dance）：源模型 V2-best 在 Dance 上原生 HOTA 8.218；用 V2-best 的同帧排序、加上 Dance-best 逐帧逐表达的输出数量，得到 31.771，接近 Dance-best 自己的 32.139；同帧 R@1 为 0.157 对 0.155。
要回答：在其他源→目标组合上，跨域性能下降的主要原因是否也是“数量”（每条表达、每帧选几个），而不是“排序”（同一帧里谁更符合）？
本任务只用 FlexHook 官方 checkpoint 推理，加上 CPU 上的诊断。不训练、不微调、不调用任何 MLLM。

## 1. 数据集与组合（固定）
三个目标数据集：
- KITTI-V2：Refer-KITTI-V2 验证集，tracker 输出 Temp-NeuralSORT-kitti2，配置 configs/infer/kitti2.yaml。
- Dance：Refer-Dance Val（25 视频、425 表达），tracker 输出 DETR-NeuralSORT-Dance，配置 configs/infer/dance.yaml。
- SportsMOT：LaMOT 中的 SportsMOT 子集（视频为 SportsMOT 官方 val 划分，表达来自 LaMOT；FlexHook seqmaps/SportsMOT.txt 为 45 视频、465 表达），tracker 输出 McByte-SportsMOT，配置 configs/infer/mcbyte-sports-infer-tiny.yaml，推理入口与 FlexHook infer.sh 中 LaMOT-best + SportsMOT 的命令一致。
三个源模型：V2-best（refer-kitti-v2-best）、Dance-best（refer-dance-best）、LaMOT-best。
同域参照 I(T)：KITTI-V2 用 V2-best；Dance 用 Dance-best；SportsMOT 用 LaMOT-best。
六个跨域组合：
  C1 V2-best → Dance（已有结果，核对 SHA 后复用）
  C2 Dance-best → KITTI-V2（最关键）
  C3 V2-best → SportsMOT
  C4 Dance-best → SportsMOT
  C5 LaMOT-best → KITTI-V2
  C6 LaMOT-best → Dance
LaMOT 的 MOT17、VisDrone、TAO 子集不纳入本任务（前两者测试表达过少，TAO 以类别名为主且规模过大）。Refer-KITTI V1 与 V2 视频重叠，不纳入。

## 2. 硬约束
1. 只用官方 checkpoint，不训练、不微调、不修改权重。
2. 每个目标使用它自己的官方推理配置、dataloader（SportsMOT 用 mix 版本）、图像预处理和官方 tracker 输出；源模型只替换权重文件。同一目标上的所有模型共用同一批候选，框、ID、帧号都不改。
3. 若源 checkpoint 与目标配置结构不兼容（load_state_dict strict 报错、类别 token 或文本处理不一致等），停止该组合并报告原因，不做任何修补。
4. 推理设置沿用 diag_attrib v2：修正后的 tokenizer 策略、FP32、完整候选库（包含每条轨迹的末尾观测）。如果 mix dataloader 中存在同样的“跳过末尾观测”问题，按 v2 的方式在完整候选上补分数，并同时保留历史覆盖下的结果。每个目标在报告中写明类别 token 和 tokenizer 的处理方式。
5. GT 只在预测封存之后的评测和 oracle 进程中读取；推理进程不读 GT。
6. 不 kill、不抢占他人 GPU 进程，沿用 audit_gpus 准入，最多同时用 2 张卡。
7. 本文件的定义和判定规则全部现在固定，全部组合都上报，不挑选。
8. 先把本文件提交为 results/multidomain_generality/PREREG.md 并 push，然后再开始推理。

## 3. P0：只读盘点（完成后先 push 盘点结果，再进入 P1）
1. 三个 checkpoint 的路径和 SHA。缺失的列出来，停下来报告，不要自行下载。
2. 三个 tracker 输出的路径，以及每个目标的视频数、表达数、候选观测数、候选–表达组合数。
3. SportsMOT：val 视频图像、LaMOT 标注、McByte-SportsMOT tracker 输出、seqmap 是否齐全并按 FlexHook DATASET.md 正确链接。缺失的列出来，停下来报告，不要自行下载。
4. 用每个源模型在每个目标上对 1 个视频做冒烟推理（含 strict 加载检查），估算全量推理的 GPU 小时。任何一个组合估算超过 6 GPU 小时，停下来报告估算，等待决定。
5. 自检门槛（历史覆盖下）：
   - V2-best 在 KITTI-V2 上必须复现 HOTA 42.526；
   - Dance-best 在 Dance 上必须复现 32.134（完整覆盖 32.139，已有结果核对 SHA 即可）；
   - LaMOT-best 在 SportsMOT 上：FlexHook 只公布了四个子集合并后的 56.756，没有 SportsMOT 单独的数字。先在论文正文、补充材料、仓库中查找 SportsMOT 子集的官方数字：找到则必须复现到 ±0.1；找不到则不设数值门槛，但必须严格按 infer.sh 的官方命令运行，并在报告中写明“无官方子集数字可核对”。
   前两项复现不了就停止并报告。

## 4. P1：推理
对每个目标 T，运行三个源模型（包括同域模型），在完整候选上导出 margin 并封存。
已有结果直接复用，核对 SHA 后不重跑：V2-best→Dance、Dance-best→Dance、V2-best→KITTI-V2（2,355,528 行）。
新增推理：Dance-best→KITTI-V2、LaMOT-best→KITTI-V2、LaMOT-best→Dance、V2-best→SportsMOT、Dance-best→SportsMOT、LaMOT-best→SportsMOT。

## 5. P2：诊断（CPU）
对每个跨域组合 (S, T)：
- NATIVE：S 的原生 margin>0 输出。
- IN_DOMAIN：I(T) 的原生输出。
- R_S×K_I：S 的同帧排序 + I(T) 逐帧逐表达的数量（复用 v2 的 topk_ids、tie 规则和公共定义）。
- R_I×K_S：I(T) 的排序 + S 的数量。
- R_S×K_ORACLE、R_I×K_ORACLE：数量取官方过滤后的 GT 数（只在 oracle 进程中计算，标为【ORACLE，仅诊断】）。
- B0_GLOBAL：按 S 的分数做全局排序，取前 N 行，N = I(T) 原生输出的总行数。
- 同帧 GT R@1：S 和 I(T) 各算一个（口径与 v2 相同）。
每个配置报告：HOTA / DetA / AssA / IDF1 / IDSW / 行数 / Query-Empty 预测行数。
KITTI-V2 上再按表达的目标类别（car / pedestrian）分别报告 NATIVE、IN_DOMAIN、R_S×K_I 的 HOTA。

每个目标数据集的结构统计（ORACLE，只做描述）：
- 每条 MOT 轨迹被判为正例的表达数分布：0 / 1 / ≥2 的比例（口径与 count_structure 相同）；
- 空表达（全视频没有 GT）占全部表达的比例；
- 在有 GT 的 (视频, 表达) 中，有 GT 的帧占全部帧的比例（均值、p50）；
- 每个视频的表达数、表达是否在视频之间共享（如 Dance 的 17 句共用）。

不确定性：Dance（25 视频）和 SportsMOT（45 视频）上做视频聚类 bootstrap（2000 次、seed 0，复用已有脚本），比较 R_S×K_I − NATIVE 和 IN_DOMAIN − R_S×K_I。KITTI-V2 只有 4 个验证视频，只报告点估计，并在表中注明。

## 6. 预注册指标与判定（逐字执行，规则没覆盖的情况如实描述，不新增规则）
对每个跨域组合 (S, T)：
- gap = HOTA(IN_DOMAIN) − HOTA(NATIVE)。若 gap < 3，记为“跨域差距过小，不判定”。
- 数量可解释比例 ρ = [HOTA(R_S×K_I) − HOTA(NATIVE)] / gap。
- 排序差距 Δ_rank = HOTA(IN_DOMAIN) − HOTA(R_S×K_I)。
- 判定：ρ ≥ 0.7 记为“数量主导”；0.3 ≤ ρ < 0.7 记为“数量与排序共同作用”；ρ < 0.3 记为“排序主导”。
汇总：
- 在可判定的组合中，若“数量主导”的个数 ≥ 2/3，并且 C2（Dance-best→KITTI-V2）也是“数量主导”，记为“现象普遍成立”；
- 若“数量主导”的个数 < 1/3，记为“现象不普遍”；
- 其余情况记为“现象部分成立”，并列出成立与不成立的组合。

## 7. 交付
results/multidomain_generality/REPORT.md，内容依次为：
1. 判定结果汇总：3×3 的 ρ 矩阵（行为源、列为目标，对角线标为同域参照）及每格的判定标签；
2. P0 盘点、冒烟与成本估算、自检复现结果；
3. 每个跨域组合的诊断表（全部配置、全部指标）；
4. KITTI-V2 按类别拆分的结果；
5. 三个目标数据集的结构统计；
6. bootstrap；
7. 成本（每次推理的 GPU 小时）；
8. GT 审计、失败与修复记录。
机器文件：inventory.json、reproduction.json、diagnostics.csv、per_category.csv、structure_stats.csv、rho_matrix.csv、bootstrap.json，以及 margin / 预测的路径与 SHA 清单。大文件不进 git。
push 后回复提交号和报告链接。完成后停止，不训练，不调用 MLLM，不开始任何方法实验。
