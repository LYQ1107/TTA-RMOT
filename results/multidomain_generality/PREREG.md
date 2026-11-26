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

---

## 修订记录 v2：2026-11-01 用户补充授权（原始预注册正文保持不变）

以下为用户补充指令原文。本修订发布并验证远端提交后，才允许本轮下载和推理。

# 补充指令：授权下载缺失资产，并恢复 PREREG 的完整范围

## 0. 授权与总体安排
我授权你下载本任务所需的公开资产：LaMOT-best 权重、LaMOT 指代表达与标注、McByte-SportsMOT tracker 输出。下载完成并通过核验后，按 PREREG（98e7f84e）的完整范围执行 C1–C6，不再缩小范围。
为节省时间，下载期间可以先并行做不依赖这些资产的部分：
- C1 V2-best → Dance：复用已有结果，核对 SHA；
- C2 Dance-best → KITTI-V2：新推理；
- 补齐 V2-best 在 KITTI-V2 上的完整候选分数（见第 4 节）。
仍然不训练、不调用 MLLM，最多同时用 2 张 GPU。先把本补充指令追加到 PREREG 的修订记录并 push，再开始下载和推理。

## 1. 需要下载的内容与来源（按优先级）
1. FlexHook 官方发布（README 第 4 步与 datasets/DATASET.md 指向的百度网盘 FlexHook_best：https://pan.baidu.com/s/1L-43y9SFDKmgl3dJNRlvNA ，提取码 d3qj）：
   - SOTA_ckpts/LaMOT-best.pth
   - tracker_outputs/McByte-SportsMOT（以及该网盘中 LaMOT 推理所需的其他 tracker 输出，只下载 SportsMOT 相关的即可）
   - datasets/ 中 LaMOT 相关部分（LaMOT-main 的目录结构与 FlexHook 整理好的 JSON 标注文件）
2. LaMOT 官方仓库 https://github.com/Nathan-Li123/LaMOT ：按其 README 获取指代表达与标注。若 FlexHook 网盘中的 datasets/LaMOT-main 不完整，用这里的官方标注补齐；两者都有时，以 FlexHook 网盘版本为准（与官方推理代码一致），并记录两者是否一致。
3. SportsMOT 图像：服务器上已有 val 的 45 个视频，不重新下载，只按 DATASET.md 建软链接。

## 2. 下载方式与安全要求
- 百度网盘：可以在本任务的独立环境里安装命令行工具（例如 BaiduPCS-Go 或 bypy），不要改动共享环境。如果需要登录凭据而你没有，不要猜测或索取账号密码，直接停止百度网盘这条路径，把需要我手动下载的文件（精确路径与文件名）列出来报告给我。
- GitHub / 其他公开链接：直接下载。
- 下载前检查目标磁盘的剩余空间，空间不足就停止并报告。
- 所有下载内容放在新目录 /data2/user/TTA-RMOT-assets/flexhook/lamot_20261101/ 下，按来源分子目录；不覆盖任何已有文件，不修改 vendor/FlexHook 代码。
- 压缩包在各自独立的空目录中解压；不执行下载内容中的任何脚本或代码。
- 每个文件记录：来源 URL、下载时间、大小、SHA256，写入 results/multidomain_generality/downloads.json。
- 记录 LaMOT 和 FlexHook 资产的许可证信息。

## 3. 下载后的核验（全部通过才能进入 SportsMOT 与 LaMOT 相关推理）
1. LaMOT-best.pth 能以 strict=True 加载进 FlexHook 的 rope-swin-tiny + roberta 结构。
2. McByte-SportsMOT 的 tracker 输出覆盖 seqmaps/SportsMOT.txt 中的全部 45 个视频，并记录候选观测数。
3. LaMOT 标注覆盖 SportsMOT.txt 的全部 466 条序列（视频+表达），文件名与 seqmap 一一对应，并列出不匹配项。
4. 软链接建好后，用 LaMOT-best 在 SportsMOT 上对 1 个视频做冒烟推理，确认 dataloader、图像路径、tracker 输出、表达都能正常读取。
5. 在论文正文、补充材料（若网盘中有）、仓库中查找 LaMOT-best 在 SportsMOT 子集上的官方数字：找到则必须复现到 ±0.1；找不到则按 PREREG 的规定，严格按 infer.sh 的官方命令运行，并注明“无官方子集数字可核对”。
任何一项不通过，就停止 SportsMOT 与 LaMOT 相关部分并报告原因；C1、C2 和 KITTI 补齐可以继续完成。
如果 McByte-SportsMOT 的官方 tracker 输出无论如何都拿不到，不要自己运行 tracker 生成候选，停止并报告，由我决定。

## 4. KITTI-V2 完整候选的补齐
- V2-best 在 KITTI-V2 上，按 diag_attrib v2 对 Dance 的做法补齐 69,854 个末尾观测的分数，得到 2,425,382 个分数的完整候选库并封存。
- 自检：在历史覆盖（原 2,355,528 个）上必须逐行复现原有分数，并复现 HOTA 42.526。
- 记录完整覆盖下 V2-best 在 KITTI-V2 上的 HOTA（作为该目标的同域参照）。
- 所有在 KITTI-V2 上推理的源模型（Dance-best、LaMOT-best）都同时导出完整覆盖和历史覆盖的结果；诊断与判定使用完整覆盖。

## 5. 之后的执行
全部资产到位后，按 PREREG 的 P1、P2、判定规则和交付要求完成 C1–C6。466 条以实际为准，在报告中记录与 PREREG 中写的 465 条的差异及原因（文件末行没有换行）。
其余约束全部不变。完成后 push，回复提交号和报告链接，然后停止。

---

## 修订记录 v3：2026-11-01 官方 SportsMOT 评测模板缺项的补建授权

下载并解压官方 FlexHook `gt_templates/SportsMOT` 后，只有418条GT文件覆盖466条seqmap，缺48条且无额外条目。检查包清单和官方目录，不将不存在的模板视为空目标，不删减seqmap。

用户明确答复：“允许按官方原始标注补齐，并严格核对已有模板”。执行规则固定如下：

1. 本修订先commit并push。原v1正文和v2补充授权保持不变，全部C1–C6和原判定不变。
2. 推理进程仍只读取纯文本、官方预测MOT和图像；所有目标GT/原始指代targets只由独立、预测完整封存后的评测准备进程读取。
3. 仅使用LaMOT官方原始指代标注和SportsMOT官方GT框生成表达对应的GT。预测候选依旧只用官方McByte，不得用GT框/ID替代。
4. 先按相同生成规则逐行数值复现已有418条官方模板（保留全部字段与顺序，另记原文件SHA）；全部通过才将同规则生成的48个缺项加入独立补齐目录。原始418文件不覆盖，466条完整seqmap不变。
5. 任一已有模板复现失败、原始标注/框缺失或视频/帧/ID对不齐，就停止SportsMOT评测并报告，不放宽门槛、不静默补空GT或删序列。
6. 报告单列418官方原文件、48补建文件的路径/SHA、逐行核对结果、GT隔离和成本；补建GT不得反馈模型或选择任何超参。

---

## 修订记录 v4：2026-11-01 用户授权统一按官方原始标注重建466条SportsMOT GT

v3已实际执行并按规则停止：418条中410条逐行全字段/顺序一致，8条不一致，涉及5个视频。8个原模板共10148行，原始LaMOT targets与SportsMOT框对应69045行；原模板全部为后者的严格子集，共少58897行，共有(frame,id)的框和字段没有变化。详见sports_GT_completion.json、sports_GT_mismatch_analysis.json/CSV和SPORTS_GT_CONFLICT_REPORT.md。尚未生成缺48条GT，未做任何SportsMOT正式评测。

用户已明确选择：“另行预注册：用官方原始标注统一重建全部 466 条 GT，保留旧模板不覆盖”。本修订先commit并push、回读验证，之后才执行新GT重建和评测。

1. v1–v3正文和v3失败JSON/日志原样保留。不得将410/418改写为418/418，不将v4结果宣称为原官方模板精确复现。
2. 使用已经核验的LaMOT官方原始逐帧targets与SportsMOT官方原始GT框，对全部466个(video,expression)统一生成GT，不混用410个旧模板与8个修补模板。原始418文件不修改、不覆盖、不删除。
3. 唯一生成规则：按原始SportsMOT gt.txt的行顺序，保留(frame,id)在相应官方targets中的行；全部数值字段、帧号、ID和框原样保留，不插值、补框、改可见性、设新过滤规则或做帧偏移。targets引用的框若缺失则停止并报告。空GT只能由实际原始targets导出，不能因旧文件缺失而默认置空。
4. 输出至独立目录results/runs/multidomain_generality_20261101/eval_assets/SportsMOT_v4_original_annotations/gt_template。新记录为sports_GT_reconstruction_v4.json，逐条保留输入/输出路径、SHA和行数，区分410条原模板一致、8条原模板不一致、48条原缺项。v3记录继续使用sports_GT_completion.json，不覆盖。
5. 复核原418模板的SHA不变、410条仍精确一致、8条差异集合与v3完全相同，单列8条变化及58897行差异。任何额外输入变化或新不一致停止并报告。
6. 所有模型分数和GT-free预测已在03cf5de30e16b6e396754112fcc3208106e22d55及此前提交封存。直接复用，不重跑GPU、不改预测。GT仅由独立CPU重建/oracle/官方TrackEval进程读取，不反馈模型、排序、数量或采样。
7. C3/C4的全部7个固定控制、同域LaMOT参照、完整/历史MOT覆盖、结构统计和45视频bootstrap统一使用这套v4 GT。标注为“PREREG_V4_ORIGINAL_ANNOTATION_GT”，不是原418模板协议的官方数值复现；不把两套GT口径的指标混用。
8. C1/C2/C5/C6已完成结果不改、不重跑。三个checkpoint、固定tracker、FP32/tokenizer、候选库、全部六组合、ρ/gap判定、bootstrap 2000次/seed0及其余约束不变。
9. 本次授权在任何SportsMOT HOTA产生之前作出，只解决已公开的GT来源冲突；不得根据之后的HOTA决定使用新旧GT、选择序列、删掉8条或调整阈值。
10. 仍然不训练、不微调、不调用MLLM。最终报告明确v3失败和v4授权，发布全部42个控制、日志、成本及SHA后停止。
