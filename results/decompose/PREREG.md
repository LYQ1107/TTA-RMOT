# 任务：跨域退化分解的第二步——MLLM 能否修排序（A）+ 选择需要什么粒度（B）

## 0. 背景与要回答的问题
仓库 LYQ1107/TTA-RMOT，从提交 01add887 出发，新建分支 research/decompose-rank-select-20261102。
上一轮（multidomain_generality）六个跨域组合的结果：
- 修正“数量”（用同域模型逐帧逐句的数量）补回了 54%–98% 的跨域差距，是每一组最大的单一因素；
- 但目标为 KITTI-V2 和 SportsMOT 时，排序也明显变差：同帧 GT R@1，KITTI-V2 上源模型 0.197 / 0.277 对同域 0.424，SportsMOT 上 0.154 / 0.131 对同域 0.225；目标为 Dance 时排序几乎不变；
- 结构统计：Dance 有 90.6% 的空表达，且几乎“一人一句”；KITTI-V2 和 SportsMOT 没有空表达，大多数轨迹同时符合多句。
本轮要回答：
A. 冻结的 MLLM 在 KITTI-V2 和 SportsMOT 上，同帧排序是否明显优于跨域源模型、接近同域模型？单帧裁剪、多帧片段、整帧加框三种输入方式中，哪种对哪类表达更有效？存在性和数量判断是否可靠？
B. 在每个跨域组合上，“选择”需要什么粒度的数量信息（全局、每条表达、视频级每帧常数、逐帧）？不用 MLLM 的校准方法能恢复多少？
本轮不训练任何学生或教师，不做任何方法实验。A 和 B 可以并行。

## 1. 硬约束
1. 不训练、不微调。教师为冻结的 Qwen3-VL-2B，使用与此前相同的 revision；原有 T1（单帧裁剪 Yes/No）代码路径不改，新的输入方式写成新函数。
2. 教师调用进程和 GT-free 生成进程禁止打开任何目标域的 GT 和指代标签，沿用打开审计和 Landlock 白名单，次数必须为 0。GT 只在预测封存之后的独立评测和 oracle 进程中读取。
3. 本文件的全部定义、采样规则、常数、判定规则现在固定。全部结果都上报，不挑选，不根据结果改规则。
4. 预测和教师输出先封存（sha256），再做 GT 评测。
5. 复用上一轮已封存的 margin（六个组合的源模型和同域模型，完整覆盖），先核对 SHA。
6. 不 kill、不抢占他人 GPU 进程，沿用 audit_gpus 准入，最多同时用 2 张卡。
7. SportsMOT 的主结果使用上一轮的 v4 GT；同时对官方模板覆盖的 418 条序列单独报告一遍，作为敏感性分析。
8. 先把本文件提交为 results/decompose/PREREG.md 并 push，再开始任何计算。

## 2. 公共定义
- 目标域与组合沿用上一轮：C1 V2→Dance、C2 Dance→KITTI-V2、C3 V2→SportsMOT、C4 Dance→SportsMOT、C5 LaMOT→KITTI-V2、C6 LaMOT→Dance。同域参照：KITTI-V2 用 V2-best（完整覆盖 42.752），Dance 用 Dance-best（32.139），SportsMOT 用 LaMOT-best（69.924）。
- q = 一个评测序列（视频 + 表达），t = 帧，τ = MOT 轨迹，g(q,t) = 官方过滤后该帧的 GT 数（只在 oracle 进程中计算）。
- 排序、tie、top-k、k 越界的处理，全部复用 diag_attrib v2 的公共定义和 topk_ids。
- 表达类型（只用表达文本，小写、合并空白后按词或短语匹配；一条表达可以同时属于多类；都不匹配的记为 OTHER）：
  · MOTION：moving, move, turning, turn, parking, parked, standing, stand, walking, walk, running, run, stopped, stop, faster, slower, same direction, opposite direction, counter direction, braking, accelerating, jumping, serving, dancing
  · SPATIAL：left, right, front, behind, back, near, far, ahead, middle, side, beside, closest
  · APPEARANCE：black, white, red, blue, green, yellow, gray, grey, silver, brown, purple, orange, pink, dark, light, shirt, t-shirt, tshirt, pants, skirt, clothes, wearing, dressed, jersey, uniform
  · ROLE：team, libero, goalkeeper, referee, setter, coach
  把每条表达的类型写入 expression_types.csv（只依赖文本，在任何评测之前生成并封存）。

## 3. Part A：MLLM 教师探针（GPU）

### 3.1 探针集（固定规则，不看 GT）
- KITTI-V2：全部 4 个验证视频；每个视频在帧号上等间隔取 8 帧（5%、17%、…、89%，取最近的有候选的帧）；每个视频按表达的目标类别（car / pedestrian）分层，用 numpy RandomState(0) 无放回抽取 60 条表达（按两类在该视频中的比例分配名额，四舍五入，不足 60 条则全取）；探针帧上的全部候选。
- SportsMOT：45 个视频按 ID 排序，取第 1、4、7、…（每隔 3 个取 1 个，共 15 个）；每个视频同样等间隔取 8 帧；该视频的全部表达；探针帧上的全部候选。
- Dance：复用 count_structure 已封存的探针集（5 视频、每视频 10 帧、17 句、全部候选）。其单帧结果（T1）直接复用，只新增其他两种输入方式。
- 在调用教师之前，先按估算的单次 forward 耗时估计总 GPU 小时。若三种输入方式合计超过 6 GPU 小时，就把 KITTI-V2 和 SportsMOT 的每视频帧数从 8 减为 4（取 5%、30%、55%、80% 处的帧），并在报告中写明。这一决定只依据耗时估算，不依据任何结果。

### 3.2 三种输入方式（每个（候选，表达）对都用三种方式各问一次）
- CROP：现有 T1 方式（320 方形裁剪、外扩 0.2、Yes/No softmax），不改。
- CLIP（时序）：同一轨迹在帧 {t−3Δ, t−2Δ, t−Δ, t} 上的 4 张裁剪（裁剪方式同 CROP），按时间从早到晚作为多图输入。Δ = max(1, round_half_up(0.25 × fps))，fps 从各数据集的 seqinfo 或配置读取，并把每个数据集实际的 Δ 写入报告。若轨迹在某个时刻缺失，就用该轨迹在该时刻之前最近的一次观测代替；可用裁剪少于 2 张时，退回 CROP 的结果，并计入“CLIP 退回数”。
  提示词：“These images show the same object at consecutive moments, from earliest to latest. Does this object match the description: "<expression>"? Answer Yes or No.”
- FRAME_BOX（上下文）：整帧图像（长边缩放到 1024），用 4 像素宽的红色矩形框出该候选，不画其他框。
  提示词：“Look at the object inside the red box. Does it match the description: "<expression>"? Answer Yes or No.”
三种方式都用 Yes / No 两个 token 的 logits 做 softmax 得到 p_yes。提示词、缩放、画框参数在本文件中固定，不调整。

### 3.3 评测（预测封存之后，在独立进程中用 GT）
在完全相同的探针条目上，比较五个打分来源：三种教师输入方式、该目标的跨域源模型（每个组合各一个）、同域模型。指标：
- 同帧 GT R@1：对每个（探针帧，表达），若该帧有 GT，取分数最高的候选，看它是否匹配 GT（IoU≥0.5 一对一匹配，口径与 v2 相同）；对有 GT 的（帧，表达）求平均。
- AUROC：全部探针（候选，表达）对。
- 存在性准确率：每个（帧，表达）上，“有没有候选 p_yes ≥ 0.5（模型分数用 margin>0）”与“g(q,t)>0”是否一致。
- 数量误差：每个（帧，表达）上 #{p_yes ≥ 0.5} 与 g(q,t) 之差的绝对值，取平均。
以上指标再按表达类型（MOTION / SPATIAL / APPEARANCE / ROLE / OTHER）分别报告。
成本：每种方式的调用次数、forward 秒数、峰值显存；CLIP 退回数。
不确定性：SportsMOT（15 个探针视频）和 Dance（5 个）上，按视频聚类 bootstrap（2000 次、seed 0）比较“每种教师方式 − 源模型”和“每种教师方式 − 同域模型”的 R@1 差值；KITTI-V2 只有 4 个视频，只报告点估计。

### 3.4 Part A 的预注册判定（逐字执行）
对每个目标 T ∈ {KITTI-V2, SportsMOT, Dance}，对每种教师方式 m：
- 若 R@1(m) ≥ R@1(同域模型) − 0.02，并且对 T 上的每个跨域源模型都有 R@1(m) ≥ R@1(源模型) + 0.05，记为“方式 m 在 T 上可以修排序”；
- 若 R@1(m) > 对 T 上所有跨域源模型的 R@1，但不满足上一条，记为“方式 m 在 T 上部分修排序”；
- 否则记为“方式 m 在 T 上不能修排序”。
报告每个 T 上判定最好的方式；若有多种方式同等，全部列出。不对表达类型做额外判定，只报告数字。

## 4. Part B：选择的粒度与 GT-free 校准（CPU）

### 4.1 Oracle 粒度阶梯（排序固定用源模型 S，标为【ORACLE，仅诊断】）
对 C1–C6 的每一个组合：
- G0_GLOBAL：对 S 的全部候选分数做全局排序，取前 N 行，N = 该目标全部 GT 检测数（Σ_q Σ_t g(q,t)）。
- G1_QUERY：对每个 q，在 q 的全部（帧，候选）上按 S 的分数排序，取前 N_q 行，N_q = Σ_t g(q,t)。（每条表达一个阈值）
- G2_V_MEAN：对每个 q，每帧取 k = m̄(q) = round_half_up(Σ_t g / #{t: g>0})；m̄(q)=0 时整条不输出。（视频级每帧常数）
- G3_F_MEAN：有 GT 的帧取 k = m̄(q)，没有 GT 的帧取 k = 0。
- G4_F_FRAME：k = g(q,t)。必须逐行复现上一轮的 R_S×K_ORACLE，否则停止并报告。
- TRACK_ORACLE：输出所有“≥50% 观测匹配 q 的 GT”的轨迹的全部观测（与 count_structure 定义相同）。
C1 中 V2-best 的 G2、G3、G4、TRACK_ORACLE 已在 count_structure 中计算过，核对后复用；其余新算。

### 4.2 GT-free 校准（不用 MLLM，排序用 S）
源统计：每个源模型只用它自己的同域评测集上的分数计算：V2-best 用 KITTI-V2 验证集（完整覆盖），Dance-best 用 Dance Val，LaMOT-best 用 SportsMOT。r_src = 原生 margin>0 的比例，μ_src = 全部分数的均值，z0 按 v2 的定义计算。在报告中注明：这些源统计来自源模型的同域评测集。
对每个组合运行：
- B1_SRC_RATE_GLOBAL、B1_SRC_RATE_QUERY、B2_Z_OFFLINE、B3_MEAN_GLOBAL、B3_MEAN_QUERY：定义与 diag_attrib v2 完全相同，只换成各组合的源模型和源统计。
- C1_TRACK_MEAN：每个 (q, τ) 取整条轨迹 margin 的均值，>0 时输出整条轨迹（定义与 count_structure 相同）。
C1 组合中 V2-best 的 B 系列已在 v2 中计算，核对后复用。

### 4.3 指标与判定
每个配置报告 HOTA / DetA / AssA / IDF1 / IDSW / 行数 / Query-Empty 预测行数。KITTI-V2 再按 car / pedestrian 报告；全部组合再按表达类型报告 HOTA。
Dance（25 视频）和 SportsMOT（45 视频）上做视频聚类 bootstrap（2000 次、seed 0）：每个粒度 − NATIVE，每个 GT-free 校准 − NATIVE；KITTI-V2 只报告点估计。
判定（逐字执行）：
- 所需粒度：在 G0、G1、G2、G3 中，取第一个满足 HOTA(G_i) ≥ HOTA(R_S×K_I) 的层级（R_S×K_I 即上一轮“源排序 + 同域数量”的结果），记为该组合的“所需粒度”；都不满足则记为“需要逐帧数量”。
- GT-free 可恢复比例：对每个 GT-free 校准 b，ρ_free(b) = [HOTA(b) − HOTA(NATIVE)] / gap，gap 沿用上一轮。逐个报告，不选最好的。

## 5. 交付
results/decompose/REPORT.md，内容依次为：
1. Part A 判定矩阵（行为输入方式，列为目标）与 Part B 的“所需粒度”汇总；
2. Part A：探针集描述、三种方式与源模型、同域模型在相同条目上的全部指标（总体和按表达类型）、bootstrap、成本；
3. Part B：粒度阶梯表、GT-free 校准表、ρ_free 表、按类别和表达类型的拆分、bootstrap；
4. SportsMOT 在官方 418 条序列上的敏感性结果；
5. GT 审计、失败与修复记录。
机器文件：expression_types.csv、probe_sets.json、teacher_probe.csv、teacher_costs.json、granularity.csv、calibrators.csv、rho_free.csv、bootstrap.json，以及教师原始输出、预测文件的路径与 SHA 清单。大文件不进 git。
push 后回复提交号和报告链接。完成后停止，不训练，不开始任何方法实验。
