# D-FIT 事后数量/存在性与同帧排序归因：预注册 v1

日期：2026-10-31。项目：LYQ1107/TTA-RMOT；分支：`research/v2dance-diag-attrib-20261031`。
唯一出发提交：`958e1d7c6165bdc257d420562d94d66916f67ddb`。
本文件先提交、推送并核对远端，之后才开始分数计算和新评测。不是新方法、盲测、Train→Val 或训练实验。

## 1. 不可变输入、隔离与停止条件

- 新输出只写 `results/runs/diag_attrib_20261031_v1/` 和 `results/diag_attrib/`。
- 旧 D-FIT 预测、margin、checkpoint、收据、结果一律只读；不训练、不优化、不重新选择 checkpoint。
- 目标固定官方 DETR+NeuralSORT 的 25 视频、425 表达、211,085 MOT 观测、3,588,445 候选–表达组合；固定官方 TrackEval 代码、seqmap、过滤配置。
- FROZEN 使用完整候选库基线 8.2180，不用旧缺末观测的 8.2060。
- D1/D2/D3 分别是 DIAG_FIT_OPT / DIAG_FIT_OPT_POS / DIAG_FIT_CAP 的 final-step 8973 封存分数，逐文件核对旧 seal SHA；D0 是旧 NAIVE_PAIRED seed17，7.4499。
- OFFICIAL_DANCE 使用 refer-dance-best.pth，SHA256 `fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c`；首先检查旧 corrected_tokenizer 结果的完整分数。若缺少，最多一张显存够、进程最少的共享 GPU，采用原推理与修正 tokenizer 补导出，不改权重或候选。
- 必须完整覆盖所有候选；不把缺失分数伪造为负数、零、UNKNOWN 或其他模型分数。OFFICIAL margin>0 必须逐行复现原官方预测 254,118 行、HOTA 32.134；不能复现则停止并报告，不能替换历史官方结果继续做矩阵。
- 所有预测都是官方 MOT 原始行的子集，框、ID、帧号不改，不插值；按 frame、数值 track_id 输出。比较旧文件时同时核对逐行规范化 MOT 字段与原始文件 SHA，任何原模型输出集合不一致都停查。
- GT-free 生成独立进程禁止打开目标 gt.txt、gt_template、labels/目标匹配注释；使用已有仅含文本的运行时 manifest，不读携带目标标签的表达文件。保存打开尝试数，非零立即失败。
- GT 只在预测封存后的独立评测进程，以及单列 ORACLE_GT/B0 诊断进程读取。ORACLE 结果永不称为方法或 GT-free 基线。GT-free 进程绝不读取从 GT 派生的帧数、目标数、presence 或排序。
- 所有原始失败日志、版本及未通过检查保留；不得换 seed、略过表达、改指标或静默放宽上述要求。健康完成单元复用。

## 2. 公共精确定义

记 q=(video, expression)，t 为官方帧，C(q,t) 为该帧预测 MOT 候选，N_q 为整个 q 的候选行数；m_M(q,t,i)=logit_yes−logit_no（已有 margin 原精度，不重新量化）。
模型名 M∈{FROZEN,D1,D2,D3,OFFICIAL_DANCE}。

- 同帧排序：`(-margin, numeric_track_id)`，精确相等才算 tie，不加 jitter/epsilon。
- 跨帧/全局分位选择：`(-margin, video_id, expression_id, frame_id, numeric_track_id)`；逐表达去掉前两项。
- 分位发射预算 n(r,N)=min(N,max(0,floor(r*N+0.5)))，选择排序前 n 项。阈值边界若并列，用上述固定次序精确补足 n；报告边界数值、并列数和 tie 选择数，不能声称所有行仅由一个严格标量阈值区分。
- 原生选择始终严格 `margin>0`；B2 按任务要求为 `z>=z0`；其他平移为严格 `margin'>0`。零不随意计正。
- k=0 输出空集合；k>|C| 输出全部并报告 overflow 帧数、截断数量。非 oracle 的 k 超界意味着实现/协议错误，停止；oracle 超界是可解释的候选覆盖限制，保留统计。
- 缺少原模型任何候选 margin、非有限 margin、GT-free 非法访问、重复候选、原预测不一致：停止，不用替代模型/默认正负/改变阈值回退。
- 仅有基线定义明确允许的无历史、零方差、无有效教师判断采用原生 FROZEN 选择；逐表达数、帧数、候选数、预测数单独记录。
- 帧级数量统计主表用官方完整视频时间轴（零候选帧 k=0），并另外报告有候选活动帧分母；主表有输出帧的均值和 p50/p90 只在 k>0 上计算。GT-present/empty 帧的均值只在独立评测进程分组，报告分母和零候选帧数。

## 3. 表 C 与每视频附表

列：GT统计、OFFICIAL_DANCE、FROZEN、D0_NAIVE、D1、D2、D3。
行：预测数；/212,541 GT检测倍数；/254,118 OFFICIAL 行数倍数；/3,588,445 选中率；有输出 q,t 数及占比；k>0 的平均/p50/p90 k；GT有目标与GT空帧平均 k；HOTA、DetA、AssA、DetRe、DetPr、AssRe、AssPr、IDF1、IDSW、官方CLEAR FP/FN、Dets、IDs。
GT列报告相应GT计数，不把GT当预测模型；无意义指标记 N/A。每视频 HOTA 按官方 sufficient statistics 精确 combine，不取表达均值；同时列行数和 /本视频 OFFICIAL 行数。官方 FP 与离线 IoU 行匹配 FP 不混用。

## 4. 排序×数量：固定 30 格

R∈{FROZEN,D1,D2,D3,OFFICIAL_DANCE}；K∈{FROZEN,D1,D2,D3,OFFICIAL_DANCE,ORACLE_GT}。
非 oracle k(q,t)=Σ_i 1[m_K(q,t,i)>0]；ORACLE_GT 的 k 为当前固定官方协议过滤后的该表达该帧 GT 数，只在独立离线 oracle 评测进程计算。
每格选 R 排序前 min(k,|C|) 个原始 MOT 行，全量425表达官方评测，报告 HOTA、DetA、AssA、IDF1、IDSW、行数及 overflow。
输出 HOTA、行数、同列各 R 减 FROZEN 的 HOTA 差值三张矩阵，oracle 列醒目标【ORACLE，使用GT，仅诊断】。

强制自检，在继续新评测前验证：

- 五个 R=K 对角逐行等于原生预测；对应 HOTA 为 8.218/29.702/22.844/27.768/32.134。
- FROZEN×D1/D2/D3 逐行等价旧 count_controls，HOTA=29.400/22.854/27.103。
- D1/D2/D3×FROZEN 逐行等价旧 count_controls，HOTA=7.8978/7.8380/8.1589。
- 复用 `scripts/diag_pf_countmatched.py:topk_ids` 相同排序规则，或对全部425表达证明新实现与旧结果逐行等价；只数值接近不足以通过。
- 完全相同且已官方评测的预测可复用对应原始官方 summary/detailed CSV，并记录等价性与来源，不重跑健康完成单元。

## 5. 固定校准基线

全部排序只用 FROZEN，不训练；所有源统计只用官方 V2 checkpoint 的已有 Refer-KITTI-V2 官方候选分数（4个官方验证视频、862表达的已释放/复现实测源域分数，非 Dance GT；明确不称为17视频源Train统计）。先核验2,355,528行、原生正750,834及原模型来源。若源缓存缺失或协议不符停止该阶段并报告，不用目标GT拟合替代。
源 margin 为原始双类 logits 差。所有均值、标准差使用 float64、总体标准差 ddof=0；source/target 每表达与全局候选分母明确保存。

### B0【ORACLE规模轴/诊断曲线，不是方法】

预算为 round-half-up(212,541×a)，a∈{0.25,0.5,0.75,1,1.25,1.5,2,3}，另加精确行数 99,704（FROZEN）、254,118（OFFICIAL）、345,302（D1）。使用全局 FROZEN 排序的前预算行；采用公共 tie 规则，记录对应 margin 边界。
GT检测数是用户提供的离线诊断参照，故整个 B0 单独标为 ORACLE/GT-informed scale。不得用于判定A候选，不选择最优点，不据曲线改变定义。
在 HOTA–行数图上加 D1/D2/D3/OFFICIAL 原生散点，不把同数量当同预测。

### B1_SRC_RATE_GLOBAL / B1_SRC_RATE_QUERY

r_src=源官方原生正行数/源全部候选行数（预期750,834/2,355,528）。
GLOBAL 在目标全部候选全局选择 n(r_src,3,588,445)；QUERY 在各 q 选 n(r_src,N_q)。源统计+目标无标签全视频分位，使用未来帧；目标GT打开0。

### B2_Z_OFFLINE / B2_Z_CAUSAL

OFFLINE：每个 q 用其全视频全部 margin 的均值 μ_q 和总体σ_q 标准化；源每个表达 z0_q=(0−μ_q)/σ_q，z0为所有σ_q>0源表达的等权均值。目标选 (m−μ_q)/σ_q>=z0。
CAUSAL：窗口固定前100个数值视频帧 [t−100,t−1]，排除当前帧，不跨表达/视频，窗口内所有可用候选按行加权计算 μ_q,t、σ_q,t；零候选历史/σ=0 时仅当前帧回退 FROZEN。源用完全相同窗口，对每个表达内有效帧的 (0−μ_q,t)/σ_q,t 先取均值，再对有有效帧的源表达等权平均得到独立 z0_causal；目标 z>=z0_causal。当前候选只用于当前选择，永不加入当前统计。
OFFLINE 用未来帧；CAUSAL 目标仅过去100帧，源离线统计固定，目标GT打开均0。无方差源表达不进入z0平均，数量单列；若无有效源表达则失败而非自定z0。

### B3_MEAN_GLOBAL / B3_MEAN_QUERY

μ_src为全部源候选的全局均值。
GLOBAL：m'=m−μ_target_global+μ_src，选m'>0。
QUERY：m'=m−μ_target_q+μ_src，选m'>0；不把不同数据集的表达强行配对。
均使用源分数和目标无标签全视频统计、含未来帧，目标GT打开0。

### B4_TEACHER_RATE / B4b_TEACHER_GATE

仅旧 Naive8500 已实际判断、有限有效 p_yes 的记录；POS严格p>=0.95，非POS不是新增硬负监督。按(video,expression,frame,track)去重，重复冲突停止，不重新调用教师。
对q，J_q=有效已判断候选数，P_q=其中POS数，N_q=该表达全视频固定候选总数；p_teacher=P_q/J_q，已判断候选占比 c_q=J_q/N_q。
明确本轮公式：r_q=p_teacher×c_q=P_q/N_q；B4选择该表达 FROZEN top n(r_q,N_q)，即总计P_q行（浮点 round-half-up）。这是对题述“已判断候选比例”的覆盖率实现，不把稀疏样本的正率外推到全部候选；不在看到GT后换公式。
B4b：J_q>0且P_q=0则该表达所有帧不输出；P_q>0保留 FROZEN 原生选择。
两者J_q=0均整条表达回退原生 FROZEN；报告表达/候选/输出行回退数。用封存MLLM判断与目标全视频分数，有未来信息；目标GT打开0，新增教师调用0，不称为纯源统计基线。

所有 B1–B4 各版本均报告完整表C指标、行数倍数、输入信息、是否未来帧、GT读取尝试数、回退与耗时，不根据评测挑新的版本。

## 6. 统计精确定义

固定25视频聚类，表达/帧全部留在视频内；paired bootstrap 2000次、numpy default_rng(0) multinomial(25,1/25)，percentile 95%区间。复用 `scripts/analyze_pf_video_uncertainty.py` 的 sufficient/paired_clusters，HOTA按官方TP加权精确汇总，核对未四舍五入的COMBINED CSV；不平均视频HOTA、不把帧当独立样本。
每个比较报告：未舍入合并HOTA差、95%区间、25个单视频差及负/零/正个数、25个leave-one-video-out合并差的[min,max]。
比较固定为：

1. D1−FROZEN×D1、D2−FROZEN×D2、D3−FROZEN×D3。
2. D1−每个B1/B2/B3/B4/B4b版本；D1−事后HOTA最高的GT-free基线（若并列按基线名称字典序，仅报告，不改定义），同时给两者行数差。B0绝不参与。
3. K为OFFICIAL_DANCE、D1、ORACLE_GT时，D3排序−FROZEN排序、D1排序−FROZEN排序。
4. D1−OFFICIAL_DANCE。

这是同一开发Val集合的描述性视频不确定性，不是盲测显著性、训练seed不确定性或经多重检验修正的新方法证明。

## 7. 预注册判定规则（按用户原文）

- 判定 A（D1 增益能否用简单校准解释）：
  - 如果存在某个 GT-free 基线 B1–B4，使 D1 − 该基线的 95% 区间包含 0 或整体为负，记为"D1 的增益可被无学习校准解释"；
  - 否则记为"学生提供了超出简单校准的存在性信息"。
- 判定 B（D3 排序信号是否真实）：
  - 在 K=OFFICIAL_DANCE 和 K=D1 两列中，如果 D3 排序 − FROZEN 排序的 95% 区间下界都 >0，记为"D3 的排序增益在合理数量下成立"；
  - 只有 K=ORACLE_GT 列成立时，记为"排序信号只在 oracle 数量下可见"；
  - 都不成立时，记为"未观察到可转化的排序增益"。
- 判定 C（数量对齐）：
  - 报告各臂行数 / OFFICIAL_DANCE 行数；
  - 0.75–1.5 倍记为"与同域官方同量级"，超出这个范围记为"过量或不足"；
  - 这只是描述性判定，不作为方法优劣的结论。
- 判定只能逐字套用上面的规则。如果出现规则没覆盖的情况，如实描述，不新增规则。

若硬性自检未通过而停止，A/B不判定，不把不完整候选对齐当科研结果。

## 8. 交付与停止

中文 REPORT 按用户第9节顺序提供Q1–Q4五句内结论、表C、三矩阵、基线表、bootstrap、B0曲线、自检、GT审计、CPU/GPU成本、真实失败修复与建议。
机器文件：attrib_matrix.csv、baselines.csv、bootstrap.json、alignment.json、每视频CSV、新预测路径与SHA清单；大预测/权重不进git。
追加research_log.md；推送后核对远端commit/tree与本地。完成后停止，不训练新学生，不启动Train→Val、留出视频、密集标签、容量/pos_weight修改或新方法。
