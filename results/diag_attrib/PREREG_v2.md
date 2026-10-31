# D-FIT 事后归因预注册 v2：完整官方参照与统一历史覆盖敏感性

日期：2026-10-31（Asia/Shanghai）。项目：`LYQ1107/TTA-RMOT`。
继续分支：`research/v2dance-diag-attrib-20261031`，起点为已推送的 `8c47309003cf961acba987d30cc8bf728a3588e9`。
本文件先 commit、push、核对远端内容，然后才开始本轮任何新评测。

## 1. 与 v1 的关系及不可变文件

用户明确批准 v2，且仅批准本文件列出的覆盖协议修订。`PREREG.md`（SHA256 `d064ca9e7762fa5ff5701fc4c66f033d64ce680a617bac1980b6153ec9a589b6`）、v1 停止报告、日志、所有历史预测/margin/checkpoint 与 vendor 代码一律不改。
v1 的公共规则、B1–B4八个版本的精确公式、tie、回退、统计、判定A/B/C和停止条件，除下述显式变更外完整继承；不因任何本轮指标修改定义。
新原始输出：`results/runs/diag_attrib_20261031_v2/`。新正式报告、CSV、PNG和精简日志：`results/diag_attrib/v2/`；不覆盖 v1 的同名文件。
全部使用同一25视频、425表达、官方DETR+NeuralSORT候选，完整211,085条MOT观测/3,588,445候选–表达。不是新方法、盲测或Train→Val；不训练、不更新参数、无新教师调用，预计完全CPU，不再运行GPU补分数。

## 2. 主参照 OFFICIAL_DANCE_COMPLETE

同一个 `refer-dance-best.pth`，SHA256 `fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c`。
复用历史corrected_tokenizer原始双类logits（原SHA `de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110`），加上v1封存的25个补分数文件；逐个核验已发布SHA，不替换已有logits。margin=Yes−No，严格margin>0，完整原生输出254,910行。
先封存全部425个预测与完整margin的SHA256，再在独立官方TrackEval进程评测；实际HOTA多少就报多少，绝不把32.134赋给完整版本。

新强制复现门槛：

1. 限于历史原本覆盖的210,393个MOT观测时，所有425表达的官方预测必须逐行等于历史254,118行（沿v1规范化数值MOT字段、按frame/数值ID对齐，并保留历史原文件SHA）。
2. 新增792行全部来自692个补分数的末观测；这些观测必须恰好等于完整候选减历史覆盖，且每个都为相应轨迹的末次观测。不得删除新正例、伪造missing为负、换阈值或换checkpoint来复现旧数量。
3. 两条均通过才接受新参照进入矩阵；任一失败停止，保留日志并报告。

主参照的对角格R=K=OFFICIAL的期望HOTA，固定为本节封存后实际官方计算所得值。其余四个主对角仍逐行复现Frozen8.218、D1 29.702、D2 22.844、D3 27.768；六个旧count_controls逐行等价门槛和既有HOTA完全不变。

## 3. 主矩阵、表C、校准与归因

主排序R∈{FROZEN,D1,D2,D3,OFFICIAL_DANCE_COMPLETE}；数量K∈同五模型再加ORACLE_GT，共30格。k按对应完整模型当前帧margin>0个数；同帧按(-margin,数值track_id)取前min(k,候选数)，oracle超界照v1统计。非oracle任何缺失或越界停止。
矩阵、表C、B0散点及精确数量点、B1–B4输出相对官方倍数、全部bootstrap、A/B/C中凡原称OFFICIAL_DANCE者，均统一指OFFICIAL_DANCE_COMPLETE，行数分母254,910；旧32.134/254,118只在历史/敏感性节出现。
相对GT分母仍为212,541；候选分母仍3,588,445，未变GT或过滤规则。

所有GT-free预测生成在独立进程启用目标GT访问审计，只读已剥离标签的runtime文本manifest、原MOT、封存分数、合法源统计/Naive8500教师判断；尝试数必须0。GT仅在独立ORACLE/B0诊断与最终评测/报告进程使用；不得让离线分组、GT数量或评测结果进入GT-free生成器。
没有任何新优化步、阈值拟合、外部教师或基于GT的候选采样。

为避免引用歧义，下列v1基线定义完全不变：

- B0：预算round-half-up(212,541×a)，a={0.25,0.5,0.75,1,1.25,1.5,2,3}，另99,704、254,910、345,302；全局Frozen top预算。GT-informed规模轴，仅曲线，不参与判定、不选最优当方法。
- B1_GLOBAL/QUERY：源官方V2已有4视频862表达、2,355,528分数中750,834个margin>0，r_src=750,834/2,355,528。全局/逐表达Frozen按该比例取分位。
- B2_OFFLINE：每表达全视频均值/总体标准差，z0为源各有效表达(0−μ_q)/σ_q等权均值，目标z>=z0；B2_CAUSAL：仅[t−100,t−1]候选统计，源同窗口逐表达内有效帧平均再对表达等权平均z0，目标当前帧不进入统计。无历史/零方差回退原生Frozen。
- B3_GLOBAL/QUERY：m'=m−目标全局/逐表达均值+源全局均值，严格m'>0。
- B4_RATE：已实际判断且有效的J_q，POS(p>=0.95)数P_q，完整候选数N_q，r_q=(P_q/J_q)(J_q/N_q)=P_q/N_q；Frozen逐表达top round-half-up(r_qN_q)，不把稀疏正率外推到全部候选。
- B4b_GATE：J_q>0且P_q=0整条表达不输出，其余按Frozen原生；两教师版本J_q=0均回退原生。未判断/无效/UNKNOWN不补负标签；重复候选冲突停止。
- 分位数量round-half-up并截断到[0,N]；跨帧tie按(-margin,video,expression,frame,数值track_id)。严格原生>0，B2使用>=；均值/总体方差float64/ddof0。未增加任何新数值超参数。

## 4. 历史210,393覆盖敏感性（只报告，不参与A/B/C）

所有五个模型统一用同一历史210,393个MOT观测（3,576,681候选–表达），移除每条轨迹末观测。通过历史官方score的key确定并证明其与末观测集合一致，不依据GT删行。
**只限制预测候选，GT、425表达、视频时间轴、帧偏移、mark/class过滤和官方评测配置完全不变。**

固定重新评测7个预测集：

- 五个R=K对角（各模型在历史候选集合内按自身margin>0）。
- 历史候选集合内K=OFFICIAL_DANCE_COMPLETE时，D1排序和D3排序两格；该K恰好等于历史官方原生数量。

另报告D1−OFFICIAL的HOTA差和同一25视频聚类区间（由上述两对角结果计算，不是第8次推理或新模型）。同时报告D1/D3排序相对同覆盖Frozen排序的说明仅限有相应结果时，不增跑未登记格子。
历史覆盖OFFICIAL对角必须逐行254,118且官方HOTA=32.134；否则停止并报告，不能在完整覆盖协议下忽略此失败。
报告单列两种覆盖的HOTA和行数；不可挑较好覆盖替代主结果，不参与主判定。

## 5. 统计与判定（不变，仅替换主官方参照）

25个完整视频为聚类单位，所有表达/帧绑定视频；2000次配对multinomial bootstrap，numpy default_rng(0)，percentile 95%区间。复用官方HOTA sufficient/paired_clusters并核对未舍入COMBINED，不能平均视频HOTA。报告leave-one-video-out范围及单视频负差数量。
必报：D1/D2/D3−Frozen@各自K；D1−八个B1–B4版本及事后最好GT-free基线（仅报告，附行数差）；K∈{OFFICIAL_DANCE_COMPLETE,D1,ORACLE_GT}中D1/D3排序−Frozen排序；D1−OFFICIAL_DANCE_COMPLETE。

判定A：存在某个B1–B4版本使D1−基线95%区间包含0或整体为负，逐字记“D1 的增益可被无学习校准解释”；否则“学生提供了超出简单校准的存在性信息”。
判定B：K=OFFICIAL_DANCE_COMPLETE和K=D1两列中D3−Frozen区间下界都>0，记“D3 的排序增益在合理数量下成立”；只有oracle列成立，记“排序信号只在 oracle 数量下可见”；都不成立记“未观察到可转化的排序增益”。未覆盖的混合情况如实写，不新增规则。
判定C：各臂/254,910在[0.75,1.5]为“与同域官方同量级”，否则“过量或不足”；仅描述，不决定方法优劣。
敏感性区间和B0不参与上述判定，禁止用目标GT修改阈值、权重或训练选择。

## 6. 执行顺序、日志、成本与停止

1. 推送本文件并验证远端commit/tree与本地内容，保存发布gate；这之前0次新评测。
2. 独立GT-free CPU进程检查旧分数/补分数SHA、692末观测与792新增行，封存完整官方与历史覆盖官方原生预测。
3. 独立官方TrackEval先计算完整官方HOTA，并重新评测历史官方，校验32.134；通过后登记主对角实际期望值。
4. CPU生成主非oracle矩阵、8个校准基线与其余6个历史敏感性预测，逐行核验四主对角与六旧count_controls；全部封存后独立评测。健康完成预测/官方CSV通过等价与SHA可复用，但7个历史敏感性按用户要求本轮重新评测。
5. ORACLE_GT五格与B0仅独立离线诊断生成/评测；不反馈到其他生成器。汇总表C、每视频、区间、A/B/C、B0图，报告全部真实成本及失败。

新输出的full_audit、SHA清单、独立进程GT打开数、CPU wall/core时间分别记录；不伪造未测CPU-core或GPU利用率。本轮新GPU调用预期0，继承补分数成本另列，不能说完全无历史GPU成本。
任何数值/协议自检失败，保留原日志，定位最小实现错误；若定义本身有问题则保留v1/v2并报告，不静默改定义。除本文件明确批准的旧覆盖复现门槛修订外，其余硬性停止条件不变。

## 7. 交付

`results/diag_attrib/v2/REPORT.md`中文报告（Q1–Q4五句内摘要、表C、三矩阵、基线、bootstrap、B0曲线、自检、GT隔离、成本、失败修复、两覆盖范围说明）；attrib_matrix.csv、baselines.csv、bootstrap.json、alignment.json、每视频CSV、敏感性CSV/区间、所有新预测SHA与可恢复路径。大预测和权重不入Git。
追加research_log.md，按独立阶段推送，核对远端commit内容。完成后停止，绝不启动新训练、Train→Val、留出视频、容量/pos_weight调整或其他新实验。
