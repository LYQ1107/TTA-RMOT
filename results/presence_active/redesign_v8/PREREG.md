# v8：修复 v7 的存在性估计偏差、慢分支样本失衡与回放未冻结（预注册版）

## 0. 背景与硬约束（不变）
- 仓库 LYQ1107/TTA-RMOT，从 commit 48ed3b6（v7 结项）切新分支 codex/presence-active-v8-20261106，结果写入 results/presence_active/redesign_v8/。
- v7 的所有文件不改动；v8 新建文件、继承 v7 类。v7 结果作为对照保留。
- 目标域 GT 只能在预测封存（seal）后由独立 CPU 评测或诊断进程读取；所有生成进程 GT_open_attempts 必须为 0。
- 不杀、不抢占他人 GPU 进程。公开资源可下载；不猜测凭据。
- 超参数六个方向统一，本轮不做逐方向调参。
- 遇到与本提示矛盾的情况，或预注册判据无法计算时，停下并报告，不得自行改判据。

## 1. v7 的四个根因（已从 v7 日志确认）
R1 π̂ 选择偏差（最主要）：OnlinePresenceEB.finish_video（flex_uda/active/v7.py）只在 presence_probability>.9 的“已确认存在”句子上累计 π 的软计数。确认本身依赖教师答 Yes，所以样本偏向 Yes。结果 π̂ 最终值 C1 .94（真值 .56）、C2 .73（真值 .12）、C5 .71（真值 .12）。π̂ 偏高使 recompute_presence 中每个 No 的对数似然比（logLR）被高估（π=.94 时每个 No 约 −1.36），C2 因此有 296/862 个有目标句被错判为空。
R2 判空自强化：rho_beta += absent.sum() 统计的是判定结果，包括仅由先验触发的判空（presence_stopped 在 value<.1 时立即置位，无需任何回答），所以 ρ̂ 越低判空越多，判空越多 ρ̂ 越低。C2 的 ρ̂ 从 .5 降到 .31（真值 1.0）；C6 有 12 个有目标句被先验直接判空，导致 271/567 条真实正轨迹丢失。另外 questions_v7 只在 .1≤p≤.9 时提存在性问题，p<.1 的句子一个问题都不会被问。
R3 慢分支样本失衡：
  - 互斥（exclusive）分支的 NONE 标签要求 np.all(pseudo[:,ti]==0)，但未提问的对标签为 −1，因此 NONE 永远无法触发，C1 的 FULL/REPLAY 都是 0 个 NONE 样本，空句的误报行数从 18,477 增加到 35,767。
  - BCE 分支没有正负下限：C4 回放 9000 步都是 0 正例 / 288000 负例；C3 回放则正例占多数，误报扩张到 230 万行。
R4 回放未冻结派生状态：SLOW_REPLAY 重新计算存在性与 EB，C4 从第 13 个视频起 ρ̂ 塌陷，后 33 个视频全部零输出（HOTA 5.11）。这测的不是慢分支的效果。

## 2. 代码改动

### 2.1 新建 flex_uda/active/v8.py

(a) class PresenceEM：两层隐变量模型 + EM，替换 OnlinePresenceEB。
模型：句子 q 是否存在 z_q ~ Bern(ρ)；若 z_q=1，档位 k 的轨迹 t 是否匹配 y_qt ~ Bern(π_k)；若 z_q=0，y_qt=0；教师对 (q,t) 的每次回答在给定 y_qt 时条件独立，敏感度/特异度取 DS 当前值 (s_v, c_v)。
档位 k ∈ {0,1,2} 直接用 state.strata[q,t]（视频开始时按初始先验排名三等分；冻结轨迹 strata=−1，不参与）。
数据：句子的全部教师回答，包括 GLANCE / PRESENCE / PRESENCE_COVER / IDENTITY / POSITIVE_RECHECK，取自 state.answer_history。不使用任何“判定结果”，不使用学生的弱排名先验。

```python
class PresenceEM:
    def __init__(self, n_strata=3, a=2., b=2., clip=(.05,.95), iters=200, tol=1e-7, track_cap=2.):
        self.records=[]          # 每句一条: dict(video,query,tracks=[(k,[(view,yes),...]),...])
        self.rho=.5; self.pi=np.full(n_strata,.5); ...
    def add_video(self, video, state):
        # 遍历本视频每个句子：收集 strata>=0 且有回答的轨迹；没有任何回答的句子不加入
    def track_terms(self, k, answers, rel):
        l_pos=sum(np.log(rel[v][0] if y else 1-rel[v][0]) for v,y in answers)
        l_neg=sum(np.log(1-rel[v][1] if y else rel[v][1]) for v,y in answers)
        m1=np.logaddexp(np.log(self.pi[k])+l_pos, np.log1p(-self.pi[k])+l_neg)
        llr=np.clip(m1-l_neg, -self.track_cap, self.track_cap)   # 每条轨迹的 LLR 截断 ±2
        r=np.exp(np.log(self.pi[k])+l_pos-m1)                     # 存在条件下该轨迹为正的后验
        return llr, r
    def fit(self, rel):
        for it in range(self.iters):
            S_g=0; N=0; num=np.zeros(3); den=np.zeros(3)
            for rec in self.records:
                terms=[(k,)+self.track_terms(k,ans,rel) for k,ans in rec['tracks']]
                g=expit(logit(self.rho)+sum(l for _,l,_ in terms))   # γ_q = P(z_q=1 | 回答)
                S_g+=g; N+=1
                for k,_,r in terms: num[k]+=g*r; den[k]+=g
            rho=np.clip((S_g+self.a-1)/(N+self.a+self.b-2), *self.clip)        # Beta(2,2) 的 MAP
            pi=np.clip((num+self.a-1)/(den+self.a+self.b-2), *self.clip)
            if max(abs(rho-self.rho), abs(pi-self.pi).max())<self.tol: break
            self.rho, self.pi = rho, pi
        return self.receipt(iterations=it+1)
```

  - 每个视频结束时调用 add_video，再用该时刻的 ds.values 对全部历史记录重新 fit。视频开始时把 (ρ̂, π̂_k) 冻结给本视频使用；视频内部不更新。
  - receipt 记录：ρ̂、π̂_0..2、迭代次数、是否收敛、记录数、每档轨迹数；另外按 v7 公式并行计算“旧估计器值”，仅作诊断。
  - 写进代码注释和报告的理由：提问选择只依赖已观测的数据（之前的回答和学生分数），满足随机缺失（MAR），基于似然的 EM 因此无需对选择过程建模就保持一致；v7 的偏差来自“按结果筛选样本”，不是来自提问选择。

(b) class BeliefStateV8(BeliefStateV7)：
  - 构造参数改为 rho_hat、pi_strata（长度为 3 的数组），不再使用 pi_hat。
  - 覆写 recompute_presence(reliability=None, queries=None)：
      total = logit(clip(ρ̂,.05,.95)) + weak_presence_logits[q] + Σ_t LLR_t
    其中求和遍历句子 q 下所有 strata≥0 且有回答的轨迹，LLR_t 按上面的 track_terms 计算（用视频开始时冻结的 π̂_k，每条轨迹截断 ±2）。
    covered_q = 有至少一条已回答轨迹的档位集合；needed_q = 该句中存在合格轨迹的档位集合。
    判空（且一旦判空就保持）：p<.1 且 covered_q ⊇ needed_q。仅凭先验不能判空。
    低存在性待定（low_tentative，新增数组）：p<.1 但覆盖不全。不判空，按 2.1(c) 优先补问。
    确认存在：p>.9。
    保留 v7 的诊断字段（logLR 截断前后之和、截断次数等），新增 covered_strata 和 low_tentative。
  - observe：沿用 v7，每次回答后对该句调用 recompute_presence。
  - 新增 presence_snapshot() 与 apply_presence_snapshot(snap)，覆盖 presence_probability、absent、confirmed_present、presence_stopped、stop_probability、low_tentative，供回放冻结使用。
  - 视频结束时，仍处于 low_tentative 的句子：输出阶段按 p<.1 视为空句，记 decision_source='PRIOR_BUDGET_EXHAUSTED'，并在报告中单独计数。这只影响输出，不影响 EM，因为 EM 只看回答。

(c) questions_v8(state, texts, lengths, reliability, round_index)：在 v7 基础上加两类高优先级问题。
  - GLANCE（扫一眼）：对没有任何回答的句子，问第 0 档中先验最高的合格轨迹（按 base_logits 排名，并列时取轨迹下标小的）。看法：前两轮用 route(text)，之后用信息增益最大的看法（同 v7）。每句至多一个 GLANCE，句子顺序为 query 下标。
  - PRESENCE_COVER（补齐覆盖）：对 low_tentative 的句子，每个未覆盖档位问该档先验最高、且尚未回答的合格轨迹。
  - low_tentative 的句子在覆盖补齐之前不提 IDENTITY 问题。
  - 其余逻辑与 questions_v7 相同：POSITIVE_RECHECK；.1≤p≤.9 时的 PRESENCE；IDENTITY 按信息增益。
  - 返回顺序：GLANCE → POSITIVE_RECHECK → PRESENCE_COVER → 其余，组内按 v7 的 order 排序。
  - 预算：所有类型都计入 budget；round_budget 和 carry 沿用 v7。如果某视频的预算小于句子数，导致扫一眼没有完成，记 GLANCE_BUDGET_SHORTFALL，并报告受影响的视频与句子数。

### 2.2 新建 flex_uda/active/v8_slow.py：SlowAdapterV8(SlowAdapterV7)
所有方向（包括 Dance 互斥组）统一使用二元 BCE，删除列表 CE 和 exclusive_training_logits 路径。
训练池按 (q,t) 对构建，每对用 sampled_frames 展开到帧：
  - DIRECT_POS / DIRECT_NEG：pseudo_labels()（.85/.15，未提问为 −1，冻结轨迹为 −1，同 v7）。
  - EXCLUSIVE_NEG（仅互斥组）：轨迹 t 有 DIRECT_POS 的句子 q* 时，同组其它句子 q'（且 (q',t) 不是 DIRECT_POS）记为 0。
  - ABSENT_NEG：absent=True 的句子（v8 下必然有回答支持）的所有合格非冻结轨迹记为 0。low_tentative 和 PRIOR_BUDGET_EXHAUSTED 的句子不产生负例。
更新门槛：distinct 正对 ≥8、distinct 负对 ≥8（负对含派生负例），且正、负帧样本各 ≥32。否则返回 status='SKIP_INSUFFICIENT_BALANCE'，并记录各池的计数。
批次构成：batch 32 = 16 正 + 16 负。负样本取 8 个直接负例 + 8 个派生负例，某一池为空时由另一池补足；不放回抽样，池不足时放回。pos_weight 固定为 1（由均衡采样代替加权）。
记录每步实际的正负数及来源构成（DIRECT_NEG / EXCLUSIVE_NEG / ABSENT_NEG）。断言：每次优化更新时 16 正 16 负。
其余沿用 v7：rank 4、alpha 8、lr 2e-4、clip 5、每轮 50 步、AMP 回退。

### 2.3 新建 scripts/active_v8_run.py（复制 active_v7_run.py 后修改）
  - 用 PresenceEM 替换 OnlinePresenceEB。BeliefStateV8 的参数为 rho_hat=em.rho、pi_strata=em.pi.copy()。视频结束时依次调用 em.add_video(video, state) 和 em.fit(ds.values)。checkpoint 保存 em。
  - live_round 的阶段顺序：[GLANCE]、[POSITIVE_RECHECK]、[PRESENCE_COVER]、[其余]；“同一句同时只有一个问题在途”的规则保留。
  - ingest：presence_question=kind in ('GLANCE','PRESENCE','PRESENCE_COVER')。
  - 每轮结束（包括第 0 轮开始前）保存 videos/<v>/presence_round{r}.npz（state.presence_snapshot()），以及 em_video_start.json（ρ̂、π̂_k）。
  - 变体：MINUS_SLOW、FULL、SLOW_REPLAY；同时为 2.6 的消融留出 --ablation 参数。
  - SLOW_REPLAY（修复 R4）：
      * 依赖对应 pair 的 MINUS_SLOW_v8 封存结果（verify_predictions），不再依赖 v7。
      * 每个视频的 ρ̂/π̂_k 直接读取 MINUS 的 em_video_start.json，不运行 EM 更新（仍可计算一份影子 EM，只记录不使用）。
      * 每轮：ingest 固定回答后，apply_presence_snapshot(MINUS 第 r 轮的快照)。最终 mask 使用 MINUS 最后一轮的存在性快照。断言回放的存在性数组与 MINUS 逐字节一致，并记录。
      * 快校准（w、a、T、β）和身份后验允许随 LoRA 分数变化，这正是“慢分支”要测的路径。报告中写明：REPLAY−MINUS = 相同问答、相同存在性判定与 ρ̂/π̂，只改变学生的打分。
      * 保留 v7 的零 adapter 输入锚定（anchor_unupdated_replay）。
  - 其余收据、封存、teacher 记账字段沿用 v7，version 字段改为 'v8'。

### 2.4 复制并改名配套脚本
active_v7_{prepare,launch,preflight,validate,continue,snapshot,report,close,smoke}.py → active_v8_*。只改版本号、路径、新字段和新报告表；执行清单（manifest）的 version 为 'v8'，allowed_registered_jobs 按第 3 节登记。

### 2.5 tests/test_active_v8.py（全部通过后才能上 GPU）
  1. EM 恢复真值：合成 200 个视频，ρ∈{.1,1.0}，π=(.8,.3,.05)，s/c=.8；按 v8 的提问规则（依赖已观测数据）选择要问的轨迹。要求 EM 估计的 |ρ̂−ρ|<.05、|π̂_k−π_k|<.07；同一数据上的 v7 旧估计器对 π_0 的偏差 >.1（回归测试）。
  2. 没有任何回答的句子不改变 EM 统计量。
  3. 覆盖不全时，即使 p<.1 也不能判空；覆盖齐全且 p<.1 时判空并保持。
  4. 预算足够时，第 1 轮内每个句子至少有一个 GLANCE。
  5. 慢分支门槛：不满足 8/8/32/32 时跳过；每次更新 16 正 16 负；Dance 互斥组能产生 EXCLUSIVE_NEG 和 ABSENT_NEG；low_tentative 的句子不产生负例。
  6. 回放：用伪造的 MINUS 快照运行两个视频，每轮存在性数组逐字节一致，ρ̂/π̂ 一致。
  7. 生成进程打开 GT 的次数为 0（沿用 runtime_guard）。

### 2.6 消融开关（第 4 阶段才运行）
  - −EM：ρ=.5、π_k=.5 固定，不做 EM。
  - −GLANCE：关闭 GLANCE。
  - V7_ESTIMATOR：用 v7 估计器的数值替代 EM（直接复现 R1/R2）。
  - RANDOM_ACQ：同类问题内随机排序（seed 0），预算相同。
  - −DS：s/c 固定为 .8/.8。
  - −ROUTE：前两轮固定使用 CROP。
  - −PRES：不使用句子存在性变量（absent 永远为 False，不做 GLANCE/COVER/PRESENCE），只问 IDENTITY。

## 3. 执行顺序与优先级

阶段 0（只用 CPU，不调用教师，不用 GPU）：离线验证估计器
  0a. 单元测试全部通过。
  0b. 用 v7 MINUS_SLOW 六个方向的 questions.json、strata、ds.receipt，离线按视频流重放 PresenceEM，得到 ρ̂/π̂_k 轨迹，与 v7 的 eb_stream 并列。先确认 v7 估计器能被离线逐值复现，误差 <1e-9。
  0c. 独立诊断进程（只读封存后的 GT，结果不回流）：计算每个方向真值 ρ 和每档真值 π_k（有目标句中、该档已回答轨迹的 GT 多数口径正例率，以及该档全部合格轨迹的正例率，两种都报）。表格：v7 估计、v8 EM、GT。
  0d. 反事实判定：用 v7 的回答 + v8 的判定规则 + 0b 的逐视频 ρ̂/π̂，重新判定每个句子的存在性。统计六个方向的“有目标句被判空”“空句被正确判空”“low_tentative”的数量，并与 v7 的 presence_decisions 对照。注意 v7 的回答里没有 GLANCE，所以这只是近似结果，报告里写明。
  0e. KITTI 专项（C2、C5）：v7 中 296/12 个错误判空句的逐句分解，包括每条已回答轨迹的档位、看法、回答、GT、使用的 π̂、LLR。再把 π 替换为 GT π_k 后重算，统计有多少句会翻转回来。这用于把错误归因到 π̂ 偏高。同时报告 FRAME_BOX 的 DS 特异度与 GT 特异度之差（C2：.76 对 .47），本轮只报告，不修复。
  阶段 0 只有两种情况停下：EM 不收敛，或 0b 无法复现 v7。其余情况报告后直接进入阶段 1，不设数值门槛。

阶段 1：冒烟测试
  C1、C2 的 MINUS_SLOW_v8 各跑前 2 个视频（--pilot-videos 2）。检查：GLANCE 覆盖率、预算缺口、EM 收据、presence_round 快照、GT_open_attempts=0。

阶段 2：主表（最高优先级）
  MINUS_SLOW_v8，C1–C6，ρ=.3。v7 中 MINUS 每个方向只需约 0.05–0.43 小时，六个方向可以并行。

阶段 3：与阶段 2 并行，不阻塞阶段 2
  FULL_v8：C1、C4、C6。
  SLOW_REPLAY_v8：C1、C4，各自在对应 pair 的 MINUS_SLOW_v8 封存后启动。

阶段 4：只有阶段 2 通过第 4 节判据才运行
  4a. 预算曲线 ρ∈{.1, 1.0}：MINUS_SLOW_v8、YT_PRIOR、UNIFORM_K，六个方向，预算逐格匹配（ρ=.3 的对照沿用已有结果）。
  4b. 2.6 的全部消融，六个方向，ρ=.3（不含 LoRA，成本低）。
  阶段 2 不通过时不运行阶段 4，只提交归因报告后停下。

## 4. 预注册判据（运行前写入 manifest，不得事后修改）
  主判据：MINUS_SLOW_v8 在 ≥5/6 个方向同时高于 YT_PRIOR（ρ=.3）和 UNIFORM_K；并且 C1 ≥ 35.724、C6 ≥ 34.469（即距 v5 FULL 37.724 / 36.469 在 2 以内）。
  对照数值：YT_PRIOR = C1 22.711、C2 24.286、C3 58.601、C4 46.099、C5 27.588、C6 20.822；UNIFORM_K = 24.108、24.413、40.804、40.804、25.674、24.108。
  慢分支判据（只决定论文写法，不影响主判据）：REPLAY_v8 − MINUS_v8 在 C1、C4 的点估计都 ≥0，且至少一个的 95% CI 下界 >0，则写进主方法；否则放入补充材料。FULL−MINUS 单独报告，并说明它还包含了选题反馈的变化。

## 5. 报告内容（results/presence_active/redesign_v8/REPORT.md）
  1. 六方向主表：NATIVE、PRES_TOPK、FULL_v5、MINUS_v7、MINUS_v8、FULL_v8、REPLAY_v8、YT_PRIOR、UNIFORM_K，HOTA 不取整。附 v8−v7 的成对 bootstrap（Dance/Sports 用视频聚类，2000 次，seed 0；KITTI 只给点估计）。
  2. 存在性表：沿用 v7 第 3 节的列，新增 decision_source（EVIDENCE / PRIOR_BUDGET_EXHAUSTED）、GLANCE 缺口、low_tentative 数量。
  3. 估计器表：逐方向的 ρ̂ 和 π̂_k 终值、v7 估计器影子值、GT。附 ρ̂/π̂_k 随视频序号变化的曲线，GT 用虚线（GT 仅在作图时读取）。
  4. 慢分支表：每轮池的大小（DIRECT_POS / DIRECT_NEG / EXCLUSIVE_NEG / ABSENT_NEG）、跳过次数及原因、实际优化步数、空句误报行数（重点看 C1 是否回到 v7 MINUS 的 18,477 附近或更低）。
  5. 回放完整性：逐轮存在性逐字节一致的断言结果；与 v7 回放的对比。
  6. 行为分析数据（论文用，先导出 CSV 和图）：
     - 每轮各类问题的占比（GLANCE / PRESENCE / PRESENCE_COVER / IDENTITY / RECHECK）；
     - 每个视频的提问数与新确认对数，随视频序号的变化；
     - 被问对在提问前的后验分布直方图，与全部对的分布对比；
     - GLANCE 第一眼的命中率：答 Yes 的比例，按 GT 有目标句 / 空句分开统计。
  7. 成本：教师新调用、缓存命中、GPU 小时、壁钟时间。
  8. 阶段 0 的全部表格（0b–0e）。

## 6. 提交与停止
  - 每完成一个阶段就 commit 并 push，commit message 写明阶段号和关键数字。
  - 阶段 2 和 3 完成后提交一份 REPORT 并停下等待指示；阶段 4 只有在判据通过时才自动继续。
  - 不运行 v5 剩余的 80 项，不扩展 v6，不运行本提示之外的任何任务。

## Amendments（实验前修订）

### A1 — 2026-11-06，用户确认的合成边界用例

原因：截断边界与合成真值冲突；本修订在任何 v8 实验结果产生之前登记。

仅修改 2.5 第 1 项中 rho=1.0 的断言为 `rho_hat >= 0.95 - 1e-9`：EM 必须收敛到截断上限。这是参数截断区间 [0.05,0.95] 的设计结果，不是估计误差。rho=0.1 仍要求 `abs(rho_hat-rho)<0.05`；每档 pi 仍要求绝对误差 <0.07；同一数据上旧 v7 估计器的 pi_0 偏差仍须 >0.1。其余验收条件、算法、数据与执行顺序均不变。

