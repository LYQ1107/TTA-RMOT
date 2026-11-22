# AGENTS.md

## 项目与用户约定

1. 我们不是一篇安全攻防论文。只做结果可信所必需的校验，禁止过度防御、为基本不可能的情况堆测试、把主要精力花在hash/格式/CI而不做研究。
2. 每次任务先确认根目录 `/data3/liuyeqiang/FlexHook-TTA`，remote `LYQ1107/TTA-RMOT`。这是从A800恢复的TTA-RMOT，不是LocateMOT或GroundJEV；其它项目命令不能照抄。
3. 用户最新明确范围：沿用官方固定候选框和Track ID，只研究跨域指代适配模块。McByte默认检测器有SportsMOT预训练，但具体候选生成权重未找回；不冒称整套前端source-only，也不擅自重建检测器。
4. 适配只用目标图片、任务查询、冻结候选和冻结Qwen指导。目标人工GT禁止参与训练、选题及自动阈值/检查点选择，只在方案固定、所有本研究输出封存后，由隔离评估/诊断进程读取。
5. 用户没有未使用新划分。C4、C1等全是开发数据；多轮开发后的区间只能是描述性证据，不能写成独立盲测或泛化定理。当前视频转导，过去/当前视频前缀可用，未来回答不可用；不称逐帧在线。
6. 用户科研Goal仍ACTIVE。目标是人类视觉方式与经典算法启发的跨域RMOT，对充分、同信息/同问答/同预算的强普通教师蒸馏有可靠改善，各保留模块贡献明确，并有配套真实成本。不能因几次测试、某个方向涨分、队列/报告完成而结项；负结果如实保留，不保证必然正结果。
7. 只用GPU0/1/2/3，分配前做一次预检，不影响其它任务；不使用4..9。NAS `/data3/liuyeqiang/nas/A800` 只读，历史results不覆盖。大数据/权重留NAS，新实验写本地 `a100_runs`；部分已完成数据和新CPU研究由原路径软链到本机 `/var/tmp/liuyeqiang-tta-rmot-a100-20261122`，不是NAS写入。
8. 不启动旧80队列、废弃v9或重复封存v8P66。A100恢复及旧三阶段见docs/A100_RESTORE_20261120.md和docs/A100_三阶段实验完整报告_20261120.txt。
9. 用户已授权正常Git推送。必要代码、TXT、精选聚合收据进入Git，原始QA/特征/预测/权重不入Git；快照和可达对象必须均<100,000,000字节，发布前运行tools/check_publication_size.py。不得force push或虚构PR。原研究分支已推送到8bc165f。本轮使用从f915e494建立的轻量发布分支codex/visual-classical-evidence-20261122，旧分支保留、不强推；当前HEAD以git为准，新增结果每次重核体积。

## 当前工作状态（2026-11-22）

- 先读代码，简短登记假设，然后实现/实验；维护research_log.md。详尽持续报告是docs/TTA_RMOT_视觉本能与经典算法_持续研究综合实证_20261122.txt，生成器scripts/a100_visual_classical_research_synthesis.py。
- 公开目录docs/a100_visual_classical_20261121的REPRODUCE.txt、各preregistration、analysis、parent_statistics、statistical_reproduction是证据入口。原始输出a100_runs/visual_classical_20261121；不能把工程执行PASS当科学验收。
- 自然768D无源残差普通头很强：未加权KD均值C4 67.082937/C1 32.074099；固定局部图+PAVA为67.638744/35.731536。旧47分和新67分条件不同，不把差值归为某创新贡献。
- 未加权最终配方四份实际冷启动完成，4526真实问答/0缓存，特征、回答、头、预测与质量研究0差。seed0 C4 66.964470→67.661313、C1 31.658042→35.696180。单方法wall C4 KD3761.595/J4437.921s，C1 KD3146.512/J3616.019s。J承担真实RGB成本，不称提速/免费/全预算最优。
- 强普通置信度加权同样使用w=1+3|2p-1|。420fit充分训练，均4000步、KL<=.005。加权普通均值C4 67.102356/C1 32.259906；旧未加权J对它C4+.536388但parent+seedCI[-.010093,1.127430]跨0，不能换窄区间宣布胜出。
- SAME_WEIGHTED_HEAD78已完成并独立复算：共享完全同一加权头的J为C4 67.580033/C1 35.727657；对KD+.477677/+3.467750，C4CI[-.069102,1.060519]跨0，主验收失败。原有PAVA在C4有独立贡献，图在C1条件贡献明确，不声称每模块每域都显著。
- CONTRACT_GRAPH138已完成并复算：PRESUPPOSED全面断No图边、保留本地No，C4相对完整J-.384532，CI[-.670867,-.074881]为负；虽净救回一些查询但误报增加，不保留。CHECKLIST原样，未用域名选规则。
- 教师留轨迹局部/整体选择V1及ONE_SE V2都未晋级。C4区间跨0，C1局部图不及同总边权完全图；不把MEDIAN/TWO_SE等事后最高分对照偷偷升为主方法。已准备的对应额外GPU冷启动均未执行。加权J的额外冷启动也因主门槛失败未启动。
- CUE144、SEARCH84、OPTIONAL150均执行和算术复算完成，但科学主门槛未通过，均不保留。SEARCH296个新真实回答改善覆盖却未改善总HOTA；OPTIONAL门控没有独立正贡献，不晋级事后STRICT控制。
- PROTOCOL_CALIBRATION114已通过预定质量及强基线门槛：C4 67.629222对共享FINAL+.477613、CI[.165546,.838314]；C1 38.429592、+5.707914、CI[1.694525,10.591625]。普通同协议对照也通过。C4两种数量投影均小幅正，C1跨0；C4 canonical query净救回0/1/-3，未解决所有漏查。全DEV，协议与域在现有数据重合，非独立泛化验证。
- 当前PROTOCOL_COLD_FINAL控制器2150067，四个独立新实际进程0/1/2/3，普通强Final给相同非空/清单协议，组合承担RGB/全局汇聚/PAVA真实成本；4526新QA/0cache预期，全部4封存后GT。代码在preregistration_protocol_cold_v1.json冻结；不要修改运行中的数值辅助文件。
- 唯一阻塞监视session71095、observer2149976；旧64759已正常0结束。动态registry和状态见a100_runs/visual_classical_20261121/supervisor/active_research_jobs_20261122.json及research_goal_active_continuation.json。接续当前监视，不启动第二个。

## 长任务规则

For long training or evaluation jobs, never perform agent-level polling.
Use exactly one blocking shell command to wait for completion. Internal shell
sleep and process checks are allowed, but do not repeatedly return to the model
to run ps, tail, wc, ls, date, or nvidia-smi.

After the blocking command returns: validate completion/integrity; on failure
identify the first actionable root cause, apply the smallest justified fix,
run a targeted check, resume the last valid checkpoint or unfinished unit.
Never hide failures, skip samples, change metrics/seeds, weaken acceptance, or
silently alter an experimental protocol. Do not modify running numerical helpers.

## 结果归因与发布

- 旧LoRA在CountMatched/RECALC剥离后无稳定纯权重增益，不等于所有LoRA普遍无效。600秒冷启动零教师调用，不支持主动政策全预算最优。
- C4可恢复候选98.193%、旧119查询未问到正确轨迹；稳定片段最终输出翻转0，不据原打分抖动盲加HMM。C1物理轨迹串人须另查，不能靠平滑掩盖。
- C1多轮改善主要含数量/基率效应；双向CountMatched有负值/跨0，不能包装普遍语义排序突破。教师Yes正匹配不是纯教师语义准确率，GT诊断率不能拿来给教师加权。
- 全部历史强软/硬、RGB、SEM、零HSV头及失败FIELD/BMA/保护/时序/锚点均保留。CLIP_KD中的CLIP是概率裁剪，不是新视觉模型。
- 质量回放0新前向不等于部署0成本；若保留配方变了，须配套新实际执行，不拼旧时间和新精度。固定前端生成成本未测，OS/NAS缓存未清理，共享主机负载如实披露。
- parent_statistics是官方HOTA聚合充分计数，可用脚本NumPy重算；精确引用、float64无损字节编码不改变指标，不是上传特征/权重。只压缩新公开产物，原始可读备份本地保留；压缩后重算全部受影响效应。
- 最小方法卡仍PROVISIONAL、candidate_finally_accepted=false。按预登记科学门槛核验，不能自动让队列PASS关闭Goal。

此前完整执行备忘录归档本地analysis/AGENTS_full_before_continuation_index_20261122.md；既有已发布历史仍在Git8bc165f。研究结果索引保留于research_log.md和各阶段TXT/JSON。

2026-11-22 COVERAGE_SEARCH_V1启动，控制器2118268。这是直接处理未问到正确人的新证据实验：保留原COVER逐query候选顺序，先每query看一次，再按已观察No概率乘积/(query调用数+1)分配PRESUPPOSED剩余预算；CHECKLIST原均匀预算。不是校准存在性后验，不用GT/未请求回答。总1160不变，私有冻结COVER缓存仅broker使用，缺失QA真实GPU0..3推理；随后270强加权软/硬头拟合及84原始/数量对照全封存后GT。普通KD获得完全相同SEARCH回答，以拆选题贡献和后处理贡献。70视频无反馈原COVER问题集合差0，GT0。0..3分配由控制器内部预检，不重复agent级GPU轮询；仍由唯一阻塞监视93426等待。全部DEV，Goal ACTIVE。

2026-11-22 CUE_CONSENSUS144及294HOTA/336效应独立复算PASS，但主科学门槛失败，不保留：C4 67.547497，比完整共享加权J-.032536，CI[-.126173,.085614]跨0；比强加权KD+.445142，CI[-.091387,1.028821]跨0。三个种子相对完整J均无查询净救回（rescue/regress均0），却FP增加4990/5347/7190。比等强度弱化为负，比随机同量隔离的差区间跨0，不支持该选择性规则有效。C1按契约完全不变，不算新贡献。没有运行额外新冷流。COVERAGE_SEARCH新证据45视频已封存：总1160、864缓存、296真实新增Qwen回答/296替换位置，教师权重不变/GT0，教师load474.124s，质量准备总651.908s，不是完整冷部署成本；三种子270fit继续，唯一监视93426，Goal ACTIVE。

2026-11-22 OPTIONAL_TRANSFER_V1启动，控制器2128080，唯一监视93426不变。教师-only旧收据显示C4 36/45视频的局部/全局CV风险均不及中性0.5（C1 1/25），不是GT结论。新IDENTITY选项部署时严格保留强加权KD而非零边原始教师锚点；最佳图须比IDENTITY降低超过一倍配对query SE才用，否则原KD后接PAVA。SE只敏感性正则，不声称独立query置信度。严格3模型最小风险、被迫用图、固定全局及ANCHOR_OPTIONAL局部锚点回填控制全保留，36原始+114数量=150输出全封存后本组GT。70视频sanity旧风险/完整图/IDENTITY→CAL重放全部0差，GT0；教师风险选择C4 44IDENTITY/1LOCAL，C1 5IDENTITY/20GLOBAL，尚无HOTA结论。0新增教师/主干/头拟合，CPU两槽，源码数值冻结。SEARCH三种子原始输出已全封存，正在数量对照，GT尚未本组评分。Goal ACTIVE，不预先宣布获胜，不将两项研究收益相加。

2026-11-22 SEARCH84官评及156HOTA/188效应独立重算PASS，但主科研门槛全部失败，不保留：新KD66.982569对COVERKD67.102356 -.119787（seed0正、17/42负，CI跨0）；新J67.239436，对自身KD+.256867和旧KD+.137080均CI跨0，对更强旧FINAL+.087827也跨0；比旧J67.580033 -.340597、三种子均负。教师真实覆盖改善：未问正确candidate query122→113、未得正确Yes147→134（同QA路径，不把三神经种子算三次覆盖）；J相对旧J净救回13/8/4query，却FP增加34603/33914/31842。只增加正确query不足以赢HOTA。J比CAL低.172124，不能用比坏完全图+4.15包装为图优于强KD的证据。270fit均4000步达KL，非故意欠拟合。没有运行该失败配方额外冷流；OPTIONAL_TRANSFER150继续，Goal ACTIVE。

2026-11-22 OPTIONAL150及300HOTA/344效应独立重算PASS，但不晋级：C4 67.462401对均值KD+.360045、CI[.013207,.763715]正；C1 37.279581对均值+5.019675、CI[1.194417,9.645308]正。然而C4对强收敛FINAL仅+.310792、CI[-.063214,.762190]跨0；门控对被迫图C4+.084777区间跨0、C1-.266020区间上界近0，独立门控贡献未成立。不改用更高的STRICT控制冒称primary通过。C4相对原完整图canonical query净救回18/17/16，但canonical标签是全候选IoU.5一对一匹配，不等同语义真值或各方法实际TP/FP。定性图已确认遮挡、模糊与框定位问题；一个teacherYes的all-black候选实际IoU仅.405，不能把所有不匹配都算语义判断错。唯一监视93426正常0结束，所有已注册研究已终止成功但Goal继续ACTIVE；未启动失败门控配方新冷流。

2026-11-22 PROTOCOL_CALIBRATION114启动，控制器2144506，CPU两槽，0新教师/编码器/头拟合。双方统一强加权收敛FINAL，不以GT选检查点；PRESUPPOSED原KD、CHECKLIST原全局汇聚，再原PAVA。普通KD也有同协议信息的非空Bernoulli条件化基线。旧均值组件结果已知，属于开发后简化非独立盲测。70视频稠密/秩一求解、旧CAL/全局重放及Finalmask校验全部0差。门槛含两域胜强Final与协议对照、模块条件贡献。唯一旧93426已结束，须记录新监视；不启动第二个。Goal ACTIVE。

2026-11-22 协议/视觉校准114已通过预先固定质量门槛：C4 67.629222 vs强FINAL+.477613、CI[.165546,.838314]；C1 38.429592、+5.707914、CI[1.694525,10.591625]。C4PAVA、C1场景汇聚各有条件独立贡献；C1排序跨0、C4query净救回0/1/-3仍是限制。全DEV非盲测。新PROTOCOL_COLD_FINAL控制器2150067，四份独立0/1/2/3冷流，普通Final也给非空/清单协议，组合承担真实RGB和PAVA。旧64759正常0结束；新唯一监视待记录。尚未验收成本/一致性，Goal仍ACTIVE。仅5份从未执行且已失败门槛的旧冷流原型移出tip，逐字保留8d6faa1和/var/tmp归档；真正实测源码未移除。
