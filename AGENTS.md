# AGENTS.md

## 项目与用户约定

1. 我们不是一篇安全攻防论文。只做结果可信所必需的校验，禁止过度防御、为基本不可能的情况堆测试、把主要精力花在hash/格式/CI而不做研究。
2. 每次任务先确认根目录 `/data3/liuyeqiang/FlexHook-TTA`，remote `LYQ1107/TTA-RMOT`。这是从A800恢复的TTA-RMOT，不是LocateMOT或GroundJEV；其它项目命令不能照抄。
3. 用户最新明确范围：沿用官方固定候选框和Track ID，只研究跨域指代适配模块。McByte默认检测器有SportsMOT预训练，但具体候选生成权重未找回；不冒称整套前端source-only，也不擅自重建检测器。
4. 适配只用目标图片、任务查询、冻结候选和冻结Qwen指导。目标人工GT禁止参与训练、选题及自动阈值/检查点选择，只在方案固定、所有本研究输出封存后，由隔离评估/诊断进程读取。
5. 用户没有未使用新划分。C4、C1等全是开发数据；多轮开发后的区间只能是描述性证据，不能写成独立盲测或泛化定理。当前视频转导，过去/当前视频前缀可用，未来回答不可用；不称逐帧在线。
6. 用户科研Goal的验收目标是人类视觉方式与经典算法启发的跨域RMOT，对充分、同信息/同问答/同预算的强普通教师蒸馏有可靠改善，各保留模块贡献明确，并有配套真实成本。不能因几次测试、某个方向涨分、队列/报告完成而结项；负结果如实保留，不保证必然正结果。
7. 只用GPU0/1/2/3，分配前做一次预检，不影响其它任务；不使用4..9。NAS `/data3/liuyeqiang/nas/A800` 只读，历史results不覆盖。大数据/权重留NAS，新实验写本地 `a100_runs`；部分已完成数据和新CPU研究由原路径软链到本机 `/var/tmp/liuyeqiang-tta-rmot-a100-20261122`，不是NAS写入。
8. 不启动旧80队列、废弃v9或重复封存v8P66。A100恢复及旧三阶段见docs/A100_RESTORE_20261120.md和docs/A100_三阶段实验完整报告_20261120.txt。
9. 用户已授权正常Git推送。必要代码、TXT、精选聚合收据进入Git，原始QA/特征/预测/权重不入Git；快照和可达对象必须均<100,000,000字节，发布前运行tools/check_publication_size.py。不得force push或虚构PR。原研究分支已推送到8bc165f。此前轻量分支codex/visual-classical-evidence-20261122保留4b8e2c0；新证据使其正常追加的可达体积100,534,621超过上限，因此本轮从f915e494保留完整当前代码树另建codex/rmot-three-principles-six-settings-20261122，正常推送新分支，不改旧分支历史、不checkout旧文件。废弃v7逐查询CSV只移出Git索引，NAS原文不变，恢复入口docs/legacy_archives/redesign_v7_per_query_decisions.txt。当前HEAD以git为准，新增结果每次重核体积。

## 当前工作状态（2026-11-22）

- 三点的当前操作化名称以scope JSON和对象—关系理论TXT为准：面向覆盖和可辨识性的主动观察；分离稳定干扰与关系变化的对比证据学习；对象—状态分解下的关系化指代。不是把DS/Huber/图传播作为三项创新。新增teacher-only理论适用条件审计复现原参数，C2/C5激励组分别1275/5343、1269/4771；八前缀都有Huber二次区外残差，不可直接套用二次闭式公式或把Gram叫Fisher信息。未读新研究质量输出、未改R3/450，两个独立数学检查通过。新审计并不完成Goal。
- 最新用户指令覆盖本文件下方的两方向结项/停止加实验说明：科研Goal重新ACTIVE，范围是C1..C6全部六个方向，要求把有根据的视觉本能/经典算法机制组合验证，并超过充分训练、同信息/同预算的普通教师蒸馏。旧C1/C4验收和31项检查仅为两方向中间证据，不代表六方向完成。不得靠削弱蒸馏或只对Native的大涨幅替代相对普通蒸馏的贡献。六方向的新统一实验已获授权，不是盲目重跑旧v8P66。
- 用户最新术语和结构更正：称“六个跨域实验设置”，以三个候选创新点组成同一框架，在六个设置验证；不是六个独立创新或按设置分别设计方法。最新进一步纠正：学习经典算法的思想，为论文提供理论支撑，不是“用经典算法校准判断”。候选主线为覆盖与疑点共同驱动的主动观察、相关性受控的证据推理、视觉与语言关系约束下的结构化适配。必须从任务问题构造机制、明确假设与数学性质、再做模块与交互实证；不把DS/PAVA/图传播名字或经典定理当成自己的创新，不把代理损失性质写成HOTA必涨。详见docs/TTA_RMOT_视觉本能与经典思想_理论设计与实证边界_20261122.txt，里面新统一目标尚未实现，现有运行只是相关机制证据。模块贡献在适用场景和交互中验证，不要求每个模块在每个设置都单独上涨。已有C1..C6标识、历史文件名和实验数据不改，正在执行的统一矩阵继续。
- 对象—关系新候选已从KITTI封存后表示诊断推进到实际teacher-only原型：同对象/同视角组内中心化消去固定加性偏置、非负Huber凸标量拟合、按当前框位置迁移关系。详见docs/TTA_RMOT_从经典思想到对象关系机制_理论与实验登记_20261122.txt。246条左右查询是保守适用门控，另四设置回退，不虚称普适。新450输出研究控制器2230493等待RGB阶段释放后运行，普通逐帧对手获得同1087D信息/所有query/所有问答并额外444fit；均值/置乱/参照/数量对照封存后GT。初次外部启动未脱离短命命令进程组，在写状态前退出；只修launcher为新session重启，数值/登记不变，现有GPU工作不动。唯一观察器以supervisor/research_goal_active_continuation.json为准（当前29635/PID2254534；观察器进度升级仅重启自身，未重启任何实验），后续以supervisor/research_goal_active_continuation.json和registry为准，不另启并发监视。理论性质不是质量已证，I1新观察设计尚未完成，Goal ACTIVE。
- I1已补计划器原型flex_uda/active/observation_design.py：先付费取对象参照，再在固定动作集上权衡覆盖和对象内最大状态激励，不把共享参照差分当独立信息累加。只对固定非负权重/已付参照/单位逻辑成本的条件代理证明单调次模与贪心近似，不称全流程/真实时间/HOTA保证。3项数学与采集约束检查和C1/C2/C4真实输入计划检查已过；整套新策略的教师采集与质量对照尚未完成。C2示例同23调用：旧21对象，新18对象+5对比，明确是覆盖与激励权衡，不虚报全面覆盖优势；C4示例旧1对象/4调用，新4对象/4调用。旧固定QA研究不验证新选题。不能把原型检查、教师预测误差下降或排队完成写作三创新/六设置Goal通过。
- 新观察全148视频x4策略清单已完成，各策略55,019逻辑请求，同预算/逐query配额/公共route；GT0/未用旧QA概率。真实输入小测试也已完成：GPU2 Qwen3-VL-2B新59前向/0cache，CROP/CLIP/FRAME_BOX实际均走通，权重SHA9a13380...与旧C2实测流一致；teacherload444.669s，服务含IO31.630s，内部forward3.237s，worker507.047s。它是四策略各29请求的共享并集采集，学生特征预付，不能当单策略部署或600秒全启动成功。原始数据本地、聚合收据在docs/a100_six_direction_20261122/observation_teacher_physical_pilot.json。
- 新输入可观测性审计（无GT/无teacher概率）发现CLIP无显式全局位置/帧偏移；KITTI46条保守明确位置+MOTION查询，旧C2/C5只有9/4条只CLIP，不能称整个KITTI主因。旧CLIP1797/1595条全部>=2不同帧，不支持重复单帧猜测。新公共route会使运动复合查询更多只CLIP，先验证保留状态的输入再全面采集。独立state_teacher_inputs.py保留相同裁剪图像，ORIGINAL/LAST/POSITION/TIME/STATE五控制，3CPU检查+真实12问题60输入构造通过。真实GPU配对已PASS：GPU1新60/0cache、同视觉SHA及400视觉token、同冻结9a13380...，GT0；LAST已改5/12、POSITION/TIME/STATE相对LAST仅各多改1/12，均Yes→No，未证明信息独立贡献或准确率，不推广事后控制，不立即全面采集新提示。需要安慰剂/任务兼容反事实区分措辞/长度与语义信息。继续使用唯一登记观察器，不改R3/RGB/450。几何不等于世界运动，STATE若使用则普通KD同样STATE答案。
- 无GT能力审计确认旧ln3有界关系原型无法翻转base<.25或>.75：C2约31.08..34.04%、C5约26.52..28.31%的适用观测受此限制；对应有教师跨帧翻转的query-track中仅32..41/322、74..84/438受限。不能说它锁死大多数，也不能把受限视为GT错误。原450实验保持不变，新增普通逐帧baseline仍必要；若失败可研究重新估计稳定成分的部分线性模型，而非按HOTA事后调大cap。尚未实现该新模型，不冒称3创新已成。
- 先读代码，简短登记假设，然后实现/实验；维护research_log.md。详尽持续报告是docs/TTA_RMOT_视觉本能与经典算法_持续研究综合实证_20261122.txt，生成器scripts/a100_visual_classical_research_synthesis.py。
- 公开目录docs/a100_visual_classical_20261121的REPRODUCE.txt、各preregistration、analysis、parent_statistics、statistical_reproduction是证据入口。原始输出a100_runs/visual_classical_20261121；不能把工程执行PASS当科学验收。
- 自然768D无源残差普通头很强：未加权KD均值C4 67.082937/C1 32.074099；固定局部图+PAVA为67.638744/35.731536。旧47分和新67分条件不同，不把差值归为某创新贡献。
- 未加权最终配方四份实际冷启动完成，4526真实问答/0缓存，特征、回答、头、预测与质量研究0差。seed0 C4 66.964470→67.661313、C1 31.658042→35.696180。单方法wall C4 KD3761.595/J4437.921s，C1 KD3146.512/J3616.019s。J承担真实RGB成本，不称提速/免费/全预算最优。
- 强普通置信度加权同样使用w=1+3|2p-1|。420fit充分训练，均4000步、KL<=.005。加权普通均值C4 67.102356/C1 32.259906；旧未加权J对它C4+.536388但parent+seedCI[-.010093,1.127430]跨0，不能换窄区间宣布胜出。
- SAME_WEIGHTED_HEAD78已完成并独立复算：共享完全同一加权头的J为C4 67.580033/C1 35.727657；对KD+.477677/+3.467750，C4CI[-.069102,1.060519]跨0，主验收失败。原有PAVA在C4有独立贡献，图在C1条件贡献明确，不声称每模块每域都显著。
- CONTRACT_GRAPH138已完成并复算：PRESUPPOSED全面断No图边、保留本地No，C4相对完整J-.384532，CI[-.670867,-.074881]为负；虽净救回一些查询但误报增加，不保留。CHECKLIST原样，未用域名选规则。
- 教师留轨迹局部/整体选择V1及ONE_SE V2都未晋级。C4区间跨0，C1局部图不及同总边权完全图；不把MEDIAN/TWO_SE等事后最高分对照偷偷升为主方法。已准备的对应额外GPU冷启动均未执行。加权J的额外冷启动也因主门槛失败未启动。
- CUE144、SEARCH84、OPTIONAL150均执行和算术复算完成，但科学主门槛未通过，均不保留。SEARCH296个新真实回答改善覆盖却未改善总HOTA；OPTIONAL门控没有独立正贡献，不晋级事后STRICT控制。
- 本轮全部实验、复现、实际收据均完成，31项最终实证检查全过：docs/a100_visual_classical_20261121/protocol_final_scientific_review.json；最小最终配方在minimal_retained_method_card.json，状态ACCEPTED_ON_EXISTING_DEV_DIRECTIONS。
- 三种子C4 67.151609→67.629222、C1 32.721677→38.429592；C4视觉对应对置乱+.561896、C1纯图边对同回填/PAVA+4.632923，两CI正；两域对简单老师回填也三seed正/CI正。C1排序仍不稳、C4并未救回所有query；全DEV且协议与域重合，非独立泛化证明。
- 四份0/1/2/3真实冷启已结束：4526新QA/0cache、140软CLIPfit充分4000步。PCD/先验/锚点/QA/教师概率和全部score/mask0差，1782个查询文件0差；末视频头权重/归一化0差。seed0 C4同协议基线66.884538→67.437980，wall3751.854→3909.972s；C1 32.603390→38.123396，3002.593→3147.103s。只对应本配方，不把时间套给三种子均值或宣称提速/全预算最优。
- 无运行中的本线程实验。控制器2150067已正常结束；唯一监视49896/PID2162819正常0结束，旧71095等待进程143退出并未重启GPU或重置计时。不要启动新监视/新实验或重跑已完成队列；继续工作应依据新用户指令及线程Goal工具状态。
- 所有失败和归因对照仍保留；11份未执行且已失败门槛的旧冷流原型源码移出tip，逐字保留Git8d6faa1及本机归档，真正实测源码未移除。执行核心精确源码在Gitaf46773及preregistration_protocol_cold_v1.json SHA列表。
- 当前综合TXT含全部HOTA/效应和历程，重复原始表指向专题TXT/JSON；--full-diagnostics输出本地单一完整逐查询计数版。旧coverage.json仅字符串外空白调整，全部token/值/类型保持，原文Gitaf46773/本机备份。
- 实证验收已通过，Root仅完成最后<100MB远端发布与线程Goal结项。此前历史日志中的ACTIVE表示当时进度，不代表还需添加新模块。

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
- parent_statistics是官方HOTA聚合充分计数，可用脚本NumPy重算；精确引用、float64无损字节编码不改变指标，不是上传特征/权重。新统计payload只压缩新公开产物，原始可读备份本地保留；压缩后重算全部受影响效应。为用户<100MB约束，可仅去掉旧JSON字符串外排版空白，保留全部token/结构/数值/类型并留原文；这不改变统计编码或运行输入。
- 最小方法卡仍PROVISIONAL、candidate_finally_accepted=false。按预登记科学门槛核验，不能自动让队列PASS关闭Goal。

此前完整执行备忘录归档本地analysis/AGENTS_full_before_continuation_index_20261122.md；既有已发布历史仍在Git8bc165f。研究结果索引保留于research_log.md和各阶段TXT/JSON。

2026-11-22 COVERAGE_SEARCH_V1启动，控制器2118268。这是直接处理未问到正确人的新证据实验：保留原COVER逐query候选顺序，先每query看一次，再按已观察No概率乘积/(query调用数+1)分配PRESUPPOSED剩余预算；CHECKLIST原均匀预算。不是校准存在性后验，不用GT/未请求回答。总1160不变，私有冻结COVER缓存仅broker使用，缺失QA真实GPU0..3推理；随后270强加权软/硬头拟合及84原始/数量对照全封存后GT。普通KD获得完全相同SEARCH回答，以拆选题贡献和后处理贡献。70视频无反馈原COVER问题集合差0，GT0。0..3分配由控制器内部预检，不重复agent级GPU轮询；仍由唯一阻塞监视93426等待。全部DEV，Goal ACTIVE。

2026-11-22 CUE_CONSENSUS144及294HOTA/336效应独立复算PASS，但主科学门槛失败，不保留：C4 67.547497，比完整共享加权J-.032536，CI[-.126173,.085614]跨0；比强加权KD+.445142，CI[-.091387,1.028821]跨0。三个种子相对完整J均无查询净救回（rescue/regress均0），却FP增加4990/5347/7190。比等强度弱化为负，比随机同量隔离的差区间跨0，不支持该选择性规则有效。C1按契约完全不变，不算新贡献。没有运行额外新冷流。COVERAGE_SEARCH新证据45视频已封存：总1160、864缓存、296真实新增Qwen回答/296替换位置，教师权重不变/GT0，教师load474.124s，质量准备总651.908s，不是完整冷部署成本；三种子270fit继续，唯一监视93426，Goal ACTIVE。

2026-11-22 OPTIONAL_TRANSFER_V1启动，控制器2128080，唯一监视93426不变。教师-only旧收据显示C4 36/45视频的局部/全局CV风险均不及中性0.5（C1 1/25），不是GT结论。新IDENTITY选项部署时严格保留强加权KD而非零边原始教师锚点；最佳图须比IDENTITY降低超过一倍配对query SE才用，否则原KD后接PAVA。SE只敏感性正则，不声称独立query置信度。严格3模型最小风险、被迫用图、固定全局及ANCHOR_OPTIONAL局部锚点回填控制全保留，36原始+114数量=150输出全封存后本组GT。70视频sanity旧风险/完整图/IDENTITY→CAL重放全部0差，GT0；教师风险选择C4 44IDENTITY/1LOCAL，C1 5IDENTITY/20GLOBAL，尚无HOTA结论。0新增教师/主干/头拟合，CPU两槽，源码数值冻结。SEARCH三种子原始输出已全封存，正在数量对照，GT尚未本组评分。Goal ACTIVE，不预先宣布获胜，不将两项研究收益相加。

2026-11-22 SEARCH84官评及156HOTA/188效应独立重算PASS，但主科研门槛全部失败，不保留：新KD66.982569对COVERKD67.102356 -.119787（seed0正、17/42负，CI跨0）；新J67.239436，对自身KD+.256867和旧KD+.137080均CI跨0，对更强旧FINAL+.087827也跨0；比旧J67.580033 -.340597、三种子均负。教师真实覆盖改善：未问正确candidate query122→113、未得正确Yes147→134（同QA路径，不把三神经种子算三次覆盖）；J相对旧J净救回13/8/4query，却FP增加34603/33914/31842。只增加正确query不足以赢HOTA。J比CAL低.172124，不能用比坏完全图+4.15包装为图优于强KD的证据。270fit均4000步达KL，非故意欠拟合。没有运行该失败配方额外冷流；OPTIONAL_TRANSFER150继续，Goal ACTIVE。

2026-11-22 OPTIONAL150及300HOTA/344效应独立重算PASS，但不晋级：C4 67.462401对均值KD+.360045、CI[.013207,.763715]正；C1 37.279581对均值+5.019675、CI[1.194417,9.645308]正。然而C4对强收敛FINAL仅+.310792、CI[-.063214,.762190]跨0；门控对被迫图C4+.084777区间跨0、C1-.266020区间上界近0，独立门控贡献未成立。不改用更高的STRICT控制冒称primary通过。C4相对原完整图canonical query净救回18/17/16，但canonical标签是全候选IoU.5一对一匹配，不等同语义真值或各方法实际TP/FP。定性图已确认遮挡、模糊与框定位问题；一个teacherYes的all-black候选实际IoU仅.405，不能把所有不匹配都算语义判断错。唯一监视93426正常0结束，所有已注册研究已终止成功但Goal继续ACTIVE；未启动失败门控配方新冷流。

2026-11-22 PROTOCOL_CALIBRATION114启动，控制器2144506，CPU两槽，0新教师/编码器/头拟合。双方统一强加权收敛FINAL，不以GT选检查点；PRESUPPOSED原KD、CHECKLIST原全局汇聚，再原PAVA。普通KD也有同协议信息的非空Bernoulli条件化基线。旧均值组件结果已知，属于开发后简化非独立盲测。70视频稠密/秩一求解、旧CAL/全局重放及Finalmask校验全部0差。门槛含两域胜强Final与协议对照、模块条件贡献。唯一旧93426已结束，须记录新监视；不启动第二个。Goal ACTIVE。

2026-11-22 协议/视觉校准114已通过预先固定质量门槛：C4 67.629222 vs强FINAL+.477613、CI[.165546,.838314]；C1 38.429592、+5.707914、CI[1.694525,10.591625]。C4PAVA、C1场景汇聚各有条件独立贡献；C1排序跨0、C4query净救回0/1/-3仍是限制。全DEV非盲测。新PROTOCOL_COLD_FINAL控制器2150067，四份独立0/1/2/3冷流，普通Final也给非空/清单协议，组合承担真实RGB和PAVA。旧64759正常0结束；新唯一监视待记录。尚未验收成本/一致性，Goal仍ACTIVE。仅5份从未执行且已失败门槛的旧冷流原型移出tip，逐字保留8d6faa1和/var/tmp归档；真正实测源码未移除。

2026-11-22 PURE_COMPONENTS66已完成，90HOTA/120效应复算PASS、原主配方/CAL0差、所有CountSelf0。纯图边C1+4.632923 CI[1.311040,8.745540]；真cue对置乱C4+.561896 CI[.250933,.913257]；对简单老师回填两域都三seed正/CI正。不要再加新方法或换更高控制，现有质量/模块归因门槛已满足，等待PROTOCOL_COLD_FINAL2150067四份实际运行与一致性/成本，以及<100MB发布。唯一监视71095/2149976继续。Goal ACTIVE；C1排序和视觉单独效应仍不稳定、所有数据DEV且协议/域重合。
