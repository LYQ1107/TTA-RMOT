# AGENTS.md

## 通用原则

1. 我们不是一篇安全攻防论文，你有权力进行校验，但是禁止禁止禁止过度防御。
2. 禁止为反复出现的基本不可能发生的 case 写防御。
3. 需要 rubric 的地方不要过度机械化。
4. 每次任务开始先确认项目身份：当前项目根目录必须是
   `/data3/liuyeqiang/FlexHook-TTA`（2026-11-20 从 A800 的
   `/data2/user/FlexHook-TTA` 恢复到 A100），Git remote 必须为 `LYQ1107/TTA-RMOT`。
   本项目与 `/data2/user/LocateMOT`、`/data2/user/LocateMOT_GroundJEV` 是不同任务；
   继承的旧文档与代码仅作来源追踪，不从本任务执行其训练命令。若收到的任务/附件指向
   LocateMOT / GroundJEV 或其它仓库，先提醒用户“这是 TTA-RMOT 任务吗”，避免复制错任务。
5. A100 恢复记录见 `docs/A100_RESTORE_20261120.md`。大型数据、checkpoint、历史结果
   保留在 `/data3/liuyeqiang/nas/A800`；`assets/` 和 `results/` 提供本机入口。
   历史脚本和封存 JSON 内的 `/dataN/user/` 是 A800 原路径，重启实验前按恢复清单适配，
   不直接执行旧启动队列。旧 v8P 的 66 项正式评测已封存；A100新实证见
   `docs/A100_三阶段实验完整报告_20261120.txt` 和 `docs/a100_three_stage_20261120/`。
   LoRA独立增量的区间跨0，固定重算数量的排序差为负；600秒冷启动没有教师反馈。
   不将这轮工程/实验完成写成LoRA语义贡献或主动选题最优已获证明。
6. 当前分支按用户要求以轻量历史发布，代码与重要实验记录须小于100,000,000字节。
   大型checkpoint、原始数据和历史归档留NAS；本轮新回答、完整预测、LoRA状态及日志
   保存在本地`a100_runs`，不进入Git；只精选小型收据、统计和图表进入docs。
   发布前运行 tools/check_publication_size.py；NAS清单和历史重建说明见
   docs/LIGHTWEIGHT_BRANCH_20261120.md，不恢复旧分片上传例外。
7. 2026-11-21机制诊断见docs/TTA_RMOT_视觉本能与经典算法_机制诊断完整报告_20261121.txt
   及docs/a100_mechanism_20261121/。已有数据全部属于开发数据，用户明确暂不做独立盲测。
   C4候选可恢复率98.193%，但有119/466查询未问到正确轨迹；最终稳定片段输出翻转0，
   不因原分数抖动盲加HMM。固定问答下DS系统级增益约8.8 HOTA，但点概率质量不一致；
   100位置视角干预没有稳定正路由收益，不将整套系统官评当成全面视角政策验证。
   Sports45片段属于13原视频；主区间按原视频聚类，旧LoRA联合增益此口径亦跨0。
   点诊断C4实际仅3个评估原视频、C5仅2个，不造置信区间；后续抽样先分原视频配额。
8. 持续研究的新方案与完整证据见
   `docs/TTA_RMOT_无标签跨域方案_完整验证与贡献结论_20261121.txt` 和
   `docs/a100_visual_classical_20261121/REPRODUCE.txt`。保留强教师KD小头、
   同证据视觉图及三观察感知执行；源主干/教师冻结，按视频前缀适配，当前视频转导。
   C4三种子图对原始软KD平均+0.756 HOTA；同概率边界后+0.528，但种子敏感性区间跨0。
   全C4真实全帧与关键观察均3508新问题/0缓存；学生前向7686.8秒降至1858.4秒，
   固定问答PCD/RGB最大差均0，但源先验/门控改变，不能宣称严格无损。
   图的数量/排序效应随条件而变；C1、强均值基线、独立新域和600秒冷启动没有全面获胜证明。
   CLIP_KD中的CLIP是教师概率裁剪，不是新视觉模型或CLIP时序视角。
   动态FIELD选题和BMA/协议保护为已封存负结果，不能当作最终有效模块。
   成本以已提供候选框/ID为前提；区分共享组件分摊与独立单方法Popen-to-exit，
   不将上游检测/物理跟踪的未测成本抹掉。

## 研究执行原则

当前状态修正：用户否决上一轮科研目标结项，持续Goal为ACTIVE。上一轮TXT仅为阶段证据。
普通KD高分核验已完成：C4直接教师44.570、当前视频KD68.341、历史+当前69.050；
C4/C1前缀重新拟合均0变化，C4独立KD/图各466查询绕过缓存官评差均0。
高分不等于图贡献；新三观察锚点对齐与同预算5%/10%/20%研究已完成404组官评。
10%预算均值KD+图相对均值KD：C1 +2.510，原视频+种子CI[0.743,4.769]；
C4 +0.608，敏感性CI[-0.028,1.269]仍跨0。C1固定KD数量差约0且区间跨0，
不能将主要数量校准收益包装为普遍排序突破。5% C4图不及硬KD400步。
三观察anchor头比POOLED软KD低1.510/2.092，外观衰减时序亦不及恒定连接，
不保留该新机制；物理串ID不等于衣着语义变化。详见最新稀疏证据阶段TXT。
2026-11-22开始相对场景关系开发实验：同10%问答下普通KD也获得相同RGB/位置特征，
比较MASKED同维度控制、完整/无位置/无图/同总权重完全图/打乱图；另查C1稀疏图特异性。
先封存全部新预测及数量反事实再GT。新阶段仍是开发迭代，尚未证明关系模块有效。
10%预算新鲜C4双分支实测已在GPU3启动，C1双分支等待关系生成释放GPU0；
新代码显式选择V2-best→Dance的源模型/源数量头、PNG与互斥，默认C4/0.3保持。
每方向内均值KD/图共享真正新源/教师流，外部双分支wall与组件分账分别报，
不冒充独立单方法冷重复，也不把历史精度配新时间。后台任务不要agent级轮询。
官方McByte默认检测器使用SportsMOT预训练，但当前候选文件的具体生成权重日志未找到；
用户已明确选择“沿用官方固定候选，研究跨域指代适配模块”。该共享前端为合法任务输入，
不再擅自扩展为重建完全source-only检测器；也不把上游性能当作我方贡献。
后续模块保留候选框和Track ID，只改变指代选择；适配继续禁止目标域人工标签。
必须继续用户完整科研目标，不能因C4单点涨幅、少数实验、队列完成或生成报告再次结项。

1. 先阅读代码 -> 写简短研究计划 -> 开始实现和实验，不要长时间停留在文档规划。
2. 仅做保证数值结果可信所必需的 sanity checks；不要把主要精力用于单元测试
   覆盖率、hash、格式化、CI、异常输入防御等。
3. 维护 `research_log.md`，简洁记录：实验假设、失败现象、原因判断、修改内容、
   结果变化、是否保留。

## Long-running experiment policy

For long training or evaluation jobs, never perform agent-level polling.

Use exactly one blocking shell command to wait for completion. Internal shell
sleep and process checks are allowed, but do not repeatedly return to the model
to run ps, tail, wc, ls, date, or nvidia-smi.

After the blocking command returns:

1. Validate completion and output integrity.
2. If the task failed, identify the first actionable root cause.
3. Apply the smallest justified fix.
5. Run a targeted regression test.
6. Resume from the latest valid checkpoint or unfinished unit.

Never hide failures, skip samples, change metrics, change seeds, weaken
acceptance criteria, or silently alter the experimental protocol.
