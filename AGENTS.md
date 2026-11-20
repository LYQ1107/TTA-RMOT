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

## 研究执行原则

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
