# TTA-RMOT

本仓库为 [LYQ1107/TTA-RMOT](https://github.com/LYQ1107/TTA-RMOT)，服务器工作目录
`/data2/user/FlexHook-TTA`。它与 LocateMOT / GroundJEV 是不同任务；不会从本库启动其训练流程。
继承的旧首页全文已[归档](docs/history/LOCATEMOT_INHERITED_README_20261025.md)，旧代码、研究记录及独立工作区均保留。

## 当前研究：P1 冻结视觉语义教师可靠性

新分支`research/v2dance-evidence-teacher-p1-20261026`。执行[预注册P1协议](docs/TEACHER_EXPERIMENT_PROTOCOL.md)：
冻结Qwen3-VL/安全复用YesTrack，比较整句、等预算多问法、分属性和证据验证，**不训练学生**。
[机制审计](docs/TEACHER_METHOD_AUDIT.md)、[数据/权重来源](docs/TEACHER_DATA_PROVENANCE.md)、
[预算公平性](docs/TEACHER_MODEL_AND_BUDGET_FAIRNESS.md)与[固定采样清单](results/teacher_probe/)分开保存。
T0权重已核验并实际运行；官方V2 LoRA未取得，按用户授权如实保留T1缺失，不以其它权重替代。
六组四对功能检查及源域203对校准已完成，POS/NEG门限仅由源开发集冻结为0.95/0.05。
完整25视频/425表达smoke六组每组1649对均已通过；完整probe正按四个整视频分片执行，每组仍为8381个固定配对。
教师无参数更新、无目标GT读取；全部预测封存后再独立评测，目前尚无新P1目标质量指标，不把工程通过当科研成功。

## 上一阶段：FlexHook V2 → Dance Offline UDA 诊断

按2026-10-26最新指令先做[25Val同集无标签适应诊断](docs/UDA_METHOD_IMPLEMENTATION.md)，
不将其称作40Train→Val UDA或未见视频泛化。目标GT不进入训练，四组单卡独立对照。
[当前进度与资产缺口](docs/UDA_PROGRESS.md)；40Train图像/表达已补齐，原协议同配置Train MOT仍缺失。

**25Val阶段四组已全部完成并停止**：各8973次LoRA更新，随后冻结checkpoint完整25视频/425表达官方评测。
HOTA：Frozen **8.2180**；Source Rehearsal **7.4352**；Mean Teacher **8.2494**；Weak MIL **6.9281**；Combined **7.7293**。
Teacher的数量匹配Frozen已达8.2479，其微小残差和视频区间不能支持可靠语义提升；Combined不优于较好单模块。
没有有效目标MIL/presence正负证据，不能把代码具备该loss称为有效目标弱监督已验证。
[完整科学报告](docs/UDA_FINAL_SCIENTIFIC_REPORT.md)、[官方指标及逐视频/表达明细](results/uda/)、
[完整预测压缩包](results/uda/full/archives/)、[checkpoint与分数恢复索引](results/uda/full/full_recovery_index.json)
及[完整运行/失败恢复日志](results/uda/full/logs/runtime_logs.tar.gz)已公开。
原40Train→25Val正式目标仍未完成，不自动扩展训练或按Val结果调参。

## 已完成研究：FlexHook V2 → Dance Positive-Free Online TTA

最新用户授权的[快速可恢复性诊断](docs/PF_RECOVERABILITY_DIAGNOSTIC_20261025.md)独立于已完成六组结果：
**已完成并停止**。完整425表达Count-Matched Frozen HOTA8.8206，Identity8.8179：
输出数量配合原Frozen排序几乎重现全部收益；这是依赖Identity数量的后验诊断，不是新可部署方法。
原模型presence有有限信号（离线AUROC0.7258/AP0.1994）；固定75帧冻结PCD保留少量可读出信息，
但GT诊断探针没有超过同样本Frozen head，也未发现独立可靠的无标签语义监督。
67项测试通过；[诊断分数与日志](results/diagnostics/pf_recoverability_20261025/)、
[425条表达完整诊断预测](results/diagnostics/pf_recoverability_20261025/archives/countmatched_425_predictions.tar.gz)
及[科研图](results/diagnostics/pf_recoverability_20261025/figures/recoverability_summary.png)均单独保留。
没有重跑六方法、源训练或新完整TTA；等待用户下一条明确指令。

固定官方 Refer-KITTI-V2 第二阶段 checkpoint、官方 Dance DETR+NeuralSORT 候选，
在全部25视频/425表达上检验相对排序与轨迹身份自监督。没有目标GT在线监督、额外语义教师或源端重训。
Source-Free 仅指第二阶段；固定跟踪器是 Dance 专用官方跟踪器，不能宣称整个系统无目标域监督。

- 匹配基线为完整211,085条MOT候选的 `ZERO_UPDATE`，HOTA **8.2180**。
  历史8.2060遗漏692条轨迹末观测，仅作旧协议对照，不能混算增益。
- 六组顺序：`ZERO_UPDATE`、`RELATIVE_ONLY`、`PF_RANK`、`PF_RANK_GUARD`、`PF_IDENTITY`、`PF_HYBRID`。
- 原配置：rank4 / LR5e-5 / update interval4 / seed0 / FP32；每episode重置学习状态，先预测后更新。
- 最多四张物理GPU。共享须用户明确授权并实际检查显存，优先进程少的卡；不终止其他任务。
- 预测完整封存后才由独立官方TrackEval和归因读取GT。本集已用于开发，不是独立盲测。

## 实验与复查入口

- [完整运行状态](docs/PF_FULL_EXPERIMENT_STATUS.md)：以这里记录的当前阶段为准，不因示例命令重启健康作业。
- [官方全量结果表](docs/PF_FULL_RESULTS.md)：不把未完成方法当作完成。
- [六方法最终科学报告](docs/PF_FINAL_SCIENTIFIC_REPORT_20261025.md)：全部25视频/425表达、官方指标、成本及完整归因。
- [错误与收益分析](docs/PF_FAILURE_ANALYSIS.md)：真实GT恢复、误检、排序与成本权衡。
- [Query-Present / Empty补充验收](docs/PF_SUPPLEMENTAL_ACCEPTANCE_20261025.md)。
- [视频聚类与不确定性](docs/PF_VIDEO_GROUP_ANALYSIS_20261025.md)：25个完整视频为单位，不把相关帧当独立样本。
- [精简机器可读结果](results/positive_free/full/)：官方summary/detailed CSV、逐表达/逐视频指标、审计、来源及离线诊断。
- [完整文本日志压缩包](results/positive_free/full/logs/runtime_logs.tar.gz)：138个准备/调度/worker/model/官方评测/原型日志，逐文件复核；不含图像、权重及大容量margin流。
- [执行协议](docs/PF_4GPU_EXECUTION_PLAN.md)与[研究记录](research_log.md)。

六组已全部完成，最优点估计为Identity HOTA8.8179（+0.5999），但视频级95%区间包含0、
AUROC下降且空查询误检上升，不能宣布稳定语义恢复。Hybrid8.3213未超过Identity。
最终报告及[静态科研图](results/positive_free/full/figures/scientific_summary.png)已生成并检查；
恢复索引位于`results/positive_free/full/manifests/full_recovery_index.json`。
已完成[presence-aware交互审计原型](docs/PF_QUERY_PRESENCE_PROTOTYPE_20261025.md)和真实GPU0/1表达/12帧验证。
它只消除加性偏置并报告UNKNOWN，尚无独立可靠的语义学习信号；0更新，不覆盖冻结预测。
58项测试通过，上轮六组及presence原型已结束；新诊断单独记录，不自动启动新的完整TTA训练。

## 代码、环境与大文件

- `flex_tta/`：FlexHook适应模块、因果episode、冻结特征复用、原子完成收据。
- `scripts/run_pf_full_suite.sh`：正式顺序执行；`run_pf_4gpu.sh`默认只打印计划，需要明确开关才启动全量。
- `scripts/audit_pf_full.py`、`summarize_pf_full.py`、`analyze_pf_full_offline.py`：封存、官方结果整理及隔离的离线归因。
- `scripts/analyze_pf_video_uncertainty.py`、`plot_pf_science.py`、`report_pf_full.py`：视频级分析与最终报告。
- 服务器Python：`/home/szgx/anaconda3/envs/dfot/bin/python`；数据优先放在`/data2/user/TTA-RMOT-assets/`。
- 原始预测、压缩margin、完整运行日志及恢复收据保留在
  `/data2/user/FlexHook-TTA/results/runs/pf_*_completebank_full_20261025*`。
- checkpoint、图像与大容量中间文件保留服务器/NAS；GitHub只保存必要代码、配置、精简记录与引用路径。

复现与恢复前请先核对实验状态、固定资产和原episode receipts；不要覆盖已完成结果。
