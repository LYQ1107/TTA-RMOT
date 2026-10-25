# TTA-RMOT

本仓库为 [LYQ1107/TTA-RMOT](https://github.com/LYQ1107/TTA-RMOT)，服务器工作目录
`/data2/user/FlexHook-TTA`。它与 LocateMOT / GroundJEV 是不同任务；不会从本库启动其训练流程。
继承的旧首页全文已[归档](docs/history/LOCATEMOT_INHERITED_README_20261025.md)，旧代码、研究记录及独立工作区均保留。

## 当前研究：FlexHook V2 → Dance Positive-Free Online TTA

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
- [执行协议](docs/PF_4GPU_EXECUTION_PLAN.md)与[研究记录](research_log.md)。

六组已全部完成，最优点估计为Identity HOTA8.8179（+0.5999），但视频级95%区间包含0、
AUROC下降且空查询误检上升，不能宣布稳定语义恢复。Hybrid8.3213未超过Identity。
最终报告及[静态科研图](results/positive_free/full/figures/scientific_summary.png)已生成并检查；
恢复索引位于`results/positive_free/full/manifests/full_recovery_index.json`。
接下来只完成已授权的presence-aware设计与有界原型，不自动启动第七种完整GPU实验。

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
