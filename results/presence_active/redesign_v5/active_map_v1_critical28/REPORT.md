# MAP v1：原矩阵累计28项关键实验

2026-11-04用户授权“先跑18项”；当前准备新增C2/C4/C5/C6的18项，ρ=0.3，配置不改。
本批未完成前不宣称28项已完成，更不宣称完整108项已完成。原40 GPU小时关卡保留，余80项不在本次授权内。

旧C1/C3十项及正负结果完整保留于[上一阶段报告](../active_map_v1_full/REPORT.md)，不重跑其GPU预测。
新方向完整封存后独立官方TrackEval；全部新增18项均上报，不按目标GT选方法、阈值或checkpoint。
Sports用v4全466及相同v4内容的418 ID敏感性，KITTI仅4视频点估计，Dance/Sports按视频聚类bootstrap。

本批实测外推25.584 GPU小时、累计28项外推35.812；实际费用、失败及恢复如实追加。
执行清单registered_jobs.json、原估算的只读副本BC_cost_estimate.json、预注册PREREG.md均在本目录。
新launcher与报告独立存放，原10项历史报告/清单/日志不覆盖。教师revision、数值参数和固定MOT不改。

## 阶段B/C已封存结果

新增18项尚未启动；启动信息将在预注册公开后写入execution_started.json。
