# MAP v1 完整视频续跑：主方法与关键对照

日期：2026-11-03。项目：LYQ1107/TTA-RMOT，method/presence-active-20261102。
用户本轮要求继续完整目标，撤销此前“小规模 pilot 后停止”，不撤销原40 GPU小时成本关卡。

## 不变的科学问题与方法

完整目标仍为三域六方向的存在性—身份解耦、主动教师与快慢适应。
本阶段不是新校准算法：严格沿用已公开 MAP v1（9ba080e9 预注册、78b197ad 实现、3d53952f 首视频报告）。
源锚单位先验、双类且教师支持的 fast 证据、证据不足 SKIP、原 slow LoRA、种子、源 checkpoint、
Qwen3-VL revision、教师三视图、预算和轮数全部不改。不得按目标指标调整阈值或选择 checkpoint。
原失稳校准和小 pilot 的全部记录保持不动。

## 预先固定的本阶段队列

1. C1 V2→Dance、C3 V2→SportsMOT 的 FULL，ρ=0.3，分别覆盖全部25/45视频、425/466表达。
2. 每个方向依次完成 MINUS_SLOW、UNIFORM_K、YESTRACK_BUDGET_RANDOM、YESTRACK_BUDGET_PRIOR，ρ=0.3。
3. 共10项。YesTrack预算逐视频精确匹配 FULL 实际调用，未覆盖对退回 Native；UNIFORM_K 保持原预算协议，报告实际调用差异。
4. FULL 从 active_map_v1 中 stream_000.pt 的 next_video=1 恢复，首视频不重跑，保留 DS、LoRA、优化器和随机数状态。
5. 对照使用独立输出与各自初始状态；缓存只复用同输入同教师结果，不借用 FULL 的适应状态。

该队列是 MAIN_AND_CRITICAL_ONLY，不等于完整108项矩阵。其余组合、预算、消融仍标 NOT_RUN，不能删去后声称完整目标完成。
先按真实 MAP pilot 重新估算108项及本队列成本，封存费用清单与队列后才启动。
完整矩阵若仍超过40 GPU小时，本阶段结束报告实测/外推成本，扩大范围须按原关卡取得决定。
不因早期 HOTA 好坏在本阶段增删方法，也不以 loss/梯度代替科研验收。

## 数据、评测与资源

- 预测生成、教师和适应的目标 GT 打开次数必须0；预测完整封存 SHA 后，独立 CPU 进程才可打开 GT。
- 复用固定官方候选、原框/ID/帧号与完整末尾覆盖。Sports主评测用v4全部466，敏感性用相同v4内容的原418 ID。
- 官方 TrackEval：HOTA/DetA/AssA/IDF1/IDSW、行数、Query-Empty行、Present HOTA。
- FULL 相对 Native、v5 PRES_TOPK 和全部本阶段对照，视频聚类 bootstrap 2000次、seed0；不把帧当独立样本。
- 报告逐轮问题/证据/更新、DS与GT事后对照、实际新调用/缓存/等价独立调用、LoRA/重打分/worker wall成本。
- 保留“未问的高置信伪标签、纯负 slow 更新、两个候选No推整句不存在、DS相关性”等已知局限，不据GT临时改算法。
- 最多2张经 select_gpu 准入的共享GPU，优先进程少且显存足够的卡；不停止其他项目。大文件仍放已授权data1目录。
- 使用单个阻塞协调器等待，逐方法封存后可独立CPU评测，不做agent级进程轮询。

## 交付与停止边界

本目录独立 REPORT、完整10项表、bootstrap、行为/可靠性/成本、预测和恢复路径/SHA、日志。
旧 active_map_v1/REPORT.md 与历史结果不改。本阶段完成后推送真实结果，依据40小时关卡决定剩余矩阵，不自动开展新方法。
