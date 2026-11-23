# MAP v1：新增18项关键实验，原矩阵累计28项

日期：2026-11-04。项目：LYQ1107/TTA-RMOT，工作区/data2/user/FlexHook-TTA。
用户明确答复“先跑18项”。本文件在新增推理/适应之前提交并推送。

## 授权与固定清单

这是既有108项矩阵中的下一批18项，不是新方法，不授权剩余80项。
保持原40 GPU小时关卡；本批按既有实测外推25.584 GPU小时，累计28项外推35.812小时。
共享资源、KITTI单位耗时尚未实测、载入和失败重试会造成偏差，报告真实计时，不把估算当实耗。

|方向|FULL|UNIFORM_K|YT_RANDOM|YT_PRIOR|MINUS_SLOW|新增数|
|---|---|---|---|---|---|---:|
|C2 Dance-best→KITTI-V2|是|是|是|是|原矩阵无此项|4|
|C4 Dance-best→SportsMOT|是|是|是|是|是|5|
|C5 LaMOT-best→KITTI-V2|是|是|是|是|原矩阵无此项|4|
|C6 LaMOT-best→Dance|是|是|是|是|是|5|

全部ρ=0.3。FULL先于同方向预算匹配YesTrack；两种YT按每视频FULL实际调用数精确匹配，
未覆盖对仍按用户已定义的Native回退。UNIFORM_K沿用原floor预算而非强行改成相同实际调用量。
C1/C3已完成10项只复用封存结果，不再运行GPU、不改其清单、报告、checkpoint和预测。
原参考提交9e46a524f1e95af414a20136d450aee98fa092e9，原结果目录active_map_v1_full保持封存。

## 方法、数据与隔离不变

- 继承已公开MAP v1：source_map_v1 fast、源锚先验、双类且教师支持证据、不足SKIP；slow LoRA/轮数/步数/采样/阈值/seed均不变。
- 使用同一官方源checkpoint和已封存Stage A源校准；不新增源训练，不根据C1/C3或新目标GT修改配置。
- 冻结Qwen3-VL-2B，revision 89644892e4d85e24eaac8bacfd4f463576704203，权重SHA256
  9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956；T1后端与视觉输入构造不改。
- 固定官方完整MOT候选，框/ID/帧号不改；不重做跟踪器、不加入新教师。只复用同输入的不可变教师缓存。
- 各方法独立初始化，跨视频状态只在同一方法/方向中按原顺序传递；不把C1/C3适应状态借给新方向。
- 推理/教师/适应目标GT打开尝试必须0，预测完整封存SHA后才在独立CPU评测中读GT。
- KITTI-V2沿用完整候选与已核验帧号规则；Dance全25视频/425表达；Sports主口径v4全45视频/466表达，
  敏感性为同一v4内容限制原418条ID，不切回有冲突的旧GT模板。
- 本轮仍为teacher-assisted、video-transductive；Val已有开发复用，不称严格帧因果或盲测。

## 执行、成本与失败恢复

- 最多2张共享物理GPU，沿用实际显存准入43008 MiB，优先进程少且显存够的卡，不停止其他任务。
- 单个独立session durable manager＋单阻塞观察命令；agent不另行轮询ps/tail/nvidia-smi。
- 失败保留日志与费用，按原episode/video检查点只恢复未完成单元；健康完成结果不重跑。
- 复用已修复的共享SQLite租约顺序，先发布自有batch再等待外部owner；缓存键/输出/原batch8不改。
- 新报告和清单放active_map_v1_critical28；新launcher日志独立目录，实际模型/特征继续使用已授权data1资产与data2链接。
- 分开实际新forward、缓存、committed等价forward、LoRA/重打分、worker wall及失败开销；共享准备不重复计费。

## 评测、判定与交付

官方TrackEval全部指标、行数、Empty/Present及固定候选完整覆盖；六个方向FULL分别与Native、PRES_TOPK和其所有关键对照比较。
Dance/Sports按视频聚类bootstrap2000次seed0；KITTI仅4视频，只报点估计，不伪造显著性。
区间仍条件于固定有序预测流，不覆盖重新排序视频后再适应及训练seed不确定性。
复用原行为/可靠性/恢复与新增FP诊断；最终belief不当作中间优化器实际用过的伪标签。
所有18项均上报，不按GT选checkpoint、预算点或方法，不预先保证正增益。

完成后累计28/108，独立报告汇总新18＋旧10，保留旧10报告；其余80项标未运行。
发布代码、正式指标、成本、全日志、原始教师响应和可恢复路径/SHA。大日志沿用原字节分片，不丢弃内容。
本批完成后停止新增GPU队列，报告实测及后续资源关卡；不自行扩展其他预算/消融或研发新方法。
