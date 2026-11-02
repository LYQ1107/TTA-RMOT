# 轨迹级排序可靠性诊断

【ORACLE / GT，仅诊断】本轮只读取封存官方完整 margin 与 GT，并重放旧 CROP 答案以估计模拟噪声率。没有新模型/教师调用、GPU 推理或训练。所有聚合使用整条轨迹，包括未来观测；不是因果在线方法，不是完整 RMOT HOTA，也不是可部署的 GT-free 选择器。

## 1. 预注册判定

总判定：**轨迹聚合不能解决排序问题**。0/6 同时通过 T1/T2，4/6 被固定规则归为‘轨迹级没有明显改善’。

| combo | target | model | common_sequences | FRAME_AUROC | TRACK_MEAN_AUROC | AUROC_delta | T1 | TRACK_prefix_F1 | T2 | joint_pass |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Dance | V2-best | 30 | 0.568000 | 0.601012 | 0.033012 | 轨迹级没有明显改善 | 0.976226 | 前缀约束可接受 | 否 |
| C2 | KITTI-V2 | Dance-best | 846 | 0.548924 | 0.594732 | 0.045809 | 轨迹级没有明显改善 | 0.392472 | 前缀约束代价过大 | 否 |
| C3 | SportsMOT | V2-best | 431 | 0.672532 | 0.723785 | 0.051253 | 轨迹级更好但仍不够可靠 | 0.806387 | 前缀约束可接受 | 否 |
| C4 | SportsMOT | Dance-best | 431 | 0.600691 | 0.600025 | -0.000666 | 轨迹级没有明显改善 | 0.748699 | 前缀约束代价过大 | 否 |
| C5 | KITTI-V2 | LaMOT-best | 846 | 0.679710 | 0.694955 | 0.015244 | 轨迹级没有明显改善 | 0.482497 | 前缀约束代价过大 | 否 |
| C6 | Dance | LaMOT-best | 30 | 0.573667 | 0.675037 | 0.101370 | 轨迹级更好但仍不够可靠 | 0.975783 | 前缀约束可接受 | 否 |


T1 的点估计严格用三粒度共同有定义 q，FRAME 先逐 q 平均；不把所有帧当独立样本。其他聚合只完整报告，不替换主聚合 MEAN 或改判定。T2 按原文使用全部 q，因此必须同时看下面的有目标/有正轨迹拆分。

时变轨迹（严格 0.1 < 匹配比例 < 0.9；分母为全部合格 q–轨迹对，非唯一物理轨迹）：

| target | q_track_pairs | time_varying_pairs | fraction | annotation |
|---|---|---|---|---|
| KITTI-V2 | 69854 | 3399 | 0.048659 |  |
| SportsMOT | 11738 | 186 | 0.015846 |  |
| Dance | 11152 | 141 | 0.012643 |  |


同一目标下的时变标签不依赖模型，按组合重复报告不会得到不同占比。原阈值只对上述 ALL 比例应用，不事后换条件分母。

| combo | target | model | q_track_pairs | time_varying_pairs | fraction | annotation |
|---|---|---|---|---|---|---|
| C1 | Dance | V2-best | 11152 | 141 | 0.012643 | 未定义 |
| C2 | KITTI-V2 | Dance-best | 69854 | 3399 | 0.048659 | 未定义 |
| C3 | SportsMOT | V2-best | 11738 | 186 | 0.015846 | 未定义 |
| C4 | SportsMOT | Dance-best | 11738 | 186 | 0.015846 | 未定义 |
| C5 | KITTI-V2 | LaMOT-best | 69854 | 3399 | 0.048659 | 未定义 |
| C6 | Dance | LaMOT-best | 11152 | 141 | 0.012643 | 未定义 |


三个目标均未超过原定20%门槛，未触发“该目标需要片段级而非轨迹级的判断”标注；这不是排除少量时变目标的证据。

| target | sequences | GT_present | GT_empty | Y_positive_sequences | GT_present_but_Y_zero | common_AUROC_q |
|---|---|---|---|---|---|---|
| Dance | 425 | 40 | 385 | 40 | 0 | 30 |
| KITTI-V2 | 862 | 862 | 0 | 846 | 16 | 846 |
| SportsMOT | 466 | 466 | 0 | 465 | 1 | 431 |


### 1.1 需要防止的误读

FRAME AUROC、SEGMENT AUROC、TRACK AUROC 的对象和多数标签不同；共同 q 控制了序列取样，但不能使三个对象定义变成同一任务。TRACK Y 会把不足50%匹配的轨迹归负，因此 GT-present 也可能没有正轨迹。前缀 F1 是相对本粒度标签的上限，不是覆盖完整 GT 的 RMOT 召回。

Dance 有385/425条空序列；即使所有有目标序列失败，ALL 前缀 F1 仍至少385/425。因此高 ALL T2 分数不能单独证明有目标轨迹可被可靠排序。完美校验者的二分也不等于前缀上限：前者只见搜索路径上的 Y，后者事后扫描所有 k 并使用 GT 选最优 k。

## 2. 三种粒度的 AUROC、全部聚合与对象数

ALL_DEFINED_FRAMES 是所有有双类帧的原始平均；ALL_DEFINED_Q 是各粒度自身有定义 q 的平均；COMMON_Q 是 FRAME/SEGMENT/TRACK 都有定义的同一批 q。主 T1 只能比较 COMMON_Q，未定义单元不补0.5。RAW/COMMON 的分母均完整列出，原始逐单元 AUROC 在机器表。

| target | model | granularity | aggregation | scope | sequences | common_sequences | defined_units | undefined_units | mean_AUROC |
|---|---|---|---|---|---|---|---|---|---|
| Dance | Dance-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 425 | 30 | 26982 | 406654 | 0.604001 |
| Dance | Dance-best | FRAME | MARGIN | ALL_DEFINED_Q | 425 | 30 | 40 | 385 | 0.593879 |
| Dance | Dance-best | SEGMENT | MEAN | ALL_DEFINED_Q | 425 | 30 | 31 | 394 | 0.617759 |
| Dance | Dance-best | TRACK | MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.641301 |
| Dance | Dance-best | TRACK | MEDIAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.653092 |
| Dance | Dance-best | TRACK | TOP25 | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.679408 |
| Dance | Dance-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.625301 |
| Dance | Dance-best | FRAME | MARGIN | COMMON_Q | 425 | 30 | 30 | 0 | 0.615834 |
| Dance | Dance-best | SEGMENT | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.634042 |
| Dance | Dance-best | TRACK | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.641301 |
| Dance | Dance-best | TRACK | MEDIAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.653092 |
| Dance | Dance-best | TRACK | TOP25 | COMMON_Q | 425 | 30 | 30 | 0 | 0.679408 |
| Dance | Dance-best | TRACK | LOGIT_MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.625301 |
| Dance | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 425 | 30 | 26982 | 406654 | 0.590232 |
| Dance | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_Q | 425 | 30 | 40 | 385 | 0.561714 |
| Dance | LaMOT-best | SEGMENT | MEAN | ALL_DEFINED_Q | 425 | 30 | 31 | 394 | 0.600771 |
| Dance | LaMOT-best | TRACK | MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.675037 |
| Dance | LaMOT-best | TRACK | MEDIAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.684731 |
| Dance | LaMOT-best | TRACK | TOP25 | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.702880 |
| Dance | LaMOT-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.674975 |
| Dance | LaMOT-best | FRAME | MARGIN | COMMON_Q | 425 | 30 | 30 | 0 | 0.573667 |
| Dance | LaMOT-best | SEGMENT | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.619446 |
| Dance | LaMOT-best | TRACK | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.675037 |
| Dance | LaMOT-best | TRACK | MEDIAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.684731 |
| Dance | LaMOT-best | TRACK | TOP25 | COMMON_Q | 425 | 30 | 30 | 0 | 0.702880 |
| Dance | LaMOT-best | TRACK | LOGIT_MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.674975 |
| Dance | V2-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 425 | 30 | 26982 | 406654 | 0.557218 |
| Dance | V2-best | FRAME | MARGIN | ALL_DEFINED_Q | 425 | 30 | 40 | 385 | 0.516404 |
| Dance | V2-best | SEGMENT | MEAN | ALL_DEFINED_Q | 425 | 30 | 31 | 394 | 0.573453 |
| Dance | V2-best | TRACK | MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.601012 |
| Dance | V2-best | TRACK | MEDIAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.602097 |
| Dance | V2-best | TRACK | TOP25 | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.611518 |
| Dance | V2-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 425 | 30 | 30 | 395 | 0.610782 |
| Dance | V2-best | FRAME | MARGIN | COMMON_Q | 425 | 30 | 30 | 0 | 0.568000 |
| Dance | V2-best | SEGMENT | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.586545 |
| Dance | V2-best | TRACK | MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.601012 |
| Dance | V2-best | TRACK | MEDIAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.602097 |
| Dance | V2-best | TRACK | TOP25 | COMMON_Q | 425 | 30 | 30 | 0 | 0.611518 |
| Dance | V2-best | TRACK | LOGIT_MEAN | COMMON_Q | 425 | 30 | 30 | 0 | 0.610782 |
| KITTI-V2 | Dance-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 862 | 846 | 189912 | 256965 | 0.515833 |
| KITTI-V2 | Dance-best | FRAME | MARGIN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.551858 |
| KITTI-V2 | Dance-best | SEGMENT | MEAN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.573037 |
| KITTI-V2 | Dance-best | TRACK | MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.594732 |
| KITTI-V2 | Dance-best | TRACK | MEDIAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.591608 |
| KITTI-V2 | Dance-best | TRACK | TOP25 | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.618212 |
| KITTI-V2 | Dance-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.611878 |
| KITTI-V2 | Dance-best | FRAME | MARGIN | COMMON_Q | 862 | 846 | 846 | 0 | 0.548924 |
| KITTI-V2 | Dance-best | SEGMENT | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.570378 |
| KITTI-V2 | Dance-best | TRACK | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.594732 |
| KITTI-V2 | Dance-best | TRACK | MEDIAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.591608 |
| KITTI-V2 | Dance-best | TRACK | TOP25 | COMMON_Q | 862 | 846 | 846 | 0 | 0.618212 |
| KITTI-V2 | Dance-best | TRACK | LOGIT_MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.611878 |
| KITTI-V2 | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 862 | 846 | 189912 | 256965 | 0.640482 |
| KITTI-V2 | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.679478 |
| KITTI-V2 | LaMOT-best | SEGMENT | MEAN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.692421 |
| KITTI-V2 | LaMOT-best | TRACK | MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.694955 |
| KITTI-V2 | LaMOT-best | TRACK | MEDIAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.691583 |
| KITTI-V2 | LaMOT-best | TRACK | TOP25 | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.714428 |
| KITTI-V2 | LaMOT-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.708779 |
| KITTI-V2 | LaMOT-best | FRAME | MARGIN | COMMON_Q | 862 | 846 | 846 | 0 | 0.679710 |
| KITTI-V2 | LaMOT-best | SEGMENT | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.693010 |
| KITTI-V2 | LaMOT-best | TRACK | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.694955 |
| KITTI-V2 | LaMOT-best | TRACK | MEDIAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.691583 |
| KITTI-V2 | LaMOT-best | TRACK | TOP25 | COMMON_Q | 862 | 846 | 846 | 0 | 0.714428 |
| KITTI-V2 | LaMOT-best | TRACK | LOGIT_MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.708779 |
| KITTI-V2 | V2-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 862 | 846 | 189912 | 256965 | 0.839947 |
| KITTI-V2 | V2-best | FRAME | MARGIN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.882284 |
| KITTI-V2 | V2-best | SEGMENT | MEAN | ALL_DEFINED_Q | 862 | 846 | 855 | 7 | 0.883822 |
| KITTI-V2 | V2-best | TRACK | MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.869759 |
| KITTI-V2 | V2-best | TRACK | MEDIAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.874041 |
| KITTI-V2 | V2-best | TRACK | TOP25 | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.879228 |
| KITTI-V2 | V2-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 862 | 846 | 846 | 16 | 0.865973 |
| KITTI-V2 | V2-best | FRAME | MARGIN | COMMON_Q | 862 | 846 | 846 | 0 | 0.881567 |
| KITTI-V2 | V2-best | SEGMENT | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.883538 |
| KITTI-V2 | V2-best | TRACK | MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.869759 |
| KITTI-V2 | V2-best | TRACK | MEDIAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.874041 |
| KITTI-V2 | V2-best | TRACK | TOP25 | COMMON_Q | 862 | 846 | 846 | 0 | 0.879228 |
| KITTI-V2 | V2-best | TRACK | LOGIT_MEAN | COMMON_Q | 862 | 846 | 846 | 0 | 0.865973 |
| SportsMOT | Dance-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 466 | 431 | 195101 | 71685 | 0.616674 |
| SportsMOT | Dance-best | FRAME | MARGIN | ALL_DEFINED_Q | 466 | 431 | 465 | 1 | 0.593853 |
| SportsMOT | Dance-best | SEGMENT | MEAN | ALL_DEFINED_Q | 466 | 431 | 433 | 33 | 0.597221 |
| SportsMOT | Dance-best | TRACK | MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.600025 |
| SportsMOT | Dance-best | TRACK | MEDIAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.596981 |
| SportsMOT | Dance-best | TRACK | TOP25 | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.619456 |
| SportsMOT | Dance-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.612324 |
| SportsMOT | Dance-best | FRAME | MARGIN | COMMON_Q | 466 | 431 | 431 | 0 | 0.600691 |
| SportsMOT | Dance-best | SEGMENT | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.597887 |
| SportsMOT | Dance-best | TRACK | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.600025 |
| SportsMOT | Dance-best | TRACK | MEDIAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.596981 |
| SportsMOT | Dance-best | TRACK | TOP25 | COMMON_Q | 466 | 431 | 431 | 0 | 0.619456 |
| SportsMOT | Dance-best | TRACK | LOGIT_MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.612324 |
| SportsMOT | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 466 | 431 | 195101 | 71685 | 0.863313 |
| SportsMOT | LaMOT-best | FRAME | MARGIN | ALL_DEFINED_Q | 466 | 431 | 465 | 1 | 0.833568 |
| SportsMOT | LaMOT-best | SEGMENT | MEAN | ALL_DEFINED_Q | 466 | 431 | 433 | 33 | 0.869675 |
| SportsMOT | LaMOT-best | TRACK | MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.879999 |
| SportsMOT | LaMOT-best | TRACK | MEDIAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.877022 |
| SportsMOT | LaMOT-best | TRACK | TOP25 | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.887616 |
| SportsMOT | LaMOT-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.862884 |
| SportsMOT | LaMOT-best | FRAME | MARGIN | COMMON_Q | 466 | 431 | 431 | 0 | 0.849429 |
| SportsMOT | LaMOT-best | SEGMENT | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.873553 |
| SportsMOT | LaMOT-best | TRACK | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.879999 |
| SportsMOT | LaMOT-best | TRACK | MEDIAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.877022 |
| SportsMOT | LaMOT-best | TRACK | TOP25 | COMMON_Q | 466 | 431 | 431 | 0 | 0.887616 |
| SportsMOT | LaMOT-best | TRACK | LOGIT_MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.862884 |
| SportsMOT | V2-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 466 | 431 | 195101 | 71685 | 0.681186 |
| SportsMOT | V2-best | FRAME | MARGIN | ALL_DEFINED_Q | 466 | 431 | 465 | 1 | 0.657666 |
| SportsMOT | V2-best | SEGMENT | MEAN | ALL_DEFINED_Q | 466 | 431 | 433 | 33 | 0.692955 |
| SportsMOT | V2-best | TRACK | MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.723785 |
| SportsMOT | V2-best | TRACK | MEDIAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.726787 |
| SportsMOT | V2-best | TRACK | TOP25 | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.748932 |
| SportsMOT | V2-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 466 | 431 | 431 | 35 | 0.718189 |
| SportsMOT | V2-best | FRAME | MARGIN | COMMON_Q | 466 | 431 | 431 | 0 | 0.672532 |
| SportsMOT | V2-best | SEGMENT | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.695283 |
| SportsMOT | V2-best | TRACK | MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.723785 |
| SportsMOT | V2-best | TRACK | MEDIAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.726787 |
| SportsMOT | V2-best | TRACK | TOP25 | COMMON_Q | 466 | 431 | 431 | 0 | 0.748932 |
| SportsMOT | V2-best | TRACK | LOGIT_MEAN | COMMON_Q | 466 | 431 | 431 | 0 | 0.718189 |


物理对象数分布（FRAME 按每个实际 tracker 帧一次，TRACK 按每视频一次；不被表达数量重复加权）：

| target | granularity | physical_units | mean | p50 | p90 | minimum | maximum | excluded_physical_tracks | excluded_physical_observations |
|---|---|---|---|---|---|---|---|---|---|
| KITTI-V2 | FRAME | 2018 | 5.362735 | 5.000000 | 10.000000 | 1 | 15 | 0 | 0 |
| KITTI-V2 | TRACK | 4 | 78.500000 | 54.500000 | 138.200000 | 32 | 173 | 0 | 0 |
| SportsMOT | FRAME | 26970 | 10.965184 | 10.000000 | 15.000000 | 1 | 21 | 32 | 48 |
| SportsMOT | TRACK | 45 | 25.066667 | 18.000000 | 45.600000 | 12 | 75 | 32 | 48 |
| Dance | FRAME | 25508 | 8.272973 | 7.000000 | 14.000000 | 1 | 22 | 36 | 58 |
| Dance | TRACK | 25 | 26.240000 | 18.000000 | 54.000000 | 5 | 93 | 36 | 58 |


逐视频过滤与分段信息：

| target | video | raw_tracks | retained_tracks | excluded_tracks | raw_observations | retained_observations | excluded_observations | frames | frames_with_zero_retained_candidates | segment_count | fps | segment_length_frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITTI-V2 | 0005 | 32 | 32 | 0 | 1064 | 1064 | 0 | 297 | 0 | 56 | 10 | 20 |
| KITTI-V2 | 0011 | 57 | 57 | 0 | 2576 | 2576 | 0 | 373 | 0 | 142 | 10 | 20 |
| KITTI-V2 | 0013 | 52 | 52 | 0 | 1018 | 1018 | 0 | 292 | 0 | 62 | 10 | 20 |
| KITTI-V2 | 0019 | 173 | 173 | 0 | 6164 | 6164 | 0 | 1056 | 0 | 376 | 10 | 20 |
| SportsMOT | v_00HRwkvvjtQ_c001 | 19 | 18 | 1 | 11168 | 11166 | 2 | 1162 | 0 | 225 | 25 | 50 |
| SportsMOT | v_00HRwkvvjtQ_c003 | 18 | 18 | 0 | 11176 | 11176 | 0 | 1161 | 0 | 224 | 25 | 50 |
| SportsMOT | v_00HRwkvvjtQ_c005 | 17 | 16 | 1 | 5345 | 5344 | 1 | 594 | 0 | 111 | 25 | 50 |
| SportsMOT | v_00HRwkvvjtQ_c007 | 15 | 14 | 1 | 6301 | 6299 | 2 | 707 | 0 | 126 | 25 | 50 |
| SportsMOT | v_00HRwkvvjtQ_c008 | 14 | 13 | 1 | 9486 | 9485 | 1 | 972 | 0 | 193 | 25 | 50 |
| SportsMOT | v_00HRwkvvjtQ_c011 | 17 | 17 | 0 | 6868 | 6868 | 0 | 734 | 0 | 142 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c004 | 16 | 16 | 0 | 6122 | 6122 | 0 | 542 | 0 | 126 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c005 | 15 | 15 | 0 | 3719 | 3719 | 0 | 349 | 0 | 75 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c006 | 19 | 18 | 1 | 3592 | 3591 | 1 | 346 | 0 | 74 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c007 | 13 | 13 | 0 | 2914 | 2914 | 0 | 271 | 0 | 57 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c008 | 14 | 14 | 0 | 2967 | 2967 | 0 | 266 | 0 | 60 | 25 | 50 |
| SportsMOT | v_0kUtTtmLaJA_c010 | 14 | 14 | 0 | 3011 | 3011 | 0 | 270 | 0 | 60 | 25 | 50 |
| SportsMOT | v_2QhNRucNC7E_c017 | 33 | 32 | 1 | 6750 | 6748 | 2 | 450 | 0 | 138 | 25 | 50 |
| SportsMOT | v_4-EmEtrturE_c009 | 12 | 12 | 0 | 2782 | 2782 | 0 | 250 | 0 | 56 | 25 | 50 |
| SportsMOT | v_4r8QL_wglzQ_c001 | 20 | 18 | 2 | 8168 | 8165 | 3 | 886 | 0 | 168 | 25 | 50 |
| SportsMOT | v_5ekaksddqrc_c001 | 15 | 15 | 0 | 5057 | 5057 | 0 | 550 | 0 | 104 | 25 | 50 |
| SportsMOT | v_5ekaksddqrc_c002 | 15 | 15 | 0 | 5710 | 5710 | 0 | 632 | 0 | 116 | 25 | 50 |
| SportsMOT | v_5ekaksddqrc_c003 | 16 | 16 | 0 | 11699 | 11699 | 0 | 1215 | 0 | 236 | 25 | 50 |
| SportsMOT | v_5ekaksddqrc_c004 | 16 | 16 | 0 | 7725 | 7725 | 0 | 824 | 0 | 154 | 25 | 50 |
| SportsMOT | v_5ekaksddqrc_c005 | 19 | 18 | 1 | 9284 | 9282 | 2 | 1000 | 0 | 190 | 25 | 50 |
| SportsMOT | v_9MHDmAMxO5I_c002 | 15 | 15 | 0 | 2579 | 2579 | 0 | 239 | 0 | 53 | 25 | 50 |
| SportsMOT | v_9MHDmAMxO5I_c003 | 15 | 15 | 0 | 2407 | 2407 | 0 | 226 | 0 | 51 | 25 | 50 |
| SportsMOT | v_9MHDmAMxO5I_c004 | 19 | 19 | 0 | 5081 | 5081 | 0 | 455 | 0 | 107 | 25 | 50 |
| SportsMOT | v_9MHDmAMxO5I_c006 | 20 | 20 | 0 | 2482 | 2482 | 0 | 269 | 0 | 49 | 25 | 50 |
| SportsMOT | v_9MHDmAMxO5I_c009 | 21 | 21 | 0 | 5420 | 5420 | 0 | 507 | 0 | 112 | 25 | 50 |
| SportsMOT | v_BgwzTUxJaeU_c008 | 18 | 17 | 1 | 4425 | 4424 | 1 | 500 | 0 | 92 | 25 | 50 |
| SportsMOT | v_BgwzTUxJaeU_c012 | 18 | 15 | 3 | 5722 | 5718 | 4 | 620 | 0 | 118 | 25 | 50 |
| SportsMOT | v_BgwzTUxJaeU_c014 | 16 | 16 | 0 | 9724 | 9724 | 0 | 1000 | 0 | 198 | 25 | 50 |
| SportsMOT | v_G-vNjfx1GGc_c004 | 51 | 46 | 5 | 8197 | 8187 | 10 | 675 | 0 | 170 | 25 | 50 |
| SportsMOT | v_G-vNjfx1GGc_c008 | 34 | 32 | 2 | 4857 | 4854 | 3 | 480 | 0 | 103 | 25 | 50 |
| SportsMOT | v_G-vNjfx1GGc_c600 | 42 | 42 | 0 | 5770 | 5770 | 0 | 500 | 0 | 120 | 25 | 50 |
| SportsMOT | v_G-vNjfx1GGc_c601 | 39 | 39 | 0 | 5804 | 5804 | 0 | 481 | 0 | 118 | 25 | 50 |
| SportsMOT | v_ITo3sCnpw_k_c007 | 53 | 53 | 0 | 9896 | 9896 | 0 | 701 | 0 | 205 | 25 | 50 |
| SportsMOT | v_ITo3sCnpw_k_c010 | 41 | 38 | 3 | 7813 | 7810 | 3 | 560 | 0 | 160 | 25 | 50 |
| SportsMOT | v_ITo3sCnpw_k_c011 | 40 | 40 | 0 | 10166 | 10166 | 0 | 776 | 0 | 208 | 25 | 50 |
| SportsMOT | v_ITo3sCnpw_k_c012 | 76 | 75 | 1 | 10753 | 10752 | 1 | 850 | 0 | 229 | 25 | 50 |
| SportsMOT | v_cC2mHWqMcjk_c007 | 20 | 19 | 1 | 7851 | 7849 | 2 | 681 | 0 | 163 | 25 | 50 |
| SportsMOT | v_cC2mHWqMcjk_c008 | 17 | 17 | 0 | 2761 | 2761 | 0 | 259 | 0 | 57 | 25 | 50 |
| SportsMOT | v_cC2mHWqMcjk_c009 | 14 | 14 | 0 | 4855 | 4855 | 0 | 425 | 0 | 102 | 25 | 50 |
| SportsMOT | v_dw7LOz17Omg_c053 | 45 | 45 | 0 | 8840 | 8840 | 0 | 550 | 0 | 184 | 25 | 50 |
| SportsMOT | v_dw7LOz17Omg_c067 | 33 | 33 | 0 | 7087 | 7087 | 0 | 450 | 0 | 145 | 25 | 50 |
| SportsMOT | v_i2_L4qquVg0_c006 | 52 | 48 | 4 | 11821 | 11815 | 6 | 888 | 0 | 241 | 25 | 50 |
| SportsMOT | v_i2_L4qquVg0_c007 | 26 | 26 | 0 | 5172 | 5172 | 0 | 451 | 0 | 107 | 25 | 50 |
| SportsMOT | v_i2_L4qquVg0_c009 | 32 | 32 | 0 | 8065 | 8065 | 0 | 488 | 0 | 166 | 25 | 50 |
| SportsMOT | v_i2_L4qquVg0_c010 | 66 | 63 | 3 | 8387 | 8383 | 4 | 758 | 0 | 180 | 25 | 50 |
| Dance | dancetrack0004 | 11 | 11 | 0 | 4371 | 4371 | 0 | 1203 | 0 | 114 | 20 | 40 |
| Dance | dancetrack0005 | 9 | 8 | 1 | 4576 | 4574 | 2 | 1203 | 0 | 118 | 20 | 40 |
| Dance | dancetrack0007 | 13 | 13 | 0 | 9111 | 9111 | 0 | 1203 | 0 | 236 | 20 | 40 |
| Dance | dancetrack0010 | 13 | 13 | 0 | 6740 | 6740 | 0 | 1203 | 0 | 174 | 20 | 40 |
| Dance | dancetrack0014 | 37 | 37 | 0 | 12289 | 12289 | 0 | 1203 | 0 | 329 | 20 | 40 |
| Dance | dancetrack0018 | 7 | 7 | 0 | 2507 | 2507 | 0 | 503 | 0 | 64 | 20 | 40 |
| Dance | dancetrack0019 | 22 | 22 | 0 | 15880 | 15880 | 0 | 2402 | 0 | 410 | 20 | 40 |
| Dance | dancetrack0025 | 22 | 21 | 1 | 6731 | 6730 | 1 | 803 | 0 | 181 | 20 | 40 |
| Dance | dancetrack0026 | 31 | 30 | 1 | 3691 | 3690 | 1 | 302 | 0 | 112 | 20 | 40 |
| Dance | dancetrack0030 | 7 | 7 | 0 | 7533 | 7533 | 0 | 1263 | 0 | 193 | 20 | 40 |
| Dance | dancetrack0034 | 37 | 36 | 1 | 10195 | 10194 | 1 | 923 | 0 | 285 | 20 | 40 |
| Dance | dancetrack0035 | 17 | 16 | 1 | 4927 | 4925 | 2 | 703 | 0 | 134 | 20 | 40 |
| Dance | dancetrack0041 | 72 | 70 | 2 | 14320 | 14316 | 4 | 1003 | 0 | 425 | 20 | 40 |
| Dance | dancetrack0043 | 15 | 15 | 0 | 1363 | 1363 | 0 | 183 | 0 | 43 | 20 | 40 |
| Dance | dancetrack0047 | 23 | 18 | 5 | 9064 | 9056 | 8 | 1203 | 0 | 238 | 20 | 40 |
| Dance | dancetrack0058 | 12 | 12 | 0 | 10798 | 10798 | 0 | 1601 | 0 | 278 | 20 | 40 |
| Dance | dancetrack0063 | 106 | 93 | 13 | 7930 | 7908 | 22 | 1000 | 0 | 260 | 20 | 40 |
| Dance | dancetrack0065 | 7 | 5 | 2 | 3486 | 3484 | 2 | 702 | 0 | 90 | 20 | 40 |
| Dance | dancetrack0073 | 31 | 31 | 0 | 6881 | 6881 | 0 | 703 | 0 | 184 | 20 | 40 |
| Dance | dancetrack0077 | 15 | 14 | 1 | 10780 | 10778 | 2 | 1203 | 0 | 275 | 20 | 40 |
| Dance | dancetrack0079 | 26 | 25 | 1 | 13997 | 13996 | 1 | 1202 | 0 | 355 | 20 | 40 |
| Dance | dancetrack0081 | 52 | 48 | 4 | 17148 | 17142 | 6 | 984 | 0 | 454 | 20 | 40 |
| Dance | dancetrack0090 | 39 | 39 | 0 | 11660 | 11660 | 0 | 1004 | 0 | 323 | 20 | 40 |
| Dance | dancetrack0094 | 61 | 58 | 3 | 10522 | 10516 | 6 | 603 | 0 | 300 | 20 | 40 |
| Dance | dancetrack0097 | 7 | 7 | 0 | 4585 | 4585 | 0 | 1203 | 0 | 118 | 20 | 40 |


## 3. FRAME / TRACK 前缀上限

FRAME 在 q 内对全部候选帧的最佳前缀 F1 平均，再跨 q 平均；TRACK 每 q 扫描全部 k。无正标签时空前缀 F1=1。GT_PRESENT/EMPTY 使用官方整序列 GT；Y_POSITIVE/ZERO 使用多数轨迹标签，不混用两种存在性。

| target | model | scope | granularity | sequences_in_scope | mean_prefix_F1 |
|---|---|---|---|---|---|
| Dance | Dance-best | ALL | FRAME | 425 | 0.981366 |
| Dance | Dance-best | ALL | TRACK | 425 | 0.973692 |
| Dance | Dance-best | GT_PRESENT | FRAME | 40 | 0.802017 |
| Dance | Dance-best | GT_PRESENT | TRACK | 40 | 0.720483 |
| Dance | Dance-best | GT_EMPTY | FRAME | 385 | 1.000000 |
| Dance | Dance-best | GT_EMPTY | TRACK | 385 | 1.000000 |
| Dance | Dance-best | Y_POSITIVE | FRAME | 40 | 0.802017 |
| Dance | Dance-best | Y_POSITIVE | TRACK | 40 | 0.720483 |
| Dance | Dance-best | Y_ZERO | FRAME | 385 | 1.000000 |
| Dance | Dance-best | Y_ZERO | TRACK | 385 | 1.000000 |
| Dance | Dance-best | COMMON_AUROC_Q | FRAME | 30 | 0.737319 |
| Dance | Dance-best | COMMON_AUROC_Q | TRACK | 30 | 0.627310 |
| Dance | LaMOT-best | ALL | FRAME | 425 | 0.981582 |
| Dance | LaMOT-best | ALL | TRACK | 425 | 0.975783 |
| Dance | LaMOT-best | GT_PRESENT | FRAME | 40 | 0.804310 |
| Dance | LaMOT-best | GT_PRESENT | TRACK | 40 | 0.742698 |
| Dance | LaMOT-best | GT_EMPTY | FRAME | 385 | 1.000000 |
| Dance | LaMOT-best | GT_EMPTY | TRACK | 385 | 1.000000 |
| Dance | LaMOT-best | Y_POSITIVE | FRAME | 40 | 0.804310 |
| Dance | LaMOT-best | Y_POSITIVE | TRACK | 40 | 0.742698 |
| Dance | LaMOT-best | Y_ZERO | FRAME | 385 | 1.000000 |
| Dance | LaMOT-best | Y_ZERO | TRACK | 385 | 1.000000 |
| Dance | LaMOT-best | COMMON_AUROC_Q | FRAME | 30 | 0.740350 |
| Dance | LaMOT-best | COMMON_AUROC_Q | TRACK | 30 | 0.656931 |
| Dance | V2-best | ALL | FRAME | 425 | 0.980552 |
| Dance | V2-best | ALL | TRACK | 425 | 0.976226 |
| Dance | V2-best | GT_PRESENT | FRAME | 40 | 0.793366 |
| Dance | V2-best | GT_PRESENT | TRACK | 40 | 0.747403 |
| Dance | V2-best | GT_EMPTY | FRAME | 385 | 1.000000 |
| Dance | V2-best | GT_EMPTY | TRACK | 385 | 1.000000 |
| Dance | V2-best | Y_POSITIVE | FRAME | 40 | 0.793366 |
| Dance | V2-best | Y_POSITIVE | TRACK | 40 | 0.747403 |
| Dance | V2-best | Y_ZERO | FRAME | 385 | 1.000000 |
| Dance | V2-best | Y_ZERO | TRACK | 385 | 1.000000 |
| Dance | V2-best | COMMON_AUROC_Q | FRAME | 30 | 0.725870 |
| Dance | V2-best | COMMON_AUROC_Q | TRACK | 30 | 0.663204 |
| KITTI-V2 | Dance-best | ALL | FRAME | 862 | 0.863726 |
| KITTI-V2 | Dance-best | ALL | TRACK | 862 | 0.392472 |
| KITTI-V2 | Dance-best | GT_PRESENT | FRAME | 862 | 0.863726 |
| KITTI-V2 | Dance-best | GT_PRESENT | TRACK | 862 | 0.392472 |
| KITTI-V2 | Dance-best | GT_EMPTY | FRAME | 0 | 未定义 |
| KITTI-V2 | Dance-best | GT_EMPTY | TRACK | 0 | 未定义 |
| KITTI-V2 | Dance-best | Y_POSITIVE | FRAME | 846 | 0.862006 |
| KITTI-V2 | Dance-best | Y_POSITIVE | TRACK | 846 | 0.380982 |
| KITTI-V2 | Dance-best | Y_ZERO | FRAME | 16 | 0.954660 |
| KITTI-V2 | Dance-best | Y_ZERO | TRACK | 16 | 1.000000 |
| KITTI-V2 | Dance-best | COMMON_AUROC_Q | FRAME | 846 | 0.862006 |
| KITTI-V2 | Dance-best | COMMON_AUROC_Q | TRACK | 846 | 0.380982 |
| KITTI-V2 | LaMOT-best | ALL | FRAME | 862 | 0.885191 |
| KITTI-V2 | LaMOT-best | ALL | TRACK | 862 | 0.482497 |
| KITTI-V2 | LaMOT-best | GT_PRESENT | FRAME | 862 | 0.885191 |
| KITTI-V2 | LaMOT-best | GT_PRESENT | TRACK | 862 | 0.482497 |
| KITTI-V2 | LaMOT-best | GT_EMPTY | FRAME | 0 | 未定义 |
| KITTI-V2 | LaMOT-best | GT_EMPTY | TRACK | 0 | 未定义 |
| KITTI-V2 | LaMOT-best | Y_POSITIVE | FRAME | 846 | 0.883858 |
| KITTI-V2 | LaMOT-best | Y_POSITIVE | TRACK | 846 | 0.472709 |
| KITTI-V2 | LaMOT-best | Y_ZERO | FRAME | 16 | 0.955691 |
| KITTI-V2 | LaMOT-best | Y_ZERO | TRACK | 16 | 1.000000 |
| KITTI-V2 | LaMOT-best | COMMON_AUROC_Q | FRAME | 846 | 0.883858 |
| KITTI-V2 | LaMOT-best | COMMON_AUROC_Q | TRACK | 846 | 0.472709 |
| KITTI-V2 | V2-best | ALL | FRAME | 862 | 0.953585 |
| KITTI-V2 | V2-best | ALL | TRACK | 862 | 0.651767 |
| KITTI-V2 | V2-best | GT_PRESENT | FRAME | 862 | 0.953585 |
| KITTI-V2 | V2-best | GT_PRESENT | TRACK | 862 | 0.651767 |
| KITTI-V2 | V2-best | GT_EMPTY | FRAME | 0 | 未定义 |
| KITTI-V2 | V2-best | GT_EMPTY | TRACK | 0 | 未定义 |
| KITTI-V2 | V2-best | Y_POSITIVE | FRAME | 846 | 0.953042 |
| KITTI-V2 | V2-best | Y_POSITIVE | TRACK | 846 | 0.645181 |
| KITTI-V2 | V2-best | Y_ZERO | FRAME | 16 | 0.982304 |
| KITTI-V2 | V2-best | Y_ZERO | TRACK | 16 | 1.000000 |
| KITTI-V2 | V2-best | COMMON_AUROC_Q | FRAME | 846 | 0.953042 |
| KITTI-V2 | V2-best | COMMON_AUROC_Q | TRACK | 846 | 0.645181 |
| SportsMOT | Dance-best | ALL | FRAME | 466 | 0.798545 |
| SportsMOT | Dance-best | ALL | TRACK | 466 | 0.748699 |
| SportsMOT | Dance-best | GT_PRESENT | FRAME | 466 | 0.798545 |
| SportsMOT | Dance-best | GT_PRESENT | TRACK | 466 | 0.748699 |
| SportsMOT | Dance-best | GT_EMPTY | FRAME | 0 | 未定义 |
| SportsMOT | Dance-best | GT_EMPTY | TRACK | 0 | 未定义 |
| SportsMOT | Dance-best | Y_POSITIVE | FRAME | 465 | 0.798112 |
| SportsMOT | Dance-best | Y_POSITIVE | TRACK | 465 | 0.748158 |
| SportsMOT | Dance-best | Y_ZERO | FRAME | 1 | 1.000000 |
| SportsMOT | Dance-best | Y_ZERO | TRACK | 1 | 1.000000 |
| SportsMOT | Dance-best | COMMON_AUROC_Q | FRAME | 431 | 0.782277 |
| SportsMOT | Dance-best | COMMON_AUROC_Q | TRACK | 431 | 0.728291 |
| SportsMOT | LaMOT-best | ALL | FRAME | 466 | 0.933612 |
| SportsMOT | LaMOT-best | ALL | TRACK | 466 | 0.928344 |
| SportsMOT | LaMOT-best | GT_PRESENT | FRAME | 466 | 0.933612 |
| SportsMOT | LaMOT-best | GT_PRESENT | TRACK | 466 | 0.928344 |
| SportsMOT | LaMOT-best | GT_EMPTY | FRAME | 0 | 未定义 |
| SportsMOT | LaMOT-best | GT_EMPTY | TRACK | 0 | 未定义 |
| SportsMOT | LaMOT-best | Y_POSITIVE | FRAME | 465 | 0.933470 |
| SportsMOT | LaMOT-best | Y_POSITIVE | TRACK | 465 | 0.928189 |
| SportsMOT | LaMOT-best | Y_ZERO | FRAME | 1 | 1.000000 |
| SportsMOT | LaMOT-best | Y_ZERO | TRACK | 1 | 1.000000 |
| SportsMOT | LaMOT-best | COMMON_AUROC_Q | FRAME | 431 | 0.928299 |
| SportsMOT | LaMOT-best | COMMON_AUROC_Q | TRACK | 431 | 0.922525 |
| SportsMOT | V2-best | ALL | FRAME | 466 | 0.839510 |
| SportsMOT | V2-best | ALL | TRACK | 466 | 0.806387 |
| SportsMOT | V2-best | GT_PRESENT | FRAME | 466 | 0.839510 |
| SportsMOT | V2-best | GT_PRESENT | TRACK | 466 | 0.806387 |
| SportsMOT | V2-best | GT_EMPTY | FRAME | 0 | 未定义 |
| SportsMOT | V2-best | GT_EMPTY | TRACK | 0 | 未定义 |
| SportsMOT | V2-best | Y_POSITIVE | FRAME | 465 | 0.839165 |
| SportsMOT | V2-best | Y_POSITIVE | TRACK | 465 | 0.805971 |
| SportsMOT | V2-best | Y_ZERO | FRAME | 1 | 1.000000 |
| SportsMOT | V2-best | Y_ZERO | TRACK | 1 | 1.000000 |
| SportsMOT | V2-best | COMMON_AUROC_Q | FRAME | 431 | 0.826566 |
| SportsMOT | V2-best | COMMON_AUROC_Q | TRACK | 431 | 0.790664 |


## 4. 完美与模拟校验者的查询代价 / F1

只做固定标准二分 BISECT_EXACT，不再使用上一轮会问全的 PBA。轨迹按 MEAN 降序、ID升序；同一 q 的噪声答案按物理 ID 生成一次，三个模型和二分/全量对照共用，不能按不同排序重新抽噪声。提问数是模拟读答案数，不是本轮真实教师调用次数。

旧 decompose CROP 的原始条件错误率（全29146个候选判断，包含 N=1；按完整候选匹配标签，不是 GT-R@1 或视频存在性错误）：

| target | candidate_pairs | TP | FN | FP | TN | positive_recall | FNR | FPR |
|---|---|---|---|---|---|---|---|---|
| KITTI-V2 | 9180 | 841 | 492 | 1655 | 6192 | 0.630908 | 0.369092 | 0.210909 |
| SportsMOT | 13540 | 6768 | 221 | 2203 | 4348 | 0.968379 | 0.031621 | 0.336285 |
| Dance | 6426 | 280 | 35 | 580 | 5531 | 0.888889 | 0.111111 | 0.094911 |


这些是单帧 CROP 的实测候选错误率，投射到轨迹仅作为独立 Bernoulli 翻转的假设诊断；未验证轨迹级教师有同样错误率或时间独立性。所有模拟结果一并上报，不挑 seed 或较好的模型。

| target | model | scope | verifier | sequences_in_scope | mean_tracks | mean_bisect_questions | mean_dense_questions | questions_ratio | mean_bisect_F1 | mean_dense_F1 | F1_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dance | Dance-best | ALL | ORACLE | 425 | 26.240000 | 4.002353 | 26.240000 | 0.152529 | 0.965628 | 1.000000 | 0.965628 |
| Dance | Dance-best | ALL | NOISY_SIMULATION | 425 | 26.240000 | 4.080000 | 26.240000 | 0.155488 | 0.657598 | 0.275805 | 2.384288 |
| Dance | Dance-best | GT_PRESENT | ORACLE | 40 | 25.425000 | 4.425000 | 25.425000 | 0.174041 | 0.634794 | 1.000000 | 0.634794 |
| Dance | Dance-best | GT_PRESENT | NOISY_SIMULATION | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.586977 | 0.830425 | 0.706840 |
| Dance | Dance-best | GT_EMPTY | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | Dance-best | GT_EMPTY | NOISY_SIMULATION | 385 | 26.324675 | 4.049351 | 26.324675 | 0.153823 | 0.664935 | 0.218182 | 3.047619 |
| Dance | Dance-best | Y_POSITIVE | ORACLE | 40 | 25.425000 | 4.425000 | 25.425000 | 0.174041 | 0.634794 | 1.000000 | 0.634794 |
| Dance | Dance-best | Y_POSITIVE | NOISY_SIMULATION | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.586977 | 0.830425 | 0.706840 |
| Dance | Dance-best | Y_ZERO | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | Dance-best | Y_ZERO | NOISY_SIMULATION | 385 | 26.324675 | 4.049351 | 26.324675 | 0.153823 | 0.664935 | 0.218182 | 3.047619 |
| Dance | Dance-best | COMMON_AUROC_Q | ORACLE | 30 | 26.333333 | 4.333333 | 26.333333 | 0.164557 | 0.513058 | 1.000000 | 0.513058 |
| Dance | Dance-best | COMMON_AUROC_Q | NOISY_SIMULATION | 30 | 26.333333 | 4.266667 | 26.333333 | 0.162025 | 0.481964 | 0.792085 | 0.608476 |
| Dance | LaMOT-best | ALL | ORACLE | 425 | 26.240000 | 3.997647 | 26.240000 | 0.152349 | 0.969825 | 1.000000 | 0.969825 |
| Dance | LaMOT-best | ALL | NOISY_SIMULATION | 425 | 26.240000 | 4.058824 | 26.240000 | 0.154681 | 0.688128 | 0.275805 | 2.494984 |
| Dance | LaMOT-best | GT_PRESENT | ORACLE | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.679390 | 1.000000 | 0.679390 |
| Dance | LaMOT-best | GT_PRESENT | NOISY_SIMULATION | 40 | 25.425000 | 4.350000 | 25.425000 | 0.171091 | 0.636362 | 0.830425 | 0.766310 |
| Dance | LaMOT-best | GT_EMPTY | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | LaMOT-best | GT_EMPTY | NOISY_SIMULATION | 385 | 26.324675 | 4.028571 | 26.324675 | 0.153034 | 0.693506 | 0.218182 | 3.178571 |
| Dance | LaMOT-best | Y_POSITIVE | ORACLE | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.679390 | 1.000000 | 0.679390 |
| Dance | LaMOT-best | Y_POSITIVE | NOISY_SIMULATION | 40 | 25.425000 | 4.350000 | 25.425000 | 0.171091 | 0.636362 | 0.830425 | 0.766310 |
| Dance | LaMOT-best | Y_ZERO | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | LaMOT-best | Y_ZERO | NOISY_SIMULATION | 385 | 26.324675 | 4.028571 | 26.324675 | 0.153034 | 0.693506 | 0.218182 | 3.178571 |
| Dance | LaMOT-best | COMMON_AUROC_Q | ORACLE | 30 | 26.333333 | 4.266667 | 26.333333 | 0.162025 | 0.572521 | 1.000000 | 0.572521 |
| Dance | LaMOT-best | COMMON_AUROC_Q | NOISY_SIMULATION | 30 | 26.333333 | 4.300000 | 26.333333 | 0.163291 | 0.528889 | 0.792085 | 0.667718 |
| Dance | V2-best | ALL | ORACLE | 425 | 26.240000 | 4.000000 | 26.240000 | 0.152439 | 0.970231 | 1.000000 | 0.970231 |
| Dance | V2-best | ALL | NOISY_SIMULATION | 425 | 26.240000 | 4.058824 | 26.240000 | 0.154681 | 0.667083 | 0.275805 | 2.418678 |
| Dance | V2-best | GT_PRESENT | ORACLE | 40 | 25.425000 | 4.400000 | 25.425000 | 0.173058 | 0.683700 | 1.000000 | 0.683700 |
| Dance | V2-best | GT_PRESENT | NOISY_SIMULATION | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.637754 | 0.830425 | 0.767985 |
| Dance | V2-best | GT_EMPTY | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | V2-best | GT_EMPTY | NOISY_SIMULATION | 385 | 26.324675 | 4.025974 | 26.324675 | 0.152935 | 0.670130 | 0.218182 | 3.071429 |
| Dance | V2-best | Y_POSITIVE | ORACLE | 40 | 25.425000 | 4.400000 | 25.425000 | 0.173058 | 0.683700 | 1.000000 | 0.683700 |
| Dance | V2-best | Y_POSITIVE | NOISY_SIMULATION | 40 | 25.425000 | 4.375000 | 25.425000 | 0.172075 | 0.637754 | 0.830425 | 0.767985 |
| Dance | V2-best | Y_ZERO | ORACLE | 385 | 26.324675 | 3.958442 | 26.324675 | 0.150370 | 1.000000 | 1.000000 | 1.000000 |
| Dance | V2-best | Y_ZERO | NOISY_SIMULATION | 385 | 26.324675 | 4.025974 | 26.324675 | 0.152935 | 0.670130 | 0.218182 | 3.071429 |
| Dance | V2-best | COMMON_AUROC_Q | ORACLE | 30 | 26.333333 | 4.300000 | 26.333333 | 0.163291 | 0.578267 | 1.000000 | 0.578267 |
| Dance | V2-best | COMMON_AUROC_Q | NOISY_SIMULATION | 30 | 26.333333 | 4.300000 | 26.333333 | 0.163291 | 0.539932 | 0.792085 | 0.681659 |
| KITTI-V2 | Dance-best | ALL | ORACLE | 862 | 81.037123 | 5.672854 | 81.037123 | 0.070003 | 0.235706 | 1.000000 | 0.235706 |
| KITTI-V2 | Dance-best | ALL | NOISY_SIMULATION | 862 | 81.037123 | 5.784223 | 81.037123 | 0.071377 | 0.182282 | 0.333913 | 0.545897 |
| KITTI-V2 | Dance-best | GT_PRESENT | ORACLE | 862 | 81.037123 | 5.672854 | 81.037123 | 0.070003 | 0.235706 | 1.000000 | 0.235706 |
| KITTI-V2 | Dance-best | GT_PRESENT | NOISY_SIMULATION | 862 | 81.037123 | 5.784223 | 81.037123 | 0.071377 | 0.182282 | 0.333913 | 0.545897 |
| KITTI-V2 | Dance-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | Dance-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | Dance-best | Y_POSITIVE | ORACLE | 846 | 81.229314 | 5.680851 | 81.229314 | 0.069936 | 0.221251 | 1.000000 | 0.221251 |
| KITTI-V2 | Dance-best | Y_POSITIVE | NOISY_SIMULATION | 846 | 81.229314 | 5.788416 | 81.229314 | 0.071260 | 0.184547 | 0.340228 | 0.542422 |
| KITTI-V2 | Dance-best | Y_ZERO | ORACLE | 16 | 70.875000 | 5.250000 | 70.875000 | 0.074074 | 1.000000 | 1.000000 | 1.000000 |
| KITTI-V2 | Dance-best | Y_ZERO | NOISY_SIMULATION | 16 | 70.875000 | 5.562500 | 70.875000 | 0.078483 | 0.062500 | 0.000000 | 未定义 |
| KITTI-V2 | Dance-best | COMMON_AUROC_Q | ORACLE | 846 | 81.229314 | 5.680851 | 81.229314 | 0.069936 | 0.221251 | 1.000000 | 0.221251 |
| KITTI-V2 | Dance-best | COMMON_AUROC_Q | NOISY_SIMULATION | 846 | 81.229314 | 5.788416 | 81.229314 | 0.071260 | 0.184547 | 0.340228 | 0.542422 |
| KITTI-V2 | LaMOT-best | ALL | ORACLE | 862 | 81.037123 | 5.726218 | 81.037123 | 0.070662 | 0.331110 | 1.000000 | 0.331110 |
| KITTI-V2 | LaMOT-best | ALL | NOISY_SIMULATION | 862 | 81.037123 | 5.769142 | 81.037123 | 0.071191 | 0.234996 | 0.333913 | 0.703766 |
| KITTI-V2 | LaMOT-best | GT_PRESENT | ORACLE | 862 | 81.037123 | 5.726218 | 81.037123 | 0.070662 | 0.331110 | 1.000000 | 0.331110 |
| KITTI-V2 | LaMOT-best | GT_PRESENT | NOISY_SIMULATION | 862 | 81.037123 | 5.769142 | 81.037123 | 0.071191 | 0.234996 | 0.333913 | 0.703766 |
| KITTI-V2 | LaMOT-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | LaMOT-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | LaMOT-best | Y_POSITIVE | ORACLE | 846 | 81.229314 | 5.735225 | 81.229314 | 0.070605 | 0.318460 | 1.000000 | 0.318460 |
| KITTI-V2 | LaMOT-best | Y_POSITIVE | NOISY_SIMULATION | 846 | 81.229314 | 5.769504 | 81.229314 | 0.071027 | 0.237077 | 0.340228 | 0.696818 |
| KITTI-V2 | LaMOT-best | Y_ZERO | ORACLE | 16 | 70.875000 | 5.250000 | 70.875000 | 0.074074 | 1.000000 | 1.000000 | 1.000000 |
| KITTI-V2 | LaMOT-best | Y_ZERO | NOISY_SIMULATION | 16 | 70.875000 | 5.750000 | 70.875000 | 0.081129 | 0.125000 | 0.000000 | 未定义 |
| KITTI-V2 | LaMOT-best | COMMON_AUROC_Q | ORACLE | 846 | 81.229314 | 5.735225 | 81.229314 | 0.070605 | 0.318460 | 1.000000 | 0.318460 |
| KITTI-V2 | LaMOT-best | COMMON_AUROC_Q | NOISY_SIMULATION | 846 | 81.229314 | 5.769504 | 81.229314 | 0.071027 | 0.237077 | 0.340228 | 0.696818 |
| KITTI-V2 | V2-best | ALL | ORACLE | 862 | 81.037123 | 5.834107 | 81.037123 | 0.071993 | 0.530017 | 1.000000 | 0.530017 |
| KITTI-V2 | V2-best | ALL | NOISY_SIMULATION | 862 | 81.037123 | 5.819026 | 81.037123 | 0.071807 | 0.357963 | 0.333913 | 1.072026 |
| KITTI-V2 | V2-best | GT_PRESENT | ORACLE | 862 | 81.037123 | 5.834107 | 81.037123 | 0.071993 | 0.530017 | 1.000000 | 0.530017 |
| KITTI-V2 | V2-best | GT_PRESENT | NOISY_SIMULATION | 862 | 81.037123 | 5.819026 | 81.037123 | 0.071807 | 0.357963 | 0.333913 | 1.072026 |
| KITTI-V2 | V2-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | V2-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| KITTI-V2 | V2-best | Y_POSITIVE | ORACLE | 846 | 81.229314 | 5.845154 | 81.229314 | 0.071959 | 0.521128 | 1.000000 | 0.521128 |
| KITTI-V2 | V2-best | Y_POSITIVE | NOISY_SIMULATION | 846 | 81.229314 | 5.822695 | 81.229314 | 0.071682 | 0.357641 | 0.340228 | 1.051181 |
| KITTI-V2 | V2-best | Y_ZERO | ORACLE | 16 | 70.875000 | 5.250000 | 70.875000 | 0.074074 | 1.000000 | 1.000000 | 1.000000 |
| KITTI-V2 | V2-best | Y_ZERO | NOISY_SIMULATION | 16 | 70.875000 | 5.625000 | 70.875000 | 0.079365 | 0.375000 | 0.000000 | 未定义 |
| KITTI-V2 | V2-best | COMMON_AUROC_Q | ORACLE | 846 | 81.229314 | 5.845154 | 81.229314 | 0.071959 | 0.521128 | 1.000000 | 0.521128 |
| KITTI-V2 | V2-best | COMMON_AUROC_Q | NOISY_SIMULATION | 846 | 81.229314 | 5.822695 | 81.229314 | 0.071682 | 0.357641 | 0.340228 | 1.051181 |
| SportsMOT | Dance-best | ALL | ORACLE | 466 | 25.188841 | 4.515021 | 25.188841 | 0.179247 | 0.633171 | 1.000000 | 0.633171 |
| SportsMOT | Dance-best | ALL | NOISY_SIMULATION | 466 | 25.188841 | 4.603004 | 25.188841 | 0.182740 | 0.603541 | 0.726693 | 0.830531 |
| SportsMOT | Dance-best | GT_PRESENT | ORACLE | 466 | 25.188841 | 4.515021 | 25.188841 | 0.179247 | 0.633171 | 1.000000 | 0.633171 |
| SportsMOT | Dance-best | GT_PRESENT | NOISY_SIMULATION | 466 | 25.188841 | 4.603004 | 25.188841 | 0.182740 | 0.603541 | 0.726693 | 0.830531 |
| SportsMOT | Dance-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | Dance-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | Dance-best | Y_POSITIVE | ORACLE | 465 | 25.174194 | 4.513978 | 25.174194 | 0.179310 | 0.632382 | 1.000000 | 0.632382 |
| SportsMOT | Dance-best | Y_POSITIVE | NOISY_SIMULATION | 465 | 25.174194 | 4.602151 | 25.174194 | 0.182812 | 0.604839 | 0.728256 | 0.830531 |
| SportsMOT | Dance-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| SportsMOT | Dance-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| SportsMOT | Dance-best | COMMON_AUROC_Q | ORACLE | 431 | 25.883991 | 4.512761 | 25.883991 | 0.174346 | 0.603382 | 1.000000 | 0.603382 |
| SportsMOT | Dance-best | COMMON_AUROC_Q | NOISY_SIMULATION | 431 | 25.883991 | 4.607889 | 25.883991 | 0.178021 | 0.574110 | 0.708298 | 0.810549 |
| SportsMOT | LaMOT-best | ALL | ORACLE | 466 | 25.188841 | 4.590129 | 25.188841 | 0.182229 | 0.888234 | 1.000000 | 0.888234 |
| SportsMOT | LaMOT-best | ALL | NOISY_SIMULATION | 466 | 25.188841 | 4.611588 | 25.188841 | 0.183081 | 0.760755 | 0.726693 | 1.046873 |
| SportsMOT | LaMOT-best | GT_PRESENT | ORACLE | 466 | 25.188841 | 4.590129 | 25.188841 | 0.182229 | 0.888234 | 1.000000 | 0.888234 |
| SportsMOT | LaMOT-best | GT_PRESENT | NOISY_SIMULATION | 466 | 25.188841 | 4.611588 | 25.188841 | 0.183081 | 0.760755 | 0.726693 | 1.046873 |
| SportsMOT | LaMOT-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | LaMOT-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | LaMOT-best | Y_POSITIVE | ORACLE | 465 | 25.174194 | 4.589247 | 25.174194 | 0.182300 | 0.887993 | 1.000000 | 0.887993 |
| SportsMOT | LaMOT-best | Y_POSITIVE | NOISY_SIMULATION | 465 | 25.174194 | 4.610753 | 25.174194 | 0.183154 | 0.762391 | 0.728256 | 1.046873 |
| SportsMOT | LaMOT-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| SportsMOT | LaMOT-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| SportsMOT | LaMOT-best | COMMON_AUROC_Q | ORACLE | 431 | 25.883991 | 4.593968 | 25.883991 | 0.177483 | 0.879157 | 1.000000 | 0.879157 |
| SportsMOT | LaMOT-best | COMMON_AUROC_Q | NOISY_SIMULATION | 431 | 25.883991 | 4.617169 | 25.883991 | 0.178379 | 0.745885 | 0.708298 | 1.053066 |
| SportsMOT | V2-best | ALL | ORACLE | 466 | 25.188841 | 4.540773 | 25.188841 | 0.180269 | 0.728801 | 1.000000 | 0.728801 |
| SportsMOT | V2-best | ALL | NOISY_SIMULATION | 466 | 25.188841 | 4.622318 | 25.188841 | 0.183507 | 0.676510 | 0.726693 | 0.930943 |
| SportsMOT | V2-best | GT_PRESENT | ORACLE | 466 | 25.188841 | 4.540773 | 25.188841 | 0.180269 | 0.728801 | 1.000000 | 0.728801 |
| SportsMOT | V2-best | GT_PRESENT | NOISY_SIMULATION | 466 | 25.188841 | 4.622318 | 25.188841 | 0.183507 | 0.676510 | 0.726693 | 0.930943 |
| SportsMOT | V2-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | V2-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| SportsMOT | V2-best | Y_POSITIVE | ORACLE | 465 | 25.174194 | 4.539785 | 25.174194 | 0.180335 | 0.728217 | 1.000000 | 0.728217 |
| SportsMOT | V2-best | Y_POSITIVE | NOISY_SIMULATION | 465 | 25.174194 | 4.621505 | 25.174194 | 0.183581 | 0.677965 | 0.728256 | 0.930943 |
| SportsMOT | V2-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| SportsMOT | V2-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 32.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| SportsMOT | V2-best | COMMON_AUROC_Q | ORACLE | 431 | 25.883991 | 4.540603 | 25.883991 | 0.175421 | 0.706777 | 1.000000 | 0.706777 |
| SportsMOT | V2-best | COMMON_AUROC_Q | NOISY_SIMULATION | 431 | 25.883991 | 4.628770 | 25.883991 | 0.178828 | 0.654214 | 0.708298 | 0.923641 |


questions_ratio = 平均提问数 / 平均轨迹数；F1_ratio = 平均二分F1 / 平均全量F1。它们不是逐 q 比例的平均。不对 T3 新增成功门槛，F1 损失和查询节省必须同时看。完美全量对照逐 q F1=1 已自检。

## 5. Dance 视频内存在性可分性

每个视频17句，标签取官方过滤后整序列 GT 是否非空；每句分数取合格轨迹聚合分数最大值。每视频 AUROC 后再平均；只有一类的视频未定义，不补0.5。四种既定聚合全列出，MEAN 为主，不改教师/阈值。

| model | aggregation | videos | defined_videos | undefined_videos | mean_AUROC | CI95 | valid_bootstrap_samples |
|---|---|---|---|---|---|---|---|
| Dance-best | LOGIT_MEAN | 25 | 25 | 0 | 0.885468 | [0.8037438492063491, 0.9502035714285713] | 2000 |
| Dance-best | MEAN | 25 | 25 | 0 | 0.863365 | [0.7734503968253968, 0.9394999999999999] | 2000 |
| Dance-best | MEDIAN | 25 | 25 | 0 | 0.856421 | [0.7613301587301589, 0.93665873015873] | 2000 |
| Dance-best | TOP25 | 25 | 25 | 0 | 0.872294 | [0.78665260989011, 0.9425670787545787] | 2000 |
| LaMOT-best | LOGIT_MEAN | 25 | 25 | 0 | 0.752925 | [0.6609236263736263, 0.8355064560439561] | 2000 |
| LaMOT-best | MEAN | 25 | 25 | 0 | 0.766601 | [0.6767039835164836, 0.8471727564102565] | 2000 |
| LaMOT-best | MEDIAN | 25 | 25 | 0 | 0.769857 | [0.6778571428571428, 0.8545083333333333] | 2000 |
| LaMOT-best | TOP25 | 25 | 25 | 0 | 0.781964 | [0.6872518772893773, 0.8707411630036629] | 2000 |
| V2-best | LOGIT_MEAN | 25 | 25 | 0 | 0.721567 | [0.6183289682539683, 0.815816758241758] | 2000 |
| V2-best | MEAN | 25 | 25 | 0 | 0.728838 | [0.6218687728937728, 0.8319574633699633] | 2000 |
| V2-best | MEDIAN | 25 | 25 | 0 | 0.724005 | [0.6179501221001221, 0.8235478174603176] | 2000 |
| V2-best | TOP25 | 25 | 25 | 0 | 0.742553 | [0.6350892094017094, 0.8380780982905982] | 2000 |


逐视频值在 presence_auc.csv / 其无损压缩件，见文件清单。Sports 和 KITTI 没有 GT-empty 序列，本节没有定义为其存在性分类实验。

## 6. 视频聚类 bootstrap

Dance25、Sports45视频为单位，2000次、seed0，整视频携带全部表达和帧；FRAME/TRACK配对用相同重采样权重。AUROC差值只在共同 q 上计算，前缀差值先逐 q 配对再汇总。KITTI4视频只报告点估计。Sports418保留同一v4内容并沿用原45视频权重，不把帧/表达误当独立样本。点判定不加入未注册的 CI 门槛。

| protocol | target | model | metric | point | CI95 | clusters | valid_bootstrap_samples |
|---|---|---|---|---|---|---|---|
| FULL_MAIN_V4 | Dance | Dance-best | TRACK_MEAN_minus_FRAME_AUROC | 0.025466 | [-0.04424752493758504, 0.1072672191301174] | 25 | 2000 |
| FULL_MAIN_V4 | Dance | Dance-best | TRACK_minus_FRAME_prefix_F1 | -0.007674 | [-0.01755767753809274, -0.0013552620810308724] | 25 | 2000 |
| FULL_MAIN_V4 | Dance | LaMOT-best | TRACK_MEAN_minus_FRAME_AUROC | 0.101370 | [0.030729653118609732, 0.19656244430319605] | 25 | 2000 |
| FULL_MAIN_V4 | Dance | LaMOT-best | TRACK_minus_FRAME_prefix_F1 | -0.005799 | [-0.015609217706201824, 0.00029143094201604966] | 25 | 2000 |
| FULL_MAIN_V4 | Dance | V2-best | TRACK_MEAN_minus_FRAME_AUROC | 0.033012 | [-0.06593246332174321, 0.12510759312979636] | 25 | 2000 |
| FULL_MAIN_V4 | Dance | V2-best | TRACK_minus_FRAME_prefix_F1 | -0.004326 | [-0.01130647586266896, 0.0008169788453368338] | 25 | 2000 |
| FULL_MAIN_V4 | SportsMOT | Dance-best | TRACK_MEAN_minus_FRAME_AUROC | -0.000666 | [-0.01524586055921563, 0.012695705321397249] | 45 | 2000 |
| FULL_MAIN_V4 | SportsMOT | Dance-best | TRACK_minus_FRAME_prefix_F1 | -0.049847 | [-0.06738250306882591, -0.03323920269302036] | 45 | 2000 |
| FULL_MAIN_V4 | SportsMOT | LaMOT-best | TRACK_MEAN_minus_FRAME_AUROC | 0.030570 | [0.019734027953703798, 0.04073092789414336] | 45 | 2000 |
| FULL_MAIN_V4 | SportsMOT | LaMOT-best | TRACK_minus_FRAME_prefix_F1 | -0.005269 | [-0.02100890919705007, 0.00647374887730398] | 45 | 2000 |
| FULL_MAIN_V4 | SportsMOT | V2-best | TRACK_MEAN_minus_FRAME_AUROC | 0.051253 | [0.036069425574549306, 0.06674270687049241] | 45 | 2000 |
| FULL_MAIN_V4 | SportsMOT | V2-best | TRACK_minus_FRAME_prefix_F1 | -0.033123 | [-0.06091036996685524, -0.007737070978746342] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | Dance-best | TRACK_MEAN_minus_FRAME_AUROC | 0.000121 | [-0.015944646783294932, 0.014963006391588004] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | Dance-best | TRACK_minus_FRAME_prefix_F1 | -0.040291 | [-0.060166801940111, -0.0221189479857832] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | LaMOT-best | TRACK_MEAN_minus_FRAME_AUROC | 0.030165 | [0.01798814516574997, 0.04113867375563248] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | LaMOT-best | TRACK_minus_FRAME_prefix_F1 | -0.007579 | [-0.024586882914708653, 0.004714948007628356] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | V2-best | TRACK_MEAN_minus_FRAME_AUROC | 0.050890 | [0.03376877995513157, 0.06888275305557814] | 45 | 2000 |
| SPORTS418_V4_CONTENT | SportsMOT | V2-best | TRACK_minus_FRAME_prefix_F1 | -0.037841 | [-0.06740320886890871, -0.010546230887289341] | 45 | 2000 |


## 7. SportsMOT 原官方418序列覆盖的敏感性

用户已确认原文第1节的章节引用歧义：本节重算 T1/T2/T3，不把仅适用于 Dance 的第5节强套 Sports。主结果仍466条；418敏感性使用相同 v4 GT 内容，仅去掉原模板缺失的48个ID，不使用已知缺58897行的8条旧模板、不补空GT、不删主结果。分数、观测标签和模拟噪声逐 q 复用，不重新匹配/抽样。

| combo | model | sequences | common_sequences | FRAME_AUROC | TRACK_MEAN_AUROC | AUROC_delta | T1 | TRACK_prefix_F1 | T2 |
|---|---|---|---|---|---|---|---|---|---|
| C3 | V2-best | 418 | 383 | 0.656168 | 0.707058 | 0.050890 | 轨迹级更好但仍不够可靠 | 0.820661 | 前缀约束可接受 |
| C4 | Dance-best | 418 | 383 | 0.620970 | 0.621091 | 0.000121 | 轨迹级没有明显改善 | 0.800468 | 前缀约束可接受 |


418条覆盖内全部既定聚合和自身/共同分母：

| model | granularity | aggregation | scope | sequences | common_sequences | defined_units | undefined_units | mean_AUROC |
|---|---|---|---|---|---|---|---|---|
| Dance-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 418 | 383 | 179793 | 69795 | 0.629000 |
| Dance-best | FRAME | MARGIN | ALL_DEFINED_Q | 418 | 383 | 417 | 1 | 0.611691 |
| Dance-best | SEGMENT | MEAN | ALL_DEFINED_Q | 418 | 383 | 385 | 33 | 0.617327 |
| Dance-best | TRACK | MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.621091 |
| Dance-best | TRACK | MEDIAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.618826 |
| Dance-best | TRACK | TOP25 | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.637613 |
| Dance-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.631417 |
| Dance-best | FRAME | MARGIN | COMMON_Q | 418 | 383 | 383 | 0 | 0.620970 |
| Dance-best | SEGMENT | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.618181 |
| Dance-best | TRACK | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.621091 |
| Dance-best | TRACK | MEDIAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.618826 |
| Dance-best | TRACK | TOP25 | COMMON_Q | 418 | 383 | 383 | 0 | 0.637613 |
| Dance-best | TRACK | LOGIT_MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.631417 |
| LaMOT-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 418 | 383 | 179793 | 69795 | 0.860295 |
| LaMOT-best | FRAME | MARGIN | ALL_DEFINED_Q | 418 | 383 | 417 | 1 | 0.827666 |
| LaMOT-best | SEGMENT | MEAN | ALL_DEFINED_Q | 418 | 383 | 385 | 33 | 0.864745 |
| LaMOT-best | TRACK | MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.875156 |
| LaMOT-best | TRACK | MEDIAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.871788 |
| LaMOT-best | TRACK | TOP25 | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.883620 |
| LaMOT-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.855946 |
| LaMOT-best | FRAME | MARGIN | COMMON_Q | 418 | 383 | 383 | 0 | 0.844991 |
| LaMOT-best | SEGMENT | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.869083 |
| LaMOT-best | TRACK | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.875156 |
| LaMOT-best | TRACK | MEDIAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.871788 |
| LaMOT-best | TRACK | TOP25 | COMMON_Q | 418 | 383 | 383 | 0 | 0.883620 |
| LaMOT-best | TRACK | LOGIT_MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.855946 |
| V2-best | FRAME | MARGIN | ALL_DEFINED_FRAMES | 418 | 383 | 179793 | 69795 | 0.670178 |
| V2-best | FRAME | MARGIN | ALL_DEFINED_Q | 418 | 383 | 417 | 1 | 0.640925 |
| V2-best | SEGMENT | MEAN | ALL_DEFINED_Q | 418 | 383 | 385 | 33 | 0.677180 |
| V2-best | TRACK | MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.707058 |
| V2-best | TRACK | MEDIAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.714031 |
| V2-best | TRACK | TOP25 | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.737930 |
| V2-best | TRACK | LOGIT_MEAN | ALL_DEFINED_Q | 418 | 383 | 383 | 35 | 0.701462 |
| V2-best | FRAME | MARGIN | COMMON_Q | 418 | 383 | 383 | 0 | 0.656168 |
| V2-best | SEGMENT | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.679717 |
| V2-best | TRACK | MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.707058 |
| V2-best | TRACK | MEDIAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.714031 |
| V2-best | TRACK | TOP25 | COMMON_Q | 418 | 383 | 383 | 0 | 0.737930 |
| V2-best | TRACK | LOGIT_MEAN | COMMON_Q | 418 | 383 | 383 | 0 | 0.701462 |


418条的前缀与分组：

| model | scope | granularity | sequences_in_scope | mean_prefix_F1 |
|---|---|---|---|---|
| Dance-best | ALL | FRAME | 418 | 0.840759 |
| Dance-best | ALL | TRACK | 418 | 0.800468 |
| Dance-best | GT_PRESENT | FRAME | 418 | 0.840759 |
| Dance-best | GT_PRESENT | TRACK | 418 | 0.800468 |
| Dance-best | GT_EMPTY | FRAME | 0 | 未定义 |
| Dance-best | GT_EMPTY | TRACK | 0 | 未定义 |
| Dance-best | Y_POSITIVE | FRAME | 417 | 0.840377 |
| Dance-best | Y_POSITIVE | TRACK | 417 | 0.799989 |
| Dance-best | Y_ZERO | FRAME | 1 | 1.000000 |
| Dance-best | Y_ZERO | TRACK | 1 | 1.000000 |
| Dance-best | COMMON_AUROC_Q | FRAME | 383 | 0.826309 |
| Dance-best | COMMON_AUROC_Q | TRACK | 383 | 0.782234 |
| LaMOT-best | ALL | FRAME | 418 | 0.944826 |
| LaMOT-best | ALL | TRACK | 418 | 0.937247 |
| LaMOT-best | GT_PRESENT | FRAME | 418 | 0.944826 |
| LaMOT-best | GT_PRESENT | TRACK | 418 | 0.937247 |
| LaMOT-best | GT_EMPTY | FRAME | 0 | 未定义 |
| LaMOT-best | GT_EMPTY | TRACK | 0 | 未定义 |
| LaMOT-best | Y_POSITIVE | FRAME | 417 | 0.944693 |
| LaMOT-best | Y_POSITIVE | TRACK | 417 | 0.937096 |
| LaMOT-best | Y_ZERO | FRAME | 1 | 1.000000 |
| LaMOT-best | Y_ZERO | TRACK | 1 | 1.000000 |
| LaMOT-best | COMMON_AUROC_Q | FRAME | 383 | 0.939872 |
| LaMOT-best | COMMON_AUROC_Q | TRACK | 383 | 0.931512 |
| V2-best | ALL | FRAME | 418 | 0.858502 |
| V2-best | ALL | TRACK | 418 | 0.820661 |
| V2-best | GT_PRESENT | FRAME | 418 | 0.858502 |
| V2-best | GT_PRESENT | TRACK | 418 | 0.820661 |
| V2-best | GT_EMPTY | FRAME | 0 | 未定义 |
| V2-best | GT_EMPTY | TRACK | 0 | 未定义 |
| V2-best | Y_POSITIVE | FRAME | 417 | 0.858163 |
| V2-best | Y_POSITIVE | TRACK | 417 | 0.820231 |
| V2-best | Y_ZERO | FRAME | 1 | 1.000000 |
| V2-best | Y_ZERO | TRACK | 1 | 1.000000 |
| V2-best | COMMON_AUROC_Q | FRAME | 383 | 0.845671 |
| V2-best | COMMON_AUROC_Q | TRACK | 383 | 0.804272 |


418条的 T3：

| model | scope | verifier | sequences_in_scope | mean_tracks | mean_bisect_questions | questions_ratio | mean_bisect_F1 | mean_dense_F1 | F1_ratio |
|---|---|---|---|---|---|---|---|---|---|
| Dance-best | ALL | ORACLE | 418 | 26.248804 | 4.617225 | 0.175902 | 0.691689 | 1.000000 | 0.691689 |
| Dance-best | ALL | NOISY_SIMULATION | 418 | 26.248804 | 4.681818 | 0.178363 | 0.657655 | 0.769393 | 0.854771 |
| Dance-best | GT_PRESENT | ORACLE | 418 | 26.248804 | 4.617225 | 0.175902 | 0.691689 | 1.000000 | 0.691689 |
| Dance-best | GT_PRESENT | NOISY_SIMULATION | 418 | 26.248804 | 4.681818 | 0.178363 | 0.657655 | 0.769393 | 0.854771 |
| Dance-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| Dance-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| Dance-best | Y_POSITIVE | ORACLE | 417 | 26.235012 | 4.616307 | 0.175960 | 0.690949 | 1.000000 | 0.690949 |
| Dance-best | Y_POSITIVE | NOISY_SIMULATION | 417 | 26.235012 | 4.681055 | 0.178428 | 0.659232 | 0.771238 | 0.854771 |
| Dance-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| Dance-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| Dance-best | COMMON_AUROC_Q | ORACLE | 383 | 27.127937 | 4.624021 | 0.170452 | 0.663514 | 1.000000 | 0.663514 |
| Dance-best | COMMON_AUROC_Q | NOISY_SIMULATION | 383 | 27.127937 | 4.694517 | 0.173051 | 0.629481 | 0.752595 | 0.836414 |
| LaMOT-best | ALL | ORACLE | 418 | 26.248804 | 4.657895 | 0.177452 | 0.900120 | 1.000000 | 0.900120 |
| LaMOT-best | ALL | NOISY_SIMULATION | 418 | 26.248804 | 4.677033 | 0.178181 | 0.798151 | 0.769393 | 1.037378 |
| LaMOT-best | GT_PRESENT | ORACLE | 418 | 26.248804 | 4.657895 | 0.177452 | 0.900120 | 1.000000 | 0.900120 |
| LaMOT-best | GT_PRESENT | NOISY_SIMULATION | 418 | 26.248804 | 4.677033 | 0.178181 | 0.798151 | 0.769393 | 1.037378 |
| LaMOT-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| LaMOT-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| LaMOT-best | Y_POSITIVE | ORACLE | 417 | 26.235012 | 4.657074 | 0.177514 | 0.899880 | 1.000000 | 0.899880 |
| LaMOT-best | Y_POSITIVE | NOISY_SIMULATION | 417 | 26.235012 | 4.676259 | 0.178245 | 0.800065 | 0.771238 | 1.037378 |
| LaMOT-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| LaMOT-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| LaMOT-best | COMMON_AUROC_Q | ORACLE | 383 | 27.127937 | 4.668407 | 0.172089 | 0.890993 | 1.000000 | 0.890993 |
| LaMOT-best | COMMON_AUROC_Q | NOISY_SIMULATION | 383 | 27.127937 | 4.689295 | 0.172859 | 0.784835 | 0.752595 | 1.042838 |
| V2-best | ALL | ORACLE | 418 | 26.248804 | 4.622010 | 0.176085 | 0.750529 | 1.000000 | 0.750529 |
| V2-best | ALL | NOISY_SIMULATION | 418 | 26.248804 | 4.686603 | 0.178545 | 0.711287 | 0.769393 | 0.924478 |
| V2-best | GT_PRESENT | ORACLE | 418 | 26.248804 | 4.622010 | 0.176085 | 0.750529 | 1.000000 | 0.750529 |
| V2-best | GT_PRESENT | NOISY_SIMULATION | 418 | 26.248804 | 4.686603 | 0.178545 | 0.711287 | 0.769393 | 0.924478 |
| V2-best | GT_EMPTY | ORACLE | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| V2-best | GT_EMPTY | NOISY_SIMULATION | 0 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 | 未定义 |
| V2-best | Y_POSITIVE | ORACLE | 417 | 26.235012 | 4.621103 | 0.176143 | 0.749931 | 1.000000 | 0.749931 |
| V2-best | Y_POSITIVE | NOISY_SIMULATION | 417 | 26.235012 | 4.685851 | 0.178611 | 0.712993 | 0.771238 | 0.924478 |
| V2-best | Y_ZERO | ORACLE | 1 | 32.000000 | 5.000000 | 0.156250 | 1.000000 | 1.000000 | 1.000000 |
| V2-best | Y_ZERO | NOISY_SIMULATION | 1 | 32.000000 | 5.000000 | 0.156250 | 0.000000 | 0.000000 | 未定义 |
| V2-best | COMMON_AUROC_Q | ORACLE | 383 | 27.127937 | 4.629243 | 0.170645 | 0.727731 | 1.000000 | 0.727731 |
| V2-best | COMMON_AUROC_Q | NOISY_SIMULATION | 383 | 27.127937 | 4.699739 | 0.173244 | 0.689375 | 0.752595 | 0.915997 |


## 8. 成本、失败修复、输入与可恢复路径

仓库 `/data2/user/FlexHook-TTA`，分支 research/track-ranking-20261102，从精确公开提交8e7d6e3创建，不进入LocateMOT。原文预注册先公开 `b98041f31860de5296f452e9c156e529c131c45c`，用户两项澄清及计算口径先公开 `2bfc9ac160ee8a3ac65c87e0cf1770e4491fce93`，才开始计算。九套完整分数的清单 Git blob `8b3fcfdbe3ff3f11f9eaa92b54acaeeab0d24464` 与用户指定的01add887完全相同，所有SHA开始前/交付前核对，末尾观测保留。

轨迹 n<3 对三粒度统一过滤；先在完整帧候选上匹配，再过滤，不重分配GT；不插值、不改框/ID/帧号。按首观测帧起每20/40/50帧划窗，空窗不补帧；短尾按时间跨度并入前一非空段。TOP25取ceil(n/4)，LOGIT_MEAN用数学等价稳定形式，不裁剪原分数。每个数据行均为 ORACLE/GT 诊断。

新增教师调用0、模型调用0、GPU小时0、优化步0。74视频、1753序列、5259个模型–序列组合；原始 FRAME AUROC 3441897行。30项定向及旧口径回归通过、compileall通过；过滤/分段观测守恒、标签多数规则、AUROC未定义规则、全量完美F1、共享噪声、二分提问界和前缀上限均通过。

CPU已完成worker累计墙钟 358.144 秒，失败worker 0.317 秒。三个目标worker并行，累计worker墙钟不是实际总等待时间或物理CPU核时；各launcher实际墙钟、日志与阶段成本在 costs.json。末次压缩/报告渲染、编码、发布耗时不混作科学计算时长。

失败修复记录：首个预检把含历史覆盖项的旧清单总行数误当成完整覆盖项数，在读GT/算指标前停止。最小修复为只筛 coverage=complete；新增针对真实清单的回归后通过九套完整分数核验。失败日志保留，没有改数据、弱化九模型覆盖要求、换种子、调阈值或隐藏失败。

验证阶段曾出现混合类型视频ID的pandas提示；读取units表时显式指定video为字符串，保留KITTI前导零，并重跑验证/打包。逐视频匹配、聚合、随机答案和全部统计未重跑或修改。

| stage | exit_code | wall_seconds | log |
|---|---|---|---|
| preflight | 1 | 0.317168 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/launchers/metrics_1793587844660860302/preflight.log |


恢复单位为完整视频：每个 targets/<target>/<video>/receipt.json 保存其文件SHA、GT来源及耗时；metrics阶段可验证后跳过完成视频。summary_completion.json索引全部机器表。没有未完成的视频、训练队列或模型推理任务。

复现命令：使用本机既有Python环境 `/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python`（只借用解释器/NumPy/SciPy，不加载教师），`CUDA_VISIBLE_DEVICES=''`；运行 scripts/run_track_ranking.py 的 --phase metrics、--phase aggregate、--phase report。汇总只读封存输出，不能把GT上限或模拟答案接回正式TTA。

文件：input_inventory.json、input_output_paths.json、noise_rates.csv、auroc_summary.csv、prefix_summary.csv、bisect_summary.csv、presence_summary.csv、bootstrap.json、validation.json；完整逐单元原始 CSV 的路径/SHA 和无损压缩件如下。大帧表、GT标签缓存、原始margin/GT/图像/checkpoint留在data2/既有资产目录，不入git。

| file | rows | bytes | raw_path | sha256 | git_copy |
|---|---|---|---|---|---|
| units.csv | 92744 | 10713972 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/units.csv | 6970cf41ecb3e70458e5dde943da3f0d53260510fc074d8513ee0dbd950ead06 | units.csv.gz |
| auroc_frame.csv | 3441897 | 429287346 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/auroc_frame.csv | 07dece2144901be6a461e177381b52b0fb57df4b96374921f8b56dfd9f7cc865 | 大文件仅外部路径 |
| auroc_segment.csv | 5259 | 640723 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/auroc_segment.csv | 624278ab716f884888893593d1c23419cc2a1a74cfe5376754a68a7e9b174f1e | auroc_segment.csv.gz |
| auroc_track.csv | 21036 | 2659983 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/auroc_track.csv | 03762d67f3882712c344725f9ade2c24de1055c70b597564301541f6cb10a275 | auroc_track.csv.gz |
| prefix_ceiling.csv | 10518 | 1466952 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/prefix_ceiling.csv | 6849ae59b687fa8880c8d649ab6ddfd61ad11c378aa741409ae6def50fb5cab6 | prefix_ceiling.csv.gz |
| bisect_track.csv | 21036 | 3820058 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/bisect_track.csv | 3d0a08848a89d88d884fad0a507519bbdcda45262fd2a6704416d034a531d6c4 | bisect_track.csv.gz |
| presence_scores.csv | 21036 | 2792012 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/presence_scores.csv | 1d1e71832719ec1c6df98be380be1b74309aa18fd4bf64e4cfa7e608b1a74320 | presence_scores.csv.gz |
| sequence_metrics.csv | 5259 | 1464502 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/sequence_metrics.csv | b08932e68cfd52064383733848916e1e8d71d1e2f183748ae88db2ccad73cccc | sequence_metrics.csv.gz |
| presence_auc.csv | 300 | 27840 | /data2/user/FlexHook-TTA/results/runs/track_ranking_20261102/machine/presence_auc.csv | a6cc66445b61cf457f7cb974e3bee344d5eb68e47347666c4c865442c704645e | presence_auc.csv.gz |


代码、完整精简日志与74视频receipt在 [code_and_logs.tar.gz](code_and_logs.tar.gz)。全部GT输入逐文件SHA随receipt保存；本任务没有创建/覆盖GT、教师缓存或旧实验结果。

完成后停止：不调用教师、不训练、不开始方法实现。即使个别聚合/子集指标更好，也不改变已注册的MEAN判定或自行运行新实验。
