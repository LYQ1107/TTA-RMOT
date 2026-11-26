# QCS-TTA final report

更新时间：2026-10-20。工程闭环已完成；本轮没有宣称 QCS-TTA 有科研收益。

## 资产与协议

- 项目根目录：`/data2/user/LocateMOT`；数据、V2 权重和运行产物均优先放在 `/data2`。
- TempRMOT source commit：`6a65640d849fdee4a32bb055945ee34c3b0edeb1`。
- V1 checkpoint SHA256：`55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`。
- V2 checkpoint SHA256：`4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`。
- runner 严格执行 predict-then-adapt；V1 完整分区包含 56 个视频-表达式 episode、5,320 帧查询，runner 未打开 evaluation manifest、未读 GT、未使用未来帧。
- 物理 GPU 2 由“显存足够后进程少优先”规则选出；GPU UUID 记录在每个 run metadata 中。

## V1 完整分区结果

官方 TempRMOT TrackEval，视频 0012/0014，56 个 sequence：

| 方法 | HOTA | DetA | AssA | IDF1 | ΔHOTA vs FROZEN |
|---|---:|---:|---:|---:|---:|
| FROZEN | 86.618 | 83.256 | 90.313 | 93.496 | 0.000 |
| EMA_ST | 83.899 | 79.981 | 88.197 | 91.483 | -2.719 |
| PROBE_CACHE | 85.543 | 82.052 | 89.376 | 92.765 | -1.075 |
| PROBE_ST | 86.627 | 83.260 | 90.328 | 93.498 | +0.009 |
| QCS_TTA | 86.531 | 83.170 | 90.223 | 93.446 | -0.087 |
| NO_STATE | 86.544 | 83.185 | 90.234 | 93.455 | -0.074 |

结论：QCS-TTA 和其无语义状态对照均略低于冻结源；PROBE_ST 的 +0.009 HOTA 低于本轮可解释的实验差异，不足以称为收益。普通 EMA_ST 和直接 probe cache 明显下降，说明额外自训练/缓存并非自动有效。

V2 另完成了视频 0005 的一个完整 297 帧表达式 episode：FROZEN HOTA 67.591，QCS-TTA 67.509（−0.082），NO_STATE 67.509（−0.082）。它是补充一致性证据，不是 V2 四视频完整分区结论。

## AerialMind 完整数据与结构复测

官方 AerialMind release 已完整下载到 `/data2`：93 个视频、24,617 个 expression JSON、45,630 个图像帧；所有 expression 引用的图像/标签均存在，17 个 VisDrone 与 13 个 UAVDT 官方测试视频均已审计。训练划分另有 63 个视频、27,044 个图像行、16,216 个 expressions，和测试视频无交集。

在完整数据上运行了 VisDrone `uav0000009_03358_v`（219 帧）和 UAVDT `M0203`（1,007 帧）的完整 episode。Refer-KITTI-V2 source 的 absolute referring gate 输出 0 行；新增 relative-rank 集合门控后输出 1,344 行，综合 HOTA/DetA/AssA/IDF1 为 `3.0240/1.2719/7.1964/0.0`，9.35 FPS、3.21 GB peak allocated。FROZEN、PROBE_CACHE、QCS_TTA、NO_STATE 四者结果完全相同，QCS optimizer steps 为 0。

判断：relative-rank 只修复了跨域概率标定造成的候选可见性，不能替代目标域 detector/tracker/referring source training；本轮没有把它宣称为 QCS-TTA 收益。完整 30-video test manifest 已生成，但 5,125,243 个 frame-query 的全量模型推理没有被不完整结果替代。

## 速度与显存

V1 多方法共享 source/probe packet 的运行记录：5,320 帧，端到端共享 source throughput 7.111 FPS，峰值显存 allocated 7,986,147,840 bytes、reserved 8,287,944,704 bytes；该数值不能解释为每个方法独立速度。

独立单方法 V2 297 帧测量（同一物理 GPU 2）：FROZEN 9.887 FPS、EMA_ST 12.256 FPS、PROBE_CACHE 9.410 FPS、PROBE_ST 9.376 FPS、QCS_TTA 9.851 FPS、NO_STATE 9.586 FPS。带 probe 的 QCS-TTA 峰值 allocated 3,079,602,688 bytes、reserved 3,238,002,688 bytes，p95 latency 0.2541 s；额外查询开销已计入。

## 实现边界与未完成数据

QCS-TTA 的 q0/q_eq/q_cf 对齐证据、候选级 BCE、等价 Bernoulli KL、互斥 hinge-square、source anchor、residual regularization、semantic replay 和 A1 fixed-TTL 均已进入实际 runner。正式 run trace 中 `L_eq`、`L_excl` 出现非零值；不是只实现了未调用的函数。

当前仍明确保留以下边界：

- `shared_raw_backbone: false`；首版使用有界 raw-frame ring，未虚构多尺度 backbone 共享收益。
- Refer-BDD、Refer-Dance 仍未纳入；AerialMind 已有完整、可核验的 expression+ID 接口，但当前 source-free checkpoint 跨域失败，后续需要独立目标域监督 source training 才能继续检验 QCS。
- V2 只完成一个完整 episode；V2 其余三个视频未作为完整分区结果宣称。
- `EMA_ST` 是高置信概率 EMA 对照，不冒充“同一 residual adapter 的完整 B2 复现”。

主要产物：`results/summary.csv`、`results/per_video.csv`、`results/per_expression.csv`、`results/state_transition_diagnostics.csv`，以及 `results/runs/v1_full_partition/` 和 `results/runs/v2_0005_complete_episode_qcs_logic/`。
