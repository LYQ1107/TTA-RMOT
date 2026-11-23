# QCS-TTA implementation notes

## Source boundary

`TempRMOTWrapper` 在 `RuntimeTrackerBase.update` 之后、QIM 之前复制 `FrameSnapshot`。快照保存 pre-QIM 的 256 维时序特征、source track-ref logit/probability、像素框、objectness、物理 ID、消失状态和 candidate UID；随后 source QIM 正常执行。`F.linear(features, source_track_ref.weight, bias)` 与快照 logit 有断言，且同一物理 ID 的 post-QIM referring probability 不能改变。

## Probe and evidence

每条 query 使用独立的 deep-copied frozen wrapper 和 tracker state。静态表达只重放当前帧，dynamic/mixed 表达最多重放最近四个已到达帧。probe 框先用 Hungarian/IoU 与主流候选对齐，再用歧义和阈值生成 `SUPPORTED_POS`、`SUPPORTED_NEG`、`UNKNOWN` 或 `CONTRADICTORY_PROBE`；未匹配不自动当负例。

第一版 rule parser 只接受安全的 appearance/spatial/dynamic 槽；OR、NOT、比较级和嵌套关系会显式标记 unsupported。当前 q_eq 是保守规范化表达，q_cf 只对注册的画面方向和 ego direction 对换生成。

## Adaptation

核心 adapter 是 `s_adapt=s_source+6*tanh(MLP_256-32-1(z)/6)`，最后一层零初始化，初始输出与 frozen 完全相同。source 参数全部 `requires_grad=False`；optimizer 只拿 external adapter 或 HEAD257。`L_ref`、Bernoulli KL、等价一致性、互斥 hinge-square、source anchor 和 residual regularization 均在 `rmot_tta/adaptation/losses.py` 中实现，并处理空 batch/全 mask。

正式 runner 会把可靠 aligned evidence 中的 q_eq/q_cf 概率连同 candidate feature 放入有界 logic cache；更新时仅对 adapter 的 q0 logit 反向传播 `L_eq` 和 `L_excl`，probe 概率保持 stop-gradient。V1/V2 正式 trace 均检查到非零 logic loss。没有合法对照或没有有效 referring 监督时跳过对应项/optimizer step。

semantic memory 将历史 replay sample 与 active support 分离；可靠正负冲突会推进 state version，但不会改写旧样本。A1 只保留相同 evidence/replay 容量并以 frame-unit TTL 过滤。

## Known scope limits

当前首版保留有限 CPU `RawFrameRing` 作为历史输入包，尚未把 source backbone 多尺度特征拆成可证明 query-independent 的共享缓存；因此配置将 `shared_raw_backbone` 明确置为 `false`，不把额外推理节省冒充已实现。多方法 pilot 可以共享只读 source/probe packet，但正式速度应按单方法独立 run 解释。
