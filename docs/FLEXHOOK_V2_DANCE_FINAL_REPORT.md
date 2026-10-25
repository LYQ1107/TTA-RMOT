# FlexHook V2 → Refer-Dance final TTA report

## Scope and protocol

This report belongs to the isolated TTA-RMOT/FlexHook line on branch
`research/flexhook-generalizable-plugin-tta`. The experiment starts from the
released `refer-kitti-v2-best.pth`, uses the official Dance DETR+NeuralSORT
candidate bank, keeps candidate boxes and tracker IDs fixed, and adapts only
the referring second stage online. Each prediction is sealed before the
causal update; every `(video, expression)` resets the adapter and optimizer.
The formal suite is the complete 25-video/425-expression Refer-Dance set.

Inference metadata records `target_gt_opened=false`,
`target_labels_opened=false`, and `future_frames_used=false`. Target GT is
used only by the post-hoc official TrackEval invocation.

## Results

| Method | HOTA | DetA | AssA | IDF1 | IDSW | Dets | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen V2 → Dance | 8.2060 | 3.1857 | 21.138 | 4.4360 | 259 | 99,316 | 1,740 |
| Gradient-free query/tracklet calibration | 8.2092 | 3.1468 | 21.417 | 4.4059 | 236 | 97,682 | 1,647 |
| PCD Residual | 8.0140 | 2.4762 | 25.938 | 3.6957 | 139 | 80,880 | 1,303 |
| PCD LoRA, rank 4 | 7.8654 | 2.3305 | 26.547 | 3.4920 | 138 | 82,248 | 1,313 |

The formal PCD LoRA run completed all 425 sequences with 108,511 optimizer
steps and 108,511 nonzero-gradient steps. Relative to Frozen, it improves
AssA by 5.409 points and reduces IDSW by 121, but loses 0.3406 HOTA,
0.8552 DetA, 0.944 IDF1, and 17,068 detections. It is therefore a negative
overall result, not a claimed TTA improvement. The residual adapter shows the
same pattern more strongly: association improves while detection and IDF1
fall.

The exact compact LoRA receipt is
`results/flexhook_v2_dance/pcd/pcd_lora_full_v2.json`; raw shard outputs are
under `/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_pcd_lora_full_v2/`.

## Attribution and controls

The D1–D3 diagnostics show fixed-candidate IoU-0.5 recall of 95.749%, but
the frozen referring gate retains only 5.046%; source-score AUROC is 0.5397
and ECE is 0.5266. Thus the main failure is cross-domain score calibration
and ranking, not merely tracker recall. The tracker switch rate is 2.408%,
so temporal identity errors are a secondary factor.

Source-free, no-gradient controls were also run on all 425 sequences:

- FROZEN: HOTA 8.2060 / IDF1 4.4360.
- ONLINE_STABILITY: HOTA 8.1741 / IDF1 4.3607.
- SOURCE_PRIOR_CALIBRATION: HOTA 8.2060 / IDF1 4.4360.
- QUERY_TRACKLET_CALIBRATION: HOTA 8.2092 / IDF1 4.4059.

The best no-gradient change is only +0.0032 HOTA and lowers IDF1, so it does
not support a calibration-only improvement claim. The rank-4 LoRA adapter is
zero-output at reset; the full run's source-output regression had zero sign
mismatches before adaptation, and the formal metadata records nonzero
gradients on every optimizer step.

## Independent visual evidence

`flex_tta/evidence/tracklet_language.py` implements an independent
RGB/HSV clothing-color plus causal-tracklet-stability evidence path. It does
not consume source logits or target labels and can be enabled with
`--evidence TRACKLET_LANGUAGE --evidence-weight 0.25`.

A one-expression, 1,202-frame source-free pilot completed successfully and
passed the fixed-candidate row audit. It produced 1,090 independent negative
evidence labels, zero positive labels, and zero unknown labels, with target GT
closed throughout. This is evidence-method validation only; it is not
reported as a full-suite metric because the formal 425-sequence LoRA result
used `evidence=NONE`.

## Reproducibility and failure analysis

The full run was executed as two complete-episode shards on GPUs 4 and 7,
then merged. The merged audit reports 25 videos and 425 prediction files,
82,248 checked prediction rows, unchanged tracker geometry and IDs, and no
target GT access. The source checkpoint SHA-256 and all hyperparameters are
in the compact receipts.

The research conclusion is deliberately negative but useful: fixed Dance
candidates are sufficiently complete, while V2's referring score gate is
poorly calibrated on Dance. Gradient-free calibration is nearly neutral;
parameter adaptation increases association consistency but suppresses too
many detections. Future work should target source-free ranking/selection and
confidence-preserving adaptation before claiming gains from a larger adapter.
