# FlexHook V2 → Refer-Dance parameter-level TTA

The new runner `scripts/run_flexhook_dance_pcd_tta.py` performs causal
episodic adaptation of the released FlexHook second stage.  The source
checkpoint is always `refer-kitti-v2-best.pth`; the visual backbone, text
encoder, original PCD blocks, matching heads, MOT boxes, and MOT IDs remain
frozen.  Each frame is predicted and sealed before an optimizer step, and
each `(video, expression)` resets the adapter and optimizer.

Two independent parameterizations are implemented:

- `PCD_RESIDUAL`: zero-gated residual adapters on the last two PCD output
  blocks.
- `PCD_LORA`: zero-output rank-4 LoRA on the Q/V projections of the last two
  PCD attention blocks.

The runner verifies zero-initialization against the frozen source stream,
records nonzero gradients and parameter drift, and has a separate GT-free
loader regression.  Full Residual output is in
`/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_pcd_residual_full_v1/`;
the compact tracked record is
`results/flexhook_v2_dance/pcd/pcd_residual_full_v1.json`.

## Accepted Residual result

| Method | HOTA | DetA | AssA | IDF1 | Dets | IDs |
|---|---:|---:|---:|---:|---:|---:|
| Frozen V2 → Dance | 8.206 | 3.1857 | 21.138 | 4.4360 | 99,316 | 1,740 |
| PCD Residual | 8.014 | 2.4762 | 25.938 | 3.6957 | 80,880 | 1,303 |

Residual adaptation improves association by 4.800 points and reduces IDSW from
259 to 139, but loses detection recall and IDF1.  It is therefore a negative
overall result, not a claimed improvement.  The run used 108,511 optimizer
steps, all with nonzero gradients, across all 425 sequences; target GT was not
opened.  The aggregate elapsed time reported in metadata is the sum of the
four independent episode-shard clocks, not wall-clock time.

The LoRA full-suite result is added here after its TrackEval run completes;
the rank-4 zero-initialization pilot has already passed.
