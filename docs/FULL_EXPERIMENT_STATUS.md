# Full TTA-RMOT experiment status

Status at 2026-10-21: **in progress; not complete**.

Completed:

- ShiftRMOT v2 was built from the full 21-video Refer-KITTI-V2 asset set.
- The official 17-video source split and 4-video/862-expression target split
  were audited, including hashes, time axes, paired visual seeds, and the
  runtime no-GT boundary.
- Independent visual, language, fusion, combination, and non-parametric TDA
  implementations were added and passed CPU synthetic regression checks.
- The official TempRMOT source-training command was prepared for 60 epochs,
  single-GPU execution, `hist_len=5`, and the official train list.

In progress / pending evidence:

- The first full 60-epoch source-training launch on physical GPU7 failed before
  emitting a training log or checkpoint: its launcher disappeared, the run
  command has no terminal status, and GPU7 was occupied by three unrelated
  compute processes at diagnosis. This attempt is recorded in
  `results/source_training/temprmot_refer_kitti_v2_gpu7_launch_failure.json`
  and is not accepted as training evidence.
- The unchanged official 60-epoch source protocol is being retried in the
  separate run directory
  `/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full_retry_gpu1` on
  physical GPU1. It is not marked complete until the completion audit confirms
  the final epoch, optimizer state, and clean command exit.
- After the Transformers 5 compatibility shim passed its tokenizer regression,
  the active retry moved to
  `/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full_retry_gpu1_compat`.
  The latest one-time log snapshot reached epoch 0, iteration 950/5168 with
  finite loss and approximately 5.3 GiB peak memory. This is progress
  evidence only; no checkpoint or completion claim is made yet.
- Official source inference and official TrackEval metrics are pending the
  validated full-training checkpoint.
- Full A0/A1/A2/A3 independent-TTA inference and official metrics are pending
  the same checkpoint.
- FlexHook's external Refer-KITTI-V2 interface has been prepared and audited
  (`labels.json`, 862 official test GT templates, and symlinked images). The
  pinned FlexHook split is 16 train videos and differs from TempRMOT's 17-video
  source split. The audit discloses 181 expression/frame/object references for
  which the raw per-frame box is absent, matching the pinned evaluator's skip
  behavior; FlexHook model training/reproduction has not started.

No `TRAINING_COMPLETE`, `REPRODUCED`, or full benchmark claim is made by this
file. Pilot numbers in older `results/runs/` directories remain historical
diagnostics and are not substituted for the v2 full result.
