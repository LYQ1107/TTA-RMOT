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

- Full 60-epoch source training is running in the external run directory
  `/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full` on the selected free
  GPU. It is not marked complete until the completion audit confirms the final
  epoch, optimizer state, and clean command exit.
- Official source inference and official TrackEval metrics are pending the
  validated full-training checkpoint.
- Full A0/A1/A2/A3 independent-TTA inference and official metrics are pending
  the same checkpoint.
- FlexHook reproduction has not started; it is a later protocol stage.

No `TRAINING_COMPLETE`, `REPRODUCED`, or full benchmark claim is made by this
file. Pilot numbers in older `results/runs/` directories remain historical
diagnostics and are not substituted for the v2 full result.
