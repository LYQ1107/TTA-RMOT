# Refer-KITTI-V2 to Refer-BDD cross-domain audit

This document records the completed source-free online experiment. The
project root and all prepared assets are on `/data2` as requested.

## Fixed protocol

- Source: Refer-KITTI-V2 checkpoint `checkpoint_rk2.pth`, source commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1`.
- Target: the complete official 8-video/673-expression Refer-BDD split; no
  target-domain supervised training was performed.
- Methods: `FROZEN`, `ENTROPY_HEAD`, `PROBE_ST`, `PROBE_CACHE`, `QCS_TTA`, and
  `NO_STATE`.
- Main inference gates: objectness `0.6`, box area `100` pixels, referring
  probability `0.4`, absolute selection. No relative-rank gate or zero-gate
  lower bound is used in the main result.
- Reset boundary: `(dataset, video, expression)`. The online runner receives
  only images and expression text, never GT, IDs, future frames, or the
  evaluator manifest.
- Evaluation: the six sealed prediction sets were evaluated after prediction
  completion with the pinned TempRMOT TrackEval fork.

## Environment and execution

GPU selection followed the requested policy. Physical GPU 6 had about 62.9
GiB free at launch and one unrelated 18.2 GiB `sglang` process; it was not
killed. The experiment used `CUDA_VISIBLE_DEVICES=6`, logical `cuda:0`, UUID
`GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2`. The hybrid runtime used system
Torch/CUDA 2.10 with the pinned TempRMOT extension and the EchoTrack OpenCV
environment; this was required to avoid the extension ABI mismatch seen with
the alternate Torch 2.3 environment.

The complete run is
`results/runs/refer_bdd_full_qcs_20261020/`. It processed 136,188 frame-query
rows in 32,513 seconds, emitted one candidate-diagnostics and one update-trace
row per frame-query, and completed all 673 episodes.

## Boundary fixes recorded during execution

1. A four-frame smoke run exposed an extension ABI mismatch; the runtime was
   switched to the compatible hybrid environment without changing model
   weights.
2. The first evaluator smoke used a `+1` frame offset and produced an invalid
   timestep. The official EchoTrack writer was checked directly; the config
   was corrected to zero offset and top-left normalized `xywh`, then the smoke
   evaluation passed for all methods.
3. The offline diagnostic initially interpreted the per-file label placeholder
   column as a frame number. It was corrected to use the manifest/frame
   filename, and the full diagnostic was regenerated with 216,613 target boxes.
4. The runner's episode-reset counters were repaired in code and the existing
   sealed output was corrected transparently from `update_trace.jsonl`; the
   previous final-episode values remain in `run_metadata.json` under
   `counter_correction`.

The machine-readable asset receipt is
`results/audits/refer_bdd_assets.json`; the final metrics are in
`results/bdd_crossdomain/summary.csv`.
