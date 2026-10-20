# QCS-TTA reference implementation map

This is the implementation boundary for the first QCS-TTA pass.  The map is
deliberately tied to the code that was actually inspected, rather than to a
README-only description.  Reference repositories are read-only inputs; no
external repository is modified by this project.

## Cutoff and pinned repositories

The literature/code audit was performed on 2026-10-20 using the latest public
commits visible at the 2026-10-19 cutoff.  The local TempRMOT source used for
execution is the same commit as the public repository:

| item | paper/version | first public date | repository / commit | inspected code |
|---|---|---:|---|---|
| DeYO | ICLR 2024, arXiv:2403.07366 | 2024-03-12 | `Jhyun17/DeYO`, `3d7d2897d864907571ef75edf31f0028d53f1432` | `methods/deyo.py`: `DeYO`, `forward_and_adapt_deyo`, `collect_params`, `configure_model` |
| TDA | CVPR 2024, arXiv:2403.18293 | 2024-03-28 | `kdiAAA/TDA`, `e697fb0c8078cdeff93daa56bcf8860702542069` | `tda_runner.py`: `update_cache`, `compute_cache_logits`, `run_test_tda` |
| DPE | NeurIPS 2024, arXiv:2410.12790 | 2024-10-16 | `zhangce01/DPE-CLIP`, `fecae5a50d7dc39fb944221be390201a56948416` | `main_dpe.py`: `TextResidue`, `PositiveCacheResidue`, `cache_key_value`, `run_test_dpe` |
| ML-TTA/BEM | ICLR 2025, arXiv:2502.03777 | 2025-02-07 | `Jinx630/ML-TTA`, `6ab9c66666b4019895643543ce89742a0c37c0a2` | `clip/custom_clip.py`: `ClipTestTimeTuning.inference`, `is_bind`, `PromptLearner.reset`; `tpt_classification.py`: selection/entropy/TTA routines |
| TCR | ICLR 2025 | 2024-12-18 project/code snapshot | `XLearning-SCU/2025-ICLR-TCR`, `226bb665c420a33b5b232a9c81f57e040908a9e3` | `models/tta_baselines/tcr.py`: `forward`, `forward_and_adapt`, `reset`; `models/blip_tta.py`: queue/loss helpers |
| UATTA | 2026, arXiv:2604.08598 | 2026-04-09 | `nkuzjh/UATTA`, `10b1131bceea937911dc4430139ccbf8224eb377` | `CMP_UATTA/tta/utils.py`: ITC selection/uncertainty; `CMP_UATTA/tta/optim.py`: model/optimizer setup |
| BCP | 2026, arXiv:2606.12925 | 2026-06-16 | paper-level reference; no official code was used or claimed | Method and Algorithm 1 only |
| TempRMOT | arXiv:2406.05039 / Refer-KITTI-V2 release | 2024-06-12 (V2 update 2024-06-19) | `zyn213/TempRMOT`, `6a65640d849fdee4a32bb055945ee34c3b0edeb1` | `models/transrmot_pro.py`, `models/spatial_temporal_reason.py`, `models/qim_pro.py`, `inference.py` |

The public V2 checkpoint is published by the RMOT release referenced by the
TempRMOT README (`wudongming97/RMOT`, release asset `checkpoint_rk2.pth`).
The local legal source run currently has the V1 release asset
`checkpoint_rk.pth`; its SHA-256 is recorded in `docs/ENV_DATA_AUDIT.md`.

## TempRMOT execution chain

The source path used by the wrapper is
`/data2/user/reference_envs/temprmot/source` (a clean copy of the pinned
repository with NumPy compatibility edits in its local TrackEval only).  The
model path is:

1. `TransRMOT._forward_single_image` runs the backbone, text encoder, vision-
   language fusion, deformable decoder, and returns `query_feats`, boxes,
   objectness logits, and referring logits.
2. `load_detection_output_into_cache` copies the decoder output into
   `track_instances.cache_*` and stores `query_feats` as `output_embedding`.
3. `_post_process_single_image` runs `SpatialTemporalReasoner`, then
   `frame_summarization`, then `RuntimeTrackerBase.update`.
4. `QueryInteractionModule` (`models/qim_pro.py`) selects active tracks,
   updates query embeddings, and concatenates fresh queries.
5. `TrackerPostProcess.forward` converts logits to `scores`/`refers`, scales
   boxes to pixels, and removes the raw `pred_*` fields.
6. `inference.py` applies objectness, area, and referring thresholds before
   writing the tracking result.

The snapshot hook is attached to `RuntimeTrackerBase.update`, which is the
stable point after temporal reasoning and physical-ID assignment but before
QIM.  It copies `output_embedding`, `cache_pred_refers`, boxes, objectness,
IDs, and disappearance state.  The hook never writes back to the model state.
This avoids relying on post-process fields that `TrackerPostProcess` deletes.

## Adaptation boundary

QCS-TTA is implemented in the project-owned `rmot_tta/` package.  The frozen
TempRMOT model remains in eval/no-grad mode.  Only an external referring
adapter is trainable:

- `rmot_tta/adaptation/residual_head.py`: `256 -> 32 -> 1` residual MLP with
  zero-initialized final layer, plus a 257-parameter head control;
- `rmot_tta/query/`: deterministic query parsing and at most one equivalent
  and one mutually-exclusive variant;
- `rmot_tta/probes/`: isolated frozen query replay and prediction-box
  alignment;
- `rmot_tta/evidence/`: set-valued POS/NEG/UNKNOWN selection;
- `rmot_tta/memory/`: bounded replay and semantic state versions;
- `rmot_tta/runtime/`: predict-then-adapt episode loop and trace schema.

The online output keeps the source boxes and physical IDs fixed.  Adaptation
only changes referring logits for future frames.  GT is opened only by the
separate evaluator after prediction files are sealed.

## What is intentionally not copied

- DeYO patch/pixel shuffle is not used; its selection/weighting idea is
  translated to query-equivalence and counterfactual checks.
- TDA's cache is not a permanent class table: keys include episode,
  expression, physical ID, timestamp, and semantic state.
- DPE's frozen-encoder/zero-residual principle is used only at the final
  referring head.
- BEM is not reproduced as a sigmoid replacement; any binding control uses a
  discovered positive set and is labeled BEM-inspired.
- TCR/UATTA offline gallery or mutual-top-1 assumptions are not used for
  one-to-many RMOT.
- BCP is marked paper-level because no official code was verified.

## Local provenance

The source-only legal smoke that produced two V1 prediction directories is
`/data2/user/reference_envs/temprmot/outputs/run_metadata.json`.  The new
QCS-TTA runs write their own source path, commit, checkpoint hash, data
manifest hashes, GPU UUID, command, and trace; they do not reuse those output
files as new measurements.
