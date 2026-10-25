# FlexHook V2→Dance: positive-label-free TTA — code-preparation handover

**Date:** 2026-10-25. **Status:** CODE_ONLY, **NO** full source training, **NO** full Dance inference or TrackEval executed by this patch's author. Tests were run on a local CPU-only Python environment with synthetic score/tracker fixtures. Real checkpoint compatibility and GPU correctness require a later smoke check on the A100 server.

## Exact scientific claim and legal data split

- Source: official `Refer-KITTI-V2` source train, **all 17 source videos**, if new source supervised training is ever authorized. The published `refer-kitti-v2-best.pth` already supplies a source-trained model; **do not re-train it just to run Source-Free TTA**.
- Source test: official V2 4 videos / 862 expressions for cross-check, not a target-training split.
- Target: **Refer-Dance validation**, official 25 videos / **425 video-expression episodes**, fixed public DETR+NeuralSORT tracking bank. Online test-time updates see only current/past images, expression, track IDs, boxes, and source-model scores. **Dance GT is never an adaptation input**.
- Original source checkpoint SHA256: `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`.
- Frozen baseline HOTA **8.206**; previous standard PCD LoRA **7.8654**; positive vs negative prior hard labels **15,047 vs 849,901**. Do **not** present any code test as a metric gain.
- Dance validation has already been inspected with GT for research diagnostics. Label it development unless an independent confirmation set is available. Offline GT analysis is never used to select the test-time optimizer's pseudo labels.

## Implemented research hypotheses and corresponding code

| Method | Runner option | Actual input / update | Risks / interpretations |
| --- | --- | --- | --- |
| Frozen PCD control | `ZERO_UPDATE` | Reconstruct original source-only PCD; **0** optimizer steps | **Must reproduce 8.206** in an authorized full run before calling any learned method comparable |
| Frozen relative score selector | `RELATIVE_ONLY` | No gradient; robust per-query current/past score tail, may predict 0 or >1 targets; starts after historical support | May increase false positives / is heuristic, not learned semantic correction |
| Pairwise structure LoRA | `PF_RANK` | Frozen-source history determines stable pair *ordering*, no target class pseudo labels; LoRA adapts Q/V | Historic relative order may already be wrong; cannot create missing clothing knowledge |
| Rank with source-positive guard | `PF_RANK_GUARD` | `PF_RANK` output OR original frozen positives | **Selection/postprocessing ablation**, not representation gain; report false positives separately |
| Track-ID feature InfoNCE | `PF_IDENTITY` | Previous physical MOT ID feature as contrastive positive; other IDs as negatives. This is **not a positive language match** | ID switches, look-alike people, query-semantic collapse; still only a hypothesis |
| Rank + self-supervision | `PF_HYBRID` | Both losses; no hard language positive/negative labels | Distinct supervision contributions need matched ablations |

Identity contrastive updates target PCD output hidden representation. Rank pseudo-pairs only preserve evidence from *prior frozen source ranks*, NEVER declare a higher rank to be a true referring positive. If no reliable pairs/identity history, skip optimization. Gradients do not touch frozen visual/text backbones or existing original PCD parameters.

**Not implemented**: REST and ReEx full source objectives, genuine ReEx LLM-based semantic repair, faithful source-train full-supervised adapter training, source label calibration, target-domain validated hyperparameters. Calling one of these implemented baselines "REST" or "ReEx" would be inaccurate; the code names are purposefully distinct.

## Patch files and merge points

Extract the patch archive into the already existing worktree `/data2/user/FlexHook-TTA`, preserving existing files. It **adds new files only**:

```
flex_tta/adaptation/positive_free.py
flex_tta/adaptation/identity_contrast.py
flex_tta/diagnostics/recoverability.py
scripts/run_flexhook_positivefree.py
scripts/pf_preflight.py
scripts/pf_recoverability_dev.py
scripts/plan_pf_source_train.py
scripts/run_pf_4gpu.sh
tests/test_positive_free.py
tests/test_pf_runner.py
tests/test_pf_identity.py
tests/test_pf_recoverability.py
docs/FLEXHOOK_POSITIVEFREE_PREP.md
```

Reuses existing `flex_tta/dance_io.py`, `flex_tta/runtime/dance_episode.py`, `flex_tta/adaptation/pcd.py`, `scripts/run_flexhook_dance_pcd_tta.py` and existing `scripts/merge_flexhook_pcd_shards.py` + `scripts/evaluate_flexhook_dance_tta.py`. **Does not edit the original project, tracker bank, weights or old evidence.**

## Execute NOW (code preparation and limited smoke only)

**First** prepare isolated branch **on the actual server** (only Codex/SSH there can do that; our CPU sandbox is not the A100 server):

```bash
cd /data2/user/FlexHook-TTA
git fetch origin
git switch research/flexhook-generalizable-plugin-tta
git switch -c research/v2dance-positivefree-prep-20261025
# Extract the patch ZIP's contents here. It adds new files only.
python -m pytest -q tests/test_positive_free.py tests/test_pf_runner.py tests/test_pf_identity.py tests/test_pf_recoverability.py
python -m compileall -q flex_tta scripts
bash -n scripts/run_pf_4gpu.sh
```

**Then** inventory all legal assets without opening target GT:

```bash
python scripts/pf_preflight.py \
  --dance-root /data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance \
  --tracker-root /data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs \
  --checkpoint /data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth \
  --check-checkpoint-hash --strict \
  --output results/pf_preflight_20261025.json
```

If available, add `--source-train-manifest PATH` to verify all 17 official V2 source-train videos; do not substitute test/val GT for source train.

**No GPU:**

```bash
python scripts/run_flexhook_positivefree.py \
  --scope plan --method RELATIVE_ONLY \
  --output-root results/runs/pf_preview
```

**Limited GPU smoke only** (choose one available physical GPU after inspecting existing processes; do not touch other jobs):

```bash
export PYTHONPATH="/data2/user/FlexHook-TTA/compat:/data2/user/FlexHook-TTA:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=<ID> python scripts/run_flexhook_positivefree.py \
  --scope smoke --method ZERO_UPDATE --max-frames 12 \
  --strict-initial-sign --output-root results/runs/pf_zero_smoke
CUDA_VISIBLE_DEVICES=<ID> python scripts/run_flexhook_positivefree.py \
  --scope smoke --method PF_RANK --max-frames 12 \
  --output-root results/runs/pf_rank_smoke
CUDA_VISIBLE_DEVICES=<ID> python scripts/run_flexhook_positivefree.py \
  --scope smoke --method PF_IDENTITY --max-frames 12 \
  --output-root results/runs/pf_identity_smoke
```

Every `--scope smoke` is forcibly capped to **one expression and 12 frames**. The actual GPU smoke may need to modify only compatibility scaffolding; all numbers must stay attached to their true commands and code versions. A smoke is not evidence of HOTA gain.

**Development-only offline recoverability (GT is opened by this command):** after frozen predictions are sealed, use `--i-allow-offline-gt` with `scripts/pf_recoverability_dev.py`. Results are NOT valid TTA pseudo labels. The diagnostic reports conditional top-1/3/5 retrieval rate versus a null expectation accounting for candidate count and multiple GT positives.

## Four-GPU future full experiment plan — NOT authorized now

Run only once the user explicitly says to start full evaluation or full source training. The launcher refuses work without two authorization switches and checks occupancy; **do not set these switches during preparation**.

```bash
PF_GPUS=4,5,6,7 PF_METHOD=PF_RANK \
  PF_REPO=/data2/user/FlexHook-TTA \
  PF_RUN_ROOT=/data2/user/FlexHook-TTA/results/runs/pf_rank_full \
  bash scripts/run_pf_4gpu.sh    # PRINTS SHARDS ONLY
```

When explicitly authorized in a future turn, set both `PF_LAUNCH_FULL=YES` and `FLEXHOOK_FULL_RUN_AUTHORIZED=YES` and run `bash scripts/run_pf_4gpu.sh --launch-full`, using four *actually available* GPU IDs. This processes all 25 target videos and all 17 expressions per video (425 episodes) without source/target supervised training; adaptations occur **online within each episode**. Run one method at a time and save a unique fresh root.

After a valid full run, merge and evaluate using existing utilities (GT only after predictions are sealed):

```bash
python scripts/merge_flexhook_pcd_shards.py \
 --shards-root /data2/user/FlexHook-TTA/results/runs/pf_rank_full \
 --output-root /data2/user/FlexHook-TTA/results/runs/pf_rank_full_merged \
 --method PF_RANK

# Ensure the official Dance GT symlink used by the existing evaluator is
# correctly configured; run the evaluator only after a completed run.
python scripts/evaluate_flexhook_dance_tta.py \
 --output-root /data2/user/FlexHook-TTA/results/runs/pf_rank_full_merged \
 --dataset-root /data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance \
 --vendor-root /data2/user/FlexHook-TTA/vendor/FlexHook \
 --methods PF_RANK
```

Optional **full source training** is a *separate*, not-yet-authorized study. `scripts/plan_pf_source_train.py` only prints the official `train-kitti2.yaml` command for all 17 source videos; it cannot start training. Four-GPU global batch 16 differs from the original two-GPU global batch 14; do not mislabel it an exact reproduction.

## Explicit acceptance gates

1. CPU synthetic tests pass; 4-GPU full launcher defaults to dry-run and refuses unauthorized launch.
2. On server: file inventory complete 25 videos ×17 expressions, official frozen checkpoint hash, RoBERTa tokenizer regression, no labels used by inference.
3. Single-GPU smoke: no exceptions, no OOM, exactly one expression ≤12 frames, sealed predict→adapt ordering, optimizer updates only when stable rank pairs or same-ID temporal features exist.
4. Before full comparisons: **ZERO_UPDATE fully reproduces Frozen HOTA 8.206** and matches major metrics and prediction counts. Source-only score mismatch in the smoke must be reported; no silent threshold changes.
5. First full non-gradient `RELATIVE_ONLY` and eventual full PCD methods are evaluated on **all 425** episodes with identical fixed MOT candidates. Report HOTA, DetA, AssA, DetRe, DetPr, IDF1, FPS, optimizer steps, UNKNOWN/skips and changes in predicted rows.
6. No improvement claim unless HOTA gains survive full coverage and do not merely arise from deleting positive tracks. Do not select final hyperparameters using exposed Dance GT and then call the same set an independent blind test.
7. Keep all full-source/full-target runs **paused until explicit user authorization**. Only push code, synthetic tests, smoke metadata and plans during preparation.

## Practical research risks

- If source PCD's per-query candidate ranking is nearly random, rank preservation cannot create lost clothing semantics; `PF_RANK` is a falsifiable negative-control hypothesis rather than a guaranteed solution.
- MOT track ID contrastive learning may stabilize person identity without improving language matching; `PF_IDENTITY` must be independently ablated.
- Per-query relative-tail selection can select false positives when the target set is empty; protect zero-target support, log FP/DetPr, and never force top-1.
- Do not claim this patch implements exact REST or full ReEx. It is an independently named, source-checkpoint-only, positive-label-free research pilot.
