# Positive-Free full-suite status — 2026-10-25

Project: TTA-RMOT only, `/data2/user/FlexHook-TTA`.
User authorized all full target-domain experiments, not source training. Latest explicit instruction
also authorizes sharing occupied GPUs when VRAM fits, preferring four cards with fewest processes.
Branch: `research/v2dance-positivefree-four-gpu-full-20261025`, from verified `a0cb27e`.

## Completed engineering

- FP32 frozen multi-scale visual/text encoder reuse; original C-Hook/PCD decode retained.
- Six real frame probes, including 22 candidates: zero selection/sign mismatches; maximum margin error 2.25067e-4.
- LoRA gradient maximum absolute difference 4.88758e-6 (gradient magnitude 0.339613).
- Weighted complete-video LPT groups: 52,688 / 53,495 / 52,621 / 52,281 tracker rows; 102 / 119 / 102 / 102 episodes.
- Initial idle-only policy used GPU5. Latest user authorization enables occupied cards with at least
  32 GiB free: 24 GiB per-process PyTorch allocator budget plus 8 GiB headroom, then fewest processes first.
  The budget is not a reservation or a guarantee against later allocations by other users.
- Atomic per-episode prediction/margin sealing and completion receipts; partial attempts preserved, whole episode restarted from seed 0.
- Same seed 0 reset at each independent episode, so resume/shard scheduling cannot change LoRA initial state.
- Integration gate: 33 focused CPU tests passed. Cache-enabled ZERO_UPDATE/PF_RANK/PF_IDENTITY/PF_HYBRID smoke passed (0/3/3/3 updates).
- Subsequent evaluator/recovery tooling checks: 37 tests passed, including physical-ID feature-dimension recovery and offline matching/duplicate-FP attribution.

## User-approved correction to the Frozen protocol

Full candidate checking discovered that the old pinned Frozen stream has 210,393 observations,
not all 211,085 fixed tracker observations. All 692 omissions are final observations of tracks;
14 one-observation tracks disappear altogether. Vendor `data/mydataloader.py` uses an exclusive
`range(sample_start, obj_last-frame_num)` and drops every final track observation.
The former preflight checked expressions on existing observations but did not detect missing observations.
That gap is fixed; old raw preflight/smoke records are not rewritten.

Four GT-free GPU probes confirmed positive source margins among omitted rows. User explicitly chose:
**complete all candidates and rebuild the matched Frozen baseline**, retaining old 8.206 as historical only.

The original score JSON and SHA256 `fbf498a24b73d6b2db6d9c757d313f660c0d5b09a7dbfec5c2635b8ca517ee69`
are unchanged. Same strictly loaded V2 checkpoint generated a separate, append-only supplement:

- `/data2/user/FlexHook-TTA/results/runs/pf_frozen_complete_20261025/source_supplement.json`
- SHA256 `3693edc6ef153b8ce0ca87aeec1b196d6b360ff775b8fe44755c4b15d7397639`
- 692 observations × 17 expressions = 11,764 actual two-class logits; 384 source-positive rows.
- Eight additional original-vs-cached complete-candidate comparisons: zero sign mismatches.
- Target GT never opened. Source checkpoint and tracker bank unchanged.

Every full method reads the same original stream + supplement. Supplement overlap is rejected.
The old 8.206 score cannot be used as a directly matched gain reference for the completed bank.

## Formal experiment sequence

| Stage | Status |
|---|---|
| Complete-bank ZERO_UPDATE | 102/425 sealed on GPU5; resume remaining 19 whole videos across four shared GPUs, preserving completed shard0; metrics pending |
| RELATIVE_ONLY (CPU) | after ZERO_UPDATE baseline audit/evaluation |
| PF_RANK | after RELATIVE_ONLY |
| PF_RANK_GUARD | validate exact CPU derivation of PF_RANK pre-update margins |
| PF_IDENTITY | after rank/guard |
| PF_HYBRID | after identity |

For each: all 25 videos / 425 expressions / all 211,085 observations per query bank;
full ordered frames, exact MOT rows, no target GT in adaptation, original defaults, FP32.
GT is opened only by independent evaluation after predictions are sealed/audited.
These are development-target results, not an independent blind test.

No scientific success, HOTA gain or final completion is claimed at this engineering stage.

Resource-policy transition: shard0 completed successfully; the coordinator then waited because
no card met the old idle-only rule. Only our waiting coordinator PID 3682643 was terminated to
apply the user's new sharing instruction. No GPU worker or foreign process was stopped.
Original manifest/status/resource snapshots remain intact; `active_manifest.json` retains shard0
and balances only unstarted videos into four additional logical groups. At most four are active.
Shared-policy regression gate: 39 focused tests passed in 10.71 s; compileall and both shell
entry-point syntax checks passed. Numerical hyperparameters and completed episode receipts are unchanged.

Official summary/detail outputs are published by `summarize_pf_full.py`; per-video pooling uses the official metric classes.
`analyze_pf_full_offline.py` reads GT only after full sealing/audit, reports IoU-0.5 target recovery/loss and duplicate-aware FP plus candidate ranking.
`verify_pf_guard_equivalence.py` selects a source-positive episode without GT and compares a complete direct GPU GUARD episode to CPU-derived results.
Only a passing equivalence enables derived full GUARD; otherwise the suite runs direct full GUARD on safe GPUs.
The legacy `skipped_no_pairs` counter includes interval-skipped frames; new counters separately report interval skips, actual update opportunities, and missing-evidence skips.
