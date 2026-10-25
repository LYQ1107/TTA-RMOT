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

The historical complete-bank comparison reads the original stream + supplement; subsequent
methods use the sealed native ZERO_UPDATE margins as the common matched reference described below.
Supplement overlap is rejected.
The old 8.206 score cannot be used as a directly matched gain reference for the completed bank.

## Formal experiment sequence

| Stage | Status |
|---|---|
| Complete-bank ZERO_UPDATE | COMPLETE 25/425: HOTA 8.2180, DetA 3.1937, AssA 21.1470, IDF1 4.4414; 99,704 rows / 1,769 IDs |
| RELATIVE_ONLY (CPU) | COMPLETE 25/425: HOTA 7.5705, DetA 4.1857, AssA 13.7120, IDF1 5.5746; 375,483 rows |
| PF_RANK | COMPLETE 25/425: HOTA 8.2433 (+0.0253), IDF1 4.4947; 100,582 rows; all four workers exited 0 |
| PF_RANK_GUARD | COMPLETE 25/425: HOTA 8.2444 (+0.0264), IDF1 4.4956; 100,613 rows; CPU-derived after real complete-episode GPU equivalence passed |
| PF_IDENTITY | RUNNING on physical GPU0/4/6/1; initial PIDs 220683/220740/220841/220941; 25 complete-video queue jobs, at most four active |
| PF_HYBRID | after identity |

For each: all 25 videos / 425 expressions / all 211,085 observations per query bank;
full ordered frames, exact MOT rows, no target GT in adaptation, original defaults, FP32.
GT is opened only by independent evaluation after predictions are sealed/audited.
These are development-target results, not an independent blind test.

Four complete methods have official results. Identity/Hybrid are unfinished; no final TTA success or goal completion is claimed.

Full ZERO validation: all 3,588,445 candidate-query observations / 433,636 active-frame steps;
six historical near-zero sign differences, each reproduced with original and cached forwards selecting
the same complete candidate set. Historical maximum margin discrepancy 0.00118494; original/cached
maximum discrepancy on these cases is 9.72748e-5. Independent CPU Frozen replay exactly reproduces
all 425 native ZERO predictions. Official HOTA 8.2180 is the matched baseline, not an adaptation gain.

Offline GT attribution (only after full sealing): 203,906 / 212,541 GT observations are candidate-retrievable
(95.9373%), but Frozen selects only 10,337 of them (conditional gate recall 5.0695%).
RELATIVE_ONLY raises matched GT to 25,210 but FP from 89,367 to 350,273. Its ranking/AUROC is unchanged.
These are completed control results, not a verdict on the still-running gradient-based methods.

Full Rank makes 105,971 finite nonzero-gradient updates, zero nonfinite/zero-gradient steps and
zero rollbacks. HOTA gains only 0.0253 points: diagnostic net GT recovery +155 (158 recovered,
3 lost), net FP +723; AUROC declines from 0.650204 to 0.648328. This is a small selection tradeoff,
not demonstrated new semantics. Guard retains 31 additional Frozen-positive rows, loses no
baseline GT observation, and adds only 0.0011 HOTA over Rank.

Guard equivalence: dancetrack0010 / dancing-person-dressed-all-in-white, full 1,203 active frames,
5,911 selected rows, direct GPU versus CPU-derived predictions byte-identical; 282 optimizer
steps in both traces, no target GT opened. Validation wall 112.43 s, peak allocated 8.905 GiB.
The full CPU derivation takes 7.34 s **in addition to** the complete Rank run (5.1463 GPU hours)
and that GPU validation; it is not a standalone zero-GPU RMOT inference method.

Rank used static groups of 119/119/102/85 expressions, wall 5,541.59 s. Source-only timings did
not predict backward/query cost accurately and left a slow tail. Fresh Identity/Hybrid methods
use 25 whole-video queue jobs, 17 independent episodes each, on at most four registered GPUs.
Model reload overhead is counted; no episode memory is split, and no GT/accuracy guides the queue.
Rank's finished plan/predictions are not rewritten.

Reporting correction only: structural trainable-parameter/PCD fields cannot be summed as
episode counter deltas. Read actual per-shard constants once per replica: Rank has 24,576
trainable parameters and PCD width 768. Raw sealed metadata and all predictions remain unchanged.
The earlier summary's parameter count 0 was a report aggregation error, not absent adaptation.
Reporting-fix regression: all 47 focused tests passed in 11.97 s; compileall and both shell
syntax checks passed. Re-summarization changed no official score or sealed prediction.

Resource-policy transition: shard0 completed successfully; the coordinator then waited because
no card met the old idle-only rule. Only our waiting coordinator PID 3682643 was terminated to
apply the user's new sharing instruction. No GPU worker or foreign process was stopped.
Original manifest/status/resource snapshots remain intact; `active_manifest.json` retains shard0
and balances only unstarted videos into four additional logical groups. At most four are active.
Shared-policy regression gate: 39 focused tests passed in 10.71 s; compileall and both shell
entry-point syntax checks passed. Numerical hyperparameters and completed episode receipts are unchanged.

Actual shared workers launched on physical 6/4/0/1, PIDs 3948080/3948148/3948216/3948326.
The four remaining groups contain 85/68/85/85 episodes; completed shard0 retains its 102.

## Matched Frozen numerical reference

An interim check on the 102 sealed ZERO episodes compared 895,696 margins and found two
historical sign differences close to zero. Maximum absolute margin discrepancy was 8.58307e-4.
Both cases are in video0081: frame908/ID915 and frame760/ID947. Real source-only GPU diagnosis
found the **uncached original forward and cached forward select the same IDs in both complete
candidate sets**. Their queried margins are respectively +1.35899e-5/+5.92172e-5 and
-3.07113e-5/-1.09524e-5, whereas old-file margins have the opposite signs.
Thus these two cases are not evidence that caching changes source decisions; legacy batching
and floating-point execution are not bitwise reproducible. No threshold/epsilon is changed.

Under the user's already approved rebuilt-matched-Frozen protocol, all subsequent methods
use the real complete ZERO_UPDATE pre-update margins as their common frozen reference.
Original historical score JSON, supplement, raw ZERO traces and predictions remain unchanged.
The new reference reuses sealed streams, not synthesized two-class logits or pseudo labels.
An independent CPU score-to-MOT replay must reproduce all 425 ZERO predictions exactly;
this is explicitly not a second independent GPU inference measurement.
All historical sign differences from the full run must first pass original-vs-cached GPU
diagnosis. Any actual original/cached sign difference still blocks the suite.
Only after these checks may independent TrackEval open GT. Original historical audit and
matched-reference audit are retained separately. 41 focused tests now pass (10.34 s).

Subsequent code review found the optional initial-sign gate checked the global optimizer count,
which could miss a later episode after an earlier episode had updated. It now uses this episode's
step count and frame-local sign differences, before backward; all matched-reference GPU methods
enable it. Full audit also rejects such mismatches. 42 focused tests pass in 10.04 s.
The final report generator requires all six complete official results and 425 attribution records
per method. Cost reporting distinguishes summed-worker FPS from scheduler-wall FPS;
the latter includes launcher waits but excludes preflight/evaluation/attribution.

Final pipeline regression: 44 tests passed in 10.77 s, compileall/shell syntax passed.
Matched runs avoid parsing the unused 394 MB legacy JSON in each worker while preserving its hash checks.
Offline GT diagnostics now apply official nonzero-mark / pedestrian-class-1 filtering, verified only on
synthetic rows and evaluator source before full sealing; no real target GT was opened for this change.
Added/removed FP rows are distinguished from total FP change when assignment status changes.

Source-only profiling showed shared-GPU throughput differs (GPU6/4 approximately 37 active frames/s,
GPU0 approximately 29; GPU1 slower). Future fresh GPU methods use whole-video LPT normalized by
these measured capacities and active-frame cost; all 25/425 and numerical defaults stay fixed.
Original/default manifests remain backward-compatible; active plans are immutable on resume.
New/unknown cards use an explicitly labeled median estimate. No GT or accuracy metric drives scheduling.
45 focused regressions pass in 10.06 s; compileall and shell syntax passed. Actual per-method
original/active manifests and launcher summaries will be copied beside published metrics.

Official summary/detail outputs are published by `summarize_pf_full.py`; per-video pooling uses the official metric classes.
`analyze_pf_full_offline.py` reads GT only after full sealing/audit, reports IoU-0.5 target recovery/loss and duplicate-aware FP plus candidate ranking.
`verify_pf_guard_equivalence.py` selects a source-positive episode without GT and compares a complete direct GPU GUARD episode to CPU-derived results.
Only a passing equivalence enables derived full GUARD; otherwise the suite runs direct full GUARD on safe GPUs.
The legacy `skipped_no_pairs` counter includes interval-skipped frames; new counters separately report interval skips, actual update opportunities, and missing-evidence skips.
