# v8 research log

## 2026-11-06 — preregistration, before computation

- Project: `/data2/user/FlexHook-TTA`, `LYQ1107/TTA-RMOT`; remote baseline `48ed3b6f0762da4a5879260b6d2102f767dad519`. The preserved local v7 commit has the identical Git tree `bdff31a527abd50cf62f8842a409fd5fb45d893c`.
- Hypotheses: all-answer, three-stratum hierarchical EM removes outcome-filtered counting; evidence coverage prevents prior-only rejection; balanced BCE prevents single-class slow updates; replay freezes existence/stream state.
- A1 records the user's sole acceptance amendment: rho=1 must reach the clipping ceiling within 1e-9; every other synthetic/main/slow acceptance condition stays fixed.
- Plan: publish preregistration and amendments; implement inherited v8 classes; run CPU tests and v7 exact/offline replay; if successful, pilot C1/C2, then six MINUS plus FULL C1/C4/C6 and REPLAY C1/C4. Stage 4 is conditional on the registered main gate.
- Initial disk audit: data1 under 1 GiB free and data2 about 20 GiB free. CPU phase does not allocate GPU feature caches. No old assets will be deleted or overwritten. GPU-stage storage must be checked against actual requirements.
- No v8 results, target-GT diagnostic, teacher/model invocation, or GPU experiment exists at registration.

## Stage 0a — registered stop, 2026-11-06

- CPU infrastructure checks: 9 passed, 2 recovery cases deselected, 5.33s. Full suite with maxfail=1: 9 passed/1 failed, 7.59s; the rho=1 boundary case was not run. Counts overlap, not additive.
- First recovery fixture (rho=.1, seed0, registered 200-video online stream) stopped after video5: the fixed 200-iteration fit did not reach tol1e-7; last_delta8.198819538840496e-7, rho_hat.23273205643681252, pi=(.43335323219726707,.15119714931447706,.5567154949779427). 85 sentence records, track counts108/477/48. This is an intermediate state, not completed 200-video accuracy.
- Keep the failed receipt and all acceptance conditions unchanged. No extra iterations, alternative fixture/seed, later rho case, real GT diagnosis, model/teacher invocation, GPU pilot, or stage2/3 was run after this stop.
- Added inherited v8 belief/EM, balanced BCE and frozen-existence runner draft plus CPU tests/offline entrypoint. Only the nine CPU contracts are validated; runner/scheduler/GPU integration is not declared complete. The normal guard subprocess opens no GT; its separate intentional denial test counts1 and is not an online GT read.
- Numerical cause observed: finite iteration budget exhausted before the registered tolerance, not the amended clipping-boundary assertion. The scalar one-step formula regression passes. General convergence/consistency or final recovery quality is not established; no hyperparameter or acceptance repair is inferred.
- Publish code, exact failure data, test transcript and stopped report; wait for explicit user direction as required by the preregistration.

## A2/A3 — preregister before any real v8 data result, 2026-11-06

- User explicitly authorizes iters5000 (tol1e-7 unchanged), diagnostic-only unclipped posterior monotonicity and last-delta contraction rate. Still stop if5000 fails.
- Per-stratum pi precision is tested only with >=200 unique answered pairs; otherwise UNIDENTIFIED_INSUFFICIENT_DATA, with decision-likelihood participation counts. Rho/A1 and old-v7 pi0 regression remain unchanged. Real six-direction phase0 tables use the same identification marker.
- User-provided expected synthetic numbers are motivation to reproduce, not this run's measured results. Do not change fixture/seed/rho gate after seeing results. Preserve all pre-A2 receipts; use amendments_A2_A3 for new CPU records.
- Plan: publish this amendment first; implement and rerun the full CPU suite. Only if it matches the amended acceptance continue exact v7/offline real-data diagnostics, then pilots and stages2/3. V7 stays immutable, no unregistered jobs.

## CPU phase0 and sampler conformance, before GPU

- A2/A3 full suite13 passed160.40s, then an added live batching check and full14 passed159.55s. Original rho=.1 point(.10297748; .80239465/.30663253/.05), rho=1(.95;.81417373/.32032843/.25619209), last stratum3 answered pairs/14 likelihood terms UNIDENTIFIED. Objective monotonicity diagnostic is false for109 fits in rho=.1; record this, do not add a gate or claim general EM monotonicity.
- Exact historical v7 replay148 videos, maximum error0. Real fixed-answer EM replay148/148 converged, maximum96 iterations, no GPU/teacher or GT opens. Independent GT diagnostics3506 query rows,18 strata,3142 KITTI answer rows; EM still overestimates real pi, and approximate KITTI absence can worsen. These have no registered stage0 numerical gate and do not authorize changing parameters.
- Code inspection (not target-GT tuning) found stale within-round candidates: a GLANCE No can make a query low_tentative after questions_v8 already built the round list. Added the original registered no-IDENTITY-before-coverage filter at dispatch in both live and synthetic loops. Same seed/priors/budgets/thresholds; this is a protocol-conformance fix, not a new acquisition rule. Add a failing-before-fix regression and rerun the complete CPU suite. Preserve the earlier passing synthetic files; new conformance-run files go to amendments_A2_A3/sampler_conformance. If amended recovery fails, stop as instructed.
- First frozen preflight omitted the explicit v8 execution-manifest env; original authorization rejected it before any output. Rerun with the proper env passed all six first videos and exact PRES_TOPK rows. New-cache cost bound61.4072 GPU-hours,51,374,543,012bytes; data1 had153,770,381,312bytes, so current fixed11 scope fits without deleting old assets.

- Conformance rerun: complete15 passed152.42s, no changed acceptance criterion. Rho=.1 is now.11737508749, pi=(.78706954532,.26551875809,.07160452323), all three identified/within.07; counts5552/10461/4937, max183 iterations,118 nonmonotone diagnostic fits. Rho=1 remains.95 with k2 only3 pairs/14 decision terms UNIDENTIFIED. The changed adaptive stream is disclosed, not relabeled as the exact old answer sequence. Previous13/14-pass receipts remain, counts overlap.
- GPU numerical code keeps the prescribed v8 values and original teacher/input paths; new helpers isolate execution, preserve first2 pilots for resume, and fix REPLAY existence snapshots. Stage2/3 plan is exactly11 jobs, not the previous v7 cost-fallback set. Additional logging records dispatched low_tentative status, whole-pool round-start posterior histograms and per-step16/16 sources; no target-GT-selected hyperparameter changes.
