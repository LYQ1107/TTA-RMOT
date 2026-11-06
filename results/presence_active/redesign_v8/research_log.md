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
