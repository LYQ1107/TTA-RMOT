# v8 research log

## 2026-11-06 — preregistration, before computation

- Project: `/data2/user/FlexHook-TTA`, `LYQ1107/TTA-RMOT`; remote baseline `48ed3b6f0762da4a5879260b6d2102f767dad519`. The preserved local v7 commit has the identical Git tree `bdff31a527abd50cf62f8842a409fd5fb45d893c`.
- Hypotheses: all-answer, three-stratum hierarchical EM removes outcome-filtered counting; evidence coverage prevents prior-only rejection; balanced BCE prevents single-class slow updates; replay freezes existence/stream state.
- A1 records the user's sole acceptance amendment: rho=1 must reach the clipping ceiling within 1e-9; every other synthetic/main/slow acceptance condition stays fixed.
- Plan: publish preregistration and amendments; implement inherited v8 classes; run CPU tests and v7 exact/offline replay; if successful, pilot C1/C2, then six MINUS plus FULL C1/C4/C6 and REPLAY C1/C4. Stage 4 is conditional on the registered main gate.
- Initial disk audit: data1 under 1 GiB free and data2 about 20 GiB free. CPU phase does not allocate GPU feature caches. No old assets will be deleted or overwritten. GPU-stage storage must be checked against actual requirements.
- No v8 results, target-GT diagnostic, teacher/model invocation, or GPU experiment exists at registration.
