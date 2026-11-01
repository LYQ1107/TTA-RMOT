# Execution record

- Identity: `/data2/user/FlexHook-TTA`, LYQ1107/TTA-RMOT. Clean base
  ecf57122; independent requested research branch; no LocateMOT work.
- Fixed protocol first published at dbd701e75a34dfbb34da0ad4545485ffd0cebe50.
  User clarified E denominator zero: empty output, separately reported.
- Six source score banks audited (2,550 original target margin files);
  X0/X1 exact byte reproduction: all six margins, 12 sets, 5,100 files PASS.
  Existing KITTI banks reused, no source model scoring/GT calibration.
- Frozen teacher manifest: 432 frame-contexts, 89,961 logical requests:
  Dance normal 29,410; early 27,693; KITTI normal 32,858. Repeated physical
  frames/requests use content cache; distinct-frame voting does not count
  duplicate slots twice. Source call caps not triggered in any video.
- Initial new CPU tests: 7 passed, 2 failed because sklearn absent.
  Installed scikit-learn 1.7.2, joblib 1.5.2, threadpoolctl 3.6.0 into
  this task's `results/runs/gated_decode_20261101/deps`, without modifying
  the shared environment. All nine new tests then passed (1.46 s).
- Real frozen T1 smoke PASS on Dance and KITTI image paths (6 calls).
  GPU 2 admitted with 81,154 MiB free, zero compute processes. Unchanged
  backend and crop builder, 320/margin .2; kernel whitelist + annotation
  audit. Full frozen sparse teacher job follows smoke, max one GPU.
- No student/teacher training, no changing thresholds/seeds, no foreign
  process interruption. All target/source gates and outputs must seal
  before new GT evaluation. Full results and final costs follow on completion.
- Integration regression: 32 passed (new 9 + prior Xquery/count_structure/v2
  tests), 1.38 s. Seven new scripts compile; git diff --check PASS. Frozen
  teacher backend, crop builder, and vendor tree unchanged.
