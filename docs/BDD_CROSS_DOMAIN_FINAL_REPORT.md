# Refer-KITTI-V2 → Refer-BDD final report

## Result

The complete official Refer-BDD split was evaluated source-free and online
with one GPU. No BDD supervised training, target-label access, future-frame
access, or evaluator access was used by the online runner.

| method | HOTA | DetA | AssA | IDF1 | ΔHOTA | ΔIDF1 | updates | probe calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 2.9788 | 0.54133 | 16.559 | 0.89107 | 0.0000 | 0.0000 | 0 | 0 |
| ENTROPY_HEAD | 1.5842 | 0.15295 | 16.507 | 0.28707 | -1.3946 | -0.6040 | 31,656 | 0 |
| PROBE_ST | 2.9788 | 0.54133 | 16.559 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 |
| PROBE_CACHE | 2.9788 | 0.54133 | 16.559 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 |
| QCS_TTA | 2.9788 | 0.54133 | 16.559 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 |
| NO_STATE | 2.9788 | 0.54133 | 16.559 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 |

The compact CSV at `results/bdd_crossdomain/summary.csv` is the authoritative
machine-readable copy of the table above.

## Completeness and reproducibility

- Assets: 8/8 videos, 673/673 expressions, 1,619 images, 1,610 official
  label files, 216,613 target-object references.
- Predictions: 6 methods × 673 sealed sequences, 136,188 frame-query rows.
- Runtime: physical GPU 6, UUID
  `GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2`; peak allocated memory 8.44 GB;
  shared-source throughput 4.191 FPS; wall time 32,513 seconds.
- Source checkpoint SHA-256:
  `4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`.

## Interpretation

This is a valid negative cross-domain result, not a data-shortcut result.
Before the fixed referring gate, offline candidate recall at IoU 0.5 was
about 26.4% for all valid tracker candidates and about 19.8% after the fixed
objectness/area gate. After the fixed source referring threshold it was only
about 0.7%, while the source score p95 was 0.3088. The QCS and probe methods
therefore had no supported update supervision; the entropy-only controller
adapted but degraded HOTA and IDF1 substantially.

The next scientifically distinct option would be to obtain/train a
domain-robust source candidate model or run a separately labeled BDD training
experiment. It would not be legitimate to silently change the main threshold
or use sealed GT to claim a TTA gain, so neither was done.

Authoritative artifacts:

- metrics: `results/bdd_crossdomain/summary.csv`, `per_video.csv`,
  `per_expression.csv`
- run receipt: `/data3/user/TTA-RMOT-runs/refer_bdd_full_qcs_20261020/run_metadata.json`
- asset receipt: `results/audits/refer_bdd_assets.json`

The online protocol has no target-label, future-frame, or evaluator access;
the external released checkpoint remains a provenance caveat rather than an
absolute no-leakage certificate. See
`results/audits/refer_bdd_pretrained_leakage_audit.json`.
