# Refer-BDD frozen baseline

The baseline is the same frozen Refer-KITTI-V2 checkpoint used by every
adaptation method. It is evaluated on all 8 videos, 673 expressions, and
136,188 frame-query rows.

| method | HOTA | DetA | AssA | IDF1 | selected rows | optimizer steps | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 2.9788 | 0.54133 | 16.559 | 0.89107 | 9,183 | 0 | 4.191 |

The baseline prediction directory is
`results/runs/refer_bdd_full_qcs_20261020/predictions/FROZEN/`; the compact
summary is `results/bdd_crossdomain/summary.csv`.

## Candidate visibility before language selection

The runner saw 565,186 valid tracker candidates across the full suite. The
offline source diagnostic compared them with 216,613 sealed target boxes only
after prediction. Weighted per-video IoU-0.5 recall from the diagnostic is
approximately:

| candidate stage | target-box recall |
|---|---:|
| all valid tracker candidates | 26.4% |
| after objectness > 0.6 and area > 100 | 19.8% |
| after the fixed referring probability > 0.4 gate | 0.7% |

The last figure is weighted from the rounded per-video diagnostic reports.
The source referring probabilities over all valid candidates have mean
`0.1262`, median `0.0908`, p95 `0.3088`, and offline IoU-label rank AUC
`0.4251`; the fixed `0.4` gate is therefore a measured source-calibration
failure on BDD, not a missing-data shortcut.

This is the correct lower-bound interpretation: the tracker still exposes
some target objects, but the source language head rejects almost all of them.
The main protocol keeps the fixed gate and reports the resulting negative
cross-domain result rather than tuning it on BDD.
