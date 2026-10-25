# Video-cluster and query-group analysis

INTERIM; 4/6 methods. Paired bootstrap: 25 whole videos, 2000 resamples, seed 0.

Descriptive development-set uncertainty, not frame-independent significance or training-seed uncertainty.

| Comparison | ΔHOTA | Video-cluster 95% interval | Up / down / tie videos | Query-Present ΔHOTA |
|---|---:|---:|---:|---:|
| RELATIVE_ONLY_vs_ZERO_UPDATE | -0.647552 | [-2.105309, +2.373998] | 20 / 5 / 0 | +2.188848 |
| PF_RANK_vs_ZERO_UPDATE | +0.025268 | [-0.008807, +0.142421] | 11 / 8 / 6 | +0.040748 |
| PF_RANK_GUARD_vs_ZERO_UPDATE | +0.026352 | [-0.008558, +0.144534] | 11 / 8 / 6 | +0.042486 |

Query-Empty HOTA is not meaningful without GT; FP and nonempty prediction rates are reported separately.
All 425 sequences remain in the primary evaluation. HOTA is pooled, not a mean of per-video HOTA.
Per-video deltas and leave-one-video-out influence are in `results/positive_free/full/video_uncertainty.json`; HOTA deltas are not additive video contributions.
