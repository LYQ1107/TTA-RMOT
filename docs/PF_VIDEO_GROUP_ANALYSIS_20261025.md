# Video-cluster and query-group analysis

COMPLETE_6_METHODS; 6/6 methods. Paired bootstrap: 25 whole videos, 2000 resamples, seed 0.

Descriptive development-set uncertainty, not frame-independent significance or training-seed uncertainty.

| Comparison | ΔHOTA | Video-cluster 95% interval | Up / down / tie videos | Query-Present ΔHOTA |
|---|---:|---:|---:|---:|
| RELATIVE_ONLY_vs_ZERO_UPDATE | -0.647552 | [-2.105309, +2.373998] | 20 / 5 / 0 | +2.188848 |
| PF_RANK_vs_ZERO_UPDATE | +0.025268 | [-0.008807, +0.142421] | 11 / 8 / 6 | +0.040748 |
| PF_RANK_GUARD_vs_ZERO_UPDATE | +0.026352 | [-0.008558, +0.144534] | 11 / 8 / 6 | +0.042486 |
| PF_IDENTITY_vs_ZERO_UPDATE | +0.599863 | [-0.083812, +2.430322] | 17 / 2 / 6 | +0.878350 |
| PF_HYBRID_vs_ZERO_UPDATE | +0.103265 | [-0.030775, +0.527028] | 14 / 5 / 6 | +0.176088 |
| PF_HYBRID_vs_PF_RANK | +0.077997 | [-0.026526, +0.397497] | 13 / 5 / 7 | +0.135340 |
| PF_HYBRID_vs_PF_IDENTITY | -0.496598 | [-1.953786, +0.068879] | 3 / 15 / 7 | -0.702262 |

Query-Empty HOTA is not meaningful without GT; FP and nonempty prediction rates are reported separately.
All 425 sequences remain in the primary evaluation. HOTA is pooled, not a mean of per-video HOTA.
Per-video deltas and leave-one-video-out influence are in `results/positive_free/full/video_uncertainty.json`; HOTA deltas are not additive video contributions.

| Method | Present accepted updates | Empty accepted updates | Present rollbacks | Empty rollbacks |
|---|---:|---:|---:|---:|
| ZERO_UPDATE | 0 | 0 | 0 | 0 |
| RELATIVE_ONLY | 0 | 0 | 0 | 0 |
| PF_RANK | 10805 | 95166 | 0 | 0 |
| PF_RANK_GUARD (Rank reused) | 10805 | 95166 | 0 | 0 |
| PF_IDENTITY | 8630 | 78048 | 2352 | 19039 |
| PF_HYBRID | 10862 | 97007 | 120 | 80 |

Rule-eligible evidence is not semantic correctness. GT group labels are offline diagnostics, never online gates; Guard rows reuse Rank rather than incur those updates again.
