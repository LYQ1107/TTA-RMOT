# Registered three-seed confirmation

25Val same-set study; not Train→Val or blind test. Fixed caches, budgets and source-val-only checkpoint selection.

| Seed | Naive HOTA | PSR HOTA | PSR−Naive | PSR−Frozen |
|---|---:|---:|---:|---:|
| 17 | 7.4499 | 8.4204 | +0.9705 | +0.2024 |
| 0 | 7.5959 | 8.8701 | +1.2742 | +0.6521 |
| 42 | 7.8064 | 8.8763 | +1.0699 | +0.6583 |

PSR mean±sample std: 8.7223 ± 0.2614.
Naive mean±sample std: 7.6174 ± 0.1792.

Post-hoc parent-weight control HOTA: 8.4255; no independent new weak labels.
Uniform/Confidence/Discovery/Geometry/Temporal/MeanTeacher remain seed17-only; do not claim three-seed superiority over them.
