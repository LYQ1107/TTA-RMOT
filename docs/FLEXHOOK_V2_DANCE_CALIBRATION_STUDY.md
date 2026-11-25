# FlexHook V2 → Refer-Dance score calibration study

All four methods use the same 25-video, 425-expression official Dance
DETR+NeuralSORT bank and the same corrected V2 source score stream.  Target
GT is unavailable to the prediction process and is opened only by the
separate TrackEval process.

| Method | HOTA | DetA | AssA | IDF1 | Dets | IDs | ΔHOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 8.2060 | 3.1857 | 21.138 | 4.4360 | 99,316 | 1,740 | 0 |
| ONLINE_STABILITY | 8.1741 | 3.1183 | 21.428 | 4.3607 | 98,374 | 1,726 | -0.0319 |
| SOURCE_PRIOR_CALIBRATION | 8.2060 | 3.1857 | 21.138 | 4.4360 | 99,316 | 1,740 | 0 |
| QUERY_TRACKLET_CALIBRATION | 8.2092 | 3.1468 | 21.417 | 4.4059 | 97,682 | 1,647 | +0.0032 |

`SOURCE_PRIOR_CALIBRATION` uses only fixed source statistics and is therefore
source-free at target inference but not source-label-free during development.
`QUERY_TRACKLET_CALIBRATION` uses causal current/history score distributions;
it does not force top-1 and does not use target positive ratios.

The conclusion is negative but useful: score-only calibration cannot explain
or recover the cross-domain gap.  The best HOTA change is +0.0032, while IDF1
falls slightly.  The full evaluator receipts are under
`results/runs/flexhook_v2_to_dance_score_tta_v1/`; the compact record is
`results/flexhook_v2_dance/calibration/calibration_v1.json`.

The GT-based threshold sweep is deliberately kept separate in
`docs/FLEXHOOK_V2_DANCE_FAILURE_ANALYSIS.md` and is not part of this table.
