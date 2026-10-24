# FlexHook V2 → Refer-Dance failure analysis

This report is for the fixed cross-domain protocol on branch
`research/flexhook-generalizable-plugin-tta`.  It uses the corrected
ByteLevel-BPE tokenizer, the `refer-kitti-v2-best.pth` checkpoint, and the
official Dance DETR + NeuralSORT candidate bank.  The 25 videos and 425
video-expression sequences are retained in every result.

## Main diagnosis

The fixed MOT candidate bank is not the primary bottleneck.  Offline GT
diagnostics find candidate recall of 95.749% at IoU 0.5 and 85.071% at IoU
0.75.  In contrast, the V2 score gate selects only 5.046% IoU-0.5 recall at
its unmodified zero threshold.  The corrected frozen result is HOTA 8.206,
DetA 3.1857, AssA 21.138, and IDF1 4.436.

The error is not a simple global temperature problem.  On 352,404 scored
candidate rows, the positive/negative score AUROC is 0.5397, average
precision is 0.6403, and ten-bin ECE is 0.5266.  The median margins of GT
positives and negatives are -10.977 and -11.023, respectively.  Only 3.464%
of rows are positive at the frozen threshold.  Thus a threshold change can
recover recall, but it also exposes a large overlapping negative population;
the ranking representation itself is badly degraded.

The tracker has 4,894 switches over 203,254 adjacent matched links, a switch
rate of 2.408%.  This is a real temporal nuisance, but it is smaller than the
referring-stage gate failure and does not justify changing tracker IDs.

## Development-only oracle threshold sweep

These numbers use target GT after prediction and are not a TTA method, were
not used to choose a formal configuration, and must not be interpreted as a
source-free result.

| Margin threshold | Selected rows | Precision | Recall |
|---:|---:|---:|---:|
| -4 | 47,143 | 75.10% | 17.70% |
| -2 | 26,701 | 79.91% | 10.67% |
| 0 | 12,209 | 84.45% | 5.15% |
| +2 | 4,858 | 93.00% | 2.26% |

The corresponding post-hoc TrackEval sweep reaches HOTA 12.438 at threshold
-4, but this is a GT-informed development control, not an accepted final
TTA number.  It confirms that calibration contributes to the gap, while the
low AUROC shows that calibration alone cannot solve it.

## Query analysis

The 17 Dance expressions are clothing/compositional queries, so the audit
uses auditable multi-label keyword categories rather than forcing one label.
Appearance, clothing, compositional, motion, relational, and unknown labels
are retained.  The aggregate source-selected recall is low in all populated
families; cross-expression top-5 ranking overlap has mean Jaccard 0.698.  This
supports a semantic rebinding/feature adaptation hypothesis, but does not
permit target labels to enter the online update.

## Scope and leakage

The target labels are never opened by prediction or adaptation.  The GT-based
files in `results/flexhook_v2_dance/diagnostics/` are offline diagnostics only;
the evaluator opens GT in a separate process after predictions are sealed.
The official tracker boxes and physical IDs are unchanged.

Machine-readable compact summaries are in
`results/flexhook_v2_dance/diagnostics/`.  The original full diagnostic
artifacts remain under the local `/data2` run directories.
