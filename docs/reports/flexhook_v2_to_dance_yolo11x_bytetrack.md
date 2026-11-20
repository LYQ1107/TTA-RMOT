# FlexHook V2 → Refer-Dance

## Protocol

This is a complete DanceTrack validation run using the released FlexHook
Refer-KITTI-V2 second-stage checkpoint:

- checkpoint: `refer-kitti-v2-best.pth`, SHA256
  `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`;
- data: 25 DanceTrack validation videos, 425 referring expressions and 25,508
  frames from the public Refer-Dance archive;
- tracker: image-only YOLO11x + ByteTrack, with no access to labels,
  expressions or GT templates;
- second stage: `rope-swin-tiny + roberta`, V2 weight fixed throughout;
- evaluation: post-hoc TrackEval over all 425 video-expression sequences.

The released test loader opens `labels.json` when constructing its dataset
object, but its Dance test branch does not index or consume those labels: the
candidate boxes come from the tracker files, and the second stage receives
images plus the referring expressions. GT templates are used only when
materializing/evaluating the post-hoc TrackEval files. This distinction is
recorded explicitly because the upstream loader does not lazily avoid opening
the annotation file.

The full model output is kept locally at
`/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_yolo11x_bytetrack_compat2`;
large raw outputs are intentionally not committed. The compact receipt is
`results/flexhook/flexhook_v2_to_dance_yolo11x_bytetrack.json`.

## Metrics

| HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | LocA | Dets | GT Dets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.2445 | 4.4082 | 6.2627 | 59.591 | 4.5369 | 10.966 | 12.786 | 89.511 | 2,791,694 | 212,541 |

The complete TrackEval summary additionally reports RHOTA 19.286, HOTA(0)
5.8337, IDF1 1.7578, 5,607 predicted IDs and 252 GT IDs. The run created all
425 prediction files and all 425 GT-template links required by the evaluator;
TrackEval exited successfully.

## Interpretation

The result is a valid zero-shot cross-domain baseline, not a TTA improvement.
The V2 checkpoint produces very low detection precision (4.5369%) and
association accuracy (6.2627%) with this Dance tracker, despite high
localization quality on matched boxes (LocA 89.511). This indicates severe
domain/calibration mismatch: too many DanceTrack candidates survive the V2
second-stage score gate. The official Dance line in the released script is
32.165 HOTA with a different `refer-dance-best` checkpoint and
DETR/NeuralSORT tracker; that number is reference-only and is not an
apples-to-apples comparison.

The failed first TrackEval invocation was caused by an upstream hardcoded
relative Dance image path. It was repaired by linking the audited Refer-Dance
image root for sequence-length discovery and rerunning TrackEval on the same
predictions; no inference output or metric protocol was changed.
