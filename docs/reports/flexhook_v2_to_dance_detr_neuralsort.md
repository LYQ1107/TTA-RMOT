# FlexHook V2 → Refer-Dance with the official Dance tracker

> **Superseded record.** The HOTA 12.254 values below were produced before a
> tokenizer compatibility defect was found. Transformers 5.4 loaded the local
> RoBERTa merge table incorrectly and split words into character-level pieces.
> The corrected canonical result is HOTA **8.206**, and the same-domain
> Dance-checkpoint comparison is documented in
> [`flexhook_dance_ceiling_vs_v2_corrected.md`](flexhook_dance_ceiling_vs_v2_corrected.md).

## Conclusion

This historical cross-domain baseline used the official Dance-specific tracker
bank, not the earlier diagnostic YOLO11x+ByteTrack bank. The tracker bank is
the released `DETR + NeuralSORT` output from FlexHook. The second-stage model
is fixed to the released `refer-kitti-v2-best.pth`; it is not retrained on
Dance.

FlexHook's source code is open at
[`buptLwz/FlexHook`](https://github.com/buptLwz/FlexHook). Its README releases
the best checkpoints and tracker results through the official Baidu share;
the exact Dance resource is `tracker_outputs/DETR-NeuralSORT-Dance`. The
repository's `infer.sh` identifies this configuration as
`refer-dance-best + DETR+NeuralSORT`, with a same-domain reference HOTA of
32.17.

## Protocol

- Dataset: Refer-Dance validation, 25 videos, 17 expressions, 425 evaluated
  video-expression sequences and 25,508 frames.
- Candidate source: 25 official Dance DETR+NeuralSORT tracker files, 211,085
  MOT rows. No YOLO detector is used in this accepted run.
- Second stage: fixed `refer-kitti-v2-best.pth`, with no Dance fine-tuning.
- Inference: source-free; the target GT and expressions were not used to form
  candidates, boxes, scores, or identities. The released loader opens its
  labels file as part of dataset setup, but the Dance test branch does not
  consume those labels for inference.
- Evaluation: GT templates are materialized only for post-hoc TrackEval. The
  historical run used two processes on physical GPUs 7 and 5. TrackEval
  evaluated all 425/425 sequences successfully.

The official tracker files and the 399 MB inference JSON stay on the local
data volume because they are binary/generated artifacts. A verified download
receipt and the compact result record are committed in this repository:

- `results/flexhook/flexhook_v2_to_dance_detr_neuralsort.json`
- `scripts/download_flexhook_official_dance_tracker.py`
- `/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/official_detr_neuralsort_receipt.json`

## Results

| Metric | V2 checkpoint + official Dance DETR/NeuralSORT |
|---|---:|
| HOTA (superseded) | 12.254 |
| DetA | 4.7043 |
| AssA | 31.950 |
| DetRe | 57.811 |
| DetPr | 4.8663 |
| AssRe | 36.485 |
| AssPr | 65.379 |
| LocA | 91.376 |
| RHOTA | 42.962 |
| IDF1 | 5.6069 |
| MOTA | -1063.1 |
| MOTP | 90.667 |
| Dets / GT_Dets | 2,524,962 / 212,541 |
| IDs / GT_IDs | 10,652 / 252 |

## Comparison with the earlier diagnostic run

| Candidate source | HOTA | DetA | AssA | LocA | IDF1 |
|---|---:|---:|---:|---:|---:|
| YOLO11x + ByteTrack, V2 second stage | 5.2445 | 4.4082 | 6.2627 | 89.511 | 1.7578 |
| Official Dance DETR + NeuralSORT, V2 second stage (superseded) | **12.254** | **4.7043** | **31.950** | **91.376** | **5.6069** |
| Official same-domain Dance checkpoint + DETR/NeuralSORT | 32.165 | — | — | — | — |

Relative to the earlier generic diagnostic run, the official tracker improves
HOTA by 7.0095 points, AssA by 25.6873 points, LocA by 1.865 points, and IDF1
by 3.8491 points, while producing 266,732 fewer detections. This confirms
that the very low first result was substantially confounded by the wrong
candidate detector/tracker. The remaining low precision and the gap to 32.165
are expected for the intended V2→Dance cross-domain test: the fixed V2
second-stage score model is not calibrated for Dance, and its association
model was not trained on Dance.

The 32.165 number is not claimed as a cross-domain result. It uses the
Dance-specific `refer-dance-best` checkpoint and is retained only as the
official same-domain reference.

## Reproducibility records

- Vendor commit: `bd1acc38634b28525d54dc6e0fcb38335f0029f9`.
- V2 checkpoint SHA256:
  `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`.
- Tracker receipt SHA256:
  `b8f3133539d885c4174b6c0f7116dcdc9bcb6961a361e501c71fc6d808a269f8`.
- Result JSON SHA256:
  `4ad3342ab7c6c0f2b61bae05ba29235b0712f4ee88f8dbf397e9bfcfae75d5ca`.
- TrackEval summary SHA256:
  `032c9efbc768dadf6cf125a465a49e0524a8244468255f68135dcf23bd60f888`.
- Full research history: [`research_log.md`](../../research_log.md).
