# FlexHook Dance same-domain ceiling versus V2 cross-domain result

## Conclusion

The original FlexHook Dance second-stage checkpoint gives the same-domain
reference/upper bound: **HOTA 32.134**. Keeping the official Dance
DETR+NeuralSORT candidate bank fixed and replacing only the second-stage
checkpoint with `refer-kitti-v2-best.pth` gives **HOTA 8.206**. The gap is
**23.928 HOTA points**. Therefore the low V2→Dance result is primarily a
cross-domain second-stage calibration/representation mismatch; the Dance
checkpoint result is an upper-bound comparison, not a TTA improvement.

The vendor's released reference is HOTA 32.165 for the Dance checkpoint and
the same tracker configuration. The rerun obtains 32.134, which is close
enough to validate the protocol while retaining the actual measured value.

## Protocol

- Dataset: Refer-Dance validation, 25 videos, 17 expressions per video, 425
  video-expression sequences, 25,508 frames, and 212,541 GT detections.
- Candidate source: the official released Dance DETR+NeuralSORT tracker bank,
  25 files, 211,085 MOT rows. Both experiments use exactly this same bank.
- Checkpoints: `refer-dance-best.pth` for the same-domain ceiling and
  `refer-kitti-v2-best.pth` for V2→Dance. No Dance fine-tuning was performed.
- Inference: two processes on physical GPUs 3 and 5, validation batch size
  240, with the original FlexHook model and post-processing.
- Leakage control: target GT and expressions were not used to make candidate
  boxes, scores, or identities. GT templates were materialized only for
  post-hoc TrackEval.

## Results

| Metric | Dance checkpoint (same domain) | V2 checkpoint → Dance | Difference |
|---|---:|---:|---:|
| HOTA | **32.134** | 8.206 | **+23.928** |
| DetA | 28.341 | 3.1857 | +25.1553 |
| AssA | 36.491 | 21.138 | +15.353 |
| DetRe | 48.210 | 4.5275 | +43.6825 |
| DetPr | 40.323 | 9.6892 | +30.6338 |
| AssRe | 42.166 | 24.572 | +17.594 |
| AssPr | 65.616 | 72.244 | -6.628 |
| LocA | 91.312 | 92.099 | -0.787 |
| RHOTA | 41.932 | 9.7826 | +32.1494 |
| HOTA(0) | 35.555 | 8.9887 | +26.5663 |
| IDF1 | 27.832 | 4.436 | +23.396 |
| MOTA | -14.824 | -37.171 | +22.347 |
| Detections | 254,118 | 99,316 | +154,802 |
| Predicted IDs | 1,412 | 1,740 | -328 |

The same-domain run produced predictions in 103 of 425 expression sequences;
the V2 run produced predictions in 185 of 425. This is consistent with the
V2 score gate being poorly calibrated for Dance, rather than with a detector
failure: the candidates are unchanged between the two runs.

## Tokenizer correction

An initial rerun under the installed Transformers 5.4 stack silently loaded
the local RoBERTa files with an empty BPE merge table. For example, `person`
was encoded as character-level pieces rather than a normal RoBERTa token.
That invalidated the initial HOTA 8.224 same-domain run and the earlier HOTA
12.254 V2 run. `compat/sitecustomize.py` now constructs the released local
ByteLevel BPE tokenizer directly from `vocab.json` and `merges.txt`; a direct
tokenization regression and both full reruns passed. The old 12.254 record is
marked superseded; the canonical corrected V2 result is 8.206.

## Reproducibility records

- Vendor commit: `bd1acc38634b28525d54dc6e0fcb38335f0029f9`.
- Dance checkpoint SHA256:
  `fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c`.
- V2 checkpoint SHA256:
  `71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`.
- Official tracker receipt SHA256:
  `b8f3133539d885c4174b6c0f7116dcdc9bcb6961a361e501c71fc6d808a269f8`.
- Same-domain merged result SHA256:
  `de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110`.
- V2 merged result SHA256:
  `fbf498a24b73d6b2db6d9c757d313f660c0d5b09a7dbfec5c2635b8ca517ee69`.
- TrackEval summary SHA256: same-domain
  `a0f7922789bad4d7fcf38e0d4b966e24c3de9496b5ce1d7d7a286d6d276aa30d`;
  V2 `aa72d9ec9a44e29dd36ab65a7abf660db2af72215186dcfb39c2ea84c6267e60`.
- Compact machine-readable record:
  `results/flexhook/flexhook_dance_ceiling_vs_v2_corrected.json`.
- Full raw result JSONs and TrackEval directories remain on the local data
  volume; the repository stores compact receipts rather than generated
  hundreds-of-megabytes artifacts.
