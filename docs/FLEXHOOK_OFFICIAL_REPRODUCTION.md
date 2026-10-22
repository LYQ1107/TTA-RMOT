# FlexHook phase of TTA-RMOT

This document belongs to the `TTA-RMOT` branch
`research/flexhook-generalizable-plugin-tta`. It is separate from the
LocateMOT/GroundJEV task. Raw datasets, model weights, tracker banks, and
large prediction directories stay outside Git under `/data2`/`/data3`.

## Fixed official reproduction

The vendor is pinned to FlexHook commit
`bd1acc38634b28525d54dc6e0fcb38335f0029f9`. The official Refer-KITTI-V2
checkpoint is the external file
`/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth`
(995,768,450 bytes; SHA256
`71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613`). The
run used the four official test videos (`0005`, `0011`, `0013`, `0019`) and all
862 expressions with the official two-process configuration on physical GPUs
1 and 4.

The accepted TrackEval summary is:

| method | HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | Dets | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official/FROZEN | 42.526 | 30.629 | 59.186 | 58.550 | 38.310 | 64.690 | 85.048 | 750,834 | 30,871 |

This reproduces the official README HOTA 42.526. The compact receipt is
`results/flexhook/official_refer_kitti_v2.json`; the large official output is
not committed.

An initial one-GPU trial on physical GPU3 reached batch 280/3722 and was
stopped because it was too slow. It produced no accepted metric. The official
two-GPU run completed with exit status 0. The two rank score files had
1,177,280 and 1,178,248 leaf scores, no overlap, and merged to 2,355,528.

## Source-free score TTA baseline

`flex_tta/` wraps the frozen referring score stream and tracker replay. The
`ONLINE_STABILITY` adapter only changes a causal affine score calibration
(scale/bias). It uses a track's previous-frame confidence as a pseudo-label;
it does not open target GT, use future frames, change tracker geometry, or
change tracker IDs. The car/pedestrian ID offset in the serialized rows is the
same normalization used by the official FlexHook merger.

The full run covered 862 expressions and 2,355,528 candidate rows:

| method | HOTA | DetA | AssA | RHOTA | Dets | IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FROZEN | 42.526 | 30.629 | 59.186 | 58.841 | 750,834 | 30,871 |
| ONLINE_STABILITY | 42.525 | 30.625 | 59.191 | 58.855 | 751,448 | 30,869 |

The online line is therefore HOTA-neutral within 0.001: association improves
slightly while detection decreases slightly. It is retained as a baseline,
not presented as an improvement. The compact result is
`results/flexhook/flexhook_score_tta_v3.json`.

Run the prediction stage with:

```bash
PYTHONPATH=. /data2/user/LocateMOT/.venv/bin/python \
  scripts/run_flexhook_score_tta.py \
  --source-result /data2/user/FlexHook-TTA/results/runs/official_refer_kitti_v2_2gpu/result.json \
  --dataset-root /data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2 \
  --tracker-root /data3/user/TTA-RMOT-assets/flexhook/tracker_outputs/Temp-NeuralSORT-kitti2 \
  --vendor-root /data2/user/FlexHook-TTA/vendor/FlexHook \
  --output-root /data2/user/FlexHook-TTA/results/runs/flexhook_score_tta_v3 \
  --methods FROZEN ONLINE_STABILITY
```

The official FlexHook implementation is a two-stage referring scorer over an
external tracker bank. No official FlexHook BDD or AerialMind checkpoint and
tracker bank were available in the audited resources, so this result does not
claim BDD/AerialMind cross-domain generalization or cross-tracker coverage.
Those remain unfinished research items rather than silently inferred results.
