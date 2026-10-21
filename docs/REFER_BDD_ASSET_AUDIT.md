# Refer-BDD asset audit

Audit date: 2026-10-20. The audit is complete and covers the official
EchoTrack Refer-BDD test split selected by `assets/seqmap_bdd.txt`.

## Sources and immutable receipts

- Official code and split: [EchoTrack](https://github.com/lab206/EchoTrack),
  [official BDD seqmap](https://raw.githubusercontent.com/lab206/EchoTrack/main/assets/seqmap_bdd.txt).
- Official asset distribution: [EchoTrack Google Drive folder](https://drive.google.com/drive/folders/12jXjL9wJ9keB-vD-kWKI8IzK6mrIfmAN?usp=sharing).
- The two downloaded archives were copied into the TTA-RMOT-owned path
  `/data2/user/TTA-RMOT/data/downloads/ReferBDD/` and passed `unzip -tqq`:

  | archive | bytes | SHA-256 |
  |---|---:|---|
  | `refer-bdd.zip` | 2,863,924,583 | `7af91415c71a8c1eb802e2b9295e86514195f745fb0b4f98af89b69140830759` |
  | `label_with_ids.zip` | 8,119,776 | `288b402c6180a96ac8b1c2653eda5b1c6f6e093651bf32ebb7620a11aa7195f6` |

- The source checkpoint is the TTA-RMOT external asset
  `/data3/user/TTA-RMOT-assets/checkpoints/temprmot_refer_kitti_v2_official_checkpoint_rk2.pth`, SHA-256
  `4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`.

## Split and format

| item | audited value |
|---|---:|
| videos | 8 |
| expressions | 673 |
| images | 1,619 |
| label files | 1,610 |
| expression-frame references | 74,924 |
| target-object references | 216,613 |
| image size | 1280x720 for every selected image |
| seqmap SHA-256 | `9cb1ba2a8dc145bfa84612dbb5a7f04948745b5a71dfa191a449a9ecdb64bdf4` |

The eight videos are `0000f77c-cb820c98`, `00abd8a7-ecd6fc56`,
`00fcfec9-7ba5dabf`, `02a17951-3d88d95e`, `002b562f-e0ac84fe`,
`00f7eac8-5193d600`, `0081da60-5fa22cc6`, and `00de601c-858a8a8d`.
Images use one-based frame numbers in the filename. Official
`label_with_ids` rows contain normalized top-left `xywh`; the first column in
each per-image file is a local placeholder and is not a global frame number.
The evaluator therefore uses the manifest filename/frame index and writes
pixel-space MOT boxes with `trackeval_frame_offset=0`.

The 9 missing label files are legitimate empty-object frames: all expression
references that require target IDs have a corresponding label file. No labels
were synthesized and no raw data, images, archives, or checkpoint are part of
the code/result publication.

## Runtime/evaluation boundary

`results/manifests/refer_bdd_runtime_manifest.jsonl` has 136,188 rows and no
label, expression-path, target-box, or target-ID fields. The evaluator-only
manifest has the same rows and opens labels only after prediction is sealed.
The online runner metadata records `gt_used_by_runner=false`,
`future_frames_used=false`, and `evaluation_manifest_opened=false`.

The machine-readable receipt is
`results/audits/refer_bdd_assets.json`.

The pretrained-checkpoint leakage boundary is recorded separately in
`results/audits/refer_bdd_pretrained_leakage_audit.json`: the runtime protocol
is GT-free, but the complete external checkpoint training history is not
independently auditable.
