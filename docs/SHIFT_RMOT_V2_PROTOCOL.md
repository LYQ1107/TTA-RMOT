# ShiftRMOT v2 protocol

This is the TTA-RMOT cross-domain benchmark. It is separate from LocateMOT and
GroundJEV. The source model is TempRMOT trained on the official Refer-KITTI-V2
train list; the target evaluation uses the official Refer-KITTI-V2 test
expressions from videos `0005`, `0011`, `0013`, and `0019`.

## Dataset and split

The builder reads the existing Refer-KITTI-V2 assets and creates compact
manifests. Images remain symlinks to the source dataset; visual shifts are
reconstructed deterministically at runtime, so the repository does not contain
duplicate image copies.

| item | value |
|---|---:|
| source videos | 17 |
| target/test videos | 4 |
| all Refer-KITTI-V2 expressions | 9,758 |
| official target expressions | 862 |
| source train image paths | 5,172 |
| base source frames in manifests | 8,008 |
| compact conditions | 22,914 |
| A0 identity | 862 |
| A1 visual shift only | 6,896 |
| A2 language shift only | 1,684 |
| A3 visual + language shift | 13,472 |

The official split artifacts are recorded with SHA256 in
`data/ShiftRMOT-v2/metadata/official_protocol.json` and in the audit receipt
`results/audits/shift_rmot_v2_audit.json`. The raw dataset and generated image
cache are intentionally ignored by Git.

## Causal protocol

The runtime loader exposes only image paths, query text, shift metadata, and a
deterministic seed basis. It does not expose label paths or expression JSON.
The runner processes the complete source-video time axis in order. Ground truth
is expanded only by `scripts/evaluate_predictions.py`, after prediction files
already exist, using the official expression label-frame range.

The visual seed basis is independent of expression and language condition IDs:

```text
hash(video_id, frame_index, shift_id, severity, global_seed)
```

Consequently A1/A3 share visual realizations, A0/A2 share identity pixels, and
A2/A3 share query text. Every condition is reset at the
`video/expression/condition` boundary, and adaptation is predict-then-observe-
then-update with no future-frame access.

The manifest label metadata is `xywh_topleft`: normalized top-left `x1, y1,
w, h`, matching the pinned official TempRMOT KITTI loader. This metadata is
used only by the post-hoc evaluator; the runtime compact manifest contains no
label or expression paths.

## Disclosed limitation

The raw target expression set has two selected same-label natural-language
variants for 825 expressions, one for 34 expressions, and none for 3
expressions. Six variants use a conservative rewrite because an existing
same-label expression was unavailable. This is retained in the audit rather
than silently dropping expressions or claiming complete language coverage.
