# ShiftRMOT-Bench-v1 pilot report

Date: 2026-10-21
Repository: `TTA-RMOT` (separate from LocateMOT)
Source model: frozen TempRMOT Refer-KITTI-V2 checkpoint `checkpoint_rk2.pth`

## Executive result

The first absolute referring gate was not viable on this source/shift pair:
three of the four videos emitted no prediction rows. The smallest justified
structure change was to expose valid frozen tracker candidates and rank the
current frame by a fixed referring-logit margin, without forcing top-1. On the
same 45 conditions and 9,915 frame queries, this changed Frozen HOTA from
`0.31255` to `2.9113` and DetA from `0.090851` to `2.3094`.

This is a candidate-visibility recovery, not a successful TTA result. The
TDA-inspired EMA, TPT-inspired query-consistency, and TCR-inspired fusion
labels made no optimizer updates and were identical to Frozen. The
SAR-inspired entropy head made 2,333 updates but decreased HOTA to `2.8306`.
The method labels are explicitly adapted/inspired rather than exact ports; see
`SHIFT_METHOD_REVIEW.md`.

## Complete data release and audit

The full selected Refer-KITTI-V2 official test asset set was built before model
execution:

| item | count |
| --- | ---: |
| videos | 4 (`0005`, `0011`, `0013`, `0019`) |
| base images | 2,069 |
| expressions / base episodes | 862 |
| expression-frame references | 220,422 |
| compact conditions | 4,540 |
| A0 / A1 / A2 / A3 | 862 / 3,448 / 46 / 184 |
| language variants | 46 |

All expression-referenced images and labels were checked. Image roots under
`data/ShiftRMOT-v1/images` are symlinks to the external data2 asset store; no
raw image or transformed image copy is committed. A1/A3 pixels are generated
deterministically at runtime. The audit receipt is
`results/audits/shift_rmot_v1_audit.json`.

`0005` is marked `dev`; `0011`, `0013`, and `0019` are marked
`confirmation_overlap_audit`. These are official test videos with historical
project use, so the protocol does not claim an unseen confirmation split.

## Sealed paired pilot

The pilot selected one language-rewritable expression from each video and ran
all of its A0/A1/A2/A3 conditions. Both structures use exactly the same 45
conditions and 9,915 frame queries. Metrics below are the pinned TrackEval
summary values; prediction counts, optimizer steps, probe calls, FPS, and VRAM
come from the causal runner metadata.

| structure | method | HOTA | DetA | AssA | IDF1 | selected rows | updates | probes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute gate | FROZEN | 0.31255 | 0.090851 | 1.2371 | 0 | 702 | 0 | 0 |
| absolute gate | TDA_RMOT | 0.31255 | 0.090851 | 1.2371 | 0 | 702 | 0 | 0 |
| absolute gate | SAR_VISUAL_INSPIRED | 0 | 0 | 0 | 0 | 84 | 2,333 | 0 |
| absolute gate | TPT_LANGUAGE_INSPIRED | 0.31255 | 0.090851 | 1.2371 | 0 | 702 | 0 | 1,860 |
| absolute gate | TCR_FUSION_INSPIRED | 0.31255 | 0.090851 | 1.2371 | 0 | 702 | 0 | 1,860 |
| relative margin | FROZEN | 2.9113 | 2.3094 | 4.2506 | 0.012108 | 28,064 | 0 | 0 |
| relative margin | TDA_RMOT | 2.9113 | 2.3094 | 4.2506 | 0.012108 | 28,064 | 0 | 0 |
| relative margin | SAR_VISUAL_INSPIRED | 2.8306 | 2.2728 | 4.0007 | 0.0083143 | 26,620 | 2,333 | 0 |
| relative margin | TPT_LANGUAGE_INSPIRED | 2.9113 | 2.3094 | 4.2506 | 0.012108 | 28,064 | 0 | 1,860 |
| relative margin | TCR_FUSION_INSPIRED | 2.9113 | 2.3094 | 4.2506 | 0.012108 | 28,064 | 0 | 1,860 |

Both completed on one A800 80GB card. Absolute pilot used GPU UUID
`GPU-0322c0e7-ffbf-e068-5bdd-587b8216a9d9`; relative pilot used
`GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2`. Peak allocated memory was 7.6214
GiB. Shared-source throughput was 2.706 FPS for the absolute run and 2.404 FPS
for the relative run. The online runner recorded no GT, future-frame, or
evaluation-manifest access.

## Condition-family analysis

The relative-run detailed per-condition averages below are percentages from
the TrackEval detailed CSV (the authoritative aggregate values are in the
table above). The visual shifts were not the main failure: A1 and A3 had
higher candidate-level HOTA than A0 in this small representative sample.

| family | Frozen HOTA | Frozen DetA | Frozen AssA | SAR-inspired HOTA |
| --- | ---: | ---: | ---: | ---: |
| A0 identity | 2.670 | 2.247 | 3.358 | 2.601 |
| A1 visual | 4.335 | 3.995 | 4.909 | 4.295 |
| A2 language | 2.903 | 2.469 | 3.582 | 2.863 |
| A3 visual+language | 4.217 | 3.960 | 4.674 | 4.176 |

The SAR-inspired update path is therefore not retained as an improvement:
it learns from source-confidence entropy but changes the candidate scores in a
direction that lowers HOTA/AssA. The probe-based methods currently have
coverage (1,860 calls) but no supported evidence reaches an optimizer update.
This distinguishes “the probe ran” from “TTA adapted.”

## Reproducibility and scope

The committed compact benchmark and receipts are sufficient to reproduce the
data audit and inspect the sealed pilot. Raw data and the 1.1 GB checkpoint
remain external local assets. The large `results/runs/` prediction trees are
ignored and are not pushed; the compact result tables in
`results/shift_rmot_pilot/` are pushed.

The execution host used a local ignored OpenCV wheel because the base Python
image did not provide `cv2`; a fresh host should install
`opencv-python-headless` and set `PYTHONPATH` to its package directory before
running the scripts.

The current result is a four-video representative pilot, not a claim that all
4,540 compact conditions have been evaluated. The next justified experiment is
to improve evidence coverage or calibrate the source referring head under the
relative candidate protocol, then run the formal dev/confirmation condition
matrix without changing metrics or using GT online.
