# TTA-RMOT full research master plan

Date: 2026-10-21
Repository: `/data2/user/TTA-RMOT`
Remote: `https://github.com/LYQ1107/TTA-RMOT`

## Project boundary

This is the TTA-RMOT task and is separate from LocateMOT/GroundJEV. All
commands, code, manifests, checkpoints, and receipts in this plan belong to
the TTA-RMOT repository. Runtime adaptation is source-free and causal:
predict on the current frame, seal the prediction, then update only from
current/history observations. Target GT is opened only by the evaluator after
prediction output is sealed.

## Current evidence and failure diagnosis

The repository already contains a compact ShiftRMOT-Bench-v1 release and a
four-video pilot. It is useful negative/development evidence, not the final
study:

- Refer-KITTI-V2 selected test scope: 4 videos, 862 expressions, 2,069 base
  images, 4,540 compact A0/A1/A2/A3 conditions.
- Only 46 conservative language rewrites were produced; the pilot selected
  one rewritable expression per video and evaluated 45 conditions/9,915 frame
  queries.
- The original absolute gate suppressed almost all candidates. The relative
  margin structure recovered candidate visibility but was not a TTA gain.
- The old runner maps TDA/SAR/TPT/TCR labels to shared or inspired adapters;
  it does not yet constitute four independent implementations. Probe/QCS/TCR
  had zero supported optimizer updates, while the entropy head updated and
  degraded the score.
- `scripts/build_shift_rmot.py` uses expression-label keys as the runtime
  frame list and seeds conditions with expression/condition identifiers. The
  new protocol will separate video processing time from evaluation labels and
  derive paired visual seeds only from `(video, frame, shift, severity,
  global_seed)`.

## Research stages

### P0 — source, data, and protocol audit

1. Hash the official TempRMOT V2 train list, test sequence map, source
   checkpoint, source code commits, and all local dataset roots.
2. Audit complete Refer-KITTI-V2 train/test video and expression counts and
   verify video-disjointness.
3. Audit Refer-Dance, AerialMind, Refer-BDD, and STORM-Bench availability,
   licenses, official splits, and label/trajectory formats. Missing assets are
   downloaded only from their official source and recorded with checksums.
4. Fix the runtime time-axis contract and paired A1/A3 pixel equivalence;
   write explicit leakage and bottom-video clustering receipts.

### P1 — controlled and real cross-domain data

Build a `data/CrossDomainRMOT`-style manifest layer using symlinks to one
physical raw-data copy and online transformations. For Refer-KITTI-V2 and
Refer-Dance, construct A0/A1/A2/A3 over all eligible test expressions with
audited rewrites and shared frame/track records. Add Refer-BDD, AerialMind
VisDrone→UAVDT, and STORM-Bench only after their official protocols are
validated. Report both all-expression coverage and strict paired-subset
coverage; never count repeated conditions as independent videos.

### P2 — official source training and reproduction

Run the official TempRMOT Refer-KITTI-V2 source protocol for all 60 epochs,
with complete source train data, resumable optimizer/scheduler/AMP/RNG state,
and a training-completion audit. First reproduce the released checkpoint with
the author's inference/post-processing/evaluator, then evaluate the completed
single-card source training under the same official protocol. Run FlexHook's
20-epoch Kitti-2 protocol separately with its own source code/configuration.
DKGTrack and additional source datasets follow only after these two lines are
complete or explicitly blocked by missing legal assets/resources.

### P3 — independent online TTA implementations

Keep `FROZEN` as the no-update control. Implement separate causal modules:

1. `T1_TDA_CACHE`: confidence-aware positive/negative temporal cache and
   retrieval/calibration, without gradients unless the registered method
   requires them.
2. `T2_VISUAL`: update a real visual candidate-feature adapter using
   reliability filtering and stability checks.
3. `T3_LANGUAGE`: update a language/text projection adapter on actual RoBERTa
   features and conservative query consistency, not a CLIP prompt alias.
4. `T4_FUSION`: update a candidate-set fusion adapter with multi-positive and
   empty-set handling.
5. `T5_COMBINED` and `T6_ROUTER`: compare registered combinations and a
   source-free shift-aware router with an explicit skip-update action.

Each episode resets adapter, optimizer, memory, and tracker state. Every
update receipt records pseudo-label provenance, positive/negative/unknown
counts, probe coverage, nonzero-gradient calls, optimizer steps, parameter
delta, candidate recall changes, and subsequent-frame effect.

### P4 — complete evaluation and independent confirmation

For every completed source checkpoint, run complete A0/A1/A2/A3 source-test
conditions and the legal real cross-domain targets through the official
TrackEval path. Publish HOTA, DetA, AssA, IDF1, DetRe, DetPr, IDSW, FPS,
VRAM, candidate recall, and evidence coverage. Do not select settings from
the historical BDD diagnostic target labels; reserve a fresh confirmation
scope for final claims.

## Completion gates

The overall goal is complete only when the data manifests and leakage audits
pass, TempRMOT 60 epochs and FlexHook 20 epochs have auditable completion
receipts and official source metrics, every claimed TTA method has a distinct
causal implementation with nonzero-update evidence or a documented failure,
and full official cross-domain tables are sealed. Until then reports must use
`PILOT`, `DEVELOPMENT`, `BLOCKED`, or `SCIENTIFIC_GAIN_UNPROVEN`; never
`TRAINING_COMPLETE` or `REPRODUCED` for partial work.

## Immediate execution order

1. Finish P0 hashes/counts and protocol tests.
2. Patch the ShiftRMOT builder and write the larger cross-domain manifest
   schema without duplicating image bytes.
3. Run the first legal source-model sanity/reproduction on one available
   card, then launch resumable TempRMOT full training only after the source
   split audit is complete.
4. In parallel, implement and unit-test independent TDA/visual/language/
   fusion adapters against the causal runner using source development data;
   do not tune on the final target labels.
5. Keep `research_log.md`, progress tables, failure receipts, and Git commits
   synchronized. Push code and lightweight results to GitHub; keep raw data,
   large checkpoints, and massive prediction trees in the local/NAS paths
   recorded by their receipts.
