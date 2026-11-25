# QCS-TTA protocol (first executable pass)

## Research plan

1. Reproduce the pinned TempRMOT source on one GPU and verify the snapshot
   linear head against the source `track_ref`.
2. Run source-only and matched controls on the complete V1 0012/0014 target
   partition, then run the same chronological stream with frozen query probes
   and an external referring adapter.
3. Seal predictions before opening evaluation labels; report official
   TrackEval metrics where the local evaluator/data contract is complete.
4. Keep all unresolved domains and missing labels explicit rather than
   filling metrics with zero.

## Online contract

For each video-expression episode, state is reset at the first frame.  The
current source frame is predicted and committed before probe evidence is
computed.  An adapter step from frame `t` is first eligible to affect frame
`t+1`.  No GT, future frame, future expression, or target interval is read by
the runner.  Query probes use independent shadow tracker state and do not
consume the main tracker ID counter.

The completed primary scope is Refer-KITTI-V1 videos `0012` and `0014`, with
56 expression episodes and 5,320 frame rows. V2 is separately audited and
has one complete 0005 episode in the current report; it is not described as a
four-video V2 result.

## Fixed source thresholds

The source control keeps the official inference thresholds: objectness
`0.6`, area `100` pixels, and referring probability `0.4`.  QCS-TTA changes
only the referring probability used for future output; boxes, IDs, objectness,
and QIM are frozen.

## Methods in the first run

The executable pilot includes `FROZEN`, `EMA_ST`, `PROBE_CACHE`, `PROBE_ST`,
`QCS_TTA`, and `NO_STATE` for the formal matched result. `HEAD257` and
`ENTROPY_HEAD` remain available as optional capacity controls. Methods share only the
read-only frozen source snapshot and probe packet.  Each method owns its
adapter, optimizer, replay, EMA, and trace.  All methods are run separately
for speed reporting; a shared-source equivalence smoke is recorded before
the multi-method pilot.

## Evaluation boundary

`runtime_manifest.jsonl` contains only image paths, frame numbers, video IDs,
expression IDs, and query text.  `evaluation_manifest.jsonl` contains the
separate label paths and expression label metadata.  The online runner never
imports the evaluator or reads the latter manifest.  The evaluator runs after
the prediction directory is sealed.

Metrics that have not been run are represented as `null` with a reason.  A
completed script is not treated as evidence of method improvement.
