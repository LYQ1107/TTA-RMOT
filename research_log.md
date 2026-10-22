Warning: truncated output (original token count: 47196)
Total output lines: 1684

# LocateMOT Research Log

Concise experimental log (latest first).

## 2026-10-20 — G2 step3000 full MOT17-02 expansion

- The preserved G2 Clean step3000 bundle and the original G0 used the same 600-frame MOT17-02-FRCNN manifest, official_scoring parser, tracking_dense decoder, Simple association and TrackEval commit. Runtime read no GT or future frames; the evaluator opened the registered GT.
- Official full-video result: G0 HOTA/DetA/AssA/IDF1 `0.357098/0.667771/0.194362/0.330325`, IDSW/FP/FN `822/1621/3212`; G2 step3000 `0.502578/0.716588/0.353559/0.495423`, `413/1664/2780`. G2 delta is HOTA `+0.145479`, IDF1 `+0.165097`, IDSW `-409`, FN `-432`, with 63/600 rejected frames versus G0 48/600.
- Judgment: this is a complete registered development diagnostic and supports a positive G2 front-end result on this video. MOT17-02 is Clean TRAIN-scope (the audit found exact frames 302–305 and 602 same-video TRAIN rows), so it is not an independent generalization claim and does not isolate JEV identity decisions. A video-disjoint development evaluation remains required.
- Receipt: `docs/GROUNDJEV/G2_FULL_MOT17_02_EVAL_20261020.{md,json}`; the protected G2 owner continued independently and reached step5447/18227.

## 2026-10-20 — V1 complete partition matched result

- Hypothesis: query probes plus a causal external referring adapter could improve referring selection while leaving TempRMOT detection, QIM, boxes, and physical IDs frozen.
- Scope: Refer-KITTI-V1 videos `0012` and `0014`, 56 video-expression episodes and 5,320 chronological frame rows. The runtime manifest had no evaluator fields; the runner reported `gt_used_by_runner=false`, `future_frames_used=false`, and `evaluation_manifest_opened=false`.
- Resource: the GPU selector chose physical GPU 2 because it had sufficient free memory and zero compute processes. The run recorded GPU UUID `GPU-c2484ded-5f87-237c-bbf0-d23fcb74fdb2`.
- Official TrackEval result: FROZEN HOTA 86.618; EMA_ST 83.899; PROBE_CACHE 85.543; PROBE_ST 86.627; QCS_TTA 86.531; NO_STATE 86.544. Deltas versus FROZEN were -2.719, -1.075, +0.009, -0.087, and -0.074 respectively.
- QCS trace: 916 optimizer steps; `L_eq` was nonzero on 880 steps and `L_excl` on 19 applicable steps (maxima 0.04486 and 0.00715). The negative result is therefore from an executed logic-loss method, not a zero-loss placeholder.
- Judgment: QCS-TTA did not improve the complete V1 partition. PROBE_ST's +0.009 is a small control fluctuation, not a method-success claim. Retain the full run and all sealed evaluator files.

## 2026-10-20 — Logic losses wired and V2 matched confirmation

- Failure found: the first controller version implemented `L_eq`/`L_excl` functions but set both terms to zero in the replay update because aligned probe probabilities were not carried into evidence records.
- Fix: CandidateEvidence now retains aligned q0/q_eq/q_cf probabilities; a bounded CPU logic cache supplies stop-gradient q_eq/q_cf targets while only adapted q0 logits receive gradients. Empty masks and missing counterfactuals remain no-op.
- Verification: compileall and 18 tests passed. In the corrected V2 0005/`automobiles-are-located-to-the-left` 297-frame trace, QCS-TTA made 75 updates and both `L_eq` and `L_excl` were nonzero on the applicable updates.
- V2 official TrackEval: FROZEN HOTA 67.591; QCS_TTA 67.509; NO_STATE 67.509. The QCS delta was -0.082, so the corrected implementation still showed no gain.
- Retain: corrected run at `results/runs/v2_0005_complete_episode_qcs_logic`; prior zero-logic run remains as a failure/fix artifact rather than being overwritten.

## 2026-10-20 — V1 launch argument failure and recovery

- Failure: an initial full V1 command passed `--episodes 0012 0014`, but the runner contract requires `video/expression` keys; it failed before model loading with a missing-episode `KeyError`.
- Fix: reran without an episode filter because the V1 runtime manifest already contains only videos 0012/0014. No prediction or evaluator output from the failed launch was reused.

## 2026-10-20 — QCS-TTA source boundary, V2 assets, and first smoke

- Hypothesis: a frozen TempRMOT pre-QIM referring snapshot can support causal, source-free external adaptation without changing detector, QIM, or physical IDs.
- Asset result: official TempRMOT source commit `6a65640d849fdee4a32bb055945ee34c3b0edeb1` and V2 `checkpoint_rk2.pth` were loaded from `/data2`; checkpoint size is `1096316824` bytes and SHA256 is `4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`. V2 test-video manifests cover 457,503 runtime/evaluation rows and runtime contains no GT fields.
- Implementation: added the frozen wrapper/snapshot, rule-only query variants, isolated probe/shadow state, IoU alignment, set-valued evidence, bounded replay and semantic state, external residual/HEAD257 adapters, causal runner, GPU selector, official TrackEval boundary, and required run/report entry points.
- Failure/fix sequence retained: direct runner import lacked project `sys.path`; source/probe score masks mixed CPU/CUDA tensors; static probe replayed excess history; each was fixed and a V2 one-frame smoke then completed with zero source checkpoint missing/unexpected keys and a successful TrackEval invocation.
- Smoke result: V2 video `0005`, expression `automobiles-are-located-to-the-left`, one frame, methods FROZEN/QCS_TTA; source parameters frozen, QCS optimizer step count 1, no GT opened by runner. TrackEval produced files, but this is not a formal accuracy result because the sequence was truncated to one frame. Retain as interface evidence only.
- Resource decision: GPU selection now filters by free memory and prefers fewer compute processes; the observed eligible choice was physical GPU 2. No other project process was stopped or modified.
- Retain: V1/V2 assets, manifests, source sanity, smoke artifacts, and all failure logs. Pending: full matched pilot, evaluator summaries, and final scientific conclusion.

## 2026-10-18 — Expanded held-out gate to all three task loops

- The held-out owner now preserves DT1203 MOT and additionally schedules the
  registered RMOT single/multi validation scopes plus TAO validation videos 4
  and 63 with official multitask metrics. It passed a dry-run at J1 step300 and
  was replaced as a CPU-only owner without changing training or GPU ownership.

## 2026-10-18 — J1 step300 and held-out gate reattached

- The live J1 three-task run reached a complete step300 checkpoint with
  adapter weights and full FP32/Adam/RNG state. MOT20-05 remains in its
  protected G0 export without a `result.json`; expression-conditioned RMOT has
  not started inference and remains gated on that release.
- The CPU-only held-out gate owner had exited while waiting, so it was
  reattached to the live J1 PID without restarting training.

## 2026-10-18 — GPU5 G1 functional retention control

- With explicit user authorization, GPU5 was shared after a memory check (about
  68.6 GiB free); the foreign TALON jobs were preserved. The corrected owner
  completed G0/G1 MOT4, RMOT single8, and RMOT multi12 controls in the OneJev
  checkout. HOTA changed 0.7485→0.7333, 0.2718→0.2117, and 0.2179→0.2284;
  this is short functional evidence, not a full-dataset promotion result.
- The G1 service was cleaned up and the evidence/documents were synchronized
  to the GroundJEV GitHub branch. The active J1/MOT20/RMOT pipeline was not
  interrupted.

## 2026-10-18 — DeepSeek Harness handoff re-audit and active continuation

- The requested GroundJEV-MOT redirection was checked against the actual
  DeepSeek Harness SSH-job records and live child processes. Harness launcher
  records with `exit_code=0` are not completion receipts: the detached
  children for the three-task SFT and MOT20 export are still alive. The live
  research checkouts remain `/data2/user/LocateMOT_OneJev_CVPR` and
  `/data2/user/LocateMOT_GroundJEV`; this root checkout is still the required
  project identity `/data2/user/LocateMOT`.
- Active work: `trackjev_3task_J1_v4_birth` (PID 3029659, ranks 3029872 and
  3029873) is a real MOT/RMOT/OVMOT run over 43,533 rows (21,168 / 16,677 /
  5,688), using GPUs 4 and 6. The latest observed log is step 280/1,200 and
  step 250 has a complete FP32-master/Adam/scheduler/RNG restore state; the
  old MOT-only training has no residual process.
- Protected export: MOT20-05 `mot20_MOT20-05_resumed_v2` (PID 3743898) is
  still restoring the exact 1,560-frame prefix and running with the explicit
  4 GiB native-feature budget after the earlier 2 GiB failure. It has no
  `result.json` yet. The expression-conditioned RMOT pilot (PID 3744099)
  remains correctly gated on that receipt and has performed zero model
  inference or optimizer updates.
- GPU5 is not idle: its active compute owner is another user's
  `/data2/OCD/TALON` process. The existing GroundAnything G1 evidence already
  contains a complete 200/200 projector adaptation and G0/G1 comparison; the
  measured result is a retained negative detector adaptation. No foreign job
  was stopped and no duplicate G1 run was started.
- Next recovery point: let the protected MOT20 export and J1 checkpoint reach
  their registered completion boundaries, then run the prepared
  expression-conditioned RMOT pilot and an independent held-out evaluation
  from the latest valid J1 adapter. No benchmark success or unified-goal
  completion is claimed in this audit.

## 2026-10-01 — latest R1 restoration completed

- Hypothesis: restore only the newest R1 execution boundary and its legal
  public-data/runtime inputs on `/data2/user/LocateMOT`; do not pretend that
  unavailable L69/L89E/R1 payloads were reproduced.
- Restored: official KITTI training images, Refer-KITTI V1, Refer-KITTI V2,
  MMDetection 3.3.0/GroundingDINO, BERT, and the direct R1 source dependency
  modules. V1 audit is complete. V2 has 157 legal expression-frame target
  references (11 repeated physical video/frame/ID pairs) pointing to IDs
  absent from corrected labels; missing frames are zero, and the anomaly is
  retained for the next adapter instead of silently filtered. The new public
  query manifest records every affected row.
- Runtime evidence: GroundingDINO loads the SHA-verified official checkpoint
  with 172,977,693 frozen parameters; one real KITTI frame/expression GPU
  inference returned 300 finite candidate boxes and finite scores. R1 imports
  and compile checks pass after path migration. The safe source loader also
  reads all 8,397 legal V1/V2 expression records from the restored
  per-expression layout without opening official-evaluation videos.
- Failure/limit: historical R1 checkpoints, L69 bank, dense indexes,
  language/visual/aligned caches and fixed manifest are unavailable. No old
  HOTA or score is claimed as reproduced; the old R1 sidecar is not extended
  with threshold/layer/learning-rate sweeps.
- Next action: construct a new legal public-data manifest/index and then a
  freshly generated candidate bank before any new correspondence training.
  The manifest is now complete. A 40-row midpoint-frame GroundingDINO probe
  found mean best IoU `.3188`, IoU@.25 `.6585`, IoU@.50 `.1463`, and mean best
  score `.0107` over 123 target boxes; the frame-zero attempt is preserved as
  a separate artifact. This supports a new bank audit but is not an RMOT
  score. Official-evaluation videos remain unread.

## 2026-08-19 Stage L11 — OVMOT temporal pseudo-track + KITTI front-end

- Hypothesis: unmatched DLA detections on expanded TAO-train OVMOT
  stream must receive high-precision temporal pseudo-track supervision
  instead of "every detection is NEW"; KITTI RMOT DetPr ~1% is a
  candidate front-end problem, fixable with category whitelist + NMS +
  query-conditioned CLIP top-k.
- Implementation:
  - Pseudo-track generator (forward + backward cycle consistency,
    appearance/motion/category filters, confidence gating) for all 500
    TAO train videos; sidecars in `outputs/l11/data/pseudo_tracks`.
  - Class-A GT upgraded from C-TAO base @0.5 IoU to base_and_novel
    @0.3 IoU (~10% -> ~28% coverage).
  - Trainer: no_unmatched_new, confidence-weighted identity/relevance
    losses, pseudo ids in the same slot/lifecycle machinery.
  - KITTI: whitelist + cross-NMS + top-30 -> ~15.3 cands/frame (was 50);
    CLIP top-12 calibration: query precision 10.2%, target recall 65%.
- Quality: pseudo same-ID precision 99.26% (8-video audit, target 90%);
  cycle pass 0.997; NEW rate 0.110 vs L10 collapse ~1.0.
- Running: 4-GPU repair training (from L9-ovmot, ~6-8 s/step),
  repaired-KITTI eval (L9-ovmot ckpt), 10-video OVMOT over-birth
  baseline: unique/rows 0.211 (L9), reused IDs 467.
- Pending: step-7000 OVMOT quick eval, full TAO TETA, KITTI TrackEval,
  L12 prompt seeding.

## 2026-08-20 continued

- Low-LR continuation (10000->11000, LR~6e-6): RMOT-Dance 32.49 ->
  34.05 HOTA; ordinary Macro 0.4855 -> 0.4924.  OVMOT full TETA on
  s11k running.
- Fresh-optimizer balance run (p_rmot 0.45/p_ovmot 0.20) REJECTED at
  step 1000 (RMOT 32.06 / ordinary 0.4663).
- L12 frozen prompt-seeded (DAVIS 2017, 10 multi-object videos):
  mask/point > box for identity robustness; joint fine-tune not
  launched (mixed signal).  Infrastructure complete.

## 2026-09-04 Stage L88 — RMOT-aware GroundingDINO LoRA adaptation

- Hypothesis: a zero-initialized rank-16/alpha-32 LoRA on the final two
  GroundingDINO fusion layers and decoder layer 0, with the frozen L86/L87
  sidecar and causal history, would improve held-out target-bag and
  expression-to-candidate correspondence without changing the L69 bank.
- Contract: label-free, distributed, and temporal regressions passed; full
  V1/V2 fit used seed `20260829`, 40 epochs, 2,640 optimizer steps, world 4,
  effective group batch 8, and 20 even checkpoints. Local MMCV required FP32
  rather than the registered BF16 autocast. Candidate deletion/truncation was
  false and all frozen-base/LoRA gradient and reload checks passed.
- Dev: all 20 checkpoints were scored on 138 video-disjoint groups; the fixed
  selection chose epoch20/Rule B before fixed validation. Checkpoint SHA is
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`.
- Fixed semantic validation: immutable L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`. L88 candidate-only
  was `recall=.4193548, precision=.1830986, FP/frame=2.4167,
  pred/positive=2.2903, hard=.8461538, multi=.3333333`; final Rule B was
  `recall=.1935484, precision=.1363636, FP/frame=1.5833,
  pred/positive=1.4194, hard=.8461538, multi=.1666667`, with empty rate
  `.5416667` and inactive false acceptance `.5`. The semantic gate failed on
  recall and multi-positive preservation; the lower volume is not a fix.
- Internal TrackEval: selected Rule B produced HOTA `26.0914` (V1) and
  `20.2386` (V2), below L86 `29.1663/21.6467` and L87-A
  `28.5752/22.1300`. No screening or official-test labels were read.
- Preserved failures/fixes: first formal causal-key drift, sparse seqinfo,
  merge, device, and score-shape retries remain in new directories with their
  provenance; no old bank/checkpoint/production entrypoint was overwritten.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; no L88 long continuation,
  screening, official test, TrackEval beyond internal validation, or
  ordinary MOT/OVMOT change is authorized. Unique next action is supervisor
  review of one new RMOT correspondence/proposal design.

## 2026-09-05 Stage L88C — corrected candidate-vs-NULL replay

- Hypothesis: the prior L88 semantic failure may include an implementation
  error in the candidate-vs-NULL gate. Corrected emission is
  `candidate_energy >= candidate_threshold` AND
  `candidate_energy-null_logit >= null_margin` AND
  `presence_logit >= presence_threshold`; candidate-only uses only the first
  condition. No training, backward, optimizer, checkpoint update, bank
  change, top-k/NMS or NULL suppression was performed.
- Implementation: refit all 20 even checkpoints on the registered B/R/P
  grids, replayed the corrected shortlist (epochs 8/20/40/30/4) over complete
  V1/V2 dev video, ran the corrected internal TrackEval matrix, froze epoch
  30 Rule B before fixed validation, and ran the fixed 16-calibration/
  24-validation replay. A first TrackEval provenance alias failure was
  preserved; the minimal `scope_key/full_video` alias fix was pushed in
  `c5a6685` and the corrected matrix is attempt2.
- Result: L88C corrected final on validation was
  `recall=.354839, precision=.207547, FP/frame=1.75,
  pred/positive=1.7097, hard=.846154, multi=.305556,
  empty=.375, inactive-FA=.666667`. Immutable L29 remains
  `.733333/.083019/10.125/8.8333/.916667/.819444` for
  recall/precision/FP/pred-positive/hard/multi. Candidate-only recall was
  `.419355`; the corrected NULL comparison removed additional positives.
- Internal dev TrackEval for the frozen epoch30/Rule B was V1
  HOTA/DetA/AssA `.296615/.216472/.408302` and V2
  `.265682/.156622/.451938`; these are dev-only, not screening or official
  results. The semantic gate failed on recall and multi-positive recall.
- Root cause: existing candidate scores have useful average pairwise ranking,
  but the lowest-scoring positives in multi-target bags do not survive the
  fixed emission boundary; V2 is worse. This is a correspondence and
  multi-positive recall failure, not a universal NULL-acceptance,
  coverage, finite/reload, or implementation-smoke failure.
- Preserved outputs: corrected replay, two TrackEval attempts, frozen
  selection, fixed semantic JSON/gate, and offline A–I diagnosis under
  `outputs/l88c/`. Code was pushed to branch
  `codex/l88c-candidate-null-corrected-replay-20260905` at `e8f5f62`.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. No screening/official-test
  labels, HOTA claim beyond internal dev TrackEval, ordinary MOT/OVMOT/TAO
  change, or new L88C training was run. The only next action is the single
  supervisor approval request in `reports/l88c/NEXT_TEST_APPROVAL_REQUEST.md`.

## 2026-09-06 — L88C recoverable storage cleanup

- While the authorized zero-training L88C final-internal replay was running,
  its open paths were checked. Three superseded L1 intermediate LoRA
  checkpoint directories (`outputs/l1_c/checkpoints/lora/checkpoint-100/200/300`,
  about 8.0G each) were not open by any process and were moved, not deleted,
  to `/data2/usr_for_deadline/locatemot_cleanup_20260906/`.
- Frozen banks, formal reports, current L88 epoch30 checkpoint/cache, and all
  ordinary MOT/OVMOT assets were left in place. `/data1` free space increased
  from about 76G to 114G. The recoverable inventory is documented in
  `reports/l88c/L88C_RESOURCE_CLEANUP_20260906.md`.

## 2026-09-07 Stage L89 — query-conditioned set correspondence decoder

- Hypothesis: a fresh QSC-D head over the frozen L84/L87-A Z1 candidate states,
  with candidate self-attention followed by cross-attention to pure frozen
  GroundingDINO/BERT language tokens, could improve same-frame correspondence
  without changing the bank, detector, tracker, or ordinary MOT/OVMOT.
- Implementation: isolated worktree at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`, base
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`; two set layers, eight heads,
  FFN 1024, hidden 256, history 8, 4,169,829 trainable parameters. Pure
  language cache has 3,418 label-free entries. Contract smoke passed after
  preserving two package/field failures. No token/span or static/motion
  supervision is verified (`UNALIGNED`).
- Fit: seed `20260829`, one GPU, BF16, 40 epochs/2,640 optimizer steps,
  finite loss and nonzero gradients, all 20 even checkpoints plus final.
  Only L49 fit supervision was used; no screening or official-test labels.
- Dev/selection: 20 checkpoints scored over 138 legal dev groups; five-item
  shortlist and full-video B/R/P matrix selected epoch 4, Rule R before fixed
  validation. Checkpoint SHA is
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- Fixed semantic: authoritative retry4 failed. L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`; L89 was
  `recall=.8709677, precision=.0794118, FP/frame=13.0417,
  pred/positive=10.9677, hard=.6923077, multi=.9027778`, with inactive
  false acceptance `1.0`. Hard-negative ranking moved, but deployable volume,
  precision, and no-match behavior failed.
- Internal TrackEval: frozen epoch4/Rule R on V1 validation (86 sequences)
  gave HOTA `2.2906`, V2 validation (537 sequences) HOTA `0.5894`, versus
  L87-A `28.5752/22.1300`. Full-video post-hoc target-row recall was only
  `.0131/.0026` and inactive acceptance was `1.0`. This is internal
  validation TrackEval evidence, not screening or official-test evidence.
- Root cause judgment: query-conditioned set representation gave limited
  frame-level ranking signal but failed calibrated persistent emission and
  absence handling. The branch is stopped pending supervisor review; no L89
  continuation, screening, official test, or ordinary MOT/OVMOT change is
  authorized. All L89 outputs are preserved under `outputs/l89/` and the new
  source/reports are on branch `codex/l89-query-conditioned-set-correspondence-20260907`.

## 2026-09-08 Stage L89C — corrected candidate-vs-NULL replay

- Hypothesis/repair: L89 training and the registered loss use `candidate_energy`
  against `null_logit`, but the old L89 evaluation/deployment path emitted
  using `presence_logit - null_logit`. L89C changed only the evaluation,
  selection, inference, and TrackEval contract to the shared
  `corrected_emission_mask`: candidate threshold, candidate-minus-NULL margin,
  and presence threshold. No training, scorer forward, bank/cache rebuild, or
  model/ordinary-MOT change was made.
- Inputs/selection: immutable L89 raw dev records (9,960; SHA recorded in
  `outputs/l89c/dev/corrected_shortlist_attempt1/`), 20 existing checkpoints,
  and the fixed manifest SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Corrected dev TrackEval evaluated 5 checkpoints × B/R/P (15 results), then
  froze epoch 2 / Rule R, checkpoint SHA
  `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
  candidate/presence/NULL thresholds `-0.75/-1.0/0.0` before fixed validation.
- Fixed semantic evidence: authoritative retry2 has 40 ordered records (16
  calibration, 24 validation), with preselection forbidden-label fields
  absent. Validation is recall `.8387097`, precision `.0869565`, FP/frame
  `11.375`, predictions/positive `9.6452`, hard violation `.6923077`,
  multi-positive recall `.8333333`, inactive false acceptance `1.0`.
  The gate is `semantic_gate_fail`: hard/recall/precision/multi floors pass,
  but volume and inactive/no-match conditions fail.
- Internal evidence: frozen epoch-2/Rule-R full-video inference covered V1
  `0004,0018` (86 sequences) and V2 `0016,0017,0020` (537 sequences).
  Corrected TrackEval HOTA is V1 `2.244718%`, V2 `0.583059%`; no screening or
  official-test labels were read. These are internal validation-scope metrics,
  not a final RMOT result. Historical buggy L89 was `2.2906%/0.5894%` and
  L87-A was `28.5752%/22.1300%` on V1/V2.
- First actionable root cause: corrected candidate-vs-NULL accounting still
  leaves no-match acceptance universal and output volume too high. All rows,
  keys, finite checks, and frozen boundaries passed; the initial GPU OOM is
  preserved as an implementation/resource attempt and the exact CPU retry
  completed. Unique next action: supervisor review and authorization of one
  new RMOT branch explicitly addressing absence/volume calibration with
  candidate correspondence. Do not extend L89/L89C or tune thresholds.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; `zero_training=true`,
  `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.

## 2026-09-08 Stage L89D — true full-video timeline repair

- Hypothesis/scope: repair only the L89C sparse-prediction versus dense-GT
  timeline mismatch. No training, model, loss, bank, tracker, threshold, or
  ordinary MOT/OVMOT change was made. New code is confined to `tools/l89d_*`
  and new evidence is under `reports/l89d/` and `outputs/l89d/`.
- Contract repair: native L69 `frame_ids` and frame pointers now drive every
  video timeline; all query/frame pairs and all candidate rows are scored.
  The multi-video query-map overwrite was fixed and covered by a targeted
  regression. Dev inference attempt3 covers 1,152,690/1,152,690 candidate
  pairs across 2,385 frame groups; internal attempt1 covers
  243,550/243,550 pairs across 1,844 frame groups. Both passed full-video and
  pair-completeness checks. The dev and internal dense Z1 supplements were
  built from frozen query-independent caches and finalized with no labels.
- Dev evidence: the valid 5-checkpoint × B/R/P TrackEval matrix selected
  epoch 8 / Rule P on the preregistered dev tuple. The frozen checkpoint is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt`,
  SHA256 `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`.
  The fixed 16-calibration/24-validation semantic replay then failed: recall
  `.5483871`, precision `.1517857`, FP/frame `3.9583333`,
  predictions/positive `3.6129032`, hard violation `.6923077`,
  multi-positive recall `.4861111`, inactive false acceptance `.6666667`.
  Recall and multi-positive floors failed; no screening/test labels were read.
- Internal evidence: fixed epoch 8 / Rule P true full-video TrackEval gives
  combined HOTA `23.8586078862%`, DetA `15.1294290119%`, AssA
  `37.9461647913%`, DetRe `57.2559333985%`, DetPr `16.9954368707%`, IDF1
  `18.6931722536%`, and IDSW `6337.5`; V1 HOTA `25.0798607896%`, V2 HOTA
  `22.6373549828%`. These are internal validation-scope TrackEval results,
  not screening, official-test, or final ordinary-RMOT evidence.
- Attempts preserved: dev ENOSPC, internal exit-137/OOM retries, the expected
  smoke full-video rejection, the selector directory-argument error, and the
  earlier multi-video GT-map mismatch. The final boundary guard passed after
  the path contract correction. The fixed manifest SHA remained
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Status: `L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Flags: `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
  Unique next action: supervisor review and authorization of one separate
  absence/volume-calibration study; do not extend L89/L89C or alter the
  production MOT/OVMOT paths.

## 2026-09-09 Stage L89E — phase-consistent temporal/history replay

- Hypothesis/scope: L89D evaluated every checkpoint with temporal history
  enabled even though L89 training used Stage-S epochs 1–8 with temporal off
  and zero history. L89E changed only this replay contract: S epochs 2/4/6/8
  use `temporal_enabled=false` and zero history; T epochs 10–20 and J epochs
  22–40 use causal last-four history. No training, model, loss, bank, tracker,
  threshold science, UIDM, ordinary MOT or OVMOT source was changed.
- Implementation: isolated branch/worktree based on L89D commit
  `d0960e1d1679414765f79203bef444f0f9968928`; phase policy and wrappers are
  in `tools/l89e_*`. The required CPU equality check passed for S and T
  `L86ClipStore` history construction. The first replay failure was a
  `FrameExample`/`_clip_history` interface mismatch; the minimal adapter fix
  passed a targeted S/T causality regression. A subsequent historical-score
  merge check found that the immutable sparse file contains S+T/J rows; the
  wrapper now filters only the historical T/J epochs and preserves their
  values. Launch-path and shortlist argument mistakes were preserved as
  `INCOMPLETE`/operator evidence; no affected model run was treated as data.
- Stage-S evidence: fresh phase-correct replay of epochs 2/4/6/8 completed
  `4 x 498 = 1,992` records, all finite, complete and zero-history. Merged
  phase-consistent fit/dev pool contains 20 epochs and 9,960 records. The
  registered maximum-five fit/dev shortlist was frozen at epochs
  `8(S),20(T),40(J),32(J),4(S)`.
- Dev selection: full native dev replay retained `11,003,527/11,003,527`
  candidate rows for all five shortlist candidates. The 15-result TrackEval
  matrix selected epoch 4 / phase S / Rule B by the registered tuple; dev
  HOTA was `0.296512` (internal fit/dev selection evidence only).
- Fixed semantic evidence: the new 16-calibration/24-validation replay used
  the frozen epoch-4/S zero-history policy and Rule B. Immutable L29 remains
  recall/precision/FP-frame/pred-positive/hard/multi
  `.7333333/.0830189/10.125/8.8333/.9166667/.8194444`. L89E validation is
  recall/precision/FP-frame/pred-positive/hard/multi
  `.4516129/.1458333/3.4166667/3.0967742/.7692308/.3611111`, with inactive
  false acceptance `.8333333` and empty rate `.2083333`. The semantic gate
  fails recall and multi-positive floors despite lower volume and higher
  precision; no threshold rescue was performed.
- Internal TrackEval: after the frozen selection, exact internal timeline
  replay covered `243,550/243,550` pairs for V1 0004/0018 and V2 0016/0017/0020.
  Phase-consistent HOTA/DetA/AssA/DetRe/DetPr were V1
  `28.7628/18.7680/44.3927/80.3045/19.5312%` and V2
  `21.8385/13.3358/35.9990/45.1160/15.8179%`; IDSW was 1,997 and 9,915.
  These are internal validation-scope TrackEval results, not screening or
  official-test results.
- Root cause: correcting S-phase temporal/history evaluation does not
  restore expression correspondence; the fixed emission remains recall- and
  multi-positive-limited, with substantial inactive acceptance. Status is
  `L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
- Preserved outputs: Stage-S attempts 1–4, merge attempts 1–2, shortlist
  attempts 1–4, dev/internal full-video and TrackEval outputs, fixed semantic
  output, and all reports under `reports/l89e/` and `outputs/l89e/`. The fixed
  manifest SHA remains
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Flags remain `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- The only next action is supervisor review of a separately authorized
  absence/volume-calibration study. Do not extend L89/L89C/L89E, alter the
  bank, or touch production MOT/OVMOT.

## R0A — safe target contract, dense native indexes, and implementation smoke — 2026-09-09

- Hypothesis: isolate only allowlisted V1/V2 target records, repair the
  frame-specific target contract, and construct native-frame R0 data without
  parsing forbidden official-eval payloads.
- Safe-source retry3 passed: lexical monolithic-JSON isolation deserialized
  only allowlisted video values and recorded
  `forbidden_payload_deserialized=false` and
  `official_test_labels_read=false`.  The exact 5,314 sparse fit rows matched
  the authoritative frame map with zero sentence/target/missing-query/invalid-
  frame mismatches.
- Dense native-frame index retry2 and its contract audit passed for V1/V2;
  all L69 rows and duplicate candidate indices were retained.  The geometry
  `image_hw` and visual-cache dataset/group-key contracts were corrected in
  R0-only files.  Retry1 dense/smoke failures remain preserved with their
  first trace; retry2 is authoritative.
- Real R0 contract smoke retry2 passed on one V1 fit frame: loss
  `3.555824041366577`, 108/108 finite nonzero trainable gradients, frozen
  GroundingDINO, no tracker optimizer parameters, strict reload max diff `0`,
  and temporary visual item removed.  This is implementation evidence only;
  no semantic, screening, official-test, HOTA, or production MOT/OVMOT result.
- Next action: commit/freeze the R0A safe code and construct the registered
  query-independent visual cache on `/data2`; no old asset or production
  entrypoint is modified.

## R0B — grouped DDP driver repair and formal legal evidence — 2026-09-09

- Hypothesis: the old Q1 query-at-a-time driver was the actionable R0 training
  contract bottleneck. A frame-centric grouped driver with complete candidate
  sets, Q=8, deterministic strata, and DDP accumulation should make the
  registered R0 experiment executable without changing the frozen bank,
  tracker, detector, or ordinary MOT/OVMOT.
- Preserved/disqualified Q1: PID 39571 was stopped gracefully; the old partial
  `outputs/r0/train/v1_formal_retry1/` remains historical and is not
  selection-eligible.
- Repairs: grouped Q8 schedule, video-local tiles, world4 DDP, accumulation 4,
  frozen detector/language inputs, complete candidate rows, present-uncovered
  masking, and resumable `locatemot-r0-tcgh-checkpoint-v2`. The legal GT
  materialization gap was repaired only in the R0 wrapper by recording and
  skipping missing annotation rows; frozen GT was not edited.
- Formal training: V1 completed 12 epochs/3,058 optimizer steps and V2
  completed 12 epochs/3,046 steps. Both were finite with nonzero gradients,
  strict reload, all four strata and both domains. Checkpoints 2/4/6/8/10/12
  are retained. Formal jobs used the free physical GPUs 4–7; physical GPUs
  0–3, which had unrelated processes, were not touched.
- Legal dev selection: V1 shortlist epochs 12/6 and V2 12/10; full dev
  TrackEval selected epoch12 for both. Dev HOTA was V1 22.8681% and V2
  19.5801% for the selected checkpoints.
- Internal validation-scope TrackEval: V1 0004/0018 HOTA 26.4582%, V2
  0016/0017/0020 HOTA 15.6097%. These are not screening or official-test
  results. No screening/official labels were read, and ordinary MOT/OVMOT,
  UIDM, L69 and production entrypoints were untouched.
- Outputs: training remains under `outputs/r0/`; large dev/inference/TrackEval
  products are under `/data2/usr_for_deadline/locatemot_r0b_*` because `/data1`
  was nearly full. No raw/dense debug cache or detector copy was created.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. This is valid grouped R0
  implementation plus legal dev/internal TrackEval evidence, not final
  ordinary-RMOT completion. The only next action is supervisor review and a
  separately authorized follow-up.

## R1 — aligned track conditioning legal-dev contract — 2026-09-10

- Hypothesis/scope: test a zero-initialized RMOT-only residual conditioned on
  aligned frozen L89E Stage-S Z0/Z1/Z4 states and causal observation history;
  the anchor, L69 bank, ordinary MOT/OVMOT, UIDM and all production paths stay
  frozen. This is not final RMOT completion.
- Read-only legal-dev audit: frozen epoch-004 Rule-B anchor selection was
  rechecked. The epoch-004 presence-only miss is `3/400=.0075`, below the
  preregistered `.05` activation threshold, so the formal R1 config freezes
  `presence_residual=false`. Existing full-video artifacts have no candidate
  score arrays for a valid EXACT_INDEX_DEDUP replay; the conservative RAW
  policy is therefore frozen without claiming a dedup comparison.
- Preserved evidence: `outputs/r1/audit/legal_dev_selection_attempt1/` and
  `reports/r1/R1_LEGAL_DEV_SELECTION.md`; no training or new checkpoint was
  created by the audit. The next action is the bounded wiring/reload smoke;
  formal fit remains conditional on that implementation gate.

## R1 — formal fit and legal-development replay — 2026-09-12

- Hypothesis/scope: a zero-initialized aligned track-conditioning residual
  over frozen L89E stage states, causal observation history, language-directed
  R0 detail and complete L69 candidate sets could improve expression-to-track
  correspondence without changing the anchor, bank, tracker or ordinary
  MOT/OVMOT paths.
- Formal fit completed in isolated worktree
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`: six epochs, 2,250
  optimizer steps, all finite with nonzero gradients, strict reload for eight
  V1/V2 checkpoints, frozen anchor, and no persistent raw/dense debug cache.
- Prediction-only replays for legal-dev V1 videos `0008/0010/0020` and V2
  videos `0000/0008/0009` completed before legal GT was opened. The aggregate
  contains 2,385 groups and eight internal TrackEval results. Registered
  selection chose epoch 4 separately for V1 and V2. Selected HOTA was
  `24.2345%` (V1) and `23.6542%` (V2), versus the L89E anchor's `28.7628%`
  and `21.8385%`; this is not a stable breakthrough and remains legal-dev
  evidence only.
- Fixed 16-calibration/24-validation diagnostic retained all 2,435 L69 rows
  and 40 keys. R1 candidate-only recall/precision/FP-frame/pred-positive/
  hard/multi were `0.1000/0.1842/0.7750/0.5429/0.7727/0.1818`; the frozen
  Rule-B replay emitted zero rows for both anchor and R1 on this slice. The
  lower volume is a recall/multi-positive collapse, not a semantic repair;
  no threshold or NULL rescue was attempted.
- Status: `R1_NO_BREAKTHROUGH / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Outputs and failed technical attempts are preserved under `outputs/r1/`
  and `reports/r1/`. Flags remain
  `screening_gt_used=false`, `official_test_labels_read=false`, and
  `ordinary_mot_ovmot_touched=false`. The only next action is supervisor
  review and explicit authorization of one separate follow-up; no automatic
  continuation, screening, official test, or production merge is allowed.

## R1 — fixed semantic diagnostic closeout — 2026-09-12

- The fixed-slice evaluator used the native L62 order and attached labels only
  after row/feature construction and scoring. It used the frozen Rule-B values
  `candidate_threshold=1.0`, `presence_threshold=0.5`, `null_margin=0.0` and
  did not fit or tune a new threshold.
- This is a diagnostic, not a preregistered semantic gate and not HOTA. The
  authoritative files are in `outputs/r1/eval/fixed_semantic_attempt1/`; the
  corresponding report is `reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`.
# 2026-10-01 — new-server restoration started

- Hypothesis: restoring public V1/V2 data and exact base inference dependencies enables a new, attributable correspondence experiment; extending the failed R1 sidecar is unjustified.
- Evidence: fetched all remote branches; latest remains R1 `99090c8`. Read chronological historical endings and R1 reports: V1 legal-dev regression, limited V2 gain, fixed-slice score mismatch/output collapse. Old trained checkpoints are unavailable.
- Changes: started resumable official KITTI S3 **direct** download (15,813,146,295 bytes); downloaded and CRC-validated V1 expression/labels archives; isolated local data-tools environment. Full inventory/short plan: `reports/RESTORE_STATUS_20261001.md`.
- Failure/repair: shell sandbox lacks bubblewrap; execution requires tool escalation. Standard venv lacks ensurepip; local virtualenv bootstrap succeeded. Neither is a scientific result.
- Results: public V1 annotations recovered; KITTI/V2/runtime/project-generated assets pending. No historical checkpoint/bank reproduction claimed. Keep recovery scripts and new provenance; preserve migration AGENTS.md edit.

## LocateMOT-U U0 — unified specification and reference audit — 2026-10-01

- Hypothesis: a single specification-conditioned foundation can absorb the
  reusable persistent-query, temporal-language, open-vocabulary and reliability
  ideas from the ten requested repositories, provided the legal video boundary
  and one-checkpoint/shared-state contract are frozen before training.
- Code: cloned and shallow-audited MOTR, TransRMOT, TempRMOT, DKGTrack,
  FlexHook, iKUN, OVTR, COVTrack, ReferDINO and Open-GroundingDino; created
  `UnifiedSpecTrack` U0 shape skeleton, shared legal-scope guards, task/sample
  contracts, loss/runtime inte…27196 tokens truncated…unting audit unchanged. The held-out gate remains waiting on the registered J1 PID and final receipt; G1 and bundle owners remain dependency-gated. No active process or checkpoint was restarted or altered.


2026-10-19 — J1 durable step1050/1200 checkpoint is now present with FP32 masters, Adam moments, scheduler and all RNG state; the latest log is step1040 with finite metrics. Clean G2 reached event step900 with finite loss and global accounting fields. Neither terminal receipt exists; held-out, G1 and bundle owners remain dependency-gated, and no active task was restarted.


2026-10-19 — J1 durable step1100/1200 is complete with FP32/Adam/scheduler/RNG state and finite step-1100 metrics. Clean G2 crossed event1000 and is now at event1014 with finite loss. Neither terminal receipt exists; the registered held-out gate, G1 GPU5 queue, and clean bundle exporter remain dependency-gated. No live process or checkpoint was modified.


2026-10-19 — J1 durable step1150/1200 is present with the full FP32/Adam/scheduler/RNG recovery state; the latest logged step is 1140 and finite. Clean G2 advanced to event1167 with finite loss. Final receipts are still absent, so the registered held-out, G1 GPU5, and bundle owners remain waiting; no live task was restarted or changed.


2026-10-19 — J1 slow-batch diagnostic at log step1160: rank0 is sleeping in futex_wait_queue_me while rank1 is actively running (state R), with the torchrun parent in do_select. This confirms an active long batch/synchronization phase rather than a two-rank deadlock; the log mtime is recent and no failure receipt exists. No process or checkpoint was interrupted.


2026-10-19 — J1 training reached a valid terminal boundary. The log contains event=done with steps=1200 and 19,143 examples; latest_training_state.json is step1200 with FP32 masters, Adam moments, scheduler and all RNG state; adapter-step1200 contains adapter_model.safetensors plus adapter/processor/tokenizer configuration files. The torchrun exited after the done event without a standalone final_receipt.json, but the pre-existing CPU gate validated the step/adapter evidence, recorded j1_final and the preserved MOT20 G0 release, and advanced to RUNNING_HELDOUT_DT1203. No duplicate gate or training restart was launched.


2026-10-19 — J1 held-out gate advanced beyond the dependency waits: it has recorded j1_final at step1200, the preserved MOT20 G0 release, the completed expression-conditioned RMOT pilot, and a successful 186-tensor LoRA merge with zero unmatched adapter keys. It is now running the DT1203 held-out evaluation; result.json and official metrics are not yet present, so no held-out quality claim is made. No duplicate evaluator was launched.

2026-10-19 — held-out gate runtime milestone after the registered blocking wait. The gate owner advanced to RUNNING_HELDOUT_RMOT_MULTI12 on GPU4. DT1203 held-out inference and official TrackEval completed successfully with the registered 1203-frame scope; its metrics receipt is explicit and reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=4586, with rejected frames retained in the official denominator (this is an observed result, not a quality claim). The registered RMOT single8 clip also completed and its official receipt reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=5. RMOT multi12 inference produced a complete 12-frame result, but all frames were rejected by the strict duplicate-coordinate parser and its official evaluator remains pending; the gate process is still alive, so no terminal verdict is recorded. Clean G2 remains alive at event step1615/18227 with finite loss 1.210435 and about 17.37 GiB peak memory; no G2 receipt exists. No active job, checkpoint, seed, metric protocol, or evaluator was restarted or changed.

2026-10-19 — RMOT multi12 held-out stage completed. The gate recorded `rmot_multi12`; the 12-frame inference archive and official TrackEval receipt are both present. The registered short functional clip reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=30, with 30 released query GT boxes and all strict parser rejections retained in scope. The gate then entered RUNNING_HELDOUT_OVMOT_VIDEO4 on GPU4; OVMOT video4 inference is live and has not produced a terminal result or official metrics yet. Clean G2 remains independent and active. No active evaluator or training process was restarted.

2026-10-19 — OVMOT video4 live-progress validation. The held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO4 with no failure; the original evaluator process and ffmpeg encoder are alive. Its frame_decisions.jsonl has advanced to frame 1561 and the run log to frame 1591, with both files modified recently; result.json and official metrics are intentionally still absent while inference continues. Clean G2 has advanced to event step2136/18227 with finite loss 1.290262 and about 17.372 GiB peak memory. No evaluator, trainer, checkpoint, or protocol was restarted.

2026-10-19 — OVMOT video4 held-out evaluation completed. The gate recorded `ovmot_video4`; inference and the official TAO OVMOT TETA evaluator both completed for the registered 40-frame legal validation scope. The official receipt reports TETA/LocA/AssocA/ClsA and all recall/precision components as 0.0, 146 GT boxes, 0 predictions, and strict perception rejections retained in the denominator. This is the observed registered-scope result, not a quality claim. The gate advanced to RUNNING_HELDOUT_OVMOT_VIDEO63 on GPU4; its inference is live and has not produced a terminal receipt. No task or checkpoint was restarted.

2026-10-19 — OVMOT video63 live-progress validation. The final held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO63 with no failure; the evaluator and encoder are alive. Its frame_decisions.jsonl has advanced to source frame 436 and the run log to frame 465, with recent writes; result.json and official metrics remain pending during inference. Clean G2 has advanced to event step2635/18227 with finite loss 1.174764 and about 17.372 GiB peak memory. No task, checkpoint, or protocol was restarted.

2026-10-19 — OVMOT video63 live-progress validation (second milestone). The held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO63 with no failure; the original evaluator and ffmpeg encoder are alive. frame_decisions.jsonl has advanced to source frame 726 and the run log to frame 755, with writes about four minutes old; result.json and official metrics remain pending during inference. Clean G2 has advanced to event step2817/18227 with finite loss 1.297216 and about 17.370 GiB peak memory. No task or checkpoint was restarted.

2026-10-19 — independent held-out gate fully completed. The registered gate is now HELDOUT_ALL_REGISTERED_TASKS_COMPLETE with failed=null. Validated receipts exist for DT1203 (1203 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=4586), RMOT single8 (8 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=5), RMOT multi12 (12 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=30), OVMOT video4 (40 frames; all official TETA components=0, 146 GT boxes, 0 predictions), and OVMOT video63 (40 frames; all official TETA components=0, 160 GT boxes, 0 predictions). Every result uses the registered evaluator scope and retains perception rejections; no quality improvement is claimed. J1 step1200, the merged 3-task model, MOT20 G0, and RMOT pilot evidence remain preserved. Clean G2 is still running at event step3043/18227 with finite loss 1.032851 and about 17.370 GiB peak memory; GPU5 G1 and the G2 bundle exporter remain dependency-gated and untouched.

2026-10-19 — held-out zero-metric root-cause audit. The official evaluators are functioning and their receipts are protocol-faithful, but the J1 held-out outputs are empty. RMOT single8 and multi12 reject 8/8 and 12/12 registered frames respectively under the strict parser, with representative `parse_error=duplicate coordinate tuples are not canonical`; `perception_failure_policy=empty_and_record` then emits empty predictions. OVMOT video4 and video63 reject all 40/40 registered annotation frames under the same parser and their official receipts explicitly report predictions=0. DT1203 has nonempty detector candidates on 1201/1203 frames, but `outputs` is empty for all frames because the J1 decision head selects NO_MATCH for every candidate; the two rejected frames show duplicate coordinate/duplicate phrase errors. Thus the zero scores reflect parser rejection plus an over-conservative J1 association/calibration failure, not an evaluator bug. Original receipts remain immutable evidence; any parser/calibration repair must use a new, separately named experiment and must not overwrite this gate.
2026-10-19 — G2 Clean midterm checkpoint/evaluation and GPU7 release: the immutable strict recovery checkpoint at step3000 was validated (manifest SHA f682519a..., base model SHA 3cf09355...), extracted with the new read-only `export_g2_midterm_checkpoint.py`, merged to `G2_CLEAN_MIDTERM_STEP003000`, and reloaded successfully. A first short-control run exposed a stale 8101 G0 service whose scheduler was defunct and timed out at frame302; its failed directory is preserved. A fresh original G0 service on GPU6/8114 and the step3000 G2 service on GPU4/8113 then completed the same MOT17-02-FRCNN 4-frame, Refer-KITTI 0007 single8, and multi12 scopes with the official_scoring parser, tracking_dense decoder policy, Simple association, and TrackEval commit 12c8791b. G0→G2 HOTA/IDF1: MOT 0.750687/0.839695→0.861959/0.894737; RMOT single 0.303853/0.400000→0.472702/0.555556; RMOT multi 0.226643/0.217391→0.448010/0.387097. Multi IDSW increased 0→11; this is short functional evidence, not a final/held-out claim. GPU7 audit found MOT20-05 G0 complete at 1658 frames, expression RMOT 24/24 complete, and no active consumer/supervisor; targeted SIGTERM stopped 8102/8109/8112, ports closed, and GPU7 memory fell 43518→0 MiB. Preserve original G0/J1 zero-score results and all failed/stale-service evidence; G2 main training remains active on GPU3/5.

2026-10-19 — J1 parser/NO_MATCH repair live evidence snapshot. The original all-zero held-out gate remains immutable. The repaired RMOT smoke replays are nonzero (single8 HOTA=0.379221, IDF1=0.666667; multi12 HOTA=0.159866, IDF1=0.146341). The repaired full DT1203 run completed 1203 frames and official TrackEval with HOTA=0.158562, DetA=0.197692, AssA=0.127243, IDF1=0.128277, IDSW=0, FP=110, FN=3560. The subsequent OVMOT replay is still running and records malformed GroundAnything decoder wire payloads (missing coordinate/box tokens, duplicate coordinates/phrases, and an invalid bbox), producing empty candidates on affected frames with a 250–281 second slow path. These errors are recorded rather than repaired by inventing boxes; a separately named decoder/schema replay is required. YesTrack has no active real process in this snapshot. The exact report and JSON are committed as `docs/GROUNDJEV/J1_ZERO_METRIC_REPAIR_REPORT_20261019.*` and `docs/GROUNDJEV/LIVE_EXPERIMENT_METRICS_20261019.*`; no model weights, videos, or caches are committed.

2026-10-19 — P0 TrackJEV dependency repair and independent verification. The real
implementation of `trackjev/__init__.py`, decision schema, candidate competition,
global assignment, losses, model, tracker decision, `global_identity_runtime.py`,
and `global_identity_core.py` was published to
`research/yestrack-onejev-verification-20261019` at commit `9e09ff9`; current
parser/native adapter/run_video files were included. A detached clean worktree
was frozen at local verification commit `b153174`. `compileall`, all TrackJEV
imports, decision/model/adapter self-tests, and a real OneJev checkpoint load
passed. The native smoke completed on real MOT17-02 frame 2 with seven
GT-derived candidates, global assignment and memory updates in 70.34 seconds;
it is plumbing validation only, not a tracking metric.

2026-10-19 — G2 evaluator identity repair. `run_g2_native_eval.py` now labels
adapted outputs `G2-CLEAN-V2`; the historical `G2-PL` identity is not reused.
The fix was pushed to `codex/groundjev-4gpu-20261013` at `9cdb7a4`.

2026-10-19 — verified OVMOT repair boundary. J1 repaired DT1203 remains
HOTA=0.1585619177, DetA=0.1976921614, AssA=0.1272432780, IDF1=0.1282768263,
IDSW=0, FP=110, FN=3560. OVMOT video4 completed 40 registered TAO frames with
40/40 strict perception rejections, 146 GT boxes, 0 predictions, all official
TETA components zero, and end-to-end FPS 0.0037955. Frame latency was
239.58--288.17 seconds due malformed decoder wire payloads. Video63 was not
started after this verified boundary so invalid output would not consume GPU
indefinitely; no video63 metric is claimed. The report and live JSON were
updated without changing model weights or the immutable original gate.

2026-10-19 — G2 live accounting update. The protected GPU3/GPU5 Clean V2
training remains running at event step 4889/18227, latest loss 1.4810857,
peak memory 17.3716 GiB. The recent 500-event mean is 14.51 seconds per step,
which gives a linear remaining estimate of 53.76 hours. No trainer, checkpoint,
optimizer, sampler, RNG, or world size was changed.

2026-10-19 — G2 short-control TRAIN overlap audit completed read-only over all
583,251 Clean V2 rows. The 24 control frames were hashed individually and
matched by normalized image path. All four MOT17-02-FRCNN frames 302--305 have
exact TRAIN rows (the video also has 602 TRAIN rows in the manifest), so that
MOT result is TRAIN-scope/non-independent. The eight RMOT single and twelve
RMOT multi Refer-KITTI 0007 frames have no exact TRAIN path matches in this
manifest. Receipt: `docs/GROUNDJEV/G2_SHORT_CONTROL_TRAIN_OVERLAP_AUDIT_20261019.{json,md}`.
No training row or manifest was changed.

2026-10-19 — J1 strict DT1203 A–E replay ablation completed on the step1200
merged checkpoint (model SHA 5cc64d1b...). All association runs use recorded
candidate/native-feature archives and no GT at runtime. A original parser +
original NO_MATCH: HOTA/IDF1=0/0, 9,755 candidates, 9,729 NO_MATCH, zero
outputs. B repaired parser only: candidate count rises to 13,853 but learned
NO_MATCH still suppresses every output, HOTA/IDF1=0/0. The original runtime
already applied the legal MOT vocabulary fallback on 972 records, so C is
explicitly non-separable from A and is reported as the same immutable receipt.
D only `NO_MATCH=only_if_omitted` on original candidates yields HOTA=0.162326,
DetA=0.202661, AssA=0.130116, IDF1=0.132145, IDSW=0, FP=69, FN=3520, with
1 NEW/1133 ACTIVE/1 REACTIVATE/1 LOST events. E full repair yields
HOTA=0.158562, DetA=0.197692, AssA=0.127243, IDF1=0.128277, IDSW=0, FP=110,
FN=3560, with 1 NEW/1135 ACTIVE events. No threshold, seed, checkpoint or
candidate was changed within each factor; parser repair is not claimed as an
identity improvement. Receipt: `docs/GROUNDJEV/J1_DT1203_A_E_STRICT_ABLATION_20261019.{json,md}`.

2026-10-20 — AerialMind full-release acquisition and audit. The official
Hugging Face release was downloaded under `data/downloads/AerialMind` and
passed CRC tests for `expression.zip` (50,089,761 bytes),
`labels_with_ids.zip` (53,789,787 bytes), and `image_02.zip`
(24,425,047,497 bytes). The complete release was extracted to
`data/AerialMind`; the audit reports 93 videos, 24,617 expression JSON files,
45,630 image frames, 45,612 label frames, 8,479,541 expression-frame
references, and 46,135,642 target references. All referenced image and label
files exist, both official 17-video VisDrone and 13-video UAVDT test
partitions are present, and no JSON/label format errors were found. The 18
image frames without standalone label files are not referenced by an
expression; persistent target IDs absent from individual frames are treated
as normal occlusion/out-of-view state, not fabricated GT. Receipt:
`results/audits/aerialmind_dataset_audit.json`.

2026-10-20 — AerialMind manifest/evaluator adaptation. The runtime now uses
the official expression filename as query text, while retaining the existing
KITTI `sentence` behavior by configuration. Evaluation manifests record
`xywh_topleft` for VisDrone, `cxcywh` for UAVDT, and zero frame offset because
AerialMind filenames are 1-based. Predictions and GT were both verified on
the same frame convention. A direct TrackEval adapter was added to supply
explicit sequence lengths and avoid the pinned TempRMOT MOT wrapper's stale
hard-coded KITTI image path. The Transformers 5 tokenizer API difference was
bridged narrowly with a `batch_encode_plus` compatibility alias; the source
model remains frozen and checkpoint loading still reports zero missing or
unexpected model keys.

2026-10-20 — complete-data AerialMind single-card experiments on physical
GPU 2 (A800 80GB, 0 compute processes, CUDA UUID
`GPU-c2484ded-5f87-237c-bbf0-d23fcb74fdb2`). Two complete episodes were run:
VisDrone `uav0000009_03358_v` (219 frames) and UAVDT `M0203` (1,007 frames),
1,226 frame queries total, with no GT or future frames available to the
runner. The formal QCS suite (`FROZEN`, `PROBE_CACHE`, `QCS_TTA`, `NO_STATE`)
completed at 9.70 FPS shared-source throughput and 3.21 GB peak allocated
memory, but all methods selected zero rows and HOTA/DetA/AssA/IDF1 were 0.
The diagnostic `referring_threshold=0.2` run also selected zero rows. A
separate all-thresholds-zero candidate lower bound emitted 1,775 rows and
obtained combined HOTA 2.7327, DetA 1.0171, AssA 7.3499, IDF1 0.0; QCS was
identical to FROZEN with zero optimizer steps. The lower-bound per-episode
result was M0203 HOTA 4.4524 / DetA 2.6987 / AssA 7.3499 and VisDrone HOTA
0.0. These results show the data/GT/evaluator path is functional but the
Refer-KITTI-V2 checkpoint is not a viable AerialMind source model: its
absolute referring scores collapse on VisDrone and its first UAVDT frame has
no valid tracked candidates. Changing only the TTA residual structure cannot
repair missing source detector/tracker calibration; a target-domain source
training checkpoint or a separately trained domain-robust candidate source is
required before claiming AerialMind QCS gains. The full official AerialMind
training claim is intentionally not started on one card because the official
repository reports about 110 hours on eight GPUs; the supervised training
path remains the next experiment, with the current zero-shot runs preserved
as negative controls.

2026-10-20 — AerialMind relative-rank structure change and complete-episode
retest. The absolute referring gate was insufficient under the measured
target-domain calibration shift, so the runner now has an explicit
`relative_margin` mode: after admitting valid frozen tracks with zero
objectness/area gates, it keeps the set of candidates within a fixed 0.75
logit margin of the best current-frame referring score and caps the set at 16;
it never reads GT, future frames, or forces a top-1 decision. The original
absolute mode is unchanged and a CPU unit test covers both paths. On the same
complete VisDrone episode (219 frames) and UAVDT episode (1,007 frames), the
relative run completed all 1,226 queries at 9.35 shared-source FPS and 3.21 GB
peak allocated memory, emitted 1,344 rows, and sealed TrackEval for all four
methods. Combined FROZEN/PROBE_CACHE/QCS_TTA/NO_STATE each reported HOTA 3.0240,
DetA 1.2719, AssA 7.1964, DetRe 5.0987, DetPr 1.4568, and IDF1 0.0; the
previous all-thresholds-zero lower bound was HOTA 2.7327 / DetA 1.0171 /
AssA 7.3499 / IDF1 0.0. QCS still made zero optimizer updates and all four
methods were identical, so the change recovers candidate visibility but does
not establish referring adaptation or a useful source-free AerialMind result.
The complete official 30-video test manifest was also generated with 5,125,243
runtime rows and 5,125,243 evaluation rows; runtime contains no GT fields.
Receipts: `results/runs/aerialmind_two_episode_relative_rank/`,
`results/manifests/aerialmind_test_{runtime,evaluation}_manifest.jsonl`, and
`results/manifests/aerialmind_test_manifest_metadata.json`.

2026-10-20 — AerialMind supervised-training boundary audit. The official
`refer-uav.train` list maps 27,044 image rows to 63 training videos and 16,216
expression files; every listed image and label exists, every training video
has an expression directory, all query JSONs are valid, and the training
videos have zero overlap with the 30-video official test partition. Receipt:
`results/audits/aerialmind_training_split_audit.json`. This makes a future
target-domain supervised checkpoint path reproducible, but it is deliberately
not used by the source-free QCS experiments; the official repository's roughly
110-hour/8-GPU training claim is not replaced by a one-card partial run.

2026-10-20 — Refer-BDD complete-data audit and manifest preparation. The
official EchoTrack seqmap and Google Drive archives were downloaded to
`/data2/user/LocateMOT/data/downloads`, checksum/unzip validated, and the
complete selected split was extracted under `data/ReferBDD`. The audit found
8 videos, 673 expressions, 1,619 images, 1,610 label files, 74,924
expression-frame references, and 216,613 target-object references. A
label-free runtime manifest and a separate evaluator-only manifest were
created; no raw data or checkpoint is intended for publication.

2026-10-20 — Refer-KITTI-V2 → Refer-BDD source-free online experiment.
Hypothesis: frozen-query probes plus external causal residual adaptation could
recover source referring decisions under BDD domain shift without target
labels. The full six-method suite (`FROZEN`, `ENTROPY_HEAD`, `PROBE_ST`,
`PROBE_CACHE`, `QCS_TTA`, `NO_STATE`) ran on physical GPU 6 using the complete
673-episode/136,188-frame-query manifest. The runner used no GT, future frame,
expression label map, or evaluator; GT was materialized only after prediction
completion. The run completed in 32,513 seconds at 4.191 shared-source FPS,
with 8.44 GB peak allocated memory.

2026-10-20 — Refer-BDD experiment diagnosis and retained result. The first
smoke exposed an environment ABI mismatch and an evaluator +1 frame-offset
mistake; both were fixed and replayed before the full run. The full sealed
result is complete for all 8 videos and 673 expressions. FROZEN obtained
HOTA/IDF1 2.9788/0.89107. ENTROPY_HEAD made 31,656 real online optimizer
updates and degraded to 1.5842/0.28707. Probe/QCS/NO_STATE made 17,616 shared
probe calls but zero supported optimizer updates and remained identical to
FROZEN. Offline candidate analysis found approximately 26.4% IoU-0.5 recall
before gates, 19.8% after objectness/area, and 0.7% after the fixed 0.4
referring gate; source-score p95 was 0.3088. This is retained as a valid
negative cross-domain result; no BDD supervised training, threshold tuning,
GT-assisted structure change, or zero-gate result was substituted.

2026-10-20 — Result-accounting repair. The runner was corrected to accumulate
optimizer/probe counters across episode resets. The existing sealed output was
repaired transparently from `update_trace.jsonl`; the previous last-episode
values (50 entropy steps and 36 probe calls) are preserved in the metadata's
`counter_correction` field, while the published summaries report the complete
suite totals. The BDD asset audit, diagnostics, metrics CSVs, and final report
are kept; unrelated GroundJEV worktree changes were not staged or overwritten.

2026-10-20 — TTA-RMOT task boundary and ShiftRMOT-Bench-v1 implementation. This
is a separate repository/task from LocateMOT; all changes in this entry are
under `/data2/user/TTA-RMOT`. Hypothesis: compact paired visual/language
conditions can test source-free RMOT adaptation without storing duplicate
transformed images or exposing GT to the online runner. The complete
Refer-KITTI-V2 official test videos (`0005`, `0011`, `0013`, `0019`) are audited
before model execution; `0005` is a dev label and the remaining videos carry
an explicit historical-overlap warning rather than an unseen-split claim.
Images are symlinked to the external data2 asset store, visual shifts are
deterministic and geometry-preserving, and language rewrites are restricted to
audited directional phrasing. The runner/evaluator now expand compact rows at
their respective boundaries. The protocol labels for TDA/SAR/TPT/TCR are
explicitly marked adapted/inspired where the open-source method cannot be
reproduced exactly on TempRMOT's architecture; no result is retained until the
manifest audit and a paired Frozen/TTA run complete.

2026-10-21 — ShiftRMOT absolute-gate pilot. Four representative expressions,
45 A0/A1/A2/A3 conditions, and 9,915 frame queries completed on one A800
(`GPU-0322c0e7-ffbf-e068-5bdd-587b8216a9d9`). Frozen HOTA/DetA/AssA/IDF1 was
0.31255/0.090851/1.2371/0. The absolute gate emitted only 702 rows, all from
the 0019 representative episode; SAR_VISUAL_INSPIRED emitted 84 rows after
2,333 entropy updates and scored zero. TDA_RMOT, TPT_LANGUAGE_INSPIRED, and
TCR_FUSION_INSPIRED matched Frozen; probe calls occurred but no supported
optimizer updates occurred. This result is retained as a negative gate
calibration observation.

2026-10-21 — ShiftRMOT relative-margin pilot and structure decision. The same
45 conditions and 9,915 frame queries were rerun with objectness/area
candidate visibility followed by a fixed 0.75 current-frame referring-logit
margin and a 16-candidate cap, without top-1 forcing. On GPU UUID
`GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2`, Frozen emitted 28,064 rows and
obtained HOTA/DetA/AssA/IDF1 2.9113/2.3094/4.2506/0.012108. The structure
change is kept because it recovered measurable candidates; it is not claimed
as a TTA gain. TDA/TPT/TCR remained identical to Frozen with zero updates;
SAR made 2,333 updates but decreased to HOTA 2.8306 and AssA 4.0007. Both
sealed evaluator outputs and the condition-level tables are retained under
`results/shift_rmot_pilot/`; the final report records that this is a
representative pilot rather than an all-4,540-condition claim.

2026-10-21 — ShiftRMOT-Bench-v2 complete Refer-KITTI-V2 data/protocol build.
The v1 audit confirmed that the runtime frame list was derived from expression
label keys and that visual seeds depended on expression/condition IDs. The
resulting risk was false negative frames during causal tracking and unequal
pixels between A1/A3 paired conditions. The v2 builder now inventories all 21
videos and 9,758 expressions, records the official train-list and seqmap
SHA256 values, keeps 8,008 source frames as external symlink targets, and
creates controlled conditions for all 862 official test expressions. Runtime
rows use every source-video frame in order; evaluator rows use only the
expression label keys after prediction. Visual seeds are independent of
expression and language and follow the registered video/frame/shift/severity/
global-seed protocol. Natural same-label-map language variants cover 825
expressions twice, 34 once, and 3 not at all; this gap is disclosed rather
than filled with unsafe rewrites. The complete manifest has A0/A1/A2/A3
counts 862/6,896/1,684/13,472. Pairing, no-GT runtime, symlink, target-set,
and time-axis audits passed; the audit status is
`PASS_WITH_DISCLOSED_LANGUAGE_COVERAGE_GAP`. Receipts:
`data/ShiftRMOT-v2/`, `results/audits/shift_rmot_v2_audit.json`,
`scripts/build_shift_rmot_v2.py`, and `scripts/audit_shift_rmot_v2.py`.

2026-10-21 — Independent v2 TTA implementation and source-training launch.
The previous v1 protocol labels were not reused as aliases. `TDA_RMOT` is now
a bounded non-parametric feature/reliability cache; visual, language, fusion,
and all pairwise/full combinations use separate zero-initialized residual
adapters over frozen TempRMOT snapshots. The language arm consumes the frozen
256-D RoBERTa-to-TempRMOT query projection, and the fusion arm includes an
explicit feature-query cross product and set-separation loss. Synthetic CPU
regression passed for all seven trainable arms with source-equivalent initial
logits, nonzero gradient, and parameter movement; TDA cache hits/updates also
passed. A wrapper sampling-length bug was fixed so inference follows the
official `hist_len=5` source-training configuration instead of the historical
hard-coded four-frame setting. The full 60-epoch official Refer-KITTI-V2
TempRMOT source run was launched on the selected free GPU 7; completion is
pending the independent checkpoint/optimizer/epoch audit. No full metric or
reproduction claim is made yet.

2026-10-21 — Official label-coordinate audit correction before metrics. A
cross-check against the pinned TempRMOT `datasets/refer_kitti.py` loader found
that Refer-KITTI KITTI label files are normalized top-left `x1,y1,w,h`, not
`cx,cy,w,h`. The first v2 manifest metadata had recorded the wrong convention;
it was corrected to `xywh_topleft`, the full manifest was regenerated, and the
audit was rerun successfully with the official time-axis, pairing, symlink,
and no-GT checks. No metric or training result based on the stale metadata is
retained or reported.

2026-10-21 — GPU7 source-training launch failure and controlled retry. The
first full TempRMOT Refer-KITTI-V2 launch was recorded with
`CUDA_VISIBLE_DEVICES=7`, but its launcher disappeared before producing any
training log, checkpoint, exit status, or end timestamp. A one-time GPU
diagnostic found three unrelated compute processes on physical GPU7, so the
card choice was not suitable for a clean single-card run; the exact kill
signal is not recoverable from the preserved files. No epoch, metric, or
completion claim is accepted from this attempt. The unchanged 60-epoch
official protocol is being retried under a new run root on physical GPU1,
which had one external process and substantially more available memory at
selection time. Receipt:
`results/source_training/temprmot_refer_kitti_v2_gpu7_launch_failure.json`.

2026-10-21 — GPU1 retry reached model initialization but failed on the first
training batch. The pinned TempRMOT code calls
`RobertaTokenizerFast.batch_encode_plus`, while the shared runtime provides
`transformers==5.2.0` without that legacy alias. The run exited with status 1
and produced no checkpoint or epoch result; this is recorded separately from
the GPU7 launch failure. A project-local `sitecustomize.py` compatibility
alias was added, and a local tokenizer regression passed before resuming the
unchanged training protocol. Receipt:
`results/source_training/temprmot_refer_kitti_v2_gpu1_transformers5_failure.json`.

2026-10-21 — Compatibility-resumed source training progress snapshot. The
third isolated run root
`/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full_retry_gpu1_compat`
passed initialization and was executing the unchanged official protocol on
physical GPU1. At the one-time snapshot it had reached epoch 0, iteration
950/5168, with average loss 9.2707 and approximately 5.3 GiB peak memory;
the logger reported 08:02:08 remaining for that epoch. This is not a final
metric or completion claim. Receipt:
`results/source_training/temprmot_refer_kitti_v2_full_retry_gpu1_compat_progress_snapshot.json`.

2026-10-21 — Official TempRMOT Refer-KITTI-V2 checkpoint recovered for a
separate fast comparison line. The stale README link was corrected by reading
the official `zyn213/TempRMOT` v1.0 release metadata; `checkpoint_rk2.pth`
downloaded successfully to external asset storage, has 1,096,316,824 bytes,
SHA256
`4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`, and
loads as epoch 49 with 969 model keys. It is being used only for an official
Frozen/source-free-TTA comparison and is not substituted for the independent
no-leakage source retraining protocol. Receipt:
`results/source_training/temprmot_refer_kitti_v2_official_checkpoint_provenance.json`.

2026-10-21 — First released-checkpoint inference launch failed during model
construction because the new single-GPU wrapper omitted the official test
sampler arguments; the pinned source calls `max(args.sampler_lengths)` and
therefore raised on `None`. No prediction or metric was produced. The wrapper
is corrected to match the pinned `configs/temp_rmot_test.sh` settings
(`sampler_steps=50,90,150`, `sampler_lengths=2,3,4,5`, `hist_len=8`, and
`extra_track_attn`), and the retry will use a separate output root. Receipt:
`results/source_training/temprmot_official_inference_gpu3_sampler_args_failure.json`.

2026-10-21 — The corrected official-checkpoint inference reached the first
 297-frame sequence but failed on its first text-encoding call because the
 wrapper's runtime path insertion did not trigger Python's startup
 `sitecustomize` import. The same Transformers 5.2 compatibility alias works
 in the training launcher, which sets `PYTHONPATH` before interpreter startup.
 The wrapper now explicitly imports `sitecustomize` after path setup; no
 complete prediction or metric is accepted from this run. Receipt:
`results/source_training/temprmot_official_inference_gpu3_testcfg_transformers5_failure.json`.

2026-10-21 — A targeted strict-load regression exposed a second official
checkpoint issue. The V2 test shell enables `extra_track_attn`, but the
released RK2 state dict has no `update_attn` or `norm4` tensors; strict loading
reports 36 missing decoder tensors. The permissive official loader would leave
these layers randomly initialized, so the long GPU3 run was stopped before
accepting metrics. The checkpoint-compatible fast line now uses the source
training architecture (`extra_track_attn=false`, `hist_len=5`, sampler lengths
`5,5,5,5`). Receipts:
`results/source_training/temprmot_official_inference_gpu3_extra_track_failure.json`
and `results/source_training/temprmot_tta_release_smoke_gpu7_extra_track_failure.json`.

2026-10-21 — After disabling the untrained extra decoder attention, the
checkpoint-compatible independent runner loaded the official RK2 weight with
zero missing/unexpected model keys and completed a 10-frame A0 regression on
GPU7. The runner recorded `gt_used_by_runner=false`,
`evaluation_manifest_opened=false`, and `future_frames_used=false`; the
post-hoc evaluator also completed on the one-expression smoke output. This is
only a compatibility/evaluator regression, not a full-dataset metric. Receipt:
`/data3/user/TTA-RMOT-runs/shift_rmot_v2_official_release_smoke_gpu7_compat2`.

2026-10-21 — Full ShiftRMOT-v2 source-free online TTA was launched from the
checkpoint-compatible released TempRMOT weight. The 22,914-condition runtime
manifest is partitioned by deterministic condition index across four
independent shards on physical GPUs 3, 5, 6, and 4; shard outputs are merged
only after all predictions finish, then evaluated with the post-hoc evaluator.
Each runner is GT-free (`gt_used_by_runner=false`, no evaluation manifest and
no future frames). This is an explicitly labeled released-weight comparison
line: checkpoint provenance is external and cannot by itself certify
no-leakage, so the separate source-retraining line remains required.
Launch receipt:
`results/tta/shift_rmot_v2_release_full_sharded_launch.json`.

2026-10-21 — Revalidated the requested Refer-KITTI-V2 → Refer-BDD line. The
existing sealed result covers all 8 official videos, 673 expressions and
136,188 frame-query rows for FROZEN, ENTROPY_HEAD, PROBE_ST, PROBE_CACHE,
QCS_TTA and NO_STATE. The Refer-BDD assets, archives and manifests were
copied from the old shared checkout into TTA-RMOT-owned `/data2` paths; the
asset audit still passes with identical archive/seqmap hashes, and the copied
prediction/evaluation artifacts are byte-identical. The runtime manifest has
no GT fields and the historical run metadata records no GT, future-frame or
evaluation-manifest access. This establishes protocol isolation, but the
publisher's complete training history for the external RK2 checkpoint cannot
be independently audited; therefore this remains a released-weight result,
not an absolute no-leakage certificate. Receipt:
`results/audits/refer_bdd_pretrained_leakage_audit.json`.

2026-10-22 — Resource-cap correction. A live process audit showed that the
TTA-RMOT source-training line (GPU1), released-checkpoint official inference
(GPU7), and four auxiliary ShiftRMOT-v2 shards (GPUs 3/4/5/6) would exceed
the user-imposed maximum of four physical GPUs. The four auxiliary shards
were interrupted through their existing handles; their partial traces remain
for failure/progress provenance but are not merged or scored. The source
training and official inference lines were left running; no new TTA-RMOT GPU
job was started after the correction. Receipt:
`results/tta/shift_rmot_v2_release_full_sharded_launch.json`.

2026-10-22 — Task-boundary correction. A FlexHook asset/inference detour was
identified as unrelated to the requested TTA-RMOT protocol
(TempRMOT Refer-KITTI-V2 -> Refer-BDD). Its single-card run was interrupted
before completion, produced no accepted metric, and its scripts/receipt were
removed from the main experiment line. The valid Refer-BDD results and the
source-training/inference evidence remain unchanged.

2026-10-22 — Explicit FlexHook task resumption in an isolated TTA-RMOT
worktree. The user clarified that this is the TTA-RMOT/FlexHook task and not
LocateMOT or GroundJEV. Work therefore continued at
`/data2/user/FlexHook-TTA` on branch
`research/flexhook-generalizable-plugin-tta`, leaving the original
`/data2/user/TTA-RMOT` dirty files untouched. The fixed vendor commit is
`bd1acc38634b28525d54dc6e0fcb38335f0029f9`.

2026-10-22 — User-requested process cleanup. The obsolete long V2 source
training chain (PIDs `2777858`, `2777863`, `2777864`, `2778107`) was stopped
with TERM, preserving its checkpoints and logs. The separate old official
TempRMOT inference PID `3105941` was also explicitly stopped after confirming
it was not the FlexHook run. No unrelated GPU process was touched.

2026-10-22 — Official FlexHook Refer-KITTI-V2 reproduction completed. A
single-card GPU3 trial reached batch 280/3722 and was stopped for runtime; no
metric from it is accepted. The official two-process protocol then completed
on physical GPUs 1 and 4 across the four test videos and 862 expressions.
TrackEval returned HOTA 42.526, DetA 30.629, AssA 59.186, with 750,834
detections and 30,871 predicted IDs. HOTA matches the official README. Rank
leaf-score counts were 1,177,280 and 1,178,248 with zero overlap and a merged
count of 2,355,528. Receipt:
`results/flexhook/official_refer_kitti_v2.json`.

2026-10-22 — Implemented a separate `flex_tta` score-plugin line. The
FROZEN adapter is an exact score-only replay baseline. `ONLINE_STABILITY`
uses only the previous frame of the same tracker ID to update a bounded affine
score calibration; it never reads target GT or future frames and never changes
boxes or IDs. A first full run exposed a serialization bug where the official
car/pedestrian ID offset was applied only to the in-memory key. The duplicate
ID failure was recorded in the ignored `flexhook_score_tta_v2` run, then fixed
by applying the same offset to MOT column 2 and rerun as v3.

2026-10-22 — Full source-free score-TTA result accepted after the corrected
FROZEN reproduction matched the official metric. Across 862 expressions and
2,355,528 candidates, FROZEN scored HOTA 42.526 / DetA 30.629 / AssA 59.186;
ONLINE_STABILITY scored HOTA 42.525 / DetA 30.625 / AssA 59.191 after
2,153,705 causal pseudo-updates. The result is HOTA-neutral within 0.001 and
is retained as a baseline, not claimed as an improvement. Compact receipt:
`results/flexhook/flexhook_score_tta_v3.json`.

2026-10-22 — Remaining scope. The audited official FlexHook resources contain
only the Refer-KITTI-V2 checkpoint and Temp-NeuralSORT tracker bank. No legal
official FlexHook BDD/AerialMind weights or corresponding tracker banks were
found, and no cross-tracker full result has been claimed. BDD/AerialMind and
cross-tracker generalization remain unfinished; this log records the limit
instead of substituting an unverified model or target labels.
