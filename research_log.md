Warning: truncated output (original token count: 46846)
Total output lines: 1659

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
  contracts, loss/runtime interfaces, parameter-sharing audit and reproducible
  audit/sanity tools. Exact source recipes and commit IDs are in
  `reports/unified/OPEN_SOURCE_TRAINING_AUDIT.md` and
  `outputs/unified/open_source_audit.json`.
- Data/protocol: froze V1/V2 fit/calibration/validation splits and reserved
  V1 `0005,0011,0013` plus V2 `0005,0011,0013,0019`; retained all 157 V2
  source target-ID-missing rows as unlabeled records. No official-test labels
  or screening ground truth were opened.
- Checkpoint/GPU: no LocateMOT-U checkpoint and no training run yet; GPU
  allocation was zero. External sanity was compile/help only (about 23 s wall
  time): 10/10 selected source files compiled, 1/10 entrypoints reached help,
  and the nine remaining failures are recorded dependency/version findings.
- Results: skeleton CPU forward finite with score shape `[1,100]`; legal guard
  smoke passed; skeleton non-head parameter fraction `0.9998259786` (not a
  final-model claim). No benchmark metric was produced. U0 scope/audit/sanity
  passed; U1 MM-GroundingDINO-B acquisition and proposal/expression gates are
  next.

## LocateMOT-U U1 — Swin-B foundation acquisition and adapter smoke — 2026-10-01

- Hypothesis: the audited GroundingDINO Swin-B CogCoOR weights can serve as the
  shared visual/specification foundation after selecting the checkpoint format
  compatible with the local MMDetection GroundingDINO implementation.
- Assets: acquired and hashed both the original release
  (`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`,
  938,057,991 bytes) and the OpenMMLab conversion
  (`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`,
  935,971,702 bytes). The MMDetection asset is the runtime candidate because
  its key layout matches the adapter; provenance is in
  `outputs/unified/protocol/foundation_asset.json`.
- Code/smoke: built the local Swin-B model (232,996,763 parameters), loaded the
  converted checkpoint with 8 expected initialization keys and 3 BERT metadata
  keys reported as non-fatal mismatch, then ran one CUDA label-free forward on
  legal V1 video `0004` frame `000063.png` using `car . pedestrian . cyclist .`.
- Results: 300 finite predictions, finite boxes and finite scores;
  `outputs/unified/u1_foundation_smoke.json` records the exact config,
  checkpoint hash, device and legal frame. This is a compatibility smoke only;
  no proposal recall, expression grounding metric or tracking result is
  claimed. Official-test labels and screening ground truth remain unread.
- Checkpoint/GPU/walltime: no LocateMOT-U checkpoint; one A800-equivalent GPU;
  model build/forward about 9 s. Next gate is label-free proposal and
  expression grounding on legal development frames.

## LocateMOT-U U1 — proposal gate diagnostic — 2026-10-01

- Hypothesis: the loaded Swin-B foundation would provide sufficiently localized
  open-vocabulary candidates for the shared semantic/tracking path on a fixed
  legal development slice.
- Protocol: 16 V1/V2 calibration/validation expression-frame rows, 50 target
  boxes, 300 candidates per row; model inference completed before each
  corresponding label file was opened. No threshold fitting, training,
  official-test labels or screening ground truth.
- Result after correcting the MMDetection BGR input contract: best-IoU
  `>=0.25` was `0.5400`, best-IoU `>=0.50` was `0.0400`, mean best IoU
  `0.2756`, and mean score at best IoU `0.01312`. Boxes and scores were finite.
  This remains below a usable proposal gate, so semantic-head/tracking training
  from this stream is disallowed. The initial RGB/BGR-swapped diagnostic was
  discarded and is not used as evidence.
- Diagnosis/next change: inspect prompt tokenization/category assignment, box
  decoding and proposal selection on this same legal slice. Preserve the split,
  rows, checkpoint and labels; do not tune a threshold or delete hard cases.
  Authoritative output: `outputs/unified/u1_proposal_gate.json` and
  `reports/unified/U1_PROPOSAL_GATE.md`.
- Selection-only check: retaining all 900 decoder queries raised IoU@0.50 to
  `0.1000` and mean best IoU to `0.3341`, so top-k selection contributes but
  does not repair localization. The top-900 output is retained separately and
  is not used as a training bank.
- A fixed class prompt (`car . pedestrian . cyclist .`) with all 900 queries
  reached IoU@0.25 `0.9000`, IoU@0.50 `0.1800`, mean best IoU `0.3635`; this is
  a useful direction for the repair but still below the stage gate and remains
  excluded from training. Output:
  `outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

## LocateMOT-U U1.3 — proposal-selection repair diagnostics finalized — 2026-10-01

- Change: made the canonical 300-query output reproducible with explicit
  `max_per_img` and prompt metadata, and synchronized the report/status values
  with the corrected BGR run. Re-ran the 300-query baseline, the 900-query
  selection diagnostic, and the fixed class prompt (`car . pedestrian .
  cyclist .`) diagnostic on the identical legal 16-row slice.
- Results: the baseline stayed at IoU@0.25 `0.5400`, IoU@0.50 `0.0400`, and
  mean best IoU `0.2756`; retaining 900 queries reached `0.1000` at IoU@0.50;
  the fixed class prompt with 900 queries reached `0.1800` at IoU@0.50 and
  `0.9000` at IoU@0.25. All boxes/scores were finite, no rows were skipped,
  and no official-test labels or screening ground truth were read.
- Interpretation: prompt/category assignment and top-k selection are concrete
  repair directions, but the proposal gate remains below the stage criterion.
  No semantic-head/tracking training bank was created and no training was
  launched. The next experiment must keep this split/checkpoint fixed while
  inspecting prompt tokenization, category assignment and box decoding.
- Verification: unified modules/tools compile; CPU skeleton smoke is finite;
  parameter-sharing audit remains skeleton-only with non-head fraction
  `0.9998259786`; legal guard accepts V1 `0004` and rejects reserved `0013`.

## LocateMOT-U U0R/U1R measurement-contract repair — 2026-10-01

- Scope: resumed from `014b8a5ec81bb83c5993b56da171089f8a440a11` on
  `codex/u0r-u1r-foundation-repair-20261001`; no U2 work and no training was
  launched. The project root was verified as `/data2/user/LocateMOT`.
- Reproducibility hypothesis: a fresh checkout should import the unified data
  package using tracked source only. The first clone failed because the broad
  `data/` ignore rule also hid `locatemot/unified/data/`. Added a narrow ignore
  exception, committed the package source, pushed it, and re-cloned the remote
  branch at `26f9c989c8b13773d681bbe6962b77e73a0026bd`. Imports, skeleton smoke,
  and foundation smoke passed in the clone. Decision: keep the repair and mark
  U0 legal-scope reproducibility PASS.
- Checkpoint hypothesis: the MMDetection Swin-B conversion must have only
  documented non-core load mismatches. `tools/unified/audit_foundation_checkpoint.py`
  now requires the exact one training-only missing key
  `dn_query_generator.label_embedding.weight`, the exact three BERT metadata
  unexpected keys, and zero backbone/encoder/decoder/bbox-head/language-fusion
  missing keys. Audit result: `PASS_CONDITIONAL_TRAINING_ONLY_MISSING`.
- Preprocessing hypothesis: the formal runtime must be equivalent to the
  official MMDetection path. Ten deterministic legal images were compared with
  the same fixed generic vocabulary; all produced 300 finite predictions,
  maximum box difference `0.0`, and maximum score difference `0.0`. Decision:
  preprocessing contract PASS; the prior hand-built tensor smoke is not formal
  evidence.
- Split hypothesis: a deterministic video-disjoint split balancing expression
  volume, physical-target volume, and motion proxy is adequate before formal
  evaluation. `tools/unified/build_legal_split.py` froze v2 with seed `20261001`
  and split SHA
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`.
  V1 fit/calibration/validation are `9/3/3` videos and V2 are `9/3/3`; V2
  source-target-ID-missing references remain `157` (fit only in this split),
  with no replacement or silent drop. Decision: freeze this split before any
  U1 adaptation or formal gate.
- Hardware/wall time: one CUDA device (A800-equivalent) was used for the
  checkpoint audit, smoke, and equivalence check; these short checks completed
  in under a minute each. Official-test labels and screening ground truth were
  not opened. External baseline execution remains incomplete and is the next
  blocking U0 task.

## LocateMOT-U U0R — TempRMOT legal-development execution — 2026-10-01

- Hypothesis: at least one published external RMOT method can be executed on a
  frozen legal-development subset without reading official-test labels or
  mixing its environment into LocateMOT-U.
- Protocol: copied TempRMOT at commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1` into the isolated environment
  `/data2/user/reference_envs/temprmot`, loaded the released checkpoint
  `checkpoint_rk.pth` (SHA-256
  `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`), and
  ran the adapter on V1 validation videos `0012` and `0014`, one legal
  expression per video. TrackEval then evaluated the two generated sequences
  with HOTA, CLEAR, and Identity.
- Result: prediction and evaluation completed with 94 prediction rows and 93
  ground-truth rows. Combined HOTA was `94.679`, MOTA `98.925`, and IDF1
  `99.465`. Official-test labels, screening ground truth, and training were not
  used. The raw outputs and environment remain outside the repository;
  `outputs/unified/external_baselines/temprmot_legal_v1.json` records the
  reproducible metadata and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`
  records the bounded scope.
- Compatibility work was confined to the isolated copy: a torchtext shim,
  current PyTorch `torch.load` behavior, disabling a redundant pretrained
  download, a CUDA scalar-type API update, and NumPy aliases in TrackEval.
  These changes do not alter the LocateMOT-U model or the frozen split.
- Decision: keep the TempRMOT run as real external-baseline evidence, while
  retaining `U0_EXTERNAL_BASELINE=INCOMPLETE` until a second independent
  method also produces predictions and evaluation. DKGTrack and FlexHook
  remain pending because their published weight links are not yet available
  through the isolated download path.

## LocateMOT-U U1R — formal generic and expression measurements — 2026-10-01

- Measurement contract: the frozen split SHA is
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`; the
  foundation checkpoint SHA is
  `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`; every
  forward used `LoadImageFromFile`, `FixScaleResize(800,1333)`,
  `PackDetInputs`, and `DetDataPreprocessor` with BGR-to-RGB conversion.
- Generic protocol: one fixed query-independent vocabulary, all legal
  calibration/validation frames with materialized labels, physical units
  `(dataset, video, frame, track_id)`, Top100/150/300/900 diagnostics, and
  small/medium/large height strata. The run covered 4,201 unique images,
  33,526 physical GT units, and 5,992 unique physical objects. Top150 recall
  was V1 `0.0803` and V2 `0.0669` at IoU 0.50 (V1/V2 IoU .25: `.5309/.4940`,
  IoU .75: `.0045/.0062`). Decision: formal generic gate FAIL; this is below
  the `.80` adaptation early-pass, so a detection adaptation probe is required
  before any grounding adaptation.
- Expression protocol: every legal calibration/validation query contributed
  one deterministic middle labeled frame and, when available, one first empty
  target frame. Units were `(dataset, video, query_id, frame, target_id)`;
  source-missing references stayed in the source denominator. The run covered
  7,160 query-frame rows and 7,199 valid target units with zero source-missing
  units in this calibration/validation sample. Top1 IoU .50 recall was V1
  `0.0030` and V2 `0.0028`; AP50 and empty-frame FP/query are recorded in
  `reports/unified/U1_EXPRESSION_GROUNDING_GATE.md`. Decision: formal
  expression gate FAIL; no expression adaptation has started.
- No official-test labels or screening ground truth were read. No U2 or
  universal track core work was started. The external-baseline state remains
  incomplete because only TempRMOT has real predictions/evaluation; DKGTrack
  and FlexHook public Baidu links require an authenticated download path.

## LocateMOT-U U1-A — protocol correction, valid adaptation, and stop — 2026-10-01

- The first eight-process launch was invalid because MMEngine's default
  `--launcher none` left the processes outside DDP; it was stopped before any
  formal evidence. A later exploratory FP32 run also used `frozen_stages=1`.
  MMDetection's Swin implementation shows that value freezes patch embedding
  and stage0 only, while the registered contract requires stages 0 and 1.
  Those exploratory metrics are excluded from the final record.
- Corrected hypothesis/protocol: a fixed legal-fit generic stream can repair
  the foundation proposal localization while freezing Swin stages 0/1 and
  BERT. The legal-fit stream has 3,606 rows and fixed `label 0 -> car`; the
  split, vocabulary, official `FixScaleResize(800,1333)` pipeline, and all
  denominators stayed unchanged. The valid run used
  `frozen_stages=2`, BF16 AMP, eight-GPU PyTorch DDP, batch 2/GPU, and no
  accumulation (global batch 16). The logged peak was 5,945 MB/GPU. A two-step
  smoke passed before the full run.
- A batch-2/two-step-accumulation attempt passed a one-step smoke but failed in
  the full run with PyTorch's `expect_autograd_hooks_` DDP reducer assertion.
  The smallest working fix was accumulation 1; this kept the recommended
  microbatch 2 and did not alter the official input resolution. This failure
  and the corrected smoke are retained in the adaptation report.
- The valid schedule completed epochs 1, 2, 4, and 6, each formally evaluated
  on the unchanged 4,201-image, 33,526-physical-unit scope. Top150
  Recall@IoU0.50 (V1/V2) was foundation `0.0803/0.0669`, then
  `0.8729/0.8515` (epoch1), `0.8873/0.8766` (epoch2),
  `0.8945/0.8843` (epoch4), and `0.8938/0.8815` (epoch6). IoU0.25 reached
  about `.99` in both domains, while epoch-6 IoU0.75 was `0.7215/0.6119`.
  Size-stratum IoU0.50 was small `0.9136`, medium `0.9445`, and large
  `0.7085`.
- Training logs contain occasional `grad_norm: nan` values, but loss values
  remained finite, all six checkpoints were written, and every formal shard
  produced finite predictions. This telemetry is reported rather than
  filtered. The final epoch-6 checkpoint SHA-256 is
  `a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
- Decision: the valid V1/V2 generic gate did not reach `.90/.90` after the
  registered six-epoch maximum. Expression adaptation was not started because
  it is downstream of the generic gate. Added the corrected adaptation report,
  failure decomposition, and four valid formal JSONs. Final status is
  `LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION` with
  `STOPPED_PENDING_SUPERVISOR_REVIEW`; U2 was not launched.

## LocateMOT-U U1R2 — measurement repair, class-aware refinement, and foundation completion — 2026-10-01

- Contract hypothesis: the remaining U1 failures were caused by a measurement/data contract mismatch rather than a need to replace the MM-GroundingDINO-B foundation. The source `labels_with_ids` fields are normalized top-left `xywh`; the repaired formal gates preserve source extents and use Top150 IoU.50 for generic proposals and Top100 IoU.50 target coverage for expressions. The previous U1-A-v1 files remain preserved as historical artifacts and were not overwritten.
- Coordinate/source audit: legal calibration/validation contains 33,526 physical GT boxes. 195 V1 boxes extend below the image at `y1>height` (166 large, 27 medium, 2 small); no x-side overflow exists. Same-frame geometry matching to official KITTI labels gives mean IoU `0.7414883`; Refer and official track IDs are separate namespaces. Decision: retain source extents and recover classes by geometry; do not clip evaluation GT.
- Data hypothesis: a class-aware physical-frame union can repair localization while removing V1/V2 duplicate training. The fit stream contains 2,545 images and 10,502 boxes; 3,424 V1/V2 overlaps were removed, 580 conflicting boxes and 21 missing class matches were recorded, and observed classes are car 6,787, pedestrian 3,700, cyclist 13, van 2. Decision: retain the stream with fixed class policy and explicit low-confidence audit.
- Numerical stability: the prior BF16 log has three finite-loss `grad_norm:nan` records (epoch/iter 1/50, 3/100, 6/100). A fresh FP32 one-step audit from epoch4 found 981 trainable parameters, total loss `40.27210998535156`, zero nonfinite gradients, and zero no-gradient parameters. Decision: R2 uses FP32/TF32, no AMP/BF16.
- Generic refinement: one bounded FP32 epoch from the frozen epoch4 checkpoint passed the corrected gate: V1/V2 Top150 IoU.50 `0.982716/0.971837`; small/medium/large `0.937467/0.989984/0.997825`. Decision: freeze the R2 epoch1 generic checkpoint and skip epoch2/mixed fallback.
- Expression hypothesis: whole-expression positive spans with all referred boxes, legal empty frames, and preserved V2 missing-target references can add language-conditioned candidate coverage without erasing generic proposals. One FP32 epoch completed from the selected generic checkpoint. Expression Top100 IoU.50 is V1/V2 `1.000000/0.961367`; the same checkpoint's generic regression is V1/V2 Top150 IoU.50 `0.986592/0.958739`.
- Language audit: on exactly 500 legal validation same-image pairs with a same-video sentence from a different target set, true versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is `0.736901/0.491159`. Both strict comparisons pass, so language conditioning is measurable rather than generic-box-only behavior.
- External evaluator: synthetic perfect/empty/ID-switch and cross-evaluator arrays pass with maximum difference `0.0`. TempRMOT evaluated videos 0012/0014 occur in its declared training pool; its HOTA 94.679 is reclassified as `PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`.
- Decision: the final shared expression checkpoint satisfies all U1 hard gates. `official_test_labels_read=false`, `screening_gt_used=false`, V2 source-target-ID-missing references remain explicit (157, silently dropped=false), and no U2/U3/U4 work started. Set `LOCATEMOT_U_READY_FOR_U2` and stop pending supervisor review.

## LocateMOT-U U2 — shared persistent track core and portability stop — 2026-10-02

- Provenance: working branch `codex/u2-shared-persistent-track-core-20261001`,
  authoritative local base `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and
  immutable U1 foundation SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
- Implementation: replaced the U0 tracking placeholders with one shared U2
  core (`PerceptionBridge`, independent identity/semantic candidate encoders,
  `UnifiedTrackState`, four-layer identity set interaction, GRU identity
  memory/anchor/EMA, association/new/no-match, lifecycle, motion, Hungarian
  rollout, and bounded OVMOT residual). The runtime is
  `locatemot/unified/runtime/u2_online_tracker.py`; no RMOT or prompt tracking
  was launched.
- Contract evidence: `tools/unified/u2_causal_contract.py` passed birth,
  occlusion, no-future, MOT specification invariance, OVMOT isolation, and
  online-state save/reload. Actual-model sharing audit passed with
  `shared_non_head_fraction=0.9378111093950003`.
- U2-P data manifest SHA-256 is
  `575bd349e00bab23634c6929c04513149afec838de419bb3f1a53469fafc7fda`.
  KITTI legal development portability passes (V1/V2 Top150 @.50
  `0.986592/0.958739`), while BDD100K, DanceTrack, MOT17, MOT20, TAO, and
  C-TAO are missing locally. Candidate-cache SHA is `not built`.
- Training command: not launched; …16846 tokens truncated…es from that exact prefix in `mot20_MOT20-05_resumed_v2` with an explicit 4 GiB per-study budget (PID 3743898), while a single-worker RMOT queue (PID 3744099) waits for that new `result.json` before touching the shared decoder/feature endpoints. No result or protected source directory was overwritten.

## 2026-10-18 21:48 CST — continuation runtime audit

J1 v4 birth remains live after the step-300 checkpoint: torchrun PID 3029659 with ranks 3029872/3029873 on GPU4 and GPU6. Both ranks are still consuming compute (rank0 at full CPU and both model GPUs at 100% utilization), so the absence of a newer log line is an active long batch rather than evidence for a restart. The authoritative restore point remains `adapter-step300` plus `latest_training_state.pt`; no checkpoint or optimizer state was overwritten.

MOT20-05 `resumed_v2` is also live and writing the protected output: `tracks.mp4`, `predictions.txt`, `frame_candidates.jsonl`, `frame_decisions.jsonl`, and `latest_identity_memory.json` advanced after the previous audit. Its final `result.json` is still absent, so the expression-conditioned RMOT queue correctly remains `WAITING_MOT20_RELEASE` with zero model-inference queries. The expanded CPU-only J1 held-out gate remains alive at PID 3791075 and waits for the J1 final receipt plus MOT20/RMOT release; no held-out metric is claimed yet. No process was restarted or stopped.
## 2026-10-18 21:51 CST — MOT20 export advanced while J1 batch remained active

The protected MOT20-05 `resumed_v2` owner made real progress after the previous audit: `frame_decisions.jsonl` reached 12,201,850,168 bytes, `latest_identity_memory.json` reached 300,275,019 bytes, and `predictions.txt` reached 11,269,005 bytes at the latest snapshot. The final `result.json` is still absent, so the export is not yet released to RMOT. J1 still has only the step-300 receipt, but both distributed ranks remain live and one rank is actively computing; the current evidence does not justify a restart. The held-out gate and expression-conditioned RMOT queue remain correctly waiting on their registered receipts.
## 2026-10-18 — selective cache policy reaffirmed

Newly downloaded datasets do not require whole-dataset native feature/candidate caches. The formal J1 SFT already consumes the registered 43,533-row three-task course with media on demand and `feature_cache=false`. Cache/export is limited to registered evidence scopes: the protected MOT20-05 G0 long export, the legal expression-conditioned RMOT pilot manifests after the MOT20 release, and the explicitly registered held-out MOT/RMOT/OVMOT evaluation manifests (DT1203, Refer-KITTI validation scope, and the two TAO validation videos). Other downloaded train/validation data remains distributed across the data volumes and is read on demand; no full TAO or full Refer-KITTI cache is started. The older pedestrian-only RMOT cache remains a rejected semantic attempt and is not promoted or reused as expression-conditioned evidence.
## 2026-10-18 — G1 continuation decision and record boundary

G1 is a completed detector-side adaptation, not an active optimizer: the registered legal TRAIN mix spans MOT17, MOT20, DanceTrack, and Refer-KITTI, with 200/200 projector steps, a resumable FP32/Adam/RNG state, and a materialized `GroundAnything-DLM-TrackProjG1` serving directory. The short unchanged G0/G1 controls are valid but mixed/mostly negative (MOT4 and RMOT single8 regress; RMOT multi12 is slightly higher), so G1-200 remains a preserved baseline and is not silently extended or promoted. The larger downloaded corpora are available, but a G1-extended run would be a new experiment: it must use legal TRAIN data only, keep G0/G200/G1-200 checkpoints, and rebuild any downstream OneJev rows before claiming an effect. Current long-running priority remains the protected MOT20 G0 receipt and the real J1 three-task checkpoint/held-out gate; GroundAnything server processes are inference services, not G1 fine-tuning.

## 2026-10-18 22:35 CST — MOT20 G0 receipt released; expression RMOT pilot started

The protected MOT20-05 resumed export produced and passed its terminal receipt: `VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, all 1,658 frames processed, 1,560-frame exact resume prefix restored, 224 decoder-rejected/strict-invalid frames retained as empty official prediction frames, 2,314,870,593-byte native candidate archive, and 4,322.94 seconds end-to-end wall time. The owner process remained alive briefly after writing the receipt while flushing its final result output, then exited without modifying the completed artifacts. The expression-conditioned RMOT queue was released only after this receipt and is now running its legal 24-query pilot; the latest owner state reports 2 completed queries, model inference started, and no failure. The J1 held-out gate remains waiting for the final step-1200 training receipt.

## 2026-10-18 22:59 CST — expression RMOT pilot progress

The released expression-conditioned RMOT pilot remains healthy: owner status RUNNING_EXPRESSION_RMOT_PILOT, 5 of 24 legal query manifests completed, model inference active, and failed=null. The queue is still single-worker over the shared GroundAnything services; no held-out or validation videos have been admitted.

## 2026-10-18 23:38 CST — expression RMOT pilot passed midpoint

The expression-conditioned RMOT pilot remains healthy at 13 of 24 legal query manifests complete. The owner is still RUNNING_EXPRESSION_RMOT_PILOT with failed=null; no held-out or validation video has entered the queue, and no optimizer update is performed by this export.
## 2026-10-18 — GroundAnything G2-A native optimization gate passed

The isolated GroundAnything G2 line was initialized from `models/GroundAnything-DLM` after auditing the pinned official source at commit `8a3b43a1b42952e117960e85d63ff173bb9fba0a`. The task text referenced `e054ab329...`, but that revision is not the checkout's actual official HEAD; the actual pinned revision and source-file hashes are recorded in the G2 audit receipts. GPU3 and GPU5 were free at the one-time resource audit; J1 on GPU4/6, protected GroundAnything services on GPU7, and the active MOT20/RMOT owners were left untouched.

G2-A passed a real native DLM forward/backward/Adam-step gate on three supervision modes: dense MOT20, referring RMOT, and TAO-Amodal OVMOT. Vision and the original language weights remained frozen; the projector exposed 27,267,584 trainable parameters and Qwen3 q/k/v/o LoRA exposed 11,796,480 trainable parameters across 36 layers. The official mixed objective remains 0.5 MDM + 0.5 causal, with projector/LoRA learning rates 2e-6/7e-6 and AdamW. Losses were 1.24458, 1.39893, and 1.53041, with finite nonzero gradients in both projector and LoRA and a completed optimizer update. Peak allocation was 39,124,467,712 bytes on one 80-GB GPU. A narrow Transformers 5.17 compatibility shim for the removed `cache_position` keyword is recorded and covered by the gate. This is a preflight receipt only; formal ten-dataset G2 training has not started while dataset coverage and legal TRAIN-only row preparation are audited.



## 2026-10-19 — GroundAnything G2 coverage audit, corpus, and DDP smoke

The resolved TRAIN-only audit follows the distributed data layout and avoids held-out media. TAO-Amodal has all 18,274 TRAIN images available across /data0, /data1, /data2, and /data3. RMOT26 has all images referenced by its primary train annotation rows, but its download receipt still records 20 failed assets, so the source remains PARTIAL_DOWNLOADS. BURST has 500 TRAIN annotation sequences but no legal TRAIN image root in this checkout; it is excluded rather than substituting TAO test frames.

The G2 row builder emitted 199,053 validated official rows with absolute image paths and bytes=null: MOT17 15,948, MOT20 8,931, DanceTrack 41,796, SportsMOT 28,574, AnimalTrack 11,543, Refer-KITTI-V2 4,333, AerialMind 45,625 dense plus 23,733 referring, RMOT26 296 referring, and TAO-Amodal 18,274. The 2,246 rejected rows are retained in a separate JSONL with source and reason; no row path is missing, no official evaluation video appears, and all rows pass the official uniform row contract. Corpus SHA is b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e.

A first two-rank DDP launch under system Python failed before model loading because that interpreter had no accelerate package. The failure is preserved as an environment issue. Re-running with the same project .venv-onejev-py310 used by G2-A passed one native optimizer step on physical GPU3/GPU5, including finite gradients, distributed synchronization, and atomic adapter-step000001/latest_training_state checkpoints. The cards were co-resident with pre-existing work and were not modified. Formal full-data training is now gated only by the registered long-run launch and later held-out evaluation; no G2 quality score is claimed yet.


## 2026-10-19 — G2 formal full-data run launched

After the G2-A native gate, ten-source coverage audit, row-contract/path audit, and one-step two-rank smoke all passed, formal G2 training launched with two DDP ranks on physical GPU3/GPU5 using `.venv-onejev-py310`, master port 29619, max_steps=1200, and checkpoint interval 50. The input manifest SHA remains b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e. The run directory is `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/formal_g2_1200`; no held-out media or existing J1/MOT20/RMOT owner is touched. Formal completion and detector/tracking evaluation remain pending.


## 2026-10-19 — G2 formal length-gate failure and minimal recovery

The first formal two-rank run reached a valid step-1 checkpoint, then stopped at row `mot20_MOT20-05_001167_dense`: the official DLM collator measured 2,193 text tokens against the temporary 2,048 preflight bound. No data, metric, seed, or split changed; the failure receipt is retained under the formal run directory. The trainer default was corrected to the official 4,096 text / 8,192 joint limits. A targeted collator check passed the failing row at 2,193 tokens, and a two-rank resume regression from the valid step-1 checkpoint completed step 2 with finite loss 1.46757, finite gradient norm 6.09755, and an atomic step-2 checkpoint. Formal 1,200-step recovery resumes from the latest valid step-1 checkpoint with the corrected length contract.

## 2026-10-19 — G2 official memory route repair and long-row regression

The corrected 4,096/8,192 length run reached the first long MOT20 row but exhausted the remaining memory because the custom trainer still used the legacy padded language loop while GPU3/GPU5 each carried an unrelated 18.5-GiB resident workload. No metric or data path changed and the valid step-1 checkpoint remains the recovery point. The trainer now follows the released DLM general memory contract: padding-free packing, non-reentrant language gradient checkpointing at every packed layer, chunk linear kernel, fused LM-head when available, and no activation CPU offload.

This environment does not provide the optional `flash_attn` or `fla` modules. The trainer therefore selects GroundAnything's official `cudnn_sdpa` packed clean-attention backend and its official `checkpointed_chunk` LM-head backend, respectively; the objective remains the same 0.5 MDM + 0.5 causal loss. A two-rank regression with one 2,193-token MOT20 row per rank completed one optimizer step on physical GPU3/GPU5 with finite loss 2.0606513, finite gradient norm 5.1535301, and atomic checkpoint/final receipts. A prior one-row DDP fixture was interrupted after rank imbalance was identified; it did not alter the trainer or corpus. Formal recovery from step 1 is now allowed to proceed; quality results remain pending.

## 2026-10-19 — G2 formal recovery relaunched

The repaired formal G2 run was relaunched from the preserved step-1 checkpoint with the unchanged 199,053-row corpus (SHA b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e), max_steps=1200, save interval 50, seed 20261019, and two DDP ranks on physical GPU3/GPU5 (master port 29633). The active command uses the packed/checkpointed route and the recorded cudnn_sdpa/checkpointed_chunk fallbacks. It is still running; no G2 quality metric or completion claim is made until the terminal receipt is written.

## 2026-10-19 — TrackJEV redirect state audit

The protected MOT20 G0 long export is terminal and complete (`VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, all 1,658 frames, exact resume prefix retained). The expression-conditioned RMOT pilot is terminal and complete for all 24 legal query manifests with `failed=null`; its proposals are recorded as native G0 expression-conditioned evidence. The actual MOT-only queue is terminal `STOPPED_BY_USER_RESEARCH_PIVOT` with `formal_track_training_launched=false`, so no abandoned MOT-only optimizer remains to be resumed.

The real OneJev three-task course is active at step 550/1200 from the registered 43,533-row manifest: identity 21,168, RMOT 16,677, and OVMOT 5,688 rows, with multimodal DDP on GPU4/GPU6, FP32/Adam/RNG checkpointing, and no feature cache. The J1 held-out gate remains `WAITING_J1_FINAL` and owns no GPU until step 1200. GroundAnything G1 is preserved separately: its apply receipt records a 200-step projector delta with five tensors replaced and byte-identical frozen vision/language weights; the GPU5 retry2 G0/G1 controls completed, while the measured G1 detector comparison is retained as a negative/mixed control rather than promoted. No held-out result or final TrackJEV claim is made before J1 reaches its terminal receipt.

## 2026-10-19 — authoritative redirect continuation audit

The protected MOT20-05 G0 long export is now terminal and complete: all 1,658
legal TRAIN frames are present, the exact 1,560-frame resume prefix is retained,
224 rejected/strict-invalid perception frames remain in the output, and runtime
GT/future-frame access is false. The evaluation-only TRAIN-scope receipt reports
HOTA 0.433441, DetA 0.638348, AssA 0.294655, and IDF1 0.441085; this is not an
independent held-out result. The expression-conditioned RMOT pilot also completed
all 24 legal query manifests with native G0 expression-conditioned proposals.

The registered MOT-only queue has no live process and remains
STOPPED_BY_USER_RESEARCH_PIVOT; no abandoned optimizer is resumed. The real J1
three-task OneJev run is live on GPU4/GPU6 at durable step700, using 43,533 rows
(identity 21,168, RMOT 16,677, OVMOT 5,688) with FP32/Adam/scheduler/RNG state.
Its final receipt and independent held-out metrics are still pending.

The clean G2 run remains active on GPU3/GPU5 at the latest observed step131/18227.
The registered fresh GPU5 G1 queue is alive and correctly waiting for the clean
G2 terminal receipt; it must not contend for GPU5 or duplicate the preserved
negative/mixed G1-200 control. The G2 bundle exporter and J1 held-out gate remain
dependency-gated. No unified quality improvement is claimed. The full evidence
map is in `docs/GROUNDJEV/EXECUTION_REDIRECT_AUDIT_20261019.md`.

## 2026-10-19 — live dependency progress revalidated

The registered clean G2 owner is still alive on GPU3/GPU5 and its event log has
advanced to optimizer step141/18227 with finite loss; no terminal receipt or
durable step-500 checkpoint exists, so it was not restarted or altered. The J1
three-task torchrun is also alive on GPU4/GPU6; its log has reached 720/1200,
while `adapter-step700` remains the latest durable FP32/Adam/RNG restore point.
The GPU5 G1 queue and clean-bundle exporter remain live dependency owners, not
failed jobs. No MOT-only process appeared, and no held-out evaluation was
started before the required terminal receipts.

## 2026-10-19 — live dependency progress revalidated (step 172 / 750)

One authoritative snapshot confirms the clean G2 torchrun is still alive on
GPU3/GPU5 at event step172/18227; its terminal receipt and durable step-500
checkpoint are absent. The active process still reports pre-v2
`samples_global=16`; the read-only global-accounting audit remains the only
corrected interpretation and no process was reloaded. J1 remains alive on
GPU4/GPU6 with durable `groundjev-full-fp32-adam-rng-v1` step750/1200 and
complete FP32/Adam/scheduler/RNG state; its final receipt is absent. The J1
held-out owner remains `WAITING_J1_FINAL`, the fresh GPU5 G1 owner remains
`WAITING_G2_CLEAN_SUCCESS`, and the clean bundle exporter remains
`WAITING_G2_AND_G1_TERMINAL`. No MOT-only process was restarted and no
optimizer, sampler, checkpoint, service, or owner state was changed. This is
execution evidence only; no quality claim is made.

2026-10-19 — Read-only G2 accounting audit generated at the active formal run directory without opening the training JSONL in write mode. The audit contains 175 valid pre-v2 events, zero malformed lines, preserves raw-log SHA `752c4ebb69de0ecf6f2fd0f77d068409934a800a333d095c1f676c61d4480df3`, and reports `raw_log_unchanged=true`. Its latest event is step175: logged `samples_global=16` and `effective_tokens_global=17079`, corrected for world size 2 to 32 samples and 34158 tokens under the registered global-batch-32 interpretation. This is an accounting correction only; no trainer process was restarted and no metric claim is made.


2026-10-19 — live dependency progress revalidated (J1 step800 / G2 step298). The real OneJev three-task torchrun remains alive on GPU4/GPU6 with the durable groundjev-full-fp32-adam-rng-v1 step800 state (260255305 bytes); no step-1200 terminal receipt or independent held-out result exists. The clean G2 torchrun remains alive on GPU3/GPU5 at event step298/18227 with finite loss 1.787952 and 17.3765 GiB latest peak memory; its pre-v2 raw event still logs samples_global=16, while the read-only accounting audit remains authoritative for the corrected global batch 32 interpretation. G1 GPU5, clean bundle export, and held-out evaluation owners remain dependency-gated. No training process, optimizer, sampler, checkpoint, or service was restarted or modified.


2026-10-19 — clean G2 crossed the registered step-300 runtime milestone. The active GPU3/GPU5 torchrun is still live at event step301/18227 with finite loss 1.813610, 16842 logged effective tokens, and 17.3718 GiB latest peak memory; no terminal receipt or durable step-500 checkpoint exists. J1 remains live on GPU4/GPU6 at durable step800/1200 with no final receipt. The existing G1 GPU5 queue, clean bundle exporter, and J1 held-out gate remain dependency owners; no process or checkpoint was restarted or altered.


2026-10-19 — read-only J1 stall diagnostic. Durable state remains step800/1200 and the final receipt is absent. The torchrun parent and both rank workers are alive; at the snapshot the parent was in do_select and both workers were sleeping in futex_wait_queue_me, with no failure/exception receipt. This is compatible with a distributed synchronization or long batch and is not evidence of a failed run, so no process, sampler, optimizer, or checkpoint was restarted.


2026-10-19 — J1 timing interpretation corrected from the read-only log: steps720–800 report 29.21–43.34 seconds per optimizer step, with finite losses/gradients and normal examples/tokens throughput. The absence of a newer durable state is therefore consistent with the registered long runtime (roughly hours for the remaining 400 steps), not a proven deadlock. The active process remains untouched.


2026-10-19 — full runtime audit after the blocking wait shows real progress. J1 log advanced through steps880/900/920 with finite loss and gradient values; the latest durable FP32/Adam/scheduler/RNG state is step900/1200 (step920 is logged but not yet a save boundary), with 68.917 GiB peak memory and no terminal receipt. Clean G2 advanced through events588/589/590 on GPU3/GPU5 with finite losses and about 17.377 GiB peak memory; no terminal receipt or step-500-only interpretation is used because the live event stream has passed it. G1 GPU5 and clean-bundle owners remain correctly waiting for G2/G1 terminal receipts, and all known processes remain alive. No active task was restarted or modified.


2026-10-19 — preserved read-only submission audit at docs/GROUNDJEV/LIVE_RUNTIME_AUDIT_20261019.json. It records G2 RUNNING at event630, J1 RUNNING with durable step900, G1 and G2-export owners waiting on registered dependencies, and the held-out gate in WAITING_J1_FINAL. The audit reports no non-finite/skip/truncation anomalies and does not mutate live runtime state.


2026-10-19 — runtime milestone update: J1 now has a durable step950/1200 FP32/Adam/scheduler/RNG state (log step940, finite loss and 43.86 s/step at that record); clean G2 has advanced to event step651 with finite loss and global accounting audit still applicable. No final receipt exists for either run. The held-out gate remains WAITING_J1_FINAL, G1 remains WAITING_G2_CLEAN_SUCCESS on GPU5, and the clean bundle exporter remains WAITING_G2_AND_G1_TERMINAL; no active task was changed.


2026-10-19 — J1 crossed the next durable boundary: latest log and FP32/Adam/scheduler/RNG state are both at step1000/1200 with finite training metrics; no final receipt exists yet. Clean G2 has advanced to event step786 with finite loss and the registered global-accounting audit unchanged. The held-out gate remains waiting on the registered J1 PID and final receipt; G1 and bundle owners remain dependency-gated. No active process or checkpoint was restarted or altered.


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

2026-10-21 — FlexHook official data interface preparation. The pinned
FlexHook `kitti-2` loader was audited separately from TempRMOT: it consumes
`labels.json` with normalized top-left boxes and frame-level `expression_raw`,
and evaluates the 16-video train / 4-video test split recorded in its own
`data/utils.py`. A converter now symlinks all 21 source image videos, joins
the official expression JSON with the raw `labels_with_ids`, and writes 862
test GT templates in external `/data3/user/TTA-RMOT-assets` storage. The
conversion found 181 expression/frame/object references whose raw per-frame
box is absent; these are counted and skipped exactly as the pinned TempRMOT
GT writer does, never filled with synthetic boxes. Audit receipt:
`results/audits/flexhook_refer_kitti_v2_audit.json`. FlexHook model training
and inference remain pending; this data preparation does not count as a
reproduction result.

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

2026-10-22 — FlexHook official asset and runtime-isolation preflight. The
public FlexHook Refer-KITTI-V2 checkpoint, Swin-RoPE-Tiny visual weight,
RoBERTa base, and 20-file Temp-NeuralSORT archive were downloaded and hashed.
The archive's eight `predict.txt` files were exposed in a separate runtime
view as symlinks; all `gt.txt` files were excluded. A runtime data view keeps
the complete image and sentence inventory but strips expression `label`
fields and replaces `labels.json` with an empty object. The official
checkpoint loaded with all state-dict keys matched, and a one-batch 0013
single-card forward passed on physical GPU3 with 1.91 GB peak allocation.
The official evaluator GT remains in a separate prepared root and is attached
only after prediction sealing. This is a smoke/protocol result, not a full
FlexHook benchmark; the full inference and TrackEval run remains pending.
Receipt: `results/audits/flexhook_official_assets_receipt.json`; runtime
builders: `scripts/prepare_flexhook_runtime.py` and
`scripts/run_flexhook_official.py`.

2026-10-22 — Resource-cap correction. A live process audit showed that the
TTA-RMOT source-training line (GPU1), released-checkpoint official inference
(GPU7), and four auxiliary ShiftRMOT-v2 shards (GPUs 3/4/5/6) would exceed
the user-imposed maximum of four physical GPUs. The four auxiliary shards
were interrupted through their existing handles; their partial traces remain
for failure/progress provenance but are not merged or scored. The source
training and official inference lines were left running, and GPU3 is reserved
for the next FlexHook single-card run. Receipt:
`results/tta/shift_rmot_v2_release_full_sharded_launch.json`.
