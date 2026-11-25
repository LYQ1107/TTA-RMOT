# Gated cross-query decoding — fixed protocol

Project LYQ1107/TTA-RMOT; base ecf57122fd96fc51c56581756251737ec48079ce;
branch research/v2dance-gated-decode-20261101. No student/teacher training,
no gradient optimization, no target-GT tuning. The explicitly requested GMM
fitting is unlabeled per-video statistical estimation, not a model update.

## Inputs and pre-evaluation gates

Verify sealed six target score arrays FROZEN, T2V_D1_s17/s23/s42,
T2V_D3_s17, SAME_SET_D1. SAME_SET is a same-Val diagnostic reference.
Verify existing source metadata and FROZEN/T2V_D1_s17 arrays, with
2,355,528 source candidate-query observations/862 expressions/4 videos.
No new FlexHook inference. Reproduce every one of the previous X0 and X1
prediction files byte-for-byte for all six margins (5,100 files); stop on
any discrepancy. Do not change old outputs or vendor/backend/builder code.
Seal all gates and predictions before a separate GT evaluator starts.

## Exact decoder inventory

Six margins × X0_C3, X1_C3_CAUSAL (12 reproduced controls).
Six margins × G0_GZ_X0, G0_GZ_X1, G1_GMM_REJECT_X0, TG_X0,
TG_X0_REASSIGN, TG_GZ_X0, TG_EARLY_X1 (42 new sets).
SA_DECODE for FROZEN and three D1 seeds (4 target sets).
TG_ONLY_FROZEN_X0 and TG_ONLY_NATIVE (2 Frozen-only references; the former
is exactly the FROZEN TG_X0 alias, not another teacher run).
Three unchanged native references: FROZEN, B1_SRC_RATE_GLOBAL,
OFFICIAL_DANCE_COMPLETE. Thus 60 decoder sets + 3 references, all 425
episodes/25 videos/211,085 physical MOT observations. Source: 2 native
references and 2 SA_DECODE sets, all 862 episodes.

X0 = argmax query of full-track raw margin mean, all track observations.
X1 = argmax query of current-frame-inclusive causal-track raw mean,
output only current observation. Queries sorted by filename; ties first.
G0: per query, pool ALL observations over ALL 25 videos (observation
weighted, not equally weighted videos); population mean/std (ddof=0),
zero std -> z=0. G0_GZ_X0 uses full-track z mean; G0_GZ_X1 uses causal
z mean but GLOBAL OFFLINE statistics, so it is NOT strictly causal.
G1: X0 raw winning-track means; sklearn GaussianMixture 1 vs 2,
random_state=0,n_init=5,covariance_type=full (other sklearn defaults).
If BIC2<BIC1, drop tracks assigned to the lower-mean component; otherwise
retain all; <4 tracks retain all. No teacher, no GT, no retuning.

## Frozen teacher and sampling

Qwen/Qwen3-VL-2B-Instruct revision
89644892e4d85e24eaac8bacfd4f463576704203, BF16, no quantization/adapter,
no updates. Unmodified yestrack_backend and VisualInputBuilder.patch,
320 square, margin .2, original official_single prompt and original BF16
Yes/No softmax p_yes (raw logits also retained). Source image access uses
a path-only frame adapter; cropping and forward code stay unchanged.

Normal anchors [.05,.17,.29,.41,.53,.65,.77,.89]; early anchors
[.025,.05,.075,.10,.125,.15,.175,.20] of the full video span.
Continuous desired frame = first + fraction*(last-first), round half up,
then select nearest frame with scored MOT candidates (tie lower frame).
Use complete legal image filename frame span, not GT. Source image frame
= original MOT frame - 1, target frame = original MOT frame. Use the
same existing historical source scored-observation universe, no new
source last-frame coverage. Duplicate anchor frames reuse identical
requests and count once for distinct-frame voting. All candidates and
all legal runtime query texts, no GT-based filtering.
Source only: if normal planned calls for one video >50,000, each frame
retains first 40 numeric track IDs, no other downsampling. Report whether
the specified 40 cap itself still exceeds 50,000.

Normal Dance gates and early Dance gates independently: each candidate
votes for lex-first argmax query if max p_yes>=.5; a query is present iff
>=2 votes from >=2 distinct frames. No mandatory positive. Cache key
includes revision, frozen weight, prompt/image/visual/bbox hashes,
dataset/video/query/frame/track. Shared teacher calls are counted once,
with requested vs unique/cache-hit and per-arm attribution reported.

TG_X0 drops tracks whose raw X0 winner is gate-absent.
TG_X0_REASSIGN changes absent winners to highest raw mean among
gate-present queries, or empty if none. TG_GZ_X0 gates G0_GZ_X0.
TG_EARLY_X1 applies the complete early gate to X1 for ALL observations:
20% video startup delay, NOT strict causal. TG_ONLY_FROZEN_X0 = Frozen
TG_X0; TG_ONLY_NATIVE = gate-filtered Frozen margin>0.

E(v) = candidates with >=2 query probabilities>=.5 divided by candidates
with >=1, using normal teacher frames. SA: E<=.5 -> TG_X0; E>.5 -> native
margin>0 WITHOUT any gate. Human clarification in this task: denominator
zero -> E undefined, output EMPTY and report separately (not E=0).
Apply source SA to Frozen and D1_s17, target SA to Frozen and 3 D1 seeds.

## Evaluation and fixed decisions

Target full official TrackEval HOTA/DetA/AssA/IDF1/IDSW, rows, Empty rows,
Present/Seen/Unseen HOTA. Seen uses exact normalized text membership in
the existing Train text-only manifest, unchanged previous definition.
Report lookahead by statistics, track decoding and gate separately;
teacher calls and GT open attempts (must be zero during generation).
Offline gates: accuracy/recall/FPR, ALL/seen/unseen; E and branch per video.
Source SA vs native HOTA difference, no bootstrap and NO GT thresholds.

2000 paired 25-video cluster bootstrap draws, seed 0, unchanged official
TP-weighted pooling. Each new decoder minus same-margin X0 and minus
Frozen native/B1/OFFICIAL_COMPLETE. Also D1 three seeds X0 minus Frozen
X0, and TG_X0 minus TG_ONLY_FROZEN_X0 / TG_ONLY_NATIVE. Report all,
no variant selection, no threshold modification.

1. Gate: all three D1 TG_X0-X0 CI lower>0 ->
   视频级教师门控带来稳定增益; otherwise 门控增益不稳定.
2. Unseen: D1 TG_X0 mean unseen HOTA>14.229 ->
   门控恢复了没见过的句子. Independently G0_GZ_X0 same strict threshold.
   Otherwise report not meeting the registered condition, no new rule.
3. GMM: all three D1 G1-X0 CI lower>0 -> GT-free 拒绝有效;
   otherwise report not meeting the registered condition.
4. SA: both source models HOTA>=native-1 AND target D1 three-seed mean
   HOTA>=26 -> 结构自适应解码在两个数据集上同时成立;
   otherwise report not meeting the registered condition.
5. Student contribution: three D1 X0-Frozen X0 differences/CIs, no verdict.

## Costs, isolation, stop

At most one physical shared-memory-fit GPU, audit_gpus admission, prefer
fewest processes, no foreign kills. Real smoke before complete teacher
calls. One blocking parent per long job, with streamed progress; resume
only unfinished/cache-missing work. Python annotation-open audit plus
teacher kernel read whitelist; no Dance/source GT in generation. All
teacher outputs and masks sealed before GT process. Record per-video
actual calls/forward seconds/peak memory, smoke/cache cost, GPU-process
wall hours, and exhaustive 3,588,445-pair forward-hour extrapolation.

Delivery REPORT.md, decoders.csv, gates.csv, exclusivity.csv,
teacher_costs.json, bootstrap.json, full metrics/logs and path/SHA index.
Large raw outputs remain server-side. Update research_log.md and push
independent stages. Finish and STOP; do not train or start new methods.
