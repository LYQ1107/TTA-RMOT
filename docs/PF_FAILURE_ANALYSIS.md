# Positive-Free failure analysis — interim, four complete methods

Project: TTA-RMOT only. Full 25 videos / 425 expressions; development-target results, not blind testing.
PF_IDENTITY and PF_HYBRID are still running/pending. No all-method final conclusion is inferred before they finish.

## Matched baseline and implementation checks

The original Frozen file omitted 692 final tracker observations. The user approved rebuilding a complete matched baseline.
Actual full ZERO_UPDATE yields HOTA **8.2180**, DetA **3.1937**, AssA **21.1470**, IDF1 **4.4414**,
DetRe **4.5441**, DetPr **9.6868**, IDSW **269**, **99,704** predicted rows and **1,769** IDs.
It covers 3,588,445 candidate-query observations and 433,636 chronological active-frame steps.

Six historical near-zero decisions differ from the old logit file. Each complete candidate set was rerun with
both uncached original and cached FP32 source inference: zero original/cached sign or ID discrepancies.
The original historical scores and native ZERO streams remain unchanged. The complete native margins
define the common frozen reference, whose independent CPU score-to-MOT replay exactly matches all 425 predictions.
That replay is not claimed as a second independent GPU model run. Historical HOTA 8.2060 is not the gain reference.

## Why the baseline is weak

Offline analysis begins only after all predictions are sealed. At IoU>=0.5, fixed MOT candidates can retrieve
203,906 of 212,541 scored GT observations (**95.9373%**), while the Frozen positive gate selects only
10,337 matches (**5.0695%** of retrievable GT) and 89,367 FP rows.
Thus a lack of candidate boxes is not the main detection-recall bottleneck in this setting; referring selection is weak.
This any-IoU candidate recall is not an oracle HOTA or a guarantee of correct track identities.

Source AUROC is **0.650204**. Conditional on frames containing a retrievable target, source Top1 hit rate
is **61.4788%**, compared with **57.8299%** for random selection; Top3 is **77.1925%** vs **76.8488%** random.
The random control matters because some referring queries match many people in the same frame.
These figures do not support describing the source ranking as reliably strong cross-domain semantics.

## Relative selection recovers targets but floods false positives

RELATIVE_ONLY: HOTA **7.5705** (delta **-0.6475**), DetA **4.1857**, AssA **13.7120**,
IDF1 **5.5746**, DetRe **11.1040**, DetPr **6.2855**, IDSW **829**; 375,483 predicted rows / 8,014 IDs.

It recovers **14,873** additional GT observations with no lost baseline match, but total FP increases by
**260,906** to 350,273. Conditional gate recall rises to **12.3635%**, while precision and association decline.
Its AUROC/TopK scores are exactly unchanged because it does not train or change the source scores.
This is a candidate-expansion control, not new semantic learning, and it does not improve overall HOTA.

Newly added selected-row FP is 260,897, not the total FP change: matching reassignment can change the FP
status of existing rows too. Both quantities are preserved in the attribution JSON; they must not be conflated.
The diagnostic GT loader follows official nonzero-mark and pedestrian-class-1 filtering.

## Query absence dominates many false positives

Under the fixed 425-episode protocol, 385 episodes have no scored GT and 40 have scored GT.
This is the annotation/evaluation distribution, not a reason to discard the GT-empty cases.
The diagnostic GT observation count (212,541) agrees with the official evaluator's GT_Dets.
The independent framewise assignment is not identical to CLEAR's temporal assignment, so its TP/FP
counts need not equal the official CLEAR TP/FP counts exactly; official metrics remain primary.

Frozen produces 87,371 diagnostic FP rows on GT-empty episodes and 1,996 on GT-present episodes.
RELATIVE_ONLY raises these to 337,014 and 13,259 respectively. Most of its false-positive expansion
therefore comes from failure to reject nonmatching video-query combinations, not merely ranking
the wrong person inside a target-present frame. Track-ID consistency alone does not supply evidence
that the language query has any matching target. This motivates checking empty-query FP separately
for all remaining methods, but no online loss, threshold or hyperparameter is changed using this information.

## Rank has only a tiny gain, without overall ranking improvement

PF_RANK: HOTA **8.2433** (delta **+0.0253 points**, about 0.31% relative), DetA **3.2336**,
AssA **21.0150**, DetRe **4.6120**, DetPr **9.7458**, IDF1 **4.4947**, IDSW **269**.
It selects 100,582 rows, recovers 158 GT observations and loses 3; diagnostic net recovery is +155.
Total diagnostic FP increases from 89,367 to 90,090 (+723). GT-empty-episode FP rises to 88,057.
AUROC declines **0.650204 → 0.648328**. Conditional Top1 changes **0.614788 → 0.615468**;
Top3 slightly declines and Top5 slightly increases. These results do not establish improved
semantic ranking, and a single-seed 0.0253-point HOTA change is not evidence of a robust gain.

This is genuine executed adaptation: 105,971 finite, nonzero-gradient steps, no nonfinite steps
or rollback. The maximum adapter delta L2 is 1.968823 (fixed limit 2). The actual per-replica
adapter has 24,576 trainable parameters and 768-wide PCD features. An initial parameter-count
summary of zero came from subtracting constant fields as episode deltas; only that reporting
field is corrected using shard statistics, with raw metadata/predictions retained unchanged.

PF_RANK_GUARD: HOTA **8.2444** (+0.0264 versus Frozen, only +0.0011 versus Rank), IDF1 **4.4956**.
It restores the 31 Frozen-positive rows that Rank removed; 100,613 rows, 158 recovered GT,
zero lost GT, diagnostic FP 90,118 (+751 versus Frozen), including 88,085 on GT-empty episodes.
Its underlying scores and AUROC are the same as Rank; this is explicit retention, not new learning.
A complete real GPU episode and CPU derivation have byte-identical predictions and matching
282-step update traces. The full Guard result reuses sealed Rank margins; its 7.34-second CPU
derivation excludes the required Rank GPU run and 112.43-second equivalence validation.

## Remaining decision

Complete the registered PF_IDENTITY and PF_HYBRID studies without tuning on GT.
Distinguish genuine GT recovery and ranking improvement from pruning/retention effects or extra outputs.
An AssA increase accompanied by DetRe loss cannot be called target-recall recovery. No overall research-success
or final negative-TTA verdict is issued before the remaining full methods finish.
