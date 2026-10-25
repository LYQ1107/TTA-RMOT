# Positive-Free failure analysis — interim, first two complete controls

Project: TTA-RMOT only. Full 25 videos / 425 expressions; development-target results, not blind testing.
PF_RANK, PF_IDENTITY and PF_HYBRID are still running/pending. Their final conclusion is not inferred from these controls.

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

## Remaining decision

Complete the registered PF_RANK, PF_RANK_GUARD, PF_IDENTITY and PF_HYBRID studies without tuning on GT.
Distinguish genuine GT recovery and ranking improvement from pruning/retention effects or extra outputs.
An AssA increase accompanied by DetRe loss cannot be called target-recall recovery. No overall research-success
or final negative-TTA verdict is issued before the remaining full methods finish.
