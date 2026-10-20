# TTA method review for ShiftRMOT

The source model is TempRMOT with its backbone, tracker, text encoder, and
source referring head frozen. The first runnable release uses the existing
causal adapter contracts in this repository and reports explicit method
provenance instead of presenting architecture-specific ports as exact
reproductions.

| protocol label | implementation in this repository | status |
| --- | --- | --- |
| `FROZEN` | source referring head, no target update | exact baseline |
| `TDA_RMOT` | bounded high-confidence temporal EMA (`EMA_ST`) | RMOT adaptation inspired by TDA; not an exact paper port |
| `SAR_VISUAL_INSPIRED` | source-free entropy-minimizing external referring residual (`ENTROPY_HEAD`) | SAR-inspired; SAR's optimizer/reliability details are not transplanted into TempRMOT |
| `TPT_LANGUAGE_INSPIRED` | causal query-consistency probe and residual adapter (`QCS_TTA`) on A2/A3 queries | TPT-inspired protocol label; no CLIP prompt parameter is silently claimed |
| `TCR_FUSION_INSPIRED` | bounded query-consistency fusion adapter (`QCS_TTA`) | TCR-inspired; no cross-modal retrieval code is copied into TempRMOT |

The public references used for the review are [TPT](https://github.com/azshue/tpt),
[SAR](https://github.com/mr-eggplant/SAR/blob/main/main.py), and
[TCR](https://github.com/XLearning-SCU/2025-ICLR-TCR). Their objectives and
backbones differ from this RMOT runner; the mapping above is therefore a
reproducible hypothesis, not a claim of exact reproduction. The experiment
record must report optimizer steps, probe calls, selected rows, HOTA, DetA,
AssA, IDF1, FPS, and peak VRAM for each paired condition.
