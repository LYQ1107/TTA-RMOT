# Refer-BDD TTA diagnostics

All diagnostic comparisons below are offline analyses of sealed predictions
and GT. They were not available to the online runner.

## Metrics and updates

| method | HOTA | IDF1 | delta HOTA | delta IDF1 | optimizer steps | shared probe calls | selected rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| FROZEN | 2.9788 | 0.89107 | 0.0000 | 0.0000 | 0 | 0 | 9,183 |
| ENTROPY_HEAD | 1.5842 | 0.28707 | -1.3946 | -0.6040 | 31,656 | 0 | 1,455 |
| PROBE_ST | 2.9788 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 | 9,183 |
| PROBE_CACHE | 2.9788 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 | 9,183 |
| QCS_TTA | 2.9788 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 | 9,183 |
| NO_STATE | 2.9788 | 0.89107 | 0.0000 | 0.0000 | 0 | 17,616 | 9,183 |

`ENTROPY_HEAD` performed real online updates on 31,656 frame steps, with
nonzero parameter deltas on all of them; its p50/p95 gradient norms were
0.7096/1.5333 and p50/p95 parameter deltas were 0.0613/0.0726. Those updates
were harmful under the fixed source calibration. Probe/QCS methods received
shared frozen-query probes, but had no supported positive/negative supervision
to update from.

## Pseudo-label and probe evidence

- ENTROPY evidence: 86 supported positives and 460,001 supported negatives.
  Offline precision was 97.7% for the positive subset and 87.6% for the
  negative subset, but the entropy controller did not use these labels as a
  supervised BDD objective.
- QCS, `PROBE_CACHE`, and `NO_STATE`: 33,531 evidence records were marked
  unknown by the alignment/evidence path; supported supervision was zero, so
  optimizer steps were zero.
- `PROBE_ST` produced no supported evidence and also made zero updates.
- Probe coverage was 8,076 due frames and 8,076 frames with results, with
  17,616 shared q0/q-equivalent/q-counterfactual calls across 673 episodes.

The complete machine-readable files are:

- `results/bdd_crossdomain/candidate_recall.json`
- `results/bdd_crossdomain/pseudo_label_quality.json`
- `results/bdd_crossdomain/probe_coverage.json`
- `results/bdd_crossdomain/update_statistics.json`
- `results/bdd_crossdomain/timing.json`

The evidence-layer conclusion is that changing the residual adapter structure
alone cannot recover a candidate set that is mostly removed by the fixed
source referring gate. A BDD-trained or domain-robust source candidate model
would be a different experiment and is not claimed here.
