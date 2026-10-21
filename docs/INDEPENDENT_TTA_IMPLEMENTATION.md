# Independent source-free TTA implementation

The v2 configuration uses independent implementations rather than aliases of
the historical pilot controller. All source TempRMOT parameters, the tracker,
backbone, and text encoder stay frozen.

| method | adaptation state | input to the external adapter |
|---|---|---|
| `FROZEN` | none | source referring score |
| `TDA_RMOT` | non-parametric feature/reliability cache | track feature and source score |
| `VISUAL_TTA` | trainable zero-initialized residual | 256-D track feature, objectness, disappearance, area |
| `LANGUAGE_TTA` | trainable zero-initialized residual | frozen 256-D RoBERTa-to-TempRMOT query projection and source score |
| `FUSION_TTA` | trainable zero-initialized residual | visual feature, query, feature-query product, source score, objectness |
| combinations | separate visual/language/fusion modules | sum of the selected independent residuals |

The adapters start exactly source-equivalent because their final residual layer
is zero initialized. Reliability-filtered evidence is collected from current
frame confidence or causal frozen-query probes. The replay stores detached
features only, is bounded, and requires distinct track-time samples before an
optimizer step. Fusion additionally has a set-separation term when a frame has
both positive and negative evidence. TDA performs no optimizer step and only
updates its bounded cache.

The synthetic regression test
`tests/test_independent_tta.py` verifies for every independent trainable arm:
source-equivalent initialization, nonzero gradient, parameter movement, and
causal update. It also verifies TDA cache hits and updates. A passing synthetic
test is an implementation check, not an end-to-end benchmark result.

## Runtime evidence

Each run writes `run_metadata.json`, `update_trace.jsonl`, and
`candidate_diagnostics.jsonl`. The metadata records method implementations,
optimizer steps, nonzero-gradient steps, parameter movement, cache hits/updates,
probe calls, memory, and the explicit `gt_used_by_runner=false` contract.

Official metrics must be generated separately with
`scripts/evaluate_predictions.py`; the evaluator materializes GT only after the
source-free runtime has finished.
