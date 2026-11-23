"""Describe the registered four-arm interaction at one common count.

Only completed public parent aggregates are consumed. No GT, predictions,
teacher queries or fitted parameters enter this secondary analysis.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.a100_visual_classical_sparse_statistics_reproduce import resolve_record, pooled

PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PLAN = PUBLIC/'conditional_interaction_decomposition_plan.json'
STATISTICS = 'conditional_observation_parent_statistics.json'
METHODS = ('FULL_FINAL', 'CONDITIONAL_BALANCED_FINAL', 'DIRECT_PROFILE_FINAL', 'DIRECT_BALANCED_FINAL')
SEEDS = (0, 17, 42)


def effects(values):
    full, condition_only, profile_only, direct = values
    return dict(factorization_given_profile=float(full-profile_only),
        factorization_without_profile=float(condition_only-direct),
        profile_given_factorization=float(full-condition_only),
        profile_without_factorization=float(profile_only-direct),
        interaction=float(full-condition_only-profile_only+direct))


def main():
    import hashlib
    plan = json.loads(PLAN.read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['source_sha256']
    parent = json.loads((PUBLIC/'conditional_observation_analysis.json').read_text())
    assert parent['all_outputs_sealed_before_this_GT'] and parent['manual_target_GT_in_adaptation'] is False
    cache = {}; settings = {}; maximum = 0.
    for pair in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
        records = []
        for index, seed in enumerate(SEEDS):
            slug = pair+f'_seed{seed}'
            names = [slug+'/'+m for m in METHODS]
            matched = [slug+'/'+m+'__AT_FULL_FINAL_COUNT' for m in METHODS]
            raw = [resolve_record(PUBLIC, STATISTICS, n, cache) for n in names]
            constrained = [resolve_record(PUBLIC, STATISTICS, n, cache) for n in matched]
            parents = raw[0]['parents']; weights = np.ones(len(parents))
            assert all(r['parents'] == parents for r in raw+constrained)
            first = effects([pooled(r, weights) for r in raw])
            second = effects([pooled(r, weights) for r in constrained])
            discrepancy = abs(first['interaction']-parent['repeated'][pair]['factorization_profile_interaction']['per_seed'][index])
            self_error = abs(float(pooled(raw[0], weights)-pooled(constrained[0], weights)))
            maximum = max(maximum, discrepancy, self_error)
            records.append(dict(neural_seed=seed, original_parent_groups=len(parents),
                raw=first, at_full_output_count=second,
                raw_minus_common_count_interaction=first['interaction']-second['interaction']))
        settings[pair] = dict(per_seed=records,
            mean_raw={k:float(np.mean([r['raw'][k] for r in records])) for k in records[0]['raw']},
            mean_at_full_output_count={k:float(np.mean([r['at_full_output_count'][k] for r in records])) for k in records[0]['raw']},
            mean_raw_minus_common_count_interaction=float(np.mean([r['raw_minus_common_count_interaction'] for r in records])),
            CI95=None)
    assert maximum < 1e-8
    result = dict(status='PASS_COMPLETED_CONDITIONAL_COMMON_COUNT_INTERACTION_DESCRIPTION',
        settings=settings, original_interaction_and_CountSelf_max_error=maximum,
        donor='Fixed original FULL_FINAL output count per query/frame; never a GT count.',
        no_new_predictions_fits_teacher_or_GT_reads=True,
        original_primary_and_registered_raw_interaction_unchanged=True,
        raw_minus_count_difference_is_bookkeeping_not_a_unique_causal_count_effect=True,
        remaining_precision_scope='These three structured comparator heads have not yet been covered by the seed0 ordinary probability-cutoff audit. A common count alone does not clear sigmoid quantization or prove semantic ordering.',
        descriptive_per_seed_means_only_no_new_interaction_CI=True,
        all_data_development=True, independent_blind_validation=False, goal_status='ACTIVE')
    (PUBLIC/'conditional_interaction_decomposition.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
