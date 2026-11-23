"""Reproduce published whole-C4 effects using only compact official statistics.

No images, GT annotations, teacher answers, checkpoints or GPU are needed.
The pooling formula is TrackEval's TP-weighted AssA/DetA aggregation, not a
mean of per-video HOTA scores. Intervals are descriptive development evidence.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    directory = ROOT / 'docs/a100_visual_classical_20261121'
    statistics = json.loads((directory / 'fresh_cold_C4_sufficient_statistics.json').read_text())
    checked = []
    max_error = 0.

    def check(name, left, right, published):
        nonlocal max_error
        assert left['parents'] == right['parents']
        n = len(left['parents'])
        weights = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=10000)

        def pooled(row, counts):
            numerator = np.asarray(row['numerator'])
            denominator = np.asarray(row['denominator'])
            return 100 * np.sqrt((counts @ numerator) / np.maximum(counts @ denominator, 1)).mean(axis=-1)

        point = float(pooled(left, np.ones(n)) - pooled(right, np.ones(n)))
        interval = np.percentile(pooled(left, weights) - pooled(right, weights), [2.5, 97.5])
        error = max(abs(point - published['delta_HOTA']),
                    float(np.max(np.abs(interval - published['original_parent_video']['CI95']))))
        assert error < 1e-8, (name, error)
        max_error = max(max_error, error)
        checked.append(name)

    for condition, stem in [('full', 'fresh_whole_C4'), ('key', 'fresh_whole_C4_sparse')]:
        stats = statistics[condition]
        report = json.loads((directory / (stem + '.json')).read_text())
        for contrast, record in report['comparisons'].items():
            left, right = contrast.split('_vs_')
            check(condition + '/' + contrast, stats[left], stats[right], record)
        report = json.loads((directory / (stem + '_countmatched.json')).read_text())
        for contrast, record in report['comparisons'].items():
            if contrast.endswith('_AT_FULL_COUNTS'):
                prefix = contrast.removesuffix('_AT_FULL_COUNTS')
                left, right = prefix.split('_vs_')
                count = left
            else:
                prefix = contrast.removesuffix('_AT_BASELINE_COUNTS')
                left, right = prefix.split('_vs_')
                count = right
            left_key = 'counts/' + left + '__AT_' + count + '_COUNT'
            right_key = 'counts/' + right + '__AT_' + count + '_COUNT'
            check(condition + '/counts/' + contrast, stats[left_key], stats[right_key], record)

    replay = json.loads((directory / 'fresh_sparse_fixedQA.json').read_text())
    for name, record in replay['comparisons'].items():
        if name.endswith('_sparse_source_vs_full_FIXED_QA'):
            method = name.removesuffix('_sparse_source_vs_full_FIXED_QA')
            left, right = statistics['fixed_QA'][method], statistics['full'][method]
        elif name.endswith('_actual_sparse_vs_fixed_QA_sparse'):
            method = name.removesuffix('_actual_sparse_vs_fixed_QA_sparse')
            left, right = statistics['key'][method], statistics['fixed_QA'][method]
        else:
            label = name.split('_same_QA_')[0]
            left = statistics['fixed_QA'][label + '_COHORT']
            right = statistics['fixed_QA'][label + '_KD']
        check('fixed_QA/' + name, left, right, record)
    independent = json.loads((directory / 'independent_single_cold_C4.json').read_text())
    check('independent_single_cold', statistics['single_cohort']['CLIP_COHORT'],
          statistics['single_KD']['CLIP_KD'], independent['comparison'])
    result = dict(status='PASS_PUBLISHED_COLD_STATISTICS_REPRODUCED',
                  comparisons_checked=len(checked), maximum_absolute_error=max_error,
                  original_parent_clusters=13, bootstrap_samples=10000, bootstrap_seed=0,
                  official_TP_weighted_pooling=True, no_raw_assets_or_GPU_required=True,
                  evaluation_clip_intervals_recomputed=False, all_development_data=True)
    atomic_json(directory / 'fresh_cold_C4_statistical_reproduction.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
