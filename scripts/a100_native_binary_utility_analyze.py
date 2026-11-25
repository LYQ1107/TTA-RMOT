"""Complete post-seal readout quality, fixed-count effects and exact reuse."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_native_binary_utility_six_setting import (
    OUT, PUBLIC, PAIRS, PRIMARY, READOUTS, EXPECTED, parent_paths, contrast_pairs, parent)
STEM = 'native_binary_utility_quality_seed0_20261124'
PARENT_STEM = 'native_binary_comparable_quality_seed0_20261124'


def main():
    barrier = json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    official = json.loads((OUT/'official.json').read_text())
    assert barrier['methods'] == official['methods'] == EXPECTED and barrier['GT_open_attempts'] == 0
    # Wait for the parent's finished public aggregate statistics, not partial
    # scores or an accidental official.json-before-analysis race.
    import time
    parent_stats = PUBLIC/(PARENT_STEM+'_parent_statistics.json')
    parent_repro = PUBLIC/(PARENT_STEM+'_reproduction.json')
    while not parent_repro.is_file():
        time.sleep(30)
    assert json.loads(parent_repro.read_text())['maximum_absolute_error'] < 1e-8
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    from scripts.a100_visual_classical_sparse_statistics_reproduce import (
        resolve_record, share_identical_statistics, share_identical_parent_groups)
    statistics, settings = {}, {}; maximum = 0.; cache = {}
    for pair in PAIRS:
        root = OUT/pair/'seed0'; raw = json.loads((root/'completion.json').read_text())
        assert raw['GT_open_attempts'] == 0 and raw['new_fits'] == raw['new_teacher_calls'] == 0
        paths = {r['method']: root/r['method'] for r in raw['methods']}
        inherited = parent_paths(pair); paths.update(inherited); assert len(paths) == 54
        rows = {n: read_stats(p) for n, p in paths.items()}
        for name, row in rows.items():
            num, den = row['parent_stats']; key = pair+'_seed0/'+name
            error = abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA'])
            maximum = max(maximum, error)
            actual = dict(parents=row['parent_videos'], HOTA=row['HOTA'],
                numerator=num.tolist(), denominator=den.tolist())
            if name in inherited:
                provider = pair+'_seed0/'+inherited[name].name
                restored = resolve_record(PUBLIC, parent_stats.name, provider, cache)
                assert all(actual[k] == restored[k] for k in actual)
                statistics[key] = dict(reference=dict(file=parent_stats.name, record=provider))
            else:
                statistics[key] = actual
        effects = {label: dict(comparison([rows[a]], [rows[b]]), methods=[a, b])
            for label, (a, b) in contrast_pairs().items()}
        replacements = {n: sum(r['rank_replacements_relative_to_count_donor'][n] for r in raw['receipts'])
            for n in READOUTS}
        settings[pair] = dict(source_target=PAIRS[pair], HOTA={n: r['HOTA'] for n, r in rows.items()},
            comparisons=effects, rank_replacements_relative_to_count_donor=replacements,
            all_donor_counts_equal_per_query_frame=True,
            new_fits=0, new_teacher_calls=0, new_backbone_calls=0,
            generation_wall_seconds=raw['wall_seconds'], generation_cost_is_not_fresh_cold_deployment=True)
    assert len(statistics) == EXPECTED and maximum < 1e-8
    filename = STEM+'_parent_statistics.json'
    packed, shared = share_identical_statistics(statistics, filename)
    packed, groups = share_identical_parent_groups(packed)
    result = dict(status='ANALYZED_ALL_SIX_FROZEN_UTILITY_READOUTS_SEED0', cohort=OUT.name,
        primary=PRIMARY, settings=settings, neural_seeds=[0], official_records=EXPECTED,
        new_official_evaluations=120, exactly_reused_records=204, paired_effects=318,
        parent_sum_max_error=maximum, shared_identical_new_records=shared, shared_parent_lists=groups,
        all_outputs_sealed_before_GT=True, all_data_DEVELOPMENT=True, independent_blind_validation=False,
        quality_replay_not_cold_deployment=True, all_live_parent_studies_unchanged=True,
        actual_conditional_set_model_not_original_pair_network_MAP=True,
        output_count_still_uses_original_pair_marginals=True,
        fixed_count_readout_change_is_not_new_feature_learning=True,
        conditional_identity_and_RSI_factorial_contributions_still_required=True,
        goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result); atomic_json(OUT/'parent_statistics_readable.json', statistics)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/filename).write_text(json.dumps(packed, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=result['status'], primary_HOTA={p: s['HOTA'][PRIMARY]
        for p, s in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
