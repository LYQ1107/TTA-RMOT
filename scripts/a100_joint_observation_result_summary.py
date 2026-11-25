"""Summarize only the fully evaluated/reproduced old joint study."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    public = ROOT/'docs/a100_six_direction_20261122'
    analysis = json.loads((public/'joint_observation_quality_seed0_analysis.json').read_text())
    reproduction = json.loads((public/'joint_observation_quality_seed0_reproduction.json').read_text())
    assert analysis['official_unique_evaluations'] == reproduction['statistical_records'] == 582
    assert reproduction['maximum_absolute_error'] < 1e-8
    primary = analysis['primary']; baseline = 'COARSE_UNIT_SOFT_DESIGN_FINAL'
    binary = 'BINARY_CONDITIONAL_REFERENCE_DESIGN_FINAL'; settings = {}
    for pair, row in analysis['settings'].items():
        ordinary = {n: v for n, v in row['raw_HOTA'].items() if n.startswith(('JOINT_KD', 'MARGINAL_KD', 'COARSE_'))}
        same = {n: v for n, v in ordinary.items() if '_DESIGN_' in n}
        def effects(name):
            return {scope: row['comparisons'][scope+'_vs_'+name] for scope in ('RAW', 'AT_PRIMARY_COUNT', 'AT_OTHER_COUNT')}
        settings[pair] = dict(primary_HOTA=row['raw_HOTA'][primary],
            fixed_display_ordinary_HOTA=row['raw_HOTA'][baseline],
            binary_channel_control_HOTA=row['raw_HOTA'][binary],
            primary_vs_fixed_display_ordinary=effects(baseline), primary_vs_binary_channel=effects(binary),
            same_evidence_ordinary_envelope_descriptive_only=dict(method=max(same, key=same.get), HOTA=max(same.values())),
            budget_matched_different_policy_envelope_descriptive_only=dict(method=max(ordinary, key=ordinary.get), HOTA=max(ordinary.values())),
            factorial={scope: {name: {k: e[k] for k in ('delta_HOTA', 'CI95', 'original_parent_groups')}
                for name, e in values.items()} for scope, values in row['factorial'].items()})
    result = dict(status='COMPLETED_JOINT_STUDY_MECHANISM_SUMMARY_GOAL_UNMET', primary=primary,
        fixed_display_ordinary=baseline, binary_channel_control=binary, settings=settings,
        neural_seeds=[0], all_582_evaluated_and_independently_reproduced=True,
        displayed_ordinary_is_one_common_readable_reference_not_a_new_preregistered_primary_baseline=True,
        all_24_ordinary_outputs_and_all_contrasts_remain_in_complete_analysis=True,
        best_ordinary_envelopes_are_descriptive_not_selected_deployment_methods=True,
        binary_channel_is_different_teacher_target_information_not_evidence_of_beating_binary_KD=True,
        current_native_RSI_studies_are_distinct_and_unchanged=True,
        CountMatched_is_diagnostic_not_additive_causal_decomposition=True,
        all_data_DEVELOPMENT_no_independent_blind_validation=True,
        goal_status='ACTIVE')
    atomic_json(public/'joint_observation_mechanism_summary_20261124.json', result)
    print(json.dumps(dict(status=result['status'], rows={p: dict(primary=s['primary_HOTA'],
        fixed_ordinary=s['fixed_display_ordinary_HOTA'], binary_channel=s['binary_channel_control_HOTA'],
        raw_delta=s['primary_vs_fixed_display_ordinary']['RAW']['delta_HOTA'],
        at_ordinary_count_delta=s['primary_vs_fixed_display_ordinary']['AT_OTHER_COUNT']['delta_HOTA'])
        for p, s in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
