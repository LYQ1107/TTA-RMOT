"""Post-seal quality, critic conditioning, and original-Trust controls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.rsi_probe_conditioning import comparable_paid_probe
from scripts.a100_native_binary_comparable_six_setting import (
    OUT, PARENT_OUT, PUBLIC, PAIRS, PRIMARY, RECIPES, METHODS, COMPARATORS,
    AFFECTED, REUSED, CROSS_NAMES, EXPECTED)
STEM = 'native_binary_comparable_quality_seed0_20261124'


def main():
    barrier = json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    official = json.loads((OUT/'official.json').read_text())
    assert barrier['methods'] == official['methods'] == EXPECTED == 318
    assert barrier['GT_open_attempts'] == 0
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    from scripts.a100_visual_classical_sparse_statistics_reproduce import pack_statistic_rows
    audit = json.loads((PUBLIC/'native_binary_effective_views_20261124.json').read_text())
    statistics, settings = {}, {}; maximum = 0.
    for pair in PAIRS:
        root = OUT/pair/'seed0'
        raw = json.loads((root/'completion.json').read_text())
        counts = json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts'] == counts['GT_open_attempts'] == 0
        assert counts['CountSelf_identical_masks_for_all_methods']
        if pair in REUSED:
            assert root.resolve() == (PARENT_OUT/pair/'seed0').resolve()
        paths = {r['method']: root/r['method'] for r in raw['methods']}
        paths.update({r['method']: root/'countmatched'/r['method'] for r in counts['methods']})
        assert len(paths) == 52
        if pair in AFFECTED:
            paths['PARENT_PRIMARY'] = PARENT_OUT/pair/'seed0'/PRIMARY
            paths.update({n: root/'probe_amendment_counts'/n for n in CROSS_NAMES})
        rows = {n: read_stats(p) for n, p in paths.items()}
        for name, row in rows.items():
            num, den = row['parent_stats']
            error = abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA'])
            maximum = max(maximum, error)
            statistics[pair+'_seed0/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'],
                numerator=num.tolist(), denominator=den.tolist())
        effects = {}
        for other in COMPARATORS:
            for label, a, b in (('RAW', PRIMARY, other),
                ('AT_PRIMARY_COUNT', PRIMARY, other+'__AT_'+PRIMARY+'_COUNT'),
                ('AT_OTHER_COUNT', PRIMARY+'__AT_'+other+'_COUNT', other)):
                effects[label+'_vs_'+other] = dict(comparison([rows[a]], [rows[b]]), methods=[a, b])
        amended = {}
        contrast = [('RAW', PRIMARY, 'PARENT_PRIMARY'),
            ('AT_PARENT_COUNT', CROSS_NAMES[0], 'PARENT_PRIMARY'),
            ('AT_CORRECTED_COUNT', PRIMARY, CROSS_NAMES[1])] if pair in AFFECTED else [
                (label, PRIMARY, PRIMARY) for label in ('RAW', 'AT_PARENT_COUNT', 'AT_CORRECTED_COUNT')]
        for label, a, b in contrast:
            amended[label] = dict(comparison([rows[a]], [rows[b]]), methods=[a, b])
        ordinary = {}; changes = []
        observers = {p: dict(updates=0, paid_requests=0, paid_probe_nodes=0, eligible_critic_pairs=0,
            excluded_actual_view_pairs=0, max_update_KL=0.) for p in ('LEARNED', 'FROZEN')}
        for receipt in raw['receipts']:
            for name, report in receipt['fits'].items():
                if RECIPES[name]['family'] != 'structured':
                    for fit in report['fits']:
                        assert fit['teacher_fit_gate_passed'] and fit['excess_risk'] <= .005
                        ordinary[name] = max(ordinary.get(name, 0.), fit['excess_risk'])
            sequences = {}
            for policy, summary in observers.items():
                folder = root/'videos'/receipt['video']/'policies'/policy
                report = receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]
                paid = {r['uid']: r for r in json.loads((folder/'paid.json').read_text())['paid']}
                sequence = json.loads((folder/'sequence.json').read_text())['sequence']
                assert len(paid) == report['teacher_budget']['unique_teacher_bundles']
                summary['paid_requests'] += len(paid)
                summary['updates'] += report['actor_actual_updates']
                for stage in range(3):
                    probe = [paid[e['uid']] for e in sequence if e['role'] == 'probe' and e['stage'] == stage]
                    eligible, rule = comparable_paid_probe(probe)
                    summary['paid_probe_nodes'] += len(probe)
                    summary['eligible_critic_pairs'] += len(eligible)//2
                    summary['excluded_actual_view_pairs'] += rule['effective_view_mismatch_pairs']
                    if pair in AFFECTED:
                        logged = report['rounds'][stage]['teacher_probe']
                        if eligible:
                            assert logged['comparable_paid_probe'] == rule
                        else:
                            assert logged is None and report['rounds'][stage]['credit']['reward'] == 0.
                    summary['max_update_KL'] = max(summary['max_update_KL'],
                        report['rounds'][stage]['actor_update'].get('mean_recorded_state_KL', 0.))
                sequences[policy] = [r['uid'] for r in sequence]
            changes.append(dict(video=receipt['video'], replaced_questions=len(
                set(sequences['LEARNED']) ^ set(sequences['FROZEN']))//2,
                full_order_identical=sequences['LEARNED'] == sequences['FROZEN']))
        for summary in observers.values():
            assert summary['excluded_actual_view_pairs'] == audit['settings'][pair]['probe_effective_view_mismatches']
        settings[pair] = dict(source_target=PAIRS[pair], raw_HOTA={n: rows[n]['HOTA'] for n in METHODS},
            comparisons=effects, correction_vs_parent=amended, observer=observers,
            question_changes=changes, ordinary_maximum_excess_risk=ordinary,
            exactly_reused_unaffected_setting=pair in REUSED,
            generation_wall_seconds=raw['wall_seconds'], generation_cost_is_not_fresh_cold_deployment=True)
    assert len(statistics) == 318 and maximum < 1e-8
    result = dict(status='ANALYZED_ALL_SIX_COMPARABLE_PROBE_CORRECTION_SEED0', cohort=OUT.name,
        primary=PRIMARY, settings=settings, neural_seeds=[0], official_records=318,
        new_official_evaluations=108, exactly_reused_official_records=210,
        own_raw_methods=108, CountSelf_aliases=108, parent_sum_max_error=maximum,
        all_outputs_sealed_before_GT=True, all_data_DEVELOPMENT=True, independent_blind_validation=False,
        quality_replay_not_cold_deployment=True, original_parent_study_preserved=True,
        exact_unaffected_reuse_is_not_independent_replication=True,
        equal_view_does_not_guarantee_equal_time_or_teacher_correctness=True,
        main_and_correction_contrasts_have_dual_per_query_frame_CountMatched=True,
        final_goal_not_closed_by_this_analysis=True)
    atomic_json(OUT/'analysis.json', result)
    atomic_json(OUT/'parent_statistics_readable.json', statistics)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(pack_statistic_rows(statistics), separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=result['status'], primary_HOTA={p: s['raw_HOTA'][PRIMARY] for p, s in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
