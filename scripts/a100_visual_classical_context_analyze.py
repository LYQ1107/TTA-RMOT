"""Post-seal original-parent statistics for the registered scene-context study."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_context_launch import conditions, count_record
from scripts.a100_visual_classical_budget_prefix import BUDGET, METHODS as BASELINES
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_sparse_observation_analyze import effect, fit_audit
from scripts.analyze_pf_video_uncertainty import pooled_hota

PUBLIC = ROOT/'docs/a100_visual_classical_20261121'


def key(record):
    return record['pair']+f"_seed{record['seed']}"


def baseline_names():
    return {name: 'ORIGINAL_MEAN' if name == 'SOFT_MEAN_400_FINAL' else
        'ORIGINAL_GRAPH' if name == 'MEAN_GRAPH' else 'BASE_'+name for name in BASELINES}


def comparisons(pair):
    result = [('ORIGINAL_GRAPH', 'ORIGINAL_COMPLETE'), ('ORIGINAL_GRAPH', 'ORIGINAL_PERMUTED')]
    if pair == 'C4':
        result += [('CONTEXT_FULL', name) for name in ('CONTEXT_MEAN', 'CONTEXT_SOFT400',
            'CONTEXT_SOFT_FINAL', 'CONTEXT_HARD400', 'CONTEXT_HARD_FINAL', 'MASKED_MEAN',
            'CONTEXT_ANCHORS', 'CONTEXT_VISUAL', 'CONTEXT_COMPLETE', 'CONTEXT_PERMUTED',
            'ORIGINAL_MEAN', 'ORIGINAL_GRAPH', 'BASE_SOFT_400', 'BASE_SOFT_FINAL', 'BASE_HARD_400', 'BASE_HARD_FINAL')]
        result += [('CONTEXT_MEAN', 'MASKED_MEAN'), ('CONTEXT_MEAN', 'ORIGINAL_MEAN'),
                   ('MASKED_MEAN', 'ORIGINAL_MEAN'), ('CONTEXT_VISUAL', 'CONTEXT_MEAN')]
    return {a+'_vs_'+b: (a, b) for a, b in result}


def count_comparisons(pair):
    full = 'ORIGINAL_GRAPH'
    result = {'COUNT_OWN_'+full+'_vs_'+other:
        (full+'__AT_'+full+'_COUNT', other+'__AT_'+full+'_COUNT')
        for other in ('ORIGINAL_COMPLETE', 'ORIGINAL_PERMUTED')}
    if pair == 'C4':
        full = 'CONTEXT_FULL'
        for other in ('CONTEXT_MEAN', 'ORIGINAL_MEAN', 'CONTEXT_ANCHORS', 'CONTEXT_VISUAL',
                      'CONTEXT_COMPLETE', 'CONTEXT_PERMUTED'):
            result['COUNT_OWN_'+full+'_vs_'+other] = (full+'__AT_'+full+'_COUNT', other+'__AT_'+full+'_COUNT')
        for other in ('CONTEXT_MEAN', 'ORIGINAL_MEAN'):
            result['COUNT_BASELINE_'+full+'_vs_'+other] = (full+'__AT_'+other+'_COUNT', other+'__AT_'+other+'_COUNT')
    return result


def query_group_points(folders, selected_methods=('ORIGINAL_MEAN', 'ORIGINAL_GRAPH', 'MASKED_MEAN', 'CONTEXT_MEAN', 'CONTEXT_VISUAL', 'CONTEXT_FULL')):
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active.acquisition import expression_types
    from flex_uda.active.color_cohort import part_mask
    from flex_uda.a100.evaluation_reuse import metrics
    from scripts.analyze_pf_video_uncertainty import unpack
    m = metadata('SportsMOT')
    groups = {name: [] for name in ('static_clothing', 'fallback_spatial', 'fallback_role', 'fallback_other')}
    for q in m['episodes']:
        types = expression_types(q['query_text'])
        name = 'static_clothing' if part_mask(q['query_text']).any() else 'fallback_spatial' if 'SPATIAL' in types else 'fallback_role' if 'ROLE' in types else 'fallback_other'
        groups[name].append(q)
    hota = metrics()[0]
    result = {}
    for name, queries in groups.items():
        wanted = {q['video_id']+'+'+q['expression_id'] for q in queries}
        row = dict(queries=len(queries), original_parents=len({q['video_id'].rsplit('_c', 1)[0] for q in queries}), HOTA={})
        for method in selected_methods:
            folder = folders[method]
            if (folder/'official_equivalent.json').is_file():
                folder = Path(json.loads((folder/'official_equivalent.json').read_text())['identical_prediction_reference'])
            with (folder/'results/pedestrian_detailed.csv').open() as handle:
                entries = {r['seq']: unpack(r, hota) for r in csv.DictReader(handle) if r['seq'] in wanted}
            assert len(entries) == len(wanted)
            row['HOTA'][method] = float(100*hota.combine_sequences(entries)['HOTA'].mean())
        result[name] = row
    return result


def main():
    official = json.loads((OUT/'analysis/context_cohort_official.json').read_text())
    assert official['status'] == 'ALL_CONTEXT_COHORT_OFFICIAL_COMPLETE' and official['methods'] == 96
    records = conditions()
    data, summary, sufficient, paths = {}, {}, {}, {}
    for record in records:
        slug, root = key(record), Path(record['folder'])
        receipt = json.loads((root/'completion.json').read_text())
        counts = json.loads((root/'countmatched/completion.json').read_text())
        assert receipt['GT_open_attempts'] == receipt['original_graph_regression_changed_rows'] == 0
        assert not any(counts['CountSelf_changed_query_files'].values())
        folders = {r['method']: root/r['method'] for r in receipt['methods']}
        folders.update({r['method']: root/'countmatched'/r['method'] for r in counts['methods']})
        old = BUDGET/f"rho0.1_seed{record['seed']}"/record['pair']
        folders.update({alias: old/name for name, alias in baseline_names().items()})
        data[slug], paths[slug] = {}, folders
        for name, folder in folders.items():
            row = read_stats(folder)
            n = len(row['parent_videos'])
            assert abs(float(pooled_hota(row['parent_stats'], np.ones(n)))-row['HOTA']) < 1e-8
            data[slug][name] = row
            sufficient[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'],
                numerator=row['parent_stats'][0].tolist(), denominator=row['parent_stats'][1].tolist())
        pairs = dict(comparisons(record['pair']), **count_comparisons(record['pair']))
        summary[slug] = dict(pair=record['pair'], seed=record['seed'], logical_teacher_calls=receipt['logical_teacher_calls'],
            raw_HOTA={name: row['HOTA'] for name, row in data[slug].items() if '__AT_' not in name},
            comparisons={name: effect([data[slug][a]], [data[slug][b]]) for name, (a, b) in pairs.items()},
            CountSelf_changed_query_files=counts['CountSelf_changed_query_files'],
            teacher_fit_audit=fit_audit(receipt['receipts']),
            actual_head_fit_seconds={name: float(sum(r['fits'][name].get('wall_seconds', 0) for r in receipt['receipts']))
                for name in receipt['receipts'][0]['fits']},
            new_teacher_calls=0, not_cold_deployment_cost=True)
    repeated = {}
    for pair in ('C4', 'C1'):
        selected = [r for r in records if r['pair'] == pair]
        slugs = [key(r) for r in selected]
        assert [r['seed'] for r in selected] == [0, 17, 42]
        pairs = dict(comparisons(pair), **count_comparisons(pair))
        repeated['CONTEXT_'+pair] = dict(neural_seeds=[0, 17, 42], condition_keys=slugs, QA_seed=0,
            mean_HOTA={name: float(np.mean([summary[k]['raw_HOTA'][name] for k in slugs]))
                for name in summary[slugs[0]]['raw_HOTA']},
            comparisons={name: effect([data[k][a] for k in slugs], [data[k][b] for k in slugs])
                for name, (a, b) in pairs.items()})
    result = dict(status='CONTEXT_COHORT_ANALYZED', conditions=summary, repeated=repeated,
        input_defined_query_group_points_seed0=query_group_points(paths['C4_seed0']),
        all_development_data=True, registered_after_previous_development_diagnosis=True,
        independent_blind_validation=False, new_teacher_calls=0, no_GT_in_adaptation=True,
        uncertainty='Descriptive parent/seed bootstrap, same fixed QA; not a correction for iteration or multiple comparisons.',
        compute='Features/QA prepaid. New context input has three head fits for controls; individual fit receipts are actual but not cold deployment times.',
        official_evaluated_methods=96, goal_status='ACTIVE')
    atomic_json(OUT/'analysis/context_cohort_analysis.json', result)
    atomic_json(PUBLIC/'context_cohort_analysis.json', result)
    (PUBLIC/'context_cohort_parent_statistics.json').write_text(json.dumps(sufficient, separators=(',', ':'))+'\n')
    print(json.dumps(dict(event='CONTEXT_ANALYSIS_COMPLETE', repeated=repeated, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
