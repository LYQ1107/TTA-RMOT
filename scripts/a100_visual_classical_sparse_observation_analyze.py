"""Exact parent-level, fixed-QA analysis; no new adaptation decisions here."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_analyze import read_stats
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_sparse_observation_launch import conditions
from scripts.a100_visual_classical_sparse_observation_evaluate import projection_pairs
from scripts.analyze_pf_video_uncertainty import paired_clusters, pooled_hota

PUBLIC = ROOT / 'docs/a100_visual_classical_20261121'


def key(record):
    return (('observation' if record['kind'] == 'OBSERVATION' else f"budget{record['fraction']:g}")+
            f"_seed{record['seed']}_{record['pair']}")


def contrasts(kind):
    if kind == 'OBSERVATION':
        result = [('ANCHOR_FULL', other) for other in ('POOLED_SOFT', 'POOLED_MEAN', 'POOLED_HARD400',
            'POOLED_HARD_FINAL', 'POOLED_GRAPH', 'ANCHOR_KD', 'ANCHOR_MEAN', 'ANCHOR_NO_GRAPH',
            'ANCHOR_VISUAL_ONLY', 'ANCHOR_TEMPORAL_ONLY', 'CONSTANT_TEMPORAL', 'MASS_MATCHED_CONSTANT_TEMPORAL')]
        result += [('ANCHOR_KD', 'POOLED_SOFT'), ('ANCHOR_MEAN_FULL', 'POOLED_MEAN'),
                   ('ANCHOR_MEAN_FULL', 'ANCHOR_MEAN')]
    else:
        result = [('GRAPH', other) for other in ('SOFT_FINAL', 'SOFT_MEAN_400_FINAL', 'HARD_400', 'HARD_FINAL', 'NO_GRAPH')]
        result += [('MEAN_GRAPH', 'SOFT_MEAN_400_FINAL')]
    return {a+'_vs_'+b: (a, b) for a, b in result}


def count_contrasts(kind):
    result = {}
    pairs = projection_pairs(kind)
    full_names = ('ANCHOR_FULL', 'ANCHOR_MEAN_FULL') if kind == 'OBSERVATION' else ('GRAPH', 'MEAN_GRAPH')
    for rank, donor in pairs:
        if rank == donor:
            continue
        full = donor if donor in full_names else rank
        direction = 'OWN' if donor == full else 'BASELINE'
        first = full+'__AT_'+donor+'_COUNT'
        second = rank+'__AT_'+donor+'_COUNT' if donor == full else donor+'__AT_'+donor+'_COUNT'
        result['COUNT_'+direction+'_'+full+'_vs_'+(rank if donor == full else donor)] = first, second
    return result


def effect(first, second):
    assert len(first) == len(second)
    parents = first[0]['parent_videos']
    assert all(a['parent_videos'] == b['parent_videos'] == parents for a, b in zip(first, second))
    rng = np.random.default_rng(0)
    counts = rng.multinomial(len(parents), np.ones(len(parents))/len(parents), size=10000)
    draws = np.stack([paired_clusters(a['parent_stats'], b['parent_stats'], counts) for a, b in zip(first, second)])
    deltas = np.asarray([a['HOTA']-b['HOTA'] for a, b in zip(first, second)])
    result = dict(delta_HOTA=float(deltas.mean()), per_neural_seed_delta=deltas.tolist(),
        original_parent_groups=len(parents), samples=10000, seed=0,
        fixed_neural_seeds_parent_CI95=np.percentile(draws.mean(0), [2.5, 97.5]).tolist())
    if len(first) > 1:
        seed_counts = rng.multinomial(len(first), np.ones(len(first))/len(first), size=10000)
        sensitivity = (seed_counts*draws.T).sum(1)/len(first)
        result['parent_and_neural_seed_sensitivity_CI95'] = np.percentile(sensitivity, [2.5, 97.5]).tolist()
    return result


def main():
    completion = json.loads((OUT / 'analysis/sparse_observation_official.json').read_text())
    assert completion['status'] == 'ALL_SPARSE_OBSERVATION_OFFICIAL_COMPLETE'
    records = conditions()
    data, summary, sufficient = {}, {}, {}
    for record in records:
        slug, source = key(record), Path(record['folder'])
        raw = json.loads((source / 'completion.json').read_text())
        methods = [(r['method'], source / r['method']) for r in raw['methods']]
        has_counts = record['kind'] == 'OBSERVATION' or record['fraction'] == .1
        if has_counts:
            counts = json.loads((source / 'countmatched/completion.json').read_text())
            assert not any(counts['CountSelf_changed_query_files'].values())
            methods += [(r['method'], source / 'countmatched' / r['method']) for r in counts['methods']]
        data[slug] = {}
        for name, folder in methods:
            d = read_stats(folder)
            reconstructed = float(pooled_hota(d['parent_stats'], np.ones(len(d['parent_videos']))))
            assert abs(reconstructed-d['HOTA']) < 1e-8
            data[slug][name] = d
            sufficient[slug+'/'+name] = dict(parents=d['parent_videos'], HOTA=d['HOTA'],
                numerator=d['parent_stats'][0].tolist(), denominator=d['parent_stats'][1].tolist())
        cs = contrasts(record['kind'])
        if has_counts:
            cs.update(count_contrasts(record['kind']))
        summary[slug] = dict(kind=record['kind'], pair=record['pair'], seed=record['seed'],
            fraction=record.get('fraction', .3), logical_teacher_calls=raw.get('logical_teacher_calls', raw['receipts'][-1]['teacher_examples']),
            raw_HOTA={r['method']: data[slug][r['method']]['HOTA'] for r in raw['methods']},
            comparisons={name: effect([data[slug][a]], [data[slug][b]]) for name, (a, b) in cs.items()},
            generation_wall_seconds=raw['wall_seconds'],
            cost_scope='Historical features and recorded QA; not a cold deployment measurement.')
    repeated = {}
    for kind in ('BUDGET', 'OBSERVATION'):
        for pair in ('C4', 'C1'):
            selected = [r for r in records if r['kind'] == kind and r['pair'] == pair and
                        (kind == 'OBSERVATION' or r['fraction'] == .1)]
            selected = sorted(selected, key=lambda r: r['seed'])
            assert [r['seed'] for r in selected] == [0, 17, 42]
            family = kind+'_'+pair
            cs = dict(contrasts(kind), **count_contrasts(kind))
            repeated[family] = dict(neural_seeds=[0, 17, 42], QA_seed=0,
                mean_HOTA={name: float(np.mean([summary[key(r)]['raw_HOTA'][name] for r in selected]))
                    for name in summary[key(selected[0])]['raw_HOTA']},
                comparisons={name: effect([data[key(r)][a] for r in selected],
                    [data[key(r)][b] for r in selected]) for name, (a, b) in cs.items()})
    preparation = {}
    for pair in ('C4', 'C1'):
        p = OUT / 'observation_alignment/inputs' / pair / 'completion.json'
        rows = json.loads(p.read_text())['receipts']
        preparation[pair] = dict(videos=len(rows), nodes=sum(r['retained_observations'] for r in rows),
            pooled_PCD_max_abs_difference=max(r['pooled_PCD_max_abs_difference'] for r in rows),
            pooled_RGB_max_abs_difference=max(r['pooled_RGB_max_abs_difference'] for r in rows),
            GT_open_attempts=sum(r['GT_open_attempts'] for r in rows))
    result = dict(status='SPARSE_EVIDENCE_AND_OBSERVATION_ANALYZED', conditions=summary,
        repeated=repeated, observation_input_equivalence=preparation,
        cross_study_feature_scope='Budget study uses the actual fresh C4 features. Observation study uses historical per-observation PCD and freshly re-pooled RGB, with its own fully refitted matched pooled KD. C4 differences are disclosed; do not connect its observation POOL .3 point as an identical-input fresh-feature budget curve.',
        all_development_data=True, independent_blind_validation=False,
        uncertainty='Descriptive parent-bootstrap of fixed outputs; fixed QA, neural seeds only. No correction for development iteration or multiple comparisons.',
        new_teacher_calls=0, official_evaluated_methods=completion['methods'],
        goal_status='ACTIVE; inspect scientific acceptance and actual deployment costs separately.')
    atomic_json(OUT / 'analysis/sparse_observation_analysis.json', result)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    atomic_json(PUBLIC / 'sparse_observation_analysis.json', result)
    # Compact parent statistics reproduce every reported HOTA/effect without
    # uploading raw predictions, target labels, checkpoints or large arrays.
    (PUBLIC / 'sparse_observation_parent_statistics.json').write_text(json.dumps(sufficient, separators=(',', ':'))+'\n')
    print(json.dumps(dict(event='SPARSE_OBSERVATION_ANALYSIS_COMPLETE', repeated=repeated,
        observation_input_equivalence=preparation, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    main()
