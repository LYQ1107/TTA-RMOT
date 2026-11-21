"""Reproduce published phase statistics with NumPy only, no raw assets or GPU.

This verifies arithmetic, not independent generalization or correction for
development iteration. Official HOTA uses pooled TP-weighted statistics.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def pooled(row, weights):
    return 100*np.sqrt((weights @ np.asarray(row['numerator'])) /
        np.maximum(weights @ np.asarray(row['denominator']), 1)).mean(axis=-1)


def methods(label):
    if label.startswith('COUNT_OWN_'):
        first, second = label.removeprefix('COUNT_OWN_').split('_vs_')
        return first+'__AT_'+first+'_COUNT', second+'__AT_'+first+'_COUNT'
    if label.startswith('COUNT_BASELINE_'):
        first, second = label.removeprefix('COUNT_BASELINE_').split('_vs_')
        return first+'__AT_'+second+'_COUNT', second+'__AT_'+second+'_COUNT'
    return label.split('_vs_')


def recompute(left, right):
    parents = left[0]['parents']
    assert all(a['parents'] == b['parents'] == parents for a, b in zip(left, right))
    n = len(parents)
    rng = np.random.default_rng(0)
    weights = rng.multinomial(n, np.ones(n)/n, size=10000)
    draws = np.stack([pooled(a, weights)-pooled(b, weights) for a, b in zip(left, right)])
    point = np.mean([pooled(a, np.ones(n))-pooled(b, np.ones(n)) for a, b in zip(left, right)])
    result = dict(delta_HOTA=float(point),
        fixed_neural_seeds_parent_CI95=np.percentile(draws.mean(0), [2.5, 97.5]).tolist())
    if len(left) > 1:
        seed_weights = rng.multinomial(len(left), np.ones(len(left))/len(left), size=10000)
        result['parent_and_neural_seed_sensitivity_CI95'] = np.percentile(
            (seed_weights*draws.T).sum(1)/len(left), [2.5, 97.5]).tolist()
    return result


def resolve_record(directory,filename,name,cache):
    path=(directory/filename).absolute();key=str(path)
    if key not in cache:cache[key]=json.loads(path.read_text())
    row=cache[key][name]
    if 'reference' in row:
        ref=row['reference']
        return resolve_record(path.parent,ref['file'],ref['record'],cache)
    return row


def share_identical_statistics(statistics,filename):
    """Keep every result key, storing identical numeric payloads only once."""
    result={};seen={};count=0
    for name,row in statistics.items():
        if 'reference' in row:result[name]=row;continue
        value=json.dumps(row,sort_keys=True,separators=(',',':'))
        if value in seen:
            result[name]=dict(reference=dict(file=filename,record=seen[value]));count+=1
        else:result[name]=row;seen[value]=name
    return result,count


def main(directory, stem='sparse_observation'):
    statistics = json.loads((directory / (stem+'_parent_statistics.json')).read_text())
    references=0;external={}
    for name,row in list(statistics.items()):
        if 'reference' in row:
            ref=row['reference'];statistics[name]=resolve_record(directory,ref['file'],ref['record'],external);references+=1
    report = json.loads((directory / (stem+'_analysis.json')).read_text())
    maximum_error, effects = 0., 0
    def compare(a, b, name):
        nonlocal maximum_error
        error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        assert error < 1e-8, (name, error)
        maximum_error = max(maximum_error, error)
    for name, row in statistics.items():
        compare(pooled(row, np.ones(len(row['parents']))), row['HOTA'], name)
    for name, row in report['conditions'].items():
        for contrast, published in row['comparisons'].items():
            a, b = methods(contrast)
            computed = recompute([statistics[name+'/'+a]], [statistics[name+'/'+b]])
            for field, value in computed.items():
                compare(value, published[field], name+'/'+contrast+'/'+field)
            effects += 1
    for family, row in report['repeated'].items():
        if 'condition_keys' in row:
            keys = row['condition_keys']
        else:
            kind, pair = family.rsplit('_', 1)
            prefix = 'observation' if kind == 'OBSERVATION' else 'budget0.1'
            keys = [prefix+f'_seed{seed}_{pair}' for seed in row['neural_seeds']]
        for method, value in row['mean_HOTA'].items():
            compare(np.mean([statistics[k+'/'+method]['HOTA'] for k in keys]), value, family+'/'+method)
        for contrast, published in row['comparisons'].items():
            a, b = methods(contrast)
            computed = recompute([statistics[k+'/'+a] for k in keys], [statistics[k+'/'+b] for k in keys])
            for field, value in computed.items():
                compare(value, published[field], family+'/'+contrast+'/'+field)
            effects += 1
    result = dict(status='PASS_PUBLISHED_'+stem.upper()+'_STATISTICS_REPRODUCED',
        HOTA_records=len(statistics), effect_records=effects, maximum_absolute_error=maximum_error,
        bootstrap_samples=10000, bootstrap_seed=0, original_parent_resampling=True,
        neural_seed_sensitivity_reproduced=True, no_raw_assets_or_GPU_required=True,
        existing_public_statistic_references=references,
        arithmetic_reproduction_is_not_independent_validation=True)
    (directory / (stem+'_statistical_reproduction.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path,
        default=Path(__file__).resolve().parents[1] / 'docs/a100_visual_classical_20261121')
    parser.add_argument('--stem', default='sparse_observation')
    args = parser.parse_args()
    main(args.directory, args.stem)
