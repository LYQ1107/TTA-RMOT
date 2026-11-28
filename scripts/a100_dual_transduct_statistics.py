"""Whole-setting official statistics and public-only reproduction for DUAL_TRANSDUCT_COEVOL."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/dual_transduct_20261127/primary'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'dual_transduct_primary_20261127'
PARENT_ALIAS = {'PARENT_SOFT_KD': 'UNIFORM_SOFT_KD288', 'PARENT_POOLED': 'UNIFORM_POOLED'}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n')


def value(row):
    a, b = np.asarray(row['numerator']), np.asarray(row['denominator'])
    assert a.shape == b.shape == (19,) and np.isfinite(a).all() and np.isfinite(b).all()
    assert (a >= 0).all() and (b >= a - 1e-8).all()
    return float(100.0 * np.sqrt(a / np.maximum(b, 1.0)).mean())


def resolve(record, *, root=PUBLIC, cache=None, stack=()):
    if 'reference' not in record:
        return record
    ref = record['reference']
    name, key, digest = ref['file'], ref['key'], ref['sha256']
    assert Path(name).name == name and name.endswith('.json')
    identity = (name, key, digest)
    if identity in stack:
        raise ValueError('Cyclic statistics reference')
    cache = {} if cache is None else cache
    if (name, digest) not in cache:
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Statistics provider SHA changed: ' + name)
        cache[name, digest] = json.loads(data)
    return resolve(cache[name, digest]['records'][key], root=root, cache=cache, stack=(*stack, identity))


def collect():
    plan = json.loads((OUT / 'registration.json').read_text())
    barrier = json.loads((OUT / 'all2960_sealed_before_GT.json').read_text())
    assert len(barrier['tasks']) == 2960 and barrier['GT_open_attempts'] == 0
    assert barrier['registration_sha256'] == hashlib.sha256((OUT / 'registration.json').read_bytes()).hexdigest()

    records = {}
    for task in barrier['tasks']:
        folder = Path(task['folder'])
        detail = folder / 'results/pedestrian_detailed.csv'
        result = json.loads((folder / 'official_completion.json').read_text())
        assert hashlib.sha256(detail.read_bytes()).hexdigest() == result['detailed_sha256']
        key = task['pair'] + '/' + task['method']
        row = records.setdefault(key, dict(numerator=np.zeros(19), denominator=np.zeros(19), videos=0, episodes=0))
        with detail.open() as stream:
            episodes = [r for r in csv.DictReader(stream) if r['seq'] != 'COMBINED']
        for ep in episodes:
            for i, thr in enumerate(range(5, 100, 5)):
                tp = float(ep['HOTA_TP___' + str(thr)])
                row['numerator'][i] += float(ep['AssA___' + str(thr)]) * tp
                row['denominator'][i] += tp + float(ep['HOTA_FP___' + str(thr)]) + float(ep['HOTA_FN___' + str(thr)])
        row['videos'] += 1
        row['episodes'] += len(episodes)

    assert len(records) == 120
    for key, row in records.items():
        row['numerator'] = row['numerator'].tolist()
        row['denominator'] = row['denominator'].tolist()
        row['HOTA'] = value(row)
        pair = key.split('/')[0]
        assert row['videos'] == sum(c['pair'] == pair for c in plan['cases'])

    old = plan['old_statistics_provider']
    old_sha = hashlib.sha256((PUBLIC / old).read_bytes()).hexdigest()
    assert old_sha == plan['old_statistics_provider_sha256']
    pairs = sorted({c['pair'] for c in plan['cases']})
    for pair in pairs:
        for alias, orig in PARENT_ALIAS.items():
            records[pair + '/' + alias] = dict(reference=dict(file=old, key=pair + '/' + orig, sha256=old_sha))
    assert len(records) == 132

    primary, controls = plan['primary'], plan['comparators']
    ord_set = set(plan['ordinary_controls'])
    cache, comparisons = {}, []
    for pair in pairs:
        def v(name):
            return value(resolve(records[pair + '/' + name], cache=cache))
        for other in controls:
            comparisons.append(dict(
                pair=pair, comparator=other,
                is_ordinary_control=(other in ord_set),
                primary_HOTA=v(primary), ordinary_HOTA=v(other),
                raw_delta_HOTA=v(primary) - v(other),
                primary_at_ordinary_count_delta_HOTA=v(primary + '__AT_' + other + '_COUNT') - v(other),
                ordinary_at_primary_count_delta_HOTA=v(primary) - v(other + '__AT_' + primary + '_COUNT'),
            ))
    ord_eligible = all(
        min(r['raw_delta_HOTA'], r['primary_at_ordinary_count_delta_HOTA'], r['ordinary_at_primary_count_delta_HOTA']) > 0
        for r in comparisons if r['is_ordinary_control']
    )
    all7_eligible = all(
        min(r['raw_delta_HOTA'], r['primary_at_ordinary_count_delta_HOTA'], r['ordinary_at_primary_count_delta_HOTA']) > 0
        for r in comparisons
    )
    analysis = dict(
        status='COMPLETE_ALL6_DUAL_TRANSDUCT_PRIMARY',
        comparisons=comparisons,
        ordinary_controls_all6_eligible=ord_eligible,
        all7_including_shared_readout_eligible=all7_eligible,
        all_data_DEV=True,
        goal_status='ACTIVE',
    )
    write(PUBLIC / (STEM + '_statistics.json'), dict(
        records=records, full_new_array_records=120, immutable_provider_references=12,
        whole_cross_domain_settings=True, never_mean_of_video_HOTA=True,
    ))
    write(PUBLIC / (STEM + '_analysis.json'), analysis)
    write(OUT / 'quality_results.json', analysis)


def reproduce():
    plan = json.loads((PUBLIC / (STEM + '_registration.json')).read_text())
    records = json.loads((PUBLIC / (STEM + '_statistics.json')).read_text())['records']
    analysis = json.loads((PUBLIC / (STEM + '_analysis.json')).read_text())
    primary, controls = plan['primary'], plan['comparators']
    ord_set = set(plan['ordinary_controls'])
    pairs = set(plan['setting_video_counts'])
    methods = [primary, *controls, *[primary + '__AT_' + c + '_COUNT' for c in controls],
               *[c + '__AT_' + primary + '_COUNT' for c in controls]]
    assert len(methods) == 22 and len(records) == 132 and len(analysis['comparisons']) == 42
    assert set(records) == {p + '/' + m for p in pairs for m in methods}
    cache, errors, seen = {}, [], set()
    for key, row in records.items():
        actual = resolve(row, cache=cache)
        assert actual['videos'] == plan['setting_video_counts'][key.split('/')[0]]
        errors.append(abs(value(actual) - actual['HOTA']))
    for row in analysis['comparisons']:
        pair, other = row['pair'], row['comparator']
        assert pair in pairs and other in controls and (pair, other) not in seen
        seen.add((pair, other))
        def v(name):
            return value(resolve(records[pair + '/' + name], cache=cache))
        errors.extend([
            abs(v(primary) - v(other) - row['raw_delta_HOTA']),
            abs(v(primary + '__AT_' + other + '_COUNT') - v(other) - row['primary_at_ordinary_count_delta_HOTA']),
            abs(v(primary) - v(other + '__AT_' + primary + '_COUNT') - row['ordinary_at_primary_count_delta_HOTA']),
        ])
    ord_eligible = all(
        min(r['raw_delta_HOTA'], r['primary_at_ordinary_count_delta_HOTA'], r['ordinary_at_primary_count_delta_HOTA']) > 0
        for r in analysis['comparisons'] if r['comparator'] in ord_set
    )
    all7_eligible = all(
        min(r['raw_delta_HOTA'], r['primary_at_ordinary_count_delta_HOTA'], r['ordinary_at_primary_count_delta_HOTA']) > 0
        for r in analysis['comparisons']
    )
    assert ord_eligible == analysis['ordinary_controls_all6_eligible']
    assert all7_eligible == analysis['all7_including_shared_readout_eligible']
    assert max(errors) < 1e-8
    result = dict(
        status='PASS_INDEPENDENT_ALL132_DUAL_TRANSDUCT_HOTA_AND126_EFFECTS',
        records=132, effects=126, maximum_absolute_error=max(errors),
        ordinary_controls_all6_eligible=ord_eligible,
        all7_including_shared_readout_eligible=all7_eligible,
        public_JSON_NumPy_only=True, immutable_provider_SHA_checked=True,
        goal_status='ACTIVE',
    )
    write(PUBLIC / (STEM + '_reproduction.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    collect() if args.collect else reproduce()
