"""Whole-setting 2x2x2 Factorial Ablation statistics & independent public-only reproduction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_factorial_20261128'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'unified_rsi_factorial_20261128'


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
    identity = (name, key, digest)
    if identity in stack:
        raise ValueError('Cyclic reference')
    cache = {} if cache is None else cache
    if (name, digest) not in cache:
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('Provider SHA changed: ' + name)
        cache[name, digest] = json.loads(data)
    return resolve(cache[name, digest]['records'][key], root=root, cache=cache, stack=(*stack, identity))


def collect():
    plan = json.loads((OUT / 'registration.json').read_text())
    barrier = json.loads((OUT / 'all1036_sealed_before_GT.json').read_text())
    assert len(barrier['tasks']) == 1036 and barrier['GT_open_attempts'] == 0
    assert barrier['registration_sha256'] == hashlib.sha256((OUT / 'registration.json').read_bytes()).hexdigest()

    records = {}
    for task in barrier['tasks']:
        folder = Path(task['folder'])
        detail = folder / 'results/pedestrian_detailed.csv'
        res = json.loads((folder / 'official_completion.json').read_text())
        assert hashlib.sha256(detail.read_bytes()).hexdigest() == res['detailed_sha256']
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

    assert len(records) == 42
    for key, row in records.items():
        row['numerator'] = row['numerator'].tolist()
        row['denominator'] = row['denominator'].tolist()
        row['HOTA'] = value(row)

    # Link all 6 existing immutable cells of the 2x2x2 factorial design
    p_fs = plan['providers']['fullscope']
    p_v3 = plan['providers']['v3']
    p_u1 = plan['providers']['u1']
    pairs = sorted({c['pair'] for c in plan['cases']})
    for pair in pairs:
        records[pair + '/CELL_000_BASE_UNIFORM'] = dict(reference=dict(file=p_fs['file'], key=pair + '/UNIFORM_POOLED', sha256=p_fs['sha256']))
        records[pair + '/CELL_100_I1_ONLY_UNIFORM'] = dict(reference=dict(file=p_v3['file'], key=pair + '/REG_SHARED_TRANSDUCT', sha256=p_v3['sha256']))
        records[pair + '/CELL_110_I1_I2_UNIFORM'] = dict(reference=dict(file=p_v3['file'], key=pair + '/DUAL_TRANSDUCT_COEVOL_V3', sha256=p_v3['sha256']))
        records[pair + '/CELL_001_I3_ONLY_ACTIVE'] = dict(reference=dict(file=p_u1['file'], key=pair + '/ACTIVE_SOFT_POOLED', sha256=p_u1['sha256']))
        records[pair + '/CELL_101_I1_I3_ACTIVE'] = dict(reference=dict(file=p_u1['file'], key=pair + '/ACTIVE_SHARED_TRANSDUCT', sha256=p_u1['sha256']))
        records[pair + '/CELL_111_FULL_I1_I2_I3'] = dict(reference=dict(file=p_u1['file'], key=pair + '/UNIFIED_RSI_RMOT', sha256=p_u1['sha256']))

    assert len(records) == 78
    cache = {}
    factorial_table = []
    for pair in pairs:
        def v(m):
            return value(resolve(records[pair + '/' + m], cache=cache))
        c000 = v('CELL_000_BASE_UNIFORM')
        c100 = v('CELL_100_I1_ONLY_UNIFORM')
        c010 = v('I2_ONLY_UNIFORM')
        c110 = v('CELL_110_I1_I2_UNIFORM')
        c001 = v('CELL_001_I3_ONLY_ACTIVE')
        c101 = v('CELL_101_I1_I3_ACTIVE')
        c011 = v('I2_PLUS_I3_ACTIVE')
        c111 = v('CELL_111_FULL_I1_I2_I3')
        c_ne = v('NO_EXISTENCE_ACTIVE')
        # Standard 2^3 Yates factorial main effects & interactions
        main_I1 = 0.25 * ((c100 - c000) + (c110 - c010) + (c101 - c001) + (c111 - c011))
        main_I2 = 0.25 * ((c010 - c000) + (c110 - c100) + (c011 - c001) + (c111 - c101))
        main_I3 = 0.25 * ((c001 - c000) + (c101 - c100) + (c011 - c010) + (c111 - c110))
        int_I1_I2 = 0.25 * (((c110 - c100) - (c010 - c000)) + ((c111 - c101) - (c011 - c001)))
        int_I1_I3 = 0.25 * (((c101 - c001) - (c100 - c000)) + ((c111 - c011) - (c110 - c010)))
        int_I2_I3 = 0.25 * (((c011 - c001) - (c010 - c000)) + ((c111 - c101) - (c110 - c100)))
        int_I1_I2_I3 = 0.25 * ((((c111 - c101) - (c011 - c001)) - ((c110 - c100) - (c010 - c000))))
        factorial_table.append(dict(
            pair=pair,
            cells=dict(c000=c000, c100=c100, c010=c010, c110=c110, c001=c001, c101=c101, c011=c011, c111=c111),
            no_existence_active=c_ne,
            checklist_existence_gain=c111 - c_ne,
            marginal_add_I1=c111 - c011,
            marginal_add_I2=c111 - c101,
            marginal_add_I3=c111 - c110,
            total_gain_vs_base=c111 - c000,
            yates_effects=dict(
                I1=main_I1, I2=main_I2, I3=main_I3,
                I1xI2=int_I1_I2, I1xI3=int_I1_I3, I2xI3=int_I2_I3, I1xI2xI3=int_I1_I2_I3,
            ),
            cm_vs_I2_ONLY_UNIFORM=dict(
                p_at_o=v('UNIFIED_RSI_RMOT__AT_I2_ONLY_UNIFORM_COUNT') - c010,
                o_at_p=c111 - v('I2_ONLY_UNIFORM__AT_UNIFIED_RSI_RMOT_COUNT'),
            ),
            cm_vs_I2_PLUS_I3_ACTIVE=dict(
                p_at_o=v('UNIFIED_RSI_RMOT__AT_I2_PLUS_I3_ACTIVE_COUNT') - c011,
                o_at_p=c111 - v('I2_PLUS_I3_ACTIVE__AT_UNIFIED_RSI_RMOT_COUNT'),
            ),
        ))

    analysis = dict(
        status='COMPLETE_ALL6_2X2X2_FACTORIAL_ABLATION',
        factorial_table=factorial_table,
        all_data_DEV=True, goal_status='ACTIVE',
    )
    write(PUBLIC / (STEM + '_statistics.json'), dict(
        records=records, full_new_array_records=42, immutable_provider_references=36,
        whole_cross_domain_settings=True, never_mean_of_video_HOTA=True,
    ))
    write(PUBLIC / (STEM + '_analysis.json'), analysis)
    write(OUT / 'factorial_results.json', analysis)


def reproduce():
    plan = json.loads((PUBLIC / (STEM + '_registration.json')).read_text())
    records = json.loads((PUBLIC / (STEM + '_statistics.json')).read_text())['records']
    analysis = json.loads((PUBLIC / (STEM + '_analysis.json')).read_text())
    assert len(records) == 78 and len(analysis['factorial_table']) == 6
    cache, errors = {}, []
    for key, row in records.items():
        actual = resolve(row, cache=cache)
        assert actual['videos'] == plan['setting_video_counts'][key.split('/')[0]]
        errors.append(abs(value(actual) - actual['HOTA']))
    for row in analysis['factorial_table']:
        pair = row['pair']
        def v(m):
            return value(resolve(records[pair + '/' + m], cache=cache))
        c = row['cells']
        errors.extend([
            abs(v('CELL_000_BASE_UNIFORM') - c['c000']),
            abs(v('CELL_100_I1_ONLY_UNIFORM') - c['c100']),
            abs(v('I2_ONLY_UNIFORM') - c['c010']),
            abs(v('CELL_110_I1_I2_UNIFORM') - c['c110']),
            abs(v('CELL_001_I3_ONLY_ACTIVE') - c['c001']),
            abs(v('CELL_101_I1_I3_ACTIVE') - c['c101']),
            abs(v('I2_PLUS_I3_ACTIVE') - c['c011']),
            abs(v('CELL_111_FULL_I1_I2_I3') - c['c111']),
        ])
    assert max(errors) < 1e-8
    result = dict(
        status='PASS_INDEPENDENT_ALL78_FACTORIAL_ABLATION_HOTA',
        records=78, factorial_cells=48, maximum_absolute_error=max(errors),
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
