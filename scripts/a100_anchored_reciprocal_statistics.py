"""Public-only NumPy reproduction of all57 anchored-primary statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/anchored_reciprocal_primary_20261125'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
STEM = 'anchored_reciprocal_primary_20261125'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n')


def value(row):
    numerator, denominator = np.asarray(row['numerator']), np.asarray(row['denominator'])
    assert numerator.shape == denominator.shape == (19,)
    assert np.isfinite(numerator).all() and np.isfinite(denominator).all()
    assert (numerator >= 0).all() and (denominator >= numerator-1e-8).all()
    return float(100*np.sqrt(numerator/np.maximum(denominator, 1)).mean())


def collect():
    barrier = json.loads((OUT/'all57_sealed_before_GT.json').read_text())
    results = json.loads((OUT/'quality_results.json').read_text())
    assert len(barrier['tasks']) == 57 and barrier['GT_open_attempts'] == 0
    records = {}
    for task in barrier['tasks']:
        pair, name, folder = task['pair'], task['method'], Path(task['folder'])
        original = results['results'][pair][name]
        path = folder/'results/pedestrian_detailed.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original['detailed_sha256']
        with path.open() as stream:
            rows = [row for row in csv.DictReader(stream) if row['seq'] != 'COMBINED']
        numerator, denominator = np.zeros(19), np.zeros(19)
        for row in rows:
            for index, threshold in enumerate(range(5, 100, 5)):
                tp = float(row['HOTA_TP___'+str(threshold)])
                numerator[index] += float(row['AssA___'+str(threshold)])*tp
                denominator[index] += tp+float(row['HOTA_FP___'+str(threshold)])+float(row['HOTA_FN___'+str(threshold)])
        record = dict(numerator=numerator.tolist(), denominator=denominator.tolist(),
                      HOTA=original['HOTA_unrounded'], episodes=len(rows))
        assert abs(value(record)-record['HOTA']) < 1e-8
        records[pair+'/'+name] = record
    write(PUBLIC/(STEM+'_statistics.json'), dict(records=records, one_video_per_setting=True, no_population_CI=True))
    write(PUBLIC/(STEM+'_analysis.json'), {k: v for k, v in results.items() if k != 'results'})
    print('EXPORTED_ALL57_ANCHORED_PRIMARY_STATISTICS', flush=True)


def reproduce():
    records = json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    analysis = json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    plan = json.loads((PUBLIC/'anchored_reciprocal_primary_registration_20261125.json').read_text())
    primary, controls = plan['primary'], plan['comparators']
    cases = [c['pair'] for c in plan['cases']]
    methods = [primary, *controls, *[primary+'__AT_'+n+'_COUNT' for n in controls],
               *[n+'__AT_'+primary+'_COUNT' for n in controls]]
    assert set(records) == {c+'/'+n for c in cases for n in methods}
    assert len(records) == 57 and len(analysis['comparisons']) == 18
    errors = [abs(value(row)-row['HOTA']) for row in records.values()]
    seen = set()
    for row in analysis['comparisons']:
        pair, other = row['pair'], row['comparator']
        assert pair in cases and other in controls and (pair, other) not in seen
        seen.add((pair, other))
        def v(name):
            return value(records[pair+'/'+name])
        errors.extend([abs(v(primary)-v(other)-row['raw_delta_HOTA']),
                       abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
                       abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible = all(min(row['raw_delta_HOTA'], row['primary_at_ordinary_count_delta_HOTA'],
                       row['ordinary_at_primary_count_delta_HOTA']) > 0 for row in analysis['comparisons'])
    assert eligible == analysis['candidate_eligible_for_larger_screen']
    maximum = max(errors)
    assert maximum < 1e-8
    result = dict(status='PASS_INDEPENDENT_ALL57_ANCHORED_HOTA_AND54_EFFECTS', records=57, effects=54,
                  maximum_absolute_error=maximum, eligibility_reproduced=eligible,
                  public_files_only_no_private_registration_GT_predictions_models_Torch_or_refits=True,
                  no_additional_neural_seeds=True, one_video_per_setting_not_population_significance=True)
    write(PUBLIC/(STEM+'_reproduction.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    else:
        reproduce()
