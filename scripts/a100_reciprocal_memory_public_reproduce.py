"""Reproduce the completed reciprocal57 study using public Git files only."""
import json
from pathlib import Path
import numpy as np

PUBLIC = Path(__file__).resolve().parents[1]/'docs/a100_six_direction_20261122'
STEM = 'reciprocal_memory_primary_20261125'


def main():
    records = json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    analysis = json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    plan = json.loads((PUBLIC/'reciprocal_memory_primary_registration_20261125.json').read_text())
    primary, controls = plan['primary'], plan['comparators']
    cases = [c['pair'] for c in plan['cases']]
    methods = [primary, *controls, *[primary+'__AT_'+n+'_COUNT' for n in controls],
               *[n+'__AT_'+primary+'_COUNT' for n in controls]]
    assert set(records) == {c+'/'+n for c in cases for n in methods}
    assert len(records) == 57 and len(analysis['comparisons']) == 18

    def value(name):
        row = records[name]
        numerator, denominator = np.asarray(row['numerator']), np.asarray(row['denominator'])
        assert numerator.shape == denominator.shape == (19,)
        assert np.isfinite(numerator).all() and np.isfinite(denominator).all()
        assert (numerator >= 0).all() and (denominator >= numerator-1e-8).all()
        return float(100*np.sqrt(numerator/np.maximum(denominator, 1)).mean())

    errors = [abs(value(name)-row['HOTA']) for name, row in records.items()]
    seen = set()
    for row in analysis['comparisons']:
        pair, other = row['pair'], row['comparator']
        assert pair in cases and other in controls and (pair, other) not in seen
        seen.add((pair, other))
        p, o = pair+'/'+primary, pair+'/'+other
        errors.extend([abs(value(p)-value(o)-row['raw_delta_HOTA']),
                       abs(value(p+'__AT_'+other+'_COUNT')-value(o)-row['primary_at_ordinary_count_delta_HOTA']),
                       abs(value(p)-value(o+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible = all(min(row['raw_delta_HOTA'], row['primary_at_ordinary_count_delta_HOTA'],
                       row['ordinary_at_primary_count_delta_HOTA']) > 0 for row in analysis['comparisons'])
    assert eligible == analysis['candidate_eligible_for_larger_screen']
    maximum = max(errors)
    assert maximum < 1e-8
    receipt = dict(status='PASS_PUBLIC_ONLY_RECIPROCAL57_STATISTICS_AND54_EFFECTS',
                   records=57, effects=54, maximum_absolute_error=maximum,
                   eligibility_reproduced=eligible, no_private_run_directory_GT_Torch_or_models=True)
    (PUBLIC/(STEM+'_public_reproduction.json')).write_text(json.dumps(receipt, separators=(',', ':'))+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
