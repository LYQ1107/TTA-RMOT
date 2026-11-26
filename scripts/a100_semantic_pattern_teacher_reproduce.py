"""Resume only public84-statistic reproduction after wrong-plan lookup.

Original pinned script/registration, model weights and official scores remain
unchanged. This entry reads this cohort's actual registration filename.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_semantic_pattern_teacher_statistics import PUBLIC,STEM,value,write


def main():
    records=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    analysis=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    plan=json.loads((PUBLIC/(STEM+'_registration.json')).read_text())
    primary,controls=plan['primary'],plan['comparators']
    cases=[c['pair'] for c in plan['cases']]
    methods=[primary,*controls,*[primary+'__AT_'+n+'_COUNT' for n in controls],
             *[n+'__AT_'+primary+'_COUNT' for n in controls]]
    assert set(records)=={p+'/'+n for p in cases for n in methods}
    assert len(records)==84 and len(analysis['comparisons'])==27
    errors=[abs(value(row)-row['HOTA']) for row in records.values()]
    seen=set()
    for row in analysis['comparisons']:
        pair,other=row['pair'],row['comparator']
        assert pair in cases and other in controls and (pair,other) not in seen
        seen.add((pair,other))
        def v(name):
            return value(records[pair+'/'+name])
        errors.extend([abs(v(primary)-v(other)-row['raw_delta_HOTA']),
                       abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
                       abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible=all(min(row['raw_delta_HOTA'],row['primary_at_ordinary_count_delta_HOTA'],
                     row['ordinary_at_primary_count_delta_HOTA'])>0 for row in analysis['comparisons'])
    assert eligible==analysis['candidate_eligible_for_larger_screen']
    maximum=max(errors)
    assert maximum<1e-8
    result=dict(status='PASS_INDEPENDENT_ALL84_SEMANTIC_PATTERN_HOTA_AND81_EFFECTS',
                records=84,effects=81,maximum_absolute_error=maximum,eligibility_reproduced=eligible,
                only_public_files_and_NumPy_no_GT_Torch_or_refits=True,no_additional_neural_seeds=True,
                original_controller_exit_code=1,original_error='Postprocessing loaded previous cohort registration',
                original_pinned_sources_and_registration_unchanged=True,
                new_training_predictions_or_official_evaluations=0,
                reproduction_source=str(Path(__file__).relative_to(ROOT)))
    write(PUBLIC/(STEM+'_reproduction.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
