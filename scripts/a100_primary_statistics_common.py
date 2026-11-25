"""Public-only HOTA collection/reproduction with explicit cohort identity."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.a100_semantic_pattern_teacher_statistics import value,write


def collect(out,public,stem,expected_records):
    barrier=json.loads((out/('all'+str(expected_records)+'_sealed_before_GT.json')).read_text())
    result=json.loads((out/'quality_results.json').read_text())
    assert len(barrier['tasks'])==expected_records and barrier['GT_open_attempts']==0
    records={}
    for task in barrier['tasks']:
        pair,name,folder=task['pair'],task['method'],Path(task['folder'])
        reference=result['results'][pair][name]
        path=folder/'results/pedestrian_detailed.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['detailed_sha256']
        with path.open() as stream: rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        numerator,denominator=np.zeros(19),np.zeros(19)
        for row in rows:
            for i,threshold in enumerate(range(5,100,5)):
                tp=float(row['HOTA_TP___'+str(threshold)])
                numerator[i]+=float(row['AssA___'+str(threshold)])*tp
                denominator[i]+=tp+float(row['HOTA_FP___'+str(threshold)])+float(row['HOTA_FN___'+str(threshold)])
        item=dict(numerator=numerator.tolist(),denominator=denominator.tolist(),HOTA=reference['HOTA_unrounded'],episodes=len(rows))
        assert abs(value(item)-item['HOTA'])<1e-8
        records[pair+'/'+name]=item
    write(public/(stem+'_statistics.json'),dict(records=records,one_video_per_setting=True,no_population_CI=True))
    write(public/(stem+'_analysis.json'),{k:v for k,v in result.items() if k!='results'})
    print('EXPORTED_ALL'+str(expected_records)+'_'+stem,flush=True)


def expected_keys(plan):
    primary,controls=plan['primary'],plan['comparators']
    methods=[primary,*controls,*[primary+'__AT_'+n+'_COUNT' for n in controls],*[n+'__AT_'+primary+'_COUNT' for n in controls]]
    return {c['pair']+'/'+n for c in plan['cases'] for n in methods}


def reproduce(public,stem,expected_records,expected_effects):
    records=json.loads((public/(stem+'_statistics.json')).read_text())['records']
    analysis=json.loads((public/(stem+'_analysis.json')).read_text())
    plan_path=public/(stem+'_registration.json')
    plan=json.loads(plan_path.read_text())
    primary,controls=plan['primary'],plan['comparators'];cases=[c['pair'] for c in plan['cases']]
    assert set(records)==expected_keys(plan) and len(records)==expected_records
    assert 3*len(analysis['comparisons'])==expected_effects
    errors=[abs(value(row)-row['HOTA']) for row in records.values()];seen=set()
    for row in analysis['comparisons']:
        pair,other=row['pair'],row['comparator']
        assert pair in cases and other in controls and (pair,other) not in seen
        seen.add((pair,other))
        def v(name): return value(records[pair+'/'+name])
        errors.extend([abs(v(primary)-v(other)-row['raw_delta_HOTA']),
                       abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
                       abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in analysis['comparisons'])
    assert eligible==analysis['candidate_eligible_for_larger_screen']
    maximum=max(errors);assert maximum<1e-8
    report=dict(status='PASS_INDEPENDENT_ALL'+str(expected_records)+'_HOTA_AND'+str(expected_effects)+'_EFFECTS',
                records=expected_records,effects=expected_effects,maximum_absolute_error=maximum,
                eligibility_reproduced=eligible,registration_file=plan_path.name,
                only_public_files_and_NumPy_no_GT_Torch_or_refits=True,no_additional_neural_seeds=True)
    write(public/(stem+'_reproduction.json'),report);print(json.dumps(report),flush=True)
