"""Independent NumPy HOTA/effect reproduction for all57 reciprocal records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/reciprocal_memory_primary_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122';STEM='reciprocal_memory_primary_20261125'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n')


def value(row):return float(100*np.sqrt(np.asarray(row['numerator'])/np.maximum(np.asarray(row['denominator']),1)).mean())


def collect():
    barrier=json.loads((OUT/'all57_sealed_before_GT.json').read_text());result=json.loads((OUT/'quality_results.json').read_text())
    assert len(barrier['tasks'])==57 and barrier['GT_open_attempts']==0
    records={}
    for task in barrier['tasks']:
        pair,name,folder=task['pair'],task['method'],Path(task['folder']);reference=result['results'][pair][name]
        path=folder/'results/pedestrian_detailed.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['detailed_sha256']
        with path.open() as stream:rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        numerator=np.zeros(19);denominator=np.zeros(19)
        for row in rows:
            for i,a in enumerate(range(5,100,5)):
                tp=float(row['HOTA_TP___'+str(a)])
                numerator[i]+=float(row['AssA___'+str(a)])*tp
                denominator[i]+=tp+float(row['HOTA_FP___'+str(a)])+float(row['HOTA_FN___'+str(a)])
        r=dict(numerator=numerator.tolist(),denominator=denominator.tolist(),HOTA=reference['HOTA_unrounded'],episodes=len(rows))
        assert abs(value(r)-r['HOTA'])<1e-8;records[pair+'/'+name]=r
    write(PUBLIC/(STEM+'_statistics.json'),dict(records=records,one_video_per_setting=True,no_population_CI=True))
    write(PUBLIC/(STEM+'_analysis.json'),{k:v for k,v in result.items() if k!='results'})
    print('EXPORTED_ALL57_RECIPROCAL_PRIMARY_STATISTICS',flush=True)


def reproduce():
    records=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    result=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text());plan=json.loads((OUT/'registration.json').read_text())
    primary=plan['primary'];errors=[];assert len(records)==57 and len(result['comparisons'])==18
    for r in records.values():errors.append(abs(value(r)-r['HOTA']))
    for r in result['comparisons']:
        pair,other=r['pair'],r['comparator']
        def v(name):return value(records[pair+'/'+name])
        errors.extend([abs(v(primary)-v(other)-r['raw_delta_HOTA']),
            abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-r['primary_at_ordinary_count_delta_HOTA']),
            abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-r['ordinary_at_primary_count_delta_HOTA'])])
    maximum=max(errors);assert maximum<1e-8
    receipt=dict(status='PASS_INDEPENDENT_ALL57_RECIPROCAL_HOTA_AND54_EFFECTS',records=57,effects=54,
        maximum_absolute_error=maximum,no_raw_GT_predictions_models_Torch_or_refits=True,
        no_additional_neural_seeds=True,screen_not_population_significance_or_six_setting_completion=True)
    write(PUBLIC/(STEM+'_reproduction.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--reproduce',action='store_true');a=p.parse_args()
    if a.collect:collect()
    else:reproduce()
