"""NumPy-only reproduction of all 39 new pilot HOTA values and comparisons.

--collect exports sufficient counts AFTER the whole cohort's evaluation.
Default reproduces that export without models, raw predictions, GT or Torch.
One video per case is a screen; no population confidence interval is claimed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/factor_rsi_primary_pilot_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='factor_rsi_primary_pilot_20261125'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,separators=(',',':'),ensure_ascii=False)+'\n')


def pooled(row):
    return float(100*np.sqrt(np.asarray(row['numerator'])/np.maximum(np.asarray(row['denominator']),1)).mean())


def collect():
    barrier=json.loads((OUT/'all_39_predictions_sealed_before_GT.json').read_text())
    quality=json.loads((OUT/'quality_results.json').read_text())
    assert len(barrier['tasks'])==39 and barrier['GT_open_attempts']==0
    records={}
    for task in barrier['tasks']:
        pair,name=task['pair'],task['method'];folder=Path(task['folder'])
        result=quality['results'][pair][name]
        path=folder/'results/pedestrian_detailed.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==result['detailed_sha256']
        assert hashlib.sha256((folder/'predictions_seal.json').read_bytes()).hexdigest()==result['prediction_seal_sha256']
        with path.open() as stream:rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        num=np.zeros(19);den=np.zeros(19)
        for row in rows:
            for i,a in enumerate(range(5,100,5)):
                tp=float(row['HOTA_TP___'+str(a)]);fp=float(row['HOTA_FP___'+str(a)]);fn=float(row['HOTA_FN___'+str(a)])
                num[i]+=float(row['AssA___'+str(a)])*tp;den[i]+=tp+fp+fn
        item=dict(numerator=num.tolist(),denominator=den.tolist(),HOTA=result['HOTA_unrounded'],
            episodes=len(rows),detailed_sha256=result['detailed_sha256'])
        assert abs(pooled(item)-item['HOTA'])<1e-8
        records[pair+'/'+name]=item
    summary={k:v for k,v in quality.items() if k!='results'}
    write(PUBLIC/(STEM+'_statistics.json'),dict(records=records,one_video_per_case=True,no_population_CI=True))
    write(PUBLIC/(STEM+'_analysis.json'),summary)
    print('EXPORTED_ALL_39_PILOT_STATISTICS_AFTER_GLOBAL_SEAL_AND_EVALUATION',flush=True)


def reproduce():
    stats=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    report=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text());errors=[]
    assert len(stats)==39 and len(report['comparisons'])==12
    for row in stats.values():errors.append(abs(pooled(row)-row['HOTA']))
    primary='FACTOR_RSI_NEURAL'
    for effect in report['comparisons']:
        pair=effect['pair'];other=effect['comparator']
        def h(name):return pooled(stats[pair+'/'+name])
        errors.extend([abs((h(primary)-h(other))-effect['raw_delta_HOTA']),
            abs((h(primary+'__AT_'+other+'_COUNT')-h(other))-effect['primary_at_ordinary_count_delta_HOTA']),
            abs((h(primary)-h(other+'__AT_'+primary+'_COUNT'))-effect['ordinary_at_primary_count_delta_HOTA'])])
    maximum=max(errors);assert maximum<1e-8
    result=dict(status='PASS_INDEPENDENT_ALL_39_PILOT_HOTA_AND_36_EFFECTS',records=39,effects=36,
        maximum_absolute_error=maximum,no_GT_models_predictions_Torch_or_refits=True,
        no_new_neural_seeds=True,not_population_significance_or_six_setting_success=True)
    write(PUBLIC/(STEM+'_reproduction.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.collect:collect()
    else:reproduce()
