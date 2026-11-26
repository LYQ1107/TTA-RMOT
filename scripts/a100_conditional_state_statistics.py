"""Whole-setting official sufficient statistics and public-only reproduction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='conditional_state_primary_20261127'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')


def value(row):
    numerator,denominator=np.asarray(row['numerator']),np.asarray(row['denominator'])
    assert numerator.shape==denominator.shape==(19,)
    assert np.isfinite(numerator).all() and np.isfinite(denominator).all()
    assert (numerator>=0).all() and (denominator>=numerator-1e-8).all()
    return float(100*np.sqrt(numerator/np.maximum(denominator,1)).mean())


def collect():
    plan=json.loads((OUT/'registration.json').read_text());barrier=json.loads((OUT/'all3256_sealed_before_GT.json').read_text())
    assert len(barrier['tasks'])==3256 and barrier['GT_open_attempts']==0
    assert barrier['registration_sha256']==hashlib.sha256((OUT/'registration.json').read_bytes()).hexdigest()
    records={}
    for task in barrier['tasks']:
        path=Path(task['folder']);detail=path/'results/pedestrian_detailed.csv'
        result=json.loads((path/'official_completion.json').read_text())
        assert hashlib.sha256(detail.read_bytes()).hexdigest()==result['detailed_sha256']
        key=task['pair']+'/'+task['method'];row=records.setdefault(key,dict(numerator=np.zeros(19),denominator=np.zeros(19),videos=0,episodes=0))
        with detail.open() as stream:episodes=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        for episode in episodes:
            for i,threshold in enumerate(range(5,100,5)):
                tp=float(episode['HOTA_TP___'+str(threshold)])
                row['numerator'][i]+=float(episode['AssA___'+str(threshold)])*tp
                row['denominator'][i]+=tp+float(episode['HOTA_FP___'+str(threshold)])+float(episode['HOTA_FN___'+str(threshold)])
        row['videos']+=1;row['episodes']+=len(episodes)
    assert len(records)==132
    for key,row in records.items():
        row['numerator']=row['numerator'].tolist();row['denominator']=row['denominator'].tolist();row['HOTA']=value(row)
        pair=key.split('/')[0];assert row['videos']==sum(c['pair']==pair for c in plan['cases'])
    primary,controls=plan['primary'],plan['comparators'];comparisons=[]
    for pair in sorted({c['pair'] for c in plan['cases']}):
        def v(name):return records[pair+'/'+name]['HOTA']
        for other in controls:
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=v(primary),ordinary_HOTA=v(other),
                raw_delta_HOTA=v(primary)-v(other),
                primary_at_ordinary_count_delta_HOTA=v(primary+'__AT_'+other+'_COUNT')-v(other),
                ordinary_at_primary_count_delta_HOTA=v(primary)-v(other+'__AT_'+primary+'_COUNT')))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    analysis=dict(status='COMPLETE_ALL6_CONDITIONAL_STATE_PRIMARY',comparisons=comparisons,
        candidate_eligible_for_contribution_validation=eligible,all_data_DEV=True,
        no_RSI_causal_three_contribution_or_fresh_cost_success_claim=True,goal_status='ACTIVE')
    write(PUBLIC/(STEM+'_statistics.json'),dict(records=records,whole_cross_domain_settings=True,never_mean_of_video_HOTA=True))
    write(PUBLIC/(STEM+'_analysis.json'),analysis);write(OUT/'quality_results.json',analysis)


def reproduce():
    # This branch reads only the public files and NumPy; no private run, GT,
    # Torch, original teacher, model refitting or official matching is needed.
    plan=json.loads((PUBLIC/(STEM+'_registration.json')).read_text())
    analysis=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    records=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    primary,controls=plan['primary'],plan['comparators'];pairs=set(plan['setting_video_counts'])
    methods=[primary,*controls,*[primary+'__AT_'+c+'_COUNT' for c in controls],*[c+'__AT_'+primary+'_COUNT' for c in controls]]
    assert len(methods)==22 and len(records)==132 and len(analysis['comparisons'])==42
    assert set(records)=={p+'/'+m for p in pairs for m in methods}
    errors=[];seen=set()
    for key,row in records.items():
        assert row['videos']==plan['setting_video_counts'][key.split('/')[0]]
        errors.append(abs(value(row)-row['HOTA']))
    for row in analysis['comparisons']:
        pair,other=row['pair'],row['comparator'];assert pair in pairs and other in controls and (pair,other) not in seen
        seen.add((pair,other))
        def v(name):return value(records[pair+'/'+name])
        errors.extend([abs(v(primary)-v(other)-row['raw_delta_HOTA']),
            abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
            abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in analysis['comparisons'])
    assert eligible==analysis['candidate_eligible_for_contribution_validation'] and max(errors)<1e-8
    result=dict(status='PASS_INDEPENDENT_ALL132_HOTA_AND126_EFFECTS',records=132,effects=126,
        maximum_absolute_error=max(errors),eligibility_reproduced=eligible,
        public_JSON_NumPy_only=True,no_new_training_or_seeds=True,goal_status='ACTIVE')
    write(PUBLIC/(STEM+'_reproduction.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    collect() if args.collect else reproduce()
