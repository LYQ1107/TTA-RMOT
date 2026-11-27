"""Retained-parent statistics; immutable providers and public NumPy checking."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/parent_control_supplement'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='conditional_state_parent_controls_20261127'
PARENT_ALIAS={'PARENT_SOFT_KD':'UNIFORM_SOFT_KD288','PARENT_POOLED':'UNIFORM_POOLED'}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')


def value(row):
    a,b=np.asarray(row['numerator']),np.asarray(row['denominator'])
    assert a.shape==b.shape==(19,) and np.isfinite(a).all() and np.isfinite(b).all()
    assert (a>=0).all() and (b>=a-1e-8).all()
    return float(100*np.sqrt(a/np.maximum(b,1)).mean())


def resolve(record,*,root=PUBLIC,cache=None,stack=()):
    """Resolve exact referenced arrays without reading GT, Torch or models."""
    if 'reference' not in record:return record
    ref=record['reference'];name=ref['file'];key=ref['key'];digest=ref['sha256']
    assert Path(name).name==name and name.endswith('.json')
    identity=(name,key,digest)
    if identity in stack:raise ValueError('Cyclic scientific statistics reference')
    cache={} if cache is None else cache
    if (name,digest) not in cache:
        data=(root/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Statistics provider hash changed: '+name)
        cache[name,digest]=json.loads(data)
    child=cache[name,digest]['records'][key]
    return resolve(child,root=root,cache=cache,stack=(*stack,identity))


def collect():
    plan=json.loads((OUT/'registration.json').read_text());barrier=json.loads((OUT/'all1036_sealed_before_GT.json').read_text())
    assert len(barrier['tasks'])==1036 and barrier['GT_open_attempts']==0
    assert barrier['registration_sha256']==hashlib.sha256((OUT/'registration.json').read_bytes()).hexdigest()
    primary,controls=plan['primary'],plan['controls'];records={}
    for task in barrier['tasks']:
        folder=Path(task['folder']);detail=folder/'results/pedestrian_detailed.csv'
        result=json.loads((folder/'official_completion.json').read_text())
        assert hashlib.sha256(detail.read_bytes()).hexdigest()==result['detailed_sha256']
        key=task['pair']+'/'+task['method'];row=records.setdefault(key,dict(numerator=np.zeros(19),denominator=np.zeros(19),videos=0,episodes=0))
        with detail.open() as stream:episodes=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        for e in episodes:
            for i,threshold in enumerate(range(5,100,5)):
                tp=float(e['HOTA_TP___'+str(threshold)])
                row['numerator'][i]+=float(e['AssA___'+str(threshold)])*tp
                row['denominator'][i]+=tp+float(e['HOTA_FP___'+str(threshold)])+float(e['HOTA_FN___'+str(threshold)])
        row['videos']+=1;row['episodes']+=len(episodes)
    assert len(records)==42
    for key,row in records.items():
        row['numerator']=row['numerator'].tolist();row['denominator']=row['denominator'].tolist();row['HOTA']=value(row)
        assert row['videos']==sum(c['pair']==key.split('/')[0] for c in plan['cases'])
    old=plan['old_statistics_provider'];current=plan['primary_statistics_provider']
    old_sha=hashlib.sha256((PUBLIC/old).read_bytes()).hexdigest();assert old_sha==plan['old_statistics_provider_sha256']
    current_sha=hashlib.sha256((PUBLIC/current).read_bytes()).hexdigest()
    for pair in sorted({c['pair'] for c in plan['cases']}):
        records[pair+'/'+primary]=dict(reference=dict(file=current,key=pair+'/'+primary,sha256=current_sha))
        for other,original in PARENT_ALIAS.items():
            records[pair+'/'+other]=dict(reference=dict(file=old,key=pair+'/'+original,sha256=old_sha))
    assert len(records)==60
    cache={};comparisons=[]
    for pair in sorted({c['pair'] for c in plan['cases']}):
        def v(name):return value(resolve(records[pair+'/'+name],cache=cache))
        for other in controls:
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=v(primary),ordinary_HOTA=v(other),
                raw_delta_HOTA=v(primary)-v(other),
                primary_at_ordinary_count_delta_HOTA=v(primary+'__AT_'+other+'_COUNT')-v(other),
                ordinary_at_primary_count_delta_HOTA=v(primary)-v(other+'__AT_'+primary+'_COUNT')))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    analysis=dict(status='COMPLETE_ALL6_RETAINED_PARENT_CONTROL_COMPARISONS',comparisons=comparisons,
        additional_parent_control_condition_met=eligible,original7control_study_unchanged=True,
        all_data_DEV=True,no_three_contribution_RSI_causal_or_fresh_cost_success_claim=True,goal_status='ACTIVE')
    write(PUBLIC/(STEM+'_statistics.json'),dict(records=records,full_new_array_records=42,immutable_provider_references=18,whole_cross_domain_settings=True))
    write(PUBLIC/(STEM+'_analysis.json'),analysis);write(OUT/'quality_results.json',analysis)


def reproduce():
    plan=json.loads((PUBLIC/(STEM+'_registration.json')).read_text())
    records=json.loads((PUBLIC/(STEM+'_statistics.json')).read_text())['records']
    analysis=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    primary,controls=plan['primary'],plan['controls'];pairs=set(plan['setting_video_counts'])
    methods=[primary,*controls,*[primary+'__AT_'+c+'_COUNT' for c in controls],*[c+'__AT_'+primary+'_COUNT' for c in controls]]
    assert len(records)==60 and len(analysis['comparisons'])==18 and len(methods)==10
    assert set(records)=={p+'/'+m for p in pairs for m in methods}
    cache={};errors=[];seen=set()
    for key,row in records.items():
        actual=resolve(row,cache=cache)
        assert actual['videos']==plan['setting_video_counts'][key.split('/')[0]]
        errors.append(abs(value(actual)-actual['HOTA']))
    for row in analysis['comparisons']:
        pair,other=row['pair'],row['comparator'];assert pair in pairs and other in controls and (pair,other) not in seen
        seen.add((pair,other))
        def v(name):return value(resolve(records[pair+'/'+name],cache=cache))
        errors.extend([abs(v(primary)-v(other)-row['raw_delta_HOTA']),
            abs(v(primary+'__AT_'+other+'_COUNT')-v(other)-row['primary_at_ordinary_count_delta_HOTA']),
            abs(v(primary)-v(other+'__AT_'+primary+'_COUNT')-row['ordinary_at_primary_count_delta_HOTA'])])
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in analysis['comparisons'])
    assert eligible==analysis['additional_parent_control_condition_met'] and max(errors)<1e-8
    result=dict(status='PASS_INDEPENDENT_ALL60_PARENT_CONTROL_HOTA_AND54_EFFECTS',records=60,effects=54,
        maximum_absolute_error=max(errors),parent_condition_reproduced=eligible,public_JSON_and_NumPy_only=True,
        immutable_provider_SHA_checked=True,no_private_run_GT_Torch_fits_or_new_seeds=True,goal_status='ACTIVE')
    write(PUBLIC/(STEM+'_reproduction.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    collect() if args.collect else reproduce()
