"""Independent event-restriction and scalar-metric check; no neural replay."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

BASE = Path('/data3/liuyeqiang/FlexHook-TTA/a100_runs/visual_classical_20261121/supervisor/uniform_teacher_holdout_audit_20261127_v3')
OWN = BASE/'independent_verification_v2_registration.json'
OUT = BASE/'independent_verification_v2.json'
ROLES = ('bootstrap_S','bootstrap_R','final_S','final_R','SOFT_KD288','SAME_ARCH_KD128')
ACTIVE = True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    assert not path.exists(),path
    with path.open('x') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    source=Path(__file__); original=read(BASE/'registration.json'); result=read(BASE/'result.json')
    assert result['status']=='COMPLETE148_UNIFORM_ONLY_TEACHER_EVENT_HOLDOUT_DIAGNOSIS'
    assert len(original['units'])==len(result['units'])==148 and tuple(original['roles'])==ROLES
    paths={source,BASE/'registration.json',BASE/'result.json'}
    for unit,row in zip(original['units'],result['units']):
        assert (unit['pair'],unit['video'])==(row['pair'],row['video'])
        for k in ('observations','queries','auxiliary_keys','active_draws','uniform_draws'):
            p=Path(unit[k]);assert sha(p)==original['inputs'][str(p)];paths.add(p)
        p=Path(row['arrays']);assert sha(p)==row['arrays_sha256'];paths.add(p)
    inputs={str(p):sha(p) for p in sorted(paths)}
    write(OWN,dict(status='REGISTERED_SCALAR_RESTRICTION_AND_MEAN_RECOMPUTATION',source_sha256=sha(source),
        inputs=inputs,scope='All148 event exclusions, array targets, support counts, and all per-unit/setting role MSE/BCE/query-macro MSE. Per-event risk algebra also checked. No independent neural-forward, semantic GT, HOTA, CI or risk-ranking reproduction claim.',time_unix=time.time()))
    allowed=set(inputs)|{str(OWN),str(OUT)}
    allowed|={os.path.realpath(p) for p in allowed}
    library_paths=(Path(sys.prefix)/'lib',Path(sys.base_prefix)/'lib',Path(np.__file__).parent.parent)
    libraries=tuple(dict.fromkeys(str(p)+'/' for a in library_paths for p in (a,a.resolve())))
    denied=[]
    def guard(event,args):
        if not ACTIVE or event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=os.path.abspath(os.fsdecode(args[0]));mode=args[1] if len(args)>1 else 'r'
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT))
        ok=p==str(OUT) if writing else p in allowed or p.startswith(libraries)
        if not ok:
            denied.append(p);raise PermissionError('Unregistered independent verification IO: '+p)
    sys.addaudithook(guard)
    checked=0; max_difference=0.; total_queries=0; uniform_events=0; retained_events=0; grouped=defaultdict(list)
    def equal_number(actual,expected):
        nonlocal checked,max_difference
        difference=abs(float(actual)-float(expected));max_difference=max(max_difference,difference);checked+=1
        assert math.isfinite(difference) and difference<=1e-10,(actual,expected)
    def verify_summary(items,query_groups,expected):
        assert expected['events']==len(items) and expected['queries']==len(query_groups)
        groups=[g for g in query_groups if g]
        assert expected['supported_queries']==len(groups)
        if not items:
            return
        for j,role in enumerate(ROLES):
            errors=[(r[0]-r[1][j])**2 for r in items]
            ce=[-r[0]*math.log(r[1][j])-(1-r[0])*math.log1p(-r[1][j]) for r in items]
            macro=[math.fsum(errors[i] for i in g)/len(g) for g in groups]
            target=expected['roles'][role]
            equal_number(math.fsum(errors)/len(errors),target['teacher_probability_MSE'])
            equal_number(math.fsum(ce)/len(ce),target['soft_teacher_BCE'])
            equal_number(math.fsum(macro)/len(macro),target['supported_query_macro_teacher_MSE'])
        for y,p,v in items:
            for stage in range(2):
                s,r=p[2*stage],p[2*stage+1]
                assert abs((y-s)**2-(v[stage]+(r-s)**2)-((y-r)**2-v[stage]+2*(y-r)*(r-s)))<=2e-14
    for unit,row in zip(original['units'],result['units']):
        obs=np.load(unit['observations'],allow_pickle=False).tolist();queries=read(unit['queries'])['queries'];n=len(obs)
        active=read(unit['active_draws'])['draws'];uniform=read(unit['uniform_draws'])['draws']
        total_queries+=len(queries);uniform_events+=len(uniform)
        key=lambda uid:(' '.join(queries[int(uid)//n]['query_text'].split()).casefold(),*map(int,obs[int(uid)%n]))
        trained=set(key(d['uid']) for d in active)
        auxiliary=set((int(k[3]),int(k[2])) for k in read(unit['auxiliary_keys'])['keys'] if k[0]==unit['video'])
        eligible=[d for d in uniform if key(d['uid']) not in trained and key(d['uid'])[1:] not in auxiliary]
        for draws in (active,uniform):
            counts=Counter(int(d['uid'])//n for d in draws)
            assert all(counts[q]==int(t['logical_budget']) for q,t in enumerate(queries))
        with np.load(row['arrays'],allow_pickle=False) as saved:
            uid=saved['uid'].tolist();qi=saved['query'].tolist();y=saved['y'].tolist();p=saved['probability'].tolist();v=saved['residual_proxy'].tolist()
        assert uid==[int(d['uid']) for d in eligible] and y==[float(d['probability']) for d in eligible]
        assert qi==[i//n for i in uid]
        assert all(len(a)==len(ROLES) and all(0<t<1 and math.isfinite(t) for t in a) for a in p)
        items=list(zip(y,p,v));groups=[[i for i,qid in enumerate(qi) if qid==q] for q in range(len(queries))]
        verify_summary(items,groups,row);retained_events+=len(items)
        grouped[unit['pair']].append((items,groups))
    for pair,batches in grouped.items():
        items=[];groups=[]
        for batch,gs in batches:
            offset=len(items);groups.extend([[i+offset for i in g] for g in gs]);items.extend(batch)
        verify_summary(items,groups,result['by_setting'][pair])
    assert total_queries==3506 and uniform_events==55019 and retained_events==result['retained_teacher_event_labels']
    for p,expected in inputs.items():
        assert sha(p)==expected,p
    write(OUT,dict(status='PASS_ALL148_EVENT_EXCLUSIONS_AND_SCALAR_ROLE_METRICS',source_sha256=sha(source),
        registration_sha256=sha(OWN),units=148,queries=total_queries,uniform_events=uniform_events,retained_events=retained_events,
        reproduced_metric_cells=checked,max_absolute_difference=max_difference,rejected_IO=denied,
        no_neural_forward_training_GT_HOTA_or_new_teacher_operations=True,time_unix=time.time()))
    print(json.dumps(read(OUT)),flush=True)


if __name__=='__main__':
    try:
        main()
    finally:
        ACTIVE=False
