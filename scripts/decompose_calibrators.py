"""GT-free full-source statistics and the six fixed score calibrators."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *


def source_stats(datasets):
    result={}
    for target,(meta,arrays,groups,records) in datasets.items():
        model=previous.IN_DOMAIN_MODEL[target];x=arrays[model];episodes=[];z=[]
        for q in meta['episodes']:
            v=np.asarray(x[q['start']:q['stop']],dtype=np.float64)
            mu,sigma=float(v.mean()),float(v.std(ddof=0))
            if sigma>0:z.append(-mu/sigma)
            episodes.append(dict(video=q['video_id'],expression=q['expression_id'],mean=mu,std=sigma,candidates=len(v)))
        result[model]=dict(model=model,in_domain_target=target,coverage='complete',source_candidate_rows=len(x),
            positive_rows=int((x>0).sum()),emission_rate=float(np.mean(x>0)),mean=float(np.mean(x,dtype=np.float64)),
            z0_offline=float(np.mean(z)),z0_definition='unweighted mean_q[-mean_q/std_q], ddof=0, std>0; same as v2',
            score_path=records[model]['path'],score_sha256=records[model]['sha256'],queries=episodes,
            GT_reads=0,statistics_from_source_model_own_in_domain_evaluation_set=True)
    return result


def generate(combos):
    require_plan();access=guard();begin=time.perf_counter()
    datasets={t:load_target(t) for t in TARGETS};stats=source_stats(datasets)
    out=RUN/'predictions';out.mkdir(parents=True,exist_ok=True)
    audit_dir=RUN/'calibrator_audit';audit_dir.mkdir(exist_ok=True)
    isolation=sandbox([], [out,PUBLIC,audit_dir])
    atomic_json(PUBLIC/'source_statistics.json',stats)
    results=[]
    for combo,model,target in PAIRS:
        if combo not in combos:continue
        meta,arrays,groups,_=datasets[target];x=arrays[model];s=stats[model]
        masks={name:np.zeros(len(x),dtype=bool) for name in CALIBRATORS}
        masks['B1_SRC_RATE_GLOBAL']=top_budget(x,round_budget(s['emission_rate'],len(x)))[0]
        masks['B3_MEAN_GLOBAL']=x-float(np.mean(x,dtype=np.float64))+s['mean']>0
        for q in meta['episodes']:
            a,b=q['start'],q['stop'];v=np.asarray(x[a:b],dtype=np.float64)
            masks['B1_SRC_RATE_QUERY'][a:b]=top_budget(v,round_budget(s['emission_rate'],len(v)))[0]
            mu,sigma=float(v.mean()),float(v.std(ddof=0))
            masks['B2_Z_OFFLINE'][a:b]=(v-mu)/sigma>=s['z0_offline'] if sigma>0 else v>0
            masks['B3_MEAN_QUERY'][a:b]=v-mu+s['mean']>0
            ids=np.asarray(meta['videos'][q['video_id']]['observations'])[:,1]
            unique,inverse=np.unique(ids,return_inverse=True)
            means=np.bincount(inverse,weights=v)/np.bincount(inverse)
            masks['C1_TRACK_MEAN'][a:b]=means[inverse]>0
        methods=[]
        for name,mask in masks.items():
            methods.append(seal(combo,name,mask,meta))
        results.append(dict(combo=combo,source=model,target=target,methods=[s['method'] for s in methods],
                            source_statistics_model=model,GT_open_attempts=0))
    record=dict(status='ALL_REQUESTED_GT_FREE_CALIBRATORS_SEALED',combos=results,GT_open_attempts=access,
                isolation=isolation,CPU_wall_seconds=time.perf_counter()-begin,
                source_statistics_path=str(PUBLIC/'source_statistics.json'),source_statistics_sha256=file_sha(PUBLIC/'source_statistics.json'))
    if access:raise ValueError('GT opened in GT-free calibrator')
    atomic_json(audit_dir/('_'.join(combos)+'.json'),record)
    print(json.dumps(dict(event='DECOMPOSE_CALIBRATORS_DONE',combos=combos,seconds=record['CPU_wall_seconds'],GT_open_attempts=0)),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--combos',nargs='+',choices=[r[0] for r in PAIRS],required=True);a=p.parse_args()
    generate(a.combos)
