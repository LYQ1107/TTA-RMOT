"""New real teacher observations on a union of GT-blind acquisition policies.

Sharing this union is an experimental economy, not a live wall-time claim.
Each method can only consume its own budgeted subset of returned answers.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
BEGIN=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from scripts.a100_visual_classical_prepare import OUT


def selected_pairs(inputs, orders, budget):
    nq=len(inputs['prior']);quotas=np.full(nq,budget//nq,dtype=int);quotas[:budget%nq]+=1
    return [(q,int(t)) for q,quota in enumerate(quotas) for t in orders[q][:quota]]


def collect(pair):
    opened=generation_guard();model,target=PAIRS[pair];m=metadata(target)
    source=OUT/'inputs'/pair;dest=OUT/'evidence'/pair;dest.mkdir(parents=True,exist_ok=True)
    assert (source/'completion.json').is_file()
    if (dest/'completion.json').is_file():
        print(json.dumps(dict(event='EVIDENCE_ALREADY_COMPLETE',pair=pair)),flush=True);return
    plan={};union={};byvideo=defaultdict(list)
    for video in sorted(m['videos']):
        folder=source/video;inputs=np.load(folder/'inputs.npz')
        receipt=json.loads((folder/'receipt.json').read_text());orders=json.loads((folder/'orders.json').read_text())
        queries=[q for q in m['episodes'] if q['video_id']==video]
        plan[video]={}
        for method in ('PRIOR','UNIFORM','UNCERTAINTY','COVER'):
            pairs=selected_pairs(inputs,orders[method+'__0'],receipt['logical_budget'])
            plan[video][method]=pairs
            assert len(pairs)==receipt['logical_budget']
            for q,t in pairs:
                key=(video,q,t)
                if key in union:continue
                request=dict(target=target,video_id=video,query_id=queries[q]['expression_id'],query_text=queries[q]['query_text'],
                    track_id=int(inputs['tracks'][t]),frame_id=int(inputs['anchors'][t]),view='CROP')
                union[key]=request;byvideo[video].append(key)
    atomic_json(dest/'acquisition_plan.json',dict(pair=pair,seed=0,positions=plan,union_requests=len(union),
        GT_open_attempts=len(opened),preregistration_sha256=file_sha(OUT/'preregistration_v1.json'),
        no_unqueried_answers_visible_to_method=True))
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    specs=[video_spec(target,v,b) for v,b in m['videos'].items()]
    print(json.dumps(dict(event='TEACHER_LOAD_BEGIN',pair=pair,physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),union_requests=len(union))),flush=True)
    start=time.perf_counter();client=teacher(specs,dest/'raw.sqlite');load=time.perf_counter()-start
    saved={};response_path=dest/'responses.jsonl'
    if response_path.is_file():
        saved={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,response_path.read_text().splitlines())}
    start=time.perf_counter()
    for video,keys in byvideo.items():
        for a in range(0,len(keys),8):
            selected=[key for key in keys[a:a+8] if key not in saved]
            if not selected:continue
            answers=client.ask_many([union[key] for key in selected])
            for key,answer in zip(selected,answers):
                row=dict(answer,local_query=key[1],local_track=key[2]);saved[key]=row;append_json(response_path,row)
        print(json.dumps(dict(event='VISUAL_TEACHER_VIDEO',pair=pair,video=video,completed=len(saved),total=len(union))),flush=True)
    service=time.perf_counter()-start;start=time.perf_counter();cost=client.close();seal=time.perf_counter()-start
    assert len(saved)==len(union) and not opened
    atomic_json(dest/'completion.json',dict(status='FROZEN_NEW_EVIDENCE_COMPLETE',pair=pair,seed=0,
        responses=len(saved),responses_sha256=file_sha(response_path),GT_open_attempts=0,
        teacher=cost,teacher_load_wall_seconds=load,teacher_service_wall_seconds=service,
        final_weight_seal_wall_seconds=seal,process_wall_seconds=time.perf_counter()-BEGIN,
        shared_pool_preparation_not_deployment_cost=True,
        per_policy_logical_calls={method:sum(len(p[method]) for p in plan.values()) for method in next(iter(plan.values()))}))
    print(json.dumps(dict(event='VISUAL_TEACHER_COMPLETE',pair=pair,responses=len(saved),wall_seconds=time.perf_counter()-BEGIN)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();collect(a.pair)
