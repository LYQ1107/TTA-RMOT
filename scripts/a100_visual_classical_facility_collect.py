"""Image-only scene coverage acquisition, followed by real frozen Qwen calls."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.distance import cdist
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.color_cohort import augmented_features
from scripts.a100_visual_classical_prepare import OUT


def facility_order(features,lengths,eligible):
    ids=np.asarray(eligible,dtype=int);x=np.asarray(features[ids],dtype=float)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    distance=np.maximum(cdist(x,x,'sqeuclidean'),0.);n=len(ids)
    if n<2:return ids.tolist(),1.
    np.fill_diagonal(distance,np.inf);k=min(3,n-1)
    scale=max(float(np.median(np.partition(distance,k-1,axis=1)[:,k-1])),1e-6)
    np.fill_diagonal(distance,0.);similarity=np.exp(-distance/scale)
    weight=np.sqrt(np.asarray(lengths,dtype=float)[ids]);coverage=np.zeros(n);chosen=[]
    for _ in range(n):
        gain=np.sum(weight[:,None]*np.maximum(similarity-coverage[:,None],0),axis=0)
        gain[chosen]=-np.inf;next_=int(np.argmax(gain));chosen.append(next_)
        coverage=np.maximum(coverage,similarity[:,next_])
    return ids[chosen].tolist(),scale


def collect(pair):
    guard=generation_guard();started=time.perf_counter();_,target=PAIRS[pair];m=metadata(target);root=OUT/'facility_evidence'/pair
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);plan={};requests={}
    for video,bank in sorted(m['videos'].items()):
        prepared=OUT/'color_features'/pair/video
        while not (prepared/'receipt.json').is_file():time.sleep(5)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:features=a['features'];tracks=a['tracks'];anchors=a['anchors']
        with np.load(prepared/'features.npz') as a:colours=a['colours']
        queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        combined,_,_=augmented_features(features,colours,texts)
        obs=np.asarray(bank['observations']);lengths=np.asarray([int((obs[:,1]==t).sum()) for t in tracks]);eligible=np.flatnonzero(lengths>=3)
        receipt=json.loads((OUT/'inputs'/pair/video/'receipt.json').read_text());budget=receipt['logical_budget']
        quota=np.full(len(queries),budget//len(queries),dtype=int);quota[:budget%len(queries)]+=1
        positions=[];scales=[]
        for qi,q in enumerate(queries):
            order,scale=facility_order(combined[qi],lengths,eligible);scales.append(scale)
            positions.extend((qi,int(t)) for t in order[:quota[qi]])
        assert len(positions)==budget
        plan[video]=dict(positions=positions,bandwidths=scales,budget=budget)
        requests[video]=[dict(target=target,video_id=video,query_id=queries[q]['expression_id'],query_text=queries[q]['query_text'],
            track_id=int(tracks[t]),frame_id=int(anchors[t]),view='CROP',local_query=q,local_track=t) for q,t in positions]
    plan_path=root/'image_only_plan.json'
    record=dict(pair=pair,policy='FACILITY',videos=plan,logical_budget=sum(p['budget'] for p in plan.values()),
        target_GT_used=False,teacher_feedback_used_for_selection=False,source_prior_rank_used=False,
        objective='duration-sqrt-weighted facility coverage in query-specific normalized PCD + real part-colour geometry',
        seed=0,GT_open_attempts=len(guard))
    if plan_path.is_file():assert json.loads(plan_path.read_text())==json.loads(json.dumps(record))
    else:atomic_json(plan_path,record)
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    specs=[video_spec(target,v,b) for v,b in m['videos'].items()]
    print(json.dumps(dict(event='FACILITY_TEACHER_LOAD_BEGIN',pair=pair,logical_budget=record['logical_budget'])),flush=True)
    t=time.perf_counter();client=teacher(specs,root/'raw.sqlite');load=time.perf_counter()-t
    response_path=root/'responses.jsonl';saved={}
    if response_path.is_file():saved={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,response_path.read_text().splitlines())}
    t=time.perf_counter()
    for video,rows in requests.items():
        for offset in range(0,len(rows),8):
            selected=[r for r in rows[offset:offset+8] if (video,r['local_query'],r['local_track']) not in saved]
            if not selected:continue
            answers=client.ask_many([{k:v for k,v in r.items() if k not in ('local_query','local_track')} for r in selected])
            for request,answer in zip(selected,answers):
                row=dict(answer,local_query=request['local_query'],local_track=request['local_track'])
                saved[video,row['local_query'],row['local_track']]=row;append_json(response_path,row)
        visible=[saved[video,r['local_query'],r['local_track']] for r in rows]
        atomic_json(OUT/'causal_teacher'/pair/'FACILITY'/(video+'.json'),dict(pair=pair,policy='FACILITY',video=video,
            answers=visible,logical_calls=len(visible),target_GT_used=False,only_this_video_and_policy=True,
            acquisition_plan_sha256=file_sha(plan_path)))
        print(json.dumps(dict(event='FACILITY_TEACHER_VIDEO',pair=pair,video=video,completed=len(saved),total=record['logical_budget'])),flush=True)
    service=time.perf_counter()-t;t=time.perf_counter();receipt=client.close();seal=time.perf_counter()-t
    assert len(saved)==record['logical_budget'] and not guard
    atomic_json(root/'completion.json',dict(status='FACILITY_REAL_FROZEN_EVIDENCE_COMPLETE',pair=pair,responses=len(saved),
        teacher=receipt,teacher_load_wall_seconds=load,teacher_service_wall_seconds=service,teacher_final_seal_wall_seconds=seal,
        process_wall_seconds=time.perf_counter()-started,GT_open_attempts=0,policy='FACILITY',
        responses_sha256=file_sha(response_path),experimental_collection_not_live_deployment_wall=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();collect(a.pair)
