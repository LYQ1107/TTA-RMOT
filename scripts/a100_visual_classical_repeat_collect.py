"""New acquisition seeds; reuse only exact already-frozen answers, ask missing ones."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from scripts.a100_visual_classical_prepare import OUT


def collect(pair):
    guard=generation_guard();begin=time.perf_counter();_,target=PAIRS[pair];m=metadata(target);root=OUT/'repeat_evidence'/pair
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True)
    old=OUT/'evidence'/pair/'responses.jsonl';existing={};origin={}
    for row in map(json.loads,old.read_text().splitlines()):
        key=row['video_id'],row['local_query'],row['local_track'];existing[key]=row;origin[key]='exact_frozen_seed0_union'
    new=root/'responses.jsonl'
    if new.is_file():
        for row in map(json.loads,new.read_text().splitlines()):
            key=row['video_id'],row['local_query'],row['local_track'];existing[key]=row;origin[key]='new_repeat_request'
    requests={};plans={}
    for seed in (17,42):
        plans[str(seed)]={}
        for video in sorted(m['videos']):
            path=OUT/'inputs'/pair/video
            with np.load(path/'inputs.npz') as a:tracks=a['tracks'];anchors=a['anchors']
            orders=json.loads((path/'orders.json').read_text())['COVER__'+str(seed)]
            budget=json.loads((path/'receipt.json').read_text())['logical_budget']
            queries=[q for q in m['episodes'] if q['video_id']==video];quota=np.full(len(queries),budget//len(queries));quota[:budget%len(queries)]+=1
            chosen=[(qi,int(ti)) for qi in range(len(queries)) for ti in orders[qi][:quota[qi]]]
            assert len(chosen)==budget;plans[str(seed)][video]=chosen
            for qi,ti in chosen:
                key=video,qi,ti
                requests[key]=dict(target=target,video_id=video,query_id=queries[qi]['expression_id'],query_text=queries[qi]['query_text'],
                    track_id=int(tracks[ti]),frame_id=int(anchors[ti]),view='CROP',local_query=qi,local_track=ti)
    record=dict(pair=pair,seeds=[17,42],policy='COVER',positions=plans,unique_positions=len(requests),
        logical_calls_per_seed={s:sum(map(len,p.values())) for s,p in plans.items()},
        original_frozen_response_pool_sha256=file_sha(old),reuse_only_identical_video_query_track_and_anchor=True,
        no_teacher_feedback_or_GT_used_for_selection=True,GT_open_attempts=len(guard))
    plan_path=root/'questions_sealed_before_teacher.json'
    if plan_path.is_file():assert json.loads(plan_path.read_text())==json.loads(json.dumps(record))
    else:atomic_json(plan_path,record)
    missing=[key for key in requests if key not in existing];load=service=seal=0.;receipt=None
    print(json.dumps(dict(event='REPEAT_TEACHER_PLAN',pair=pair,unique_positions=len(requests),exact_reused=len(requests)-len(missing),new_positions=len(missing))),flush=True)
    if missing:
        from scripts.active_run import video_spec
        from flex_uda.a100.runtime import teacher
        start=time.perf_counter();client=teacher([video_spec(target,v,b) for v,b in m['videos'].items()],root/'raw.sqlite');load=time.perf_counter()-start
        start=time.perf_counter()
        for offset in range(0,len(missing),8):
            keys=missing[offset:offset+8];batch=[requests[k] for k in keys]
            answers=client.ask_many([{k:v for k,v in row.items() if k not in ('local_query','local_track')} for row in batch])
            for key,row in zip(keys,answers):
                value=dict(row,local_query=key[1],local_track=key[2]);existing[key]=value;origin[key]='new_repeat_request';append_json(new,value)
            if offset%256==0:print(json.dumps(dict(event='REPEAT_REAL_TEACHER_PROGRESS',pair=pair,completed=offset+len(keys),total=len(missing))),flush=True)
        service=time.perf_counter()-start;start=time.perf_counter();receipt=client.close();seal=time.perf_counter()-start
    reused={}
    for seed,plan in plans.items():
        reused[seed]=0
        for video,chosen in plan.items():
            rows=[]
            for qi,ti in chosen:
                key=video,qi,ti;row=existing[key];request=requests[key]
                assert row['frame_id']==request['frame_id'] and row['track_id']==request['track_id'] and row['query_id']==request['query_id']
                rows.append(dict(row,evidence_origin=origin[key]));reused[seed]+=origin[key]=='exact_frozen_seed0_union'
            atomic_json(OUT/'causal_teacher'/pair/('COVER_S'+seed)/(video+'.json'),dict(pair=pair,policy='COVER',seed=int(seed),video=video,
                answers=rows,logical_calls=len(rows),target_GT_used=False,only_this_video_policy_seed=True,
                acquisition_plan_sha256=file_sha(plan_path)))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_REPEAT_EVIDENCE',pair=pair,seeds=[17,42],GT_open_attempts=0,
        logical_calls_per_seed=record['logical_calls_per_seed'],existing_response_reuse_per_seed=reused,
        new_union_positions=len(missing),teacher=receipt,teacher_load_seconds=load,teacher_service_seconds=service,
        teacher_final_weight_seal_seconds=seal,process_wall_seconds=time.perf_counter()-begin,
        exact_frozen_answer_reuse_is_experimental_preparation_not_live_cache_speedup=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();collect(a.pair)
