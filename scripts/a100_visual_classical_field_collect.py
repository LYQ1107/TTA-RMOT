"""Adaptive versus response-blind image-graph search, actual frozen Qwen broker."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.graph_search import GraphSearch,select_batch
from flex_uda.active.color_cohort import augmented_features,part_mask
from scripts.a100_visual_classical_prepare import OUT


def make_fields(pair,video,m,adaptive):
    with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:features=a['features'];prior=a['prior'];tracks=a['tracks'];anchors=a['anchors'];lengths=a['lengths']
    with np.load(OUT/'color_features'/pair/video/'features.npz') as a:colour=a['colours']
    queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
    geometry,_,_=augmented_features(features,colour,texts);fields=[]
    for qi,text in enumerate(texts):
        selected=part_mask(text)>0;visible=np.all(colour[:,selected].sum(2)>0,axis=1) if selected.any() else None
        fields.append(GraphSearch(prior[qi],geometry[qi],visible,adaptive=adaptive))
    eligible=np.flatnonzero(lengths>=3);budget=int(np.floor(.3*len(queries)*len(eligible)))
    return fields,eligible,budget,tracks,anchors,queries


def collect(pair):
    guard=generation_guard();started=time.perf_counter();_,target=PAIRS[pair];m=metadata(target);root=OUT/'field_evidence'/pair
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);static={};planning_seconds=0.
    for video in sorted(m['videos']):
        start=time.perf_counter();fields,eligible,budget,*_=make_fields(pair,video,m,False)
        static[video]=select_batch(fields,eligible,budget);planning_seconds+=time.perf_counter()-start
    static_path=root/'static_questions_sealed_before_teacher.json'
    plan=dict(pair=pair,policy='FIELD_STATIC',positions=static,GT_open_attempts=len(guard),
        no_teacher_feedback_in_plan=True,source_precision=.1,straddle_multiplier=2.,minimum_teacher_precision=1.,
        variance_only_fantasy_increment=.9,initial_query_floor=1,source_prior_logit_cap=2.)
    if static_path.is_file():assert json.loads(static_path.read_text())==json.loads(json.dumps(plan))
    else:atomic_json(static_path,plan)
    # A copied local frozen cache is an opaque question-answer broker. Neither
    # selection function receives this cache or any unrequested response.
    cache_path=root/'raw.sqlite';cache_source=OUT/'evidence'/pair/'raw.sqlite'
    if not cache_path.is_file():shutil.copyfile(cache_source,cache_path)
    from scripts.active_run import video_spec
    from flex_uda.a100.runtime import teacher
    start=time.perf_counter();client=teacher([video_spec(target,v,b) for v,b in m['videos'].items()],cache_path);load=time.perf_counter()-start
    receipts=[];service_begin=time.perf_counter()
    for video in sorted(m['videos']):
        for policy,adaptive in [('FIELD_STATIC',False),('FIELD',True)]:
            shard=OUT/'causal_teacher'/pair/policy/(video+'.json')
            if shard.is_file():
                old=json.loads(shard.read_text());receipts.append(dict(video=video,policy=policy,logical_calls=len(old['answers']),resumed_sealed_video=True));continue
            fields,eligible,budget,tracks,anchors,queries=make_fields(pair,video,m,adaptive);answers=[];acquisition_seconds=0.
            for offset in range(0,budget,8):
                start=time.perf_counter()
                chosen=select_batch(fields,eligible,min(8,budget-offset)) if adaptive else static[video][offset:offset+8]
                acquisition_seconds+=time.perf_counter()-start;requests=[]
                for item in chosen:
                    qi,ti=item['local_query'],item['local_track'];q=queries[qi]
                    requests.append(dict(target=target,video_id=video,query_id=q['expression_id'],query_text=q['query_text'],track_id=int(tracks[ti]),
                        frame_id=int(anchors[ti]),view='CROP'))
                append_json(root/'request_journal.jsonl',dict(stage='SELECTED_BEFORE_ANSWERS',video=video,policy=policy,offset=offset,positions=chosen,requests=requests))
                returned=client.ask_many(requests)
                for item,value in zip(chosen,returned):
                    qi,ti=item['local_query'],item['local_track'];fields[qi].update(ti,value['yes_probability'])
                    row=dict(value,local_query=qi,local_track=ti,acquisition_score=item['acquisition_score'])
                    answers.append(row);append_json(root/'responses.jsonl',dict(row,policy=policy))
                append_json(root/'request_journal.jsonl',dict(stage='ANSWERS_RETURNED',video=video,policy=policy,offset=offset,
                    request_ids=[v['request_id'] for v in returned]))
            assert len(answers)==budget and len({(r['local_query'],r['local_track']) for r in answers})==budget
            counts=np.bincount([r['local_query'] for r in answers],minlength=len(queries))
            if budget>=len(queries):assert np.all(counts>=1)
            atomic_json(shard,dict(pair=pair,policy=policy,video=video,answers=answers,logical_calls=len(answers),target_GT_used=False,
                only_this_video_and_policy=True,teacher_feedback_used_in_selection=adaptive,
                static_plan_sha256=file_sha(static_path),per_query_calls=counts.tolist()))
            receipts.append(dict(video=video,policy=policy,logical_calls=budget,acquisition_CPU_seconds=acquisition_seconds,per_query_calls=counts.tolist()))
            print(json.dumps(dict(event='GRAPH_FIELD_TEACHER_VIDEO',pair=pair,video=video,policy=policy,logical_calls=budget,
                cache_hits=sum(r['cache_hit'] for r in answers),per_query_calls=counts.tolist())),flush=True)
    service=time.perf_counter()-service_begin;start=time.perf_counter();cost=client.close();seal=time.perf_counter()-start
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_REAL_FIELD_TEACHER_EVIDENCE',pair=pair,GT_open_attempts=0,
        teacher=cost,teacher_load_wall_seconds=load,teacher_service_wall_seconds=service,teacher_final_weight_seal_seconds=seal,
        static_image_only_planning_seconds=planning_seconds,video_receipts=receipts,
        per_policy_logical_calls={p:sum(r['logical_calls'] for r in receipts if r['policy']==p) for p in ('FIELD','FIELD_STATIC')},
        process_wall_seconds=time.perf_counter()-started,selection_sees_only_requested_answers=True,
        copied_cache_is_experimental_evidence_reuse_not_a_cold_deployment_claim=True,
        source_cache=str(cache_source),actual_teacher_forwards_for_missing_questions=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();collect(a.pair)
