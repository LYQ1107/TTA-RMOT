"""Prepare lazy all-frame action support for all6experimental settings.

Failure-driven next input direction, NOT a new trained model or revived
queue. Old policies can only ask3source frames/track. Store full metadata
without a query-expanded feature tensor, teacher values or target GT.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX
from scripts.a100_observation_design_prepare import OUT as OLD_REQUESTS,POLICIES

OUT=SIX/'full_temporal_action_support_20261126'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
SOURCES=('scripts/a100_full_temporal_support.py','scripts/a100_observation_design_prepare.py',
    'flex_uda/active/acquisition.py','flex_uda/a100/mechanism.py','flex_uda/a100/runtime.py')


def request_for(doc,observations,index):
    """Full original query, fixed requested view, any provided observation."""
    n=len(observations);q,oi=divmod(int(index),n)
    query=doc['queries'][q];frame,track=map(int,observations[oi])
    assert 0<=q<len(doc['queries'])
    return dict(target=doc['target'],video_id=doc['video'],query_id=query['query_id'],
        query_text=query['query_text'],local_query=q,track_id=track,frame_id=frame,
        view=query['view'],observation_row=oi,allocation_kind='FULL_OBSERVED_TEMPORAL_SUPPORT')


def worker(case):
    from flex_uda.a100.mechanism import metadata,generation_guard
    from flex_uda.active.acquisition import route
    guard=generation_guard();pair,target,video=case['pair'],case['target'],case['video']
    md=metadata(target);obs=np.asarray(md['videos'][video]['observations'],dtype='int64')
    queries=[q for q in md['episodes'] if q['video_id']==video]
    prior=json.loads((OLD_REQUESTS/pair/video/'completion.json').read_text())
    quota=prior['query_quotas'];assert len(queries)==len(quota) and len(obs)>0
    old=set();lookup={tuple(map(int,row)):i for i,row in enumerate(obs)}
    old_frames={};request_sha={}
    for policy in POLICIES:
        p=OLD_REQUESTS/pair/video/(policy+'_requests.json');request_sha[policy]=file_sha(p)
        requests=json.loads(p.read_text())['requests']
        for r in requests:
            q=int(r['local_query']);assert r['query_id']==queries[q]['expression_id'] and r['query_text']==queries[q]['query_text']
            oi=lookup[int(r['frame_id']),int(r['track_id'])]
            old.add(q*len(obs)+oi)
            old_frames.setdefault(int(r['track_id']),set()).add(int(r['frame_id']))
    assert max(map(len,old_frames.values()),default=0)<=3
    doc=dict(pair=pair,target=target,video=video,queries=[dict(query_id=q['expression_id'],query_text=q['query_text'],
        view=route(q['query_text']),logical_budget=int(quota[i])) for i,q in enumerate(queries)],
        observations=len(obs),conceptual_actions=len(obs)*len(queries),old_unique_actions=len(old),
        definition='Allprovidedcandidateobservations forEVERYquery. Requestedview follows unchangedroute; actualteacherfallback/effectiveview must be logged by futurebroker.',
        no_target_labels_teacher_values_or_RMOT_predictions=True)
    folder=OUT/pair/video;folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'observations.npy',obs);atomic_json(folder/'queries.json',doc)
    # Exact real metadata reconstruction, not a label or probability test.
    for index in (0,doc['conceptual_actions']//2,doc['conceptual_actions']-1):
        r=request_for(doc,obs,index);q=index//len(obs)
        assert r['query_text']==queries[q]['query_text']
        assert (r['frame_id'],r['track_id']) in lookup
    pertrack={int(t):int((obs[:,1]==t).sum()) for t in np.unique(obs[:,1])}
    result=dict(pair=pair,target=target,video=video,query_count=len(queries),candidate_observations=len(obs),tracks=len(pertrack),
        conceptual_full_actions=doc['conceptual_actions'],historical_union_actions=len(old),
        positive_fixed_query_budgets=int(sum(n>0 for n in quota)),zero_fixed_query_budgets=int(sum(n==0 for n in quota)),
        logical_budget=sum(quota),maximum_full_observations_per_track=max(pertrack.values()),
        maximum_old_requested_frames_per_track=max(map(len,old_frames.values()),default=0),
        request_metadata_sha256=request_sha,
        output_sha256={n:file_sha(folder/n) for n in ('observations.npy','queries.json')},GT_open_attempts=len(guard))
    assert not guard
    atomic_json(folder/'completion.json',result)
    return result


def main():
    from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
    guard=generation_guard();started=time.perf_counter();cases=[]
    for pair,(_,target) in PAIRS.items():
        for video in sorted(metadata(target)['videos']):cases.append(dict(pair=pair,target=target,video=video))
    assert len(cases)==148
    plan=dict(status='REGISTERED_ALL6_FULL_TEMPORAL_SUPPORT_PREPARATION',cases=cases,
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},
        motivation='Prior postsealC2diagnosis has17positivequeries with no correctcandidate in oldpolicyunion. Sourcecode and metadata show only3sourceframes/track. This preparation removes that structural zero-support restriction, without using labels to selectqueries/frames.',
        scope='Only fixedprovidedcandidates and actualobservedframes, not missedfront-endobjects. No claimed taskaccuracy, teachertruth or solvedcross-domainRMOT.',
        sampling_hypothesis='Future.2uniformoverALLobservations mixed with learnedstudent-errorfocus has positivewithinqueryframe support. This sampler/freshteacherbroker is NOT implemented here.',
        mathematical_limit='For fixedtargetsubset ofmoutofNactions andBwithreplacementdraws withconditionaluniformfloor epsilon, P(discover)>=1-(1-epsilon*m/N)^B. m is unknown toadaptation and never supplied. No semantic,HOTA or largegain guarantee.',
        scaling='Lazyq*Nobs+obsindex; storefullqueries andNx2observations, no terabytequery-expanded9352tensors. Independentcontentalignment can averagephysicalROIs once because itsrepresentation isquery-independent.',
        all_source_questions_and_fixed_quotas_preserved=True,no_teacher_value_prediction_GT_or_neural_fit=True,
        no_old_queue_launch_or_new_primary_selection=True,CPU_workers=3,all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'registration.json',plan)
    with ProcessPoolExecutor(max_workers=3) as pool:records=list(pool.map(worker,cases))
    summaries={}
    for pair,(_,target) in PAIRS.items():
        r=[x for x in records if x['pair']==pair]
        summaries[pair]=dict(target=target,videos=len(r),queries=sum(x['query_count'] for x in r),
            full_conceptual_actions=sum(x['conceptual_full_actions'] for x in r),historical_union_actions=sum(x['historical_union_actions'] for x in r),
            logical_budget=sum(x['logical_budget'] for x in r),zero_budget_queries=sum(x['zero_fixed_query_budgets'] for x in r),
            maximum_old_frames_per_track=max(x['maximum_old_requested_frames_per_track'] for x in r),
            maximum_full_observations_per_track=max(x['maximum_full_observations_per_track'] for x in r))
        print(json.dumps(dict(status='FULL_TEMPORAL_METADATA_SETTING_COMPLETE',pair=pair,**summaries[pair])),flush=True)
    assert not guard
    result=dict(status='COMPLETE_ALL6_LAZY_FULL_TEMPORAL_ACTION_METADATA',settings=summaries,video_units=len(records),
        total_conceptual_actions=sum(r['conceptual_full_actions'] for r in records),wall_seconds=time.perf_counter()-started,
        detailed_video_units_local_not_Git=True,raw_9352_tensors_materialized=0,new_GPU_teacher_or_neural_calls=0,
        GT_open_attempts=0,source_sha256=plan['source_sha256'],
        no_fresh_teacher_broker_sampler_learning_or_quality_result_yet=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',dict(result,video_units_records=records))
    atomic_json(PUBLIC/'full_temporal_support_preparation_20261126.json',result)
    print(json.dumps(dict(status=result['status'],videos=148,actions=result['total_conceptual_actions'])),flush=True)


if __name__=='__main__':main()
