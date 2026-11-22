"""Prepare six-setting active-observation manifests without teacher answers.

This is a future evidence experiment's input preparation, not a quality claim
or permission to consume old QA values. Policies share frozen image/query
features, exact video/query budgets, anchor frames and expression routing.
Only the historical actual request count is used to fix each video's budget.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.acquisition import route
from flex_uda.active.observation_design import plan
from flex_uda.active.relational_transport import horizontal_direction
from flex_uda.active.relational_features import geometry
from scripts.a100_six_direction import OUT as SIX,PUBLIC,source
from scripts.a100_relation_transport import coarse_features

OUT=SIX/'active_observation_design/preparation'
STATE=OUT/'supervisor/state.json'
POLICIES=('DESIGN','COVERAGE_ONLY','PRIOR','UNIFORM')


def seed(video,query):return int(hashlib.sha256((video+':'+query+':observation-design').encode()).hexdigest()[:8],16)


def simple_order(kind,prior,samples,anchors,rng,budget):
    n=len(samples)
    order=sorted(range(n),key=lambda t:(-float(prior[t]),t)) if kind=='PRIOR' else list(map(int,rng.permutation(n)))
    actions=[dict(track_index=t,frame=int(anchors[t]),kind='OBJECT_ANCHOR') for t in order]
    # Strong object-uniform baseline: take distinct objects before revisiting
    # them. Extra paid states, if needed, remain actual source observations.
    extras=[dict(track_index=t,frame=int(f),kind='EXTRA_STATE') for t in order for f in sorted(set(map(int,samples[t]))-{int(anchors[t])})]
    if kind=='UNIFORM':extras=[extras[i] for i in rng.permutation(len(extras))]
    result=(actions+extras)[:budget];assert len(result)==budget
    return result


def main():
    started=time.perf_counter();guard=generation_guard();registration=json.loads((PUBLIC/'observation_design_manifest_registration.json').read_text())
    for name,sha in registration['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    results={};completed=0
    def progress(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=completed,total_videos=148,goal_status='ACTIVE'))
    progress('PREPARING_ACTIVE_OBSERVATION_MANIFESTS_NO_GT_NO_TEACHER')
    for pair,(_,target) in PAIRS.items():
        m=metadata(target);totals={p:Counter() for p in POLICIES};videos=[]
        for video,bank in sorted(m['videos'].items()):
            folder=OUT/pair/video
            if (folder/'completion.json').is_file():
                receipt=json.loads((folder/'completion.json').read_text())
                for policy in POLICIES:totals[policy].update(receipt['policies'][policy])
                videos.append(receipt);completed+=1;progress('PREPARING_ACTIVE_OBSERVATION_MANIFESTS_NO_GT_NO_TEACHER');continue
            start=time.perf_counter();obs=np.asarray(bank['observations']);coords,_,_=geometry(bank,target)
            queries=[q for q in m['episodes'] if q['video_id']==video]
            with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks,anchors=[a[k] for k in ('pcd','colours','prior','tracks','anchors')]
            raw=coarse_features(pcd,colours,prior,tracks,obs,[q['query_text'] for q in queries]);samples=[];positions=[];lengths=[]
            for track in tracks:
                ids=np.flatnonzero(obs[:,1]==track);chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
                samples.append([int(obs[i,0]) for i in chosen]);positions.append({int(obs[i,0]):float(coords[i,0]) for i in chosen});lengths.append(len(ids))
            # Do not extract probabilities, beliefs, votes or old predictions.
            historical=json.loads((history(pair)/'videos'/video/'questions.json').read_text());budget=len(historical);del historical
            nq=len(queries);quota=np.full(nq,budget//nq,int)
            quota_order=sorted(range(nq),key=lambda q:seed(video,queries[q]['expression_id']))
            quota[quota_order[:budget%nq]]+=1
            assert int(quota.sum())==budget and quota.max()<=sum(map(len,samples))
            requests={p:[] for p in POLICIES};counts={p:Counter() for p in POLICIES};view_counts={p:Counter() for p in POLICIES}
            for qi,q in enumerate(queries):
                count=int(quota[qi]);direction=horizontal_direction(q['query_text']);view=route(q['query_text'])
                for policy in POLICIES:
                    if policy in ('DESIGN','COVERAGE_ONLY'):
                        actions,_=plan(raw[qi],np.asarray(lengths,float),prior[qi],samples,anchors,positions,count,direction if policy=='DESIGN' else 0)
                    else:actions=simple_order(policy,prior[qi],samples,anchors,np.random.default_rng(seed(video,q['expression_id'])),count)
                    assert len(actions)==count and len({(a['track_index'],a['frame']) for a in actions})==count
                    counts[policy].update(logical_requests=count,query_episodes=1,zero_request_queries=int(count==0),
                        distinct_query_track_cells=len({a['track_index'] for a in actions}),
                        repeated_object_requests=count-len({a['track_index'] for a in actions}))
                    view_counts[policy][view]+=count
                    for index,action in enumerate(actions):
                        ti=action['track_index'];frame=action['frame'];assert frame in samples[ti]
                        requests[policy].append(dict(target=target,video_id=video,query_id=q['expression_id'],query_text=q['query_text'],
                            track_id=int(tracks[ti]),frame_id=int(frame),view=view,local_query=qi,local_track=ti,
                            order_within_query=index,allocation_kind=action['kind']))
            assert all(len(requests[p])==budget for p in POLICIES)
            assert all(view_counts[p]==view_counts['DESIGN'] for p in POLICIES)
            for policy in POLICIES:
                atomic_json(folder/(policy+'_requests.json'),dict(requests=requests[policy],GT_open_attempts=len(guard),teacher_answers_used=False,not_to_Git=True))
                totals[policy].update(counts[policy])
            receipt=dict(status='SEALED_UNLABELLED_OBSERVATION_REQUEST_MANIFESTS',pair=pair,video=video,queries=nq,tracks=len(tracks),
                actual_historical_request_count_as_fixed_budget=budget,query_quotas=quota.tolist(),policies={p:dict(counts[p]) for p in POLICIES},
                common_view_counts=dict(view_counts['DESIGN']),same_budget_query_quotas_views_and_source_samples=True,
                teacher_answers_used=False,manual_GT_open_attempts=len(guard),new_teacher_GPU_calls=0,wall_seconds=time.perf_counter()-start)
            atomic_json(folder/'completion.json',receipt);videos.append(receipt)
            completed+=1;progress('PREPARING_ACTIVE_OBSERVATION_MANIFESTS_NO_GT_NO_TEACHER')
            print(json.dumps(dict(event='OBSERVATION_MANIFEST_VIDEO',pair=pair,completed=len(videos),total=len(m['videos']))),flush=True)
        results[pair]=dict(source=PAIRS[pair][0],target=target,videos=len(videos),policies={p:dict(totals[p]) for p in POLICIES})
    assert not guard
    report=dict(status='SIX_SETTINGS_ACTIVE_OBSERVATION_MANIFESTS_PREPARED',pairs=results,policies=POLICIES,
        manual_GT_open_attempts=0,new_teacher_encoder_or_GPU_calls=0,neural_fits=0,new_candidate_predictions=0,
        teacher_answers_used=False,all_request_manifests_local_not_to_Git=True,
        view_routing='Common existing acquisition.route for every policy; routing itself not a new contribution',
        budgets='Historical actual call count per video only; equalized query quotas with hash tie order; same counts and views in all policies',
        temporal_scope='Current-video transductive; source samples from current video only; no future video features or teacher answers',
        interpretation='Prepared requests only. No new-policy QA, model adaptation, HOTA or deployment-cost evidence yet.',
        all_data_development=True,goal_status='ACTIVE',wall_seconds=time.perf_counter()-started)
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'observation_design_manifests.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    assert completed==148,completed
    progress('SIX_SETTINGS_ACTIVE_OBSERVATION_MANIFESTS_PREPARED')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        old=json.loads(STATE.read_text()) if STATE.is_file() else {};old.update(status='FAILED_OBSERVATION_MANIFEST_PREPARATION',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,old);raise
