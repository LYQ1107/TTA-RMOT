"""Unlabelled finite action pools for a new residual-learning RSI candidate.

Never load the prepaid response library here. Only the existing broker may
reveal a requested teacher response. The pool is the declared four-policy
action union, not every possible dense RMOT observation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX

OUT=SIX/'residual_rsi_inputs_20261125'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
SOURCES=('scripts/a100_residual_rsi_inputs.py','flex_uda/active/native_binary_inputs.py',
 'scripts/a100_observation_design_collect.py','scripts/a100_observation_design_prepare.py',
 'scripts/a100_observation_teacher_pilot.py','flex_uda/active/full_appearance_learning_inputs.py',
 'flex_uda/active/appearance_learning_inputs.py','scripts/a100_dino_fullsix_collect.py',
 'scripts/a100_dino_appearance_collect.py','flex_uda/active/joint_observation_inputs.py',
 'flex_uda/active/observation_learning_inputs.py','flex_uda/active/factor_rsi_learning.py',
 'scripts/a100_observation_local_cache.py','flex_uda/a100/mechanism.py','flex_uda/a100/runtime.py')


def prepare():
    from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
    guard=generation_guard();assert not PLAN.exists()
    full=SIX/'fullsix_learning_common_20261125/completion.json'
    assert json.loads(full.read_text())['main_video_units']==148
    cases=[dict(pair=p,target=PAIRS[p][1],video=sorted(metadata(PAIRS[p][1])['videos'])[0]) for p in ('C1','C2','C4')]
    report=dict(status='REGISTERED_UNLABELED_FINITE_ACTION_POOLS_FOR_RESIDUAL_LEARNING',cases=cases,
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},fullsix_learning_inputs_sha256=file_sha(full),
        definition='Existing DESIGN/YT_PRIOR/UNIFORM and other registered policy request union, deduplicated by exact request key; no unrequested teacher value is read.',
        support_limit='Any risk identity is only for this finite action pool and its declared query weighting, not outside-library dense RMOT or human semantic truth.',
        features='Same9352D for all future learners, original5508D retained plus complete shared image-only descriptors.',
        auxiliary='Separate unchanged100target-collection teacher events from fullsix_learning_common; all comparators get the same auxiliary information.',
        no_teacher_responses_read_for_action_pool_selection=True,
        no_neural_training_RMOT_predictions_or_GT=True,all_data_DEV=True,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,report);atomic_json(PUBLIC/'residual_rsi_inputs_registration_20261125.json',report)
    print(json.dumps(dict(status=report['status'],cases=3)),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    return plan


def worker(case):
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.native_binary_inputs import inventory
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.full_appearance_learning_inputs import augment_paid
    plan=registration();guard=generation_guard();use_verified_local_features()
    pair,video,target=case['pair'],case['video'],case['target'];folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
    final=folder/'completion.json'
    if final.exists():
        saved=json.loads(final.read_text());assert saved['registration_sha256']==file_sha(PLAN)
        for leaf,sha in saved['output_sha256'].items():assert file_sha(folder/leaf)==sha
        return saved
    started=time.perf_counter();actions,quotas=inventory(pair,video);order=sorted(actions)
    observation=JointObservationVideo(pair,video);inputs=Inputs(observation)
    records=[];keys=[];query=[]
    for uid in order:
        action=actions[uid];r=action['case'];q,t=action['local_query'],action['local_track']
        assert observation.queries[q]['expression_id']==r['query_id']
        assert int(observation.tracks[t])==int(r['track_id'])
        records.append(dict(local_query=q,local_track=t,track_id=int(r['track_id']),frame_id=int(r['frame_id'])))
        keys.append([video,r['query_id'],int(r['track_id']),int(r['frame_id'])]);query.append(q)
    a=inputs.arrays(records)
    base=np.column_stack((a['bags'].reshape(len(order),-1),a['present'],a['current'],a['coarse'])).astype('float32')
    features=augment_paid(base,keys,target)
    assert features.shape==(len(order),9352) and np.array_equal(features[:,:5508],base)
    assert set(query)==set(quotas) and min(quotas.values())>=1
    np.save(folder/'features.npy',features);np.save(folder/'query.npy',np.asarray(query,dtype='int64'))
    atomic_json(folder/'actions.json',dict(order=order,actions=actions,quotas=quotas,physical_row_keys=keys))
    result=dict(pair=pair,target=target,video=video,pool_actions=len(order),logical_budget=sum(quotas.values()),
        queries=len(quotas),quota_by_query=quotas,input_dimension=9352,
        all_query_strata_have_actions_and_nonzero_fixed_budget=True,
        original5508_coordinates_byte_equal=True,teacher_answers_or_GT_read=0,
        new_teacher_or_neural_updates=0,wall_seconds=time.perf_counter()-started,
        registration_sha256=file_sha(PLAN),output_sha256={n:file_sha(folder/n) for n in ('features.npy','query.npy','actions.json')})
    assert not guard
    atomic_json(final,result);print(json.dumps(dict(status='UNLABELED_RESIDUAL_POOL_READY',**result)),flush=True)
    return result


def main():
    plan=registration();started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:reports=list(pool.map(worker,plan['cases']))
    result=dict(status='COMPLETE_UNLABELED_RESIDUAL_RSI_POOLS',cases=reports,wall_seconds=time.perf_counter()-started,
        no_new_teacher_answers_training_RMOT_predictions_or_GT=True,all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result);atomic_json(PUBLIC/'residual_rsi_inputs_completion_20261125.json',result)
    print(json.dumps(dict(status=result['status'],cases=3)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:main()
