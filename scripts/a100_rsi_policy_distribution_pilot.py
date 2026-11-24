"""Two-video, teacher-only test of observable RSI policy changes; seed0.

No RMOT output/GT. Existing six-setting SGD/FROZEN prefixes are immutable
comparators; this process trains two independent new policies on CPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_native_binary_six_setting import OUT as PARENT,PLAN as PARENT_PLAN

OUT=PARENT.parent/'rsi_policy_distribution_20261124'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
PLAN=PUBLIC/'rsi_policy_distribution_registration_20261124.json'
VIDEOS=('dancetrack0004','dancetrack0005')
BRANCHES=('INFO_GATED_SGD','INFO_GATED_TRUST')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_plan():
    plan=json.loads(PLAN.read_text())
    assert plan['neural_seeds']==[0] and plan['videos']==list(VIDEOS)
    assert plan['parent_registration_sha256']==sha(PARENT_PLAN)
    for name,expected in plan['source_sha256'].items():assert sha(ROOT/name)==expected,name
    for name,expected in json.loads(PARENT_PLAN.read_text())['source_sha256'].items():assert sha(ROOT/name)==expected,name
    return plan


def worker(branch):
    check_plan()
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.native_binary_inputs import inventory,NativeVisual,prefix_arrays
    from flex_uda.active import native_binary_curriculum as curriculum
    from flex_uda.active.native_binary_fit import fit_prefix
    from flex_uda.active.rsi_policy_distribution import recorded_selection,gated_update,SGD_UPDATE,trust_update
    torch.set_num_threads(2);guard=generation_guard();use_verified_local_features()
    original_credit=curriculum.progress_reward;latest_credit={}
    def credit(*args):
        latest_credit.clear();latest_credit.update(original_credit(*args));return dict(latest_credit)
    def update(*args,**kwargs):
        return gated_update(SGD_UPDATE if branch=='INFO_GATED_SGD' else trust_update,latest_credit,*args,**kwargs)
    curriculum.select_round=recorded_selection;curriculum.progress_reward=credit;curriculum.update_challenger=update
    total=np.zeros(1101);square=np.zeros(1101);number=0;past=[];previous=None;actor=None;rows=[]
    root=OUT/branch;root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    for video in VIDEOS:
        v=JointObservationVideo('C1',video);actions,quotas=inventory('C1',video);visual=NativeVisual(v,actions)
        total+=visual.norm_sum;square+=visual.norm_square;number+=visual.norm_rows
        mean=total/number;std=np.sqrt(np.maximum(square/number-mean**2,0)).clip(min=.01)
        local=root/video;local.mkdir(parents=True,exist_ok=True)
        def callback(stage,state):
            print(json.dumps(dict(branch=branch,video=video,round=stage,reward=state['rounds'][-1]['credit']['reward'],
                update=state['rounds'][-1]['actor_update'])),flush=True)
        result=curriculum.observe_and_solve(visual,actions,quotas,mean,std,past=past,
            previous=previous,actor_state=actor,learned=True,device='cpu',callback=callback)
        previous,actor=result['checkpoint'],result['actor']
        torch.save(dict(solver=previous,actor=actor),local/'closed.pt')
        atomic_json(local/'paid.json',dict(paid=result['paid'],GT_open_attempts=0))
        atomic_json(local/'sequence.json',dict(sequence=result['sequence']))
        record=result['record'];all_records=[*past,record]
        ordinary={}
        for name,recipe in [('JOINT',dict(family='pair',risk='joint')),('COARSE_UNIT_SOFT',dict(family='coarse'))]:
            _,_,_,receipt=fit_prefix(prefix_arrays(all_records,mean,std),recipe,mean,std,device='cpu')
            assert receipt['teacher_fit_gate_passed'];ordinary[name]=receipt
        report=dict(result['report'],ordinary_all_paid_teacher_fit=ordinary,GT_open_attempts=len(guard))
        atomic_json(local/'receipt.json',report);rows.append(dict(video=video,report=report))
        past.append(record)
    assert not guard
    atomic_json(root/'completion.json',dict(status='COMPLETE_TEACHER_ONLY_POLICY_DISTRIBUTION_PILOT',
        branch=branch,neural_seeds=[0],rows=rows,GT_open_attempts=0,new_teacher_GPU_calls=0,
        RMOT_predictions_generated=False,wall_seconds=time.perf_counter()-started))


def analyze():
    check_plan();rows=[]
    for video in VIDEOS:
        policies={}
        for name in ('ORIGINAL_FROZEN','ORIGINAL_SGD',*BRANCHES):
            local=PARENT/'C1/seed0/videos'/video/'policies'/('FROZEN' if name=='ORIGINAL_FROZEN' else 'LEARNED') if name.startswith('ORIGINAL_') else OUT/name/video
            report=json.loads((local/'receipt.json').read_text());paid=json.loads((local/'paid.json').read_text())['paid'];seq=json.loads((local/'sequence.json').read_text())['sequence']
            policies[name]=dict(report=report,uids={r['uid'] for r in paid},order=[r['uid'] for r in seq])
        row=dict(video=video,policies={})
        for name,d in policies.items():
            rep=d['report'];row['policies'][name]=dict(budget=rep['teacher_budget']['unique_teacher_bundles'],
                rewards=[r['credit']['reward'] for r in rep['rounds']],
                teacher_information=[r['credit']['information'] for r in rep['rounds']],
                actor_updates=[r['actor_update'] for r in rep['rounds']],
                replaced_vs_original_SGD=len(d['uids']^policies['ORIGINAL_SGD']['uids'])//2,
                same_order_as_original_SGD=d['order']==policies['ORIGINAL_SGD']['order'])
        assert len({v['budget'] for v in row['policies'].values()})==1
        rows.append(row)
    result=dict(status='ANALYZED_TEACHER_ONLY_POLICY_DISTRIBUTION_PILOT',rows=rows,
        neural_seeds=[0],all_data_DEVELOPMENT=True,manual_GT_or_HOTA_used=False,
        actual_new_teacher_GPU_calls=0,RMOT_quality_claim=False,
        original_six_setting_protocol_unchanged=True,
        limitations='Two videos; finite pretrained teacher library; teacher progress and question changes do not establish RMOT quality or positive transfer.')
    atomic_json(OUT/'analysis.json',result)
    atomic_json(PUBLIC/'rsi_policy_distribution_pilot_20261124.json',result)
    print(json.dumps(dict(status=result['status'],replacements={r['video']:{p:d['replaced_vs_original_SGD'] for p,d in r['policies'].items()} for r in rows})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--branch',choices=BRANCHES);parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    if args.branch:worker(args.branch)
    elif args.analyze:analyze()
    else:
        import subprocess
        check_plan();start=time.perf_counter()
        print(json.dumps(dict(status='RUNNING_INDEPENDENT_CPU_RSI_POLICY_DISTRIBUTION_PILOT',
            pid=os.getpid(),branches=BRANCHES,videos=VIDEOS,GT_used=False)),flush=True)
        for branch in BRANCHES:
            subprocess.run([sys.executable,'-u',str(Path(__file__)),'--branch',branch],cwd=ROOT,check=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        analyze();print(json.dumps(dict(status='PASS_POLICY_DISTRIBUTION_ENGINEERING_NOT_GOAL_COMPLETE',wall_seconds=time.perf_counter()-start)),flush=True)
