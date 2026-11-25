"""Same-information teacher-risk fitting for the cross-candidate pilot.

Three target videos are fit independently, with seed0 and no manual labels.
The checks concern learning and optimizer adequacy, never RMOT quality.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.cross_candidate_learning import CrossCandidateHead, OrdinaryCrossCandidateKD
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from scripts.a100_cross_candidate_teacher_pilot import OUT as TEACHER, PLAN as TEACHER_PLAN
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX/'cross_candidate_fit_pilot_20261123'
PLAN = PUBLIC/'cross_candidate_fit_pilot_plan.json'
ARMS = ('CONDITIONAL_IDENTITY_REFERENCE', 'FULL_LIKELIHOOD_IDENTITY_REFERENCE',
        'CONDITIONAL_NO_STABLE_IDENTITY', 'ORDINARY_JOINT_KD')
KEYS = ('anchors_A','present_A','anchors_B','present_B','first','second')


def prepare():
    files = ('scripts/a100_cross_candidate_fit_pilot.py', 'flex_uda/active/cross_candidate_learning.py',
             'flex_uda/active/joint_observation_inputs.py','flex_uda/active/observation_learning_inputs.py')
    plan = dict(status='PROSPECTIVE_CROSS_CANDIDATE_FIT_PILOT', arms=list(ARMS),
        source_sha256={name:file_sha(ROOT/name) for name in files},
        teacher_plan_sha256=file_sha(TEACHER_PLAN), neural_seeds=[0],
        each_target_fitted_independently=True, inputs_identical_1101D_and_two_candidate_contexts=True,
        same_AB_BA_symmetrized_teacher_and_arithmetic_three_view_mixture=True,
        teacher_signal_scope='Joint AB/BA only for this engineering fit. Binary channels are audited separately, unused by every fit arm. This is not the final all-paid-information RMOT baseline.',
        optimizer=dict(Adam_lr=.01, minimum_steps=1000,maximum_steps=4000, ordinary_LBFGS_max_iterations=500),
        ordinary_capacity=[128,256,512], structured_capacity=64,
        ordinary_required_all_joint_positive_weighted_and_exactly_one_KL_at_most=.005,
        no_manual_GT_no_RMOT_predictions_no_HOTA=True,
        synthetic_contract='cross_candidate_learning_contract.json',
        scientific_scope='Engineering teacher-risk fit only; not a generalization comparison or final method selection.')
    atomic_json(PLAN,plan);print(json.dumps(plan),flush=True)


def load_video(data, pair, video):
    v = JointObservationVideo(pair,video)
    by = {(r['case'],r['mode']):r for r in data['responses']}
    grouped = {}
    for ci,c in enumerate(data['cases']):
        if c['pair'] != pair or c['video'] != video:
            continue
        key = c['query_id'],c['track_A'],c['track_B'],c['frame']
        q = (np.asarray(by[ci,'JOINT_AB']['choice_probabilities'])+
             np.asarray(by[ci,'JOINT_BA']['choice_probabilities'])[[1,0,2,3]])/2
        grouped.setdefault(key,[]).append(q)
    track = {int(t):i for i,t in enumerate(v.tracks)}
    query = {q['expression_id']:i for i,q in enumerate(v.queries)}
    keys = sorted(grouped); qi = np.asarray([query[k[0]] for k in keys])
    ta = np.asarray([track[k[1]] for k in keys]);tb = np.asarray([track[k[2]] for k in keys])
    ra = np.asarray([v.positions[k[3],k[1]] for k in keys]);rb = np.asarray([v.positions[k[3],k[2]] for k in keys])
    raw = dict(anchors_A=v.raw_bags[qi,ta],present_A=v.present[ta],
        anchors_B=v.raw_bags[qi,tb],present_B=v.present[tb],first=v.dense_rows(qi,ra),second=v.dense_rows(qi,rb))
    values = np.concatenate((raw['anchors_A'][raw['present_A']],raw['anchors_B'][raw['present_B']],raw['first'],raw['second'])).astype('float64')
    mean, std = values.mean(0),values.std(0).clip(min=.01)
    arrays = {}
    for k,a in raw.items():
        if k.startswith('present'):
            arrays[k] = torch.as_tensor(a)
        else:
            x = ((a-mean)/std).astype('float32')
            if k.startswith('anchors'):
                x[~raw['present_'+k[-1]]] = 0
            arrays[k] = torch.as_tensor(x)
    target = torch.as_tensor(np.stack([np.mean(grouped[k],axis=0) for k in keys]),dtype=torch.float64)
    target /= target.sum(-1,keepdim=True)
    return arrays,target,dict(visual_rows=len(keys), paid_joint_teacher_examples=6*len(keys),
        class_mean=target.mean(0).tolist(), no_GT_open=True),mean,std


def diagnostics(logp, target):
    target = target.double();logp = logp.double()
    joint = (torch.xlogy(target,target)-target*logp).sum(-1)
    one = target[:,:2].sum(-1)
    q = target[:,:2]/one[:,None].clamp_min(1e-300)
    conditional = (torch.xlogy(q,q)-q*torch.log_softmax(logp[:,:2],-1)).sum(-1)
    positive = 1-target[:,-1]
    return dict(joint_KL=float(joint.mean().detach()),
        positive_weighted_joint_KL=float(((joint*positive).sum()/positive.sum().clamp_min(1e-300)).detach()),
        exactly_one_conditional_KL=float(((conditional*one).sum()/one.sum().clamp_min(1e-300)).detach()),
        per_teacher_argmax_KL={str(int(k)):float(joint[target.argmax(-1)==k].mean().detach()) for k in target.argmax(-1).unique()})


def adequate(d):
    return max(d['joint_KL'],d['positive_weighted_joint_KL'],d['exactly_one_conditional_KL'],
               *d['per_teacher_argmax_KL'].values()) <= .005


def fit(name, arrays, target, width):
    torch.manual_seed(0);start = time.perf_counter();d = arrays['first'].shape[-1]
    if name == 'ORDINARY_JOINT_KD':
        model = OrdinaryCrossCandidateKD(d,width)
    else:
        model = CrossCandidateHead(d,width,
            conditional_estimation=name!='FULL_LIKELIHOOD_IDENTITY_REFERENCE',
            stable_identity=name!='CONDITIONAL_NO_STABLE_IDENTITY')
    optimizer = torch.optim.Adam(model.parameters(),lr=.01);milestones=[]
    for step in range(1,4001):
        optimizer.zero_grad(set_to_none=True)
        logp = model(**arrays).double(); loss = -(target*logp).sum(-1).mean()
        assert bool(torch.isfinite(loss));loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step % 250 == 0:
            with torch.no_grad():
                receipt = diagnostics(model(**arrays),target)
            milestones.append(dict(step=step,**receipt))
            if step >= 1000 and adequate(receipt):
                break
    lbfgs_iterations=0
    if name == 'ORDINARY_JOINT_KD' and not adequate(receipt):
        model=model.double(); double={k:a.double() if a.dtype!=torch.bool else a for k,a in arrays.items()}
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
        for i in range(5):
            def closure():
                optimizer.zero_grad();loss=-(target*model(**double)).sum(-1).mean();loss.backward();return loss
            optimizer.step(closure);lbfgs_iterations+=100
            deployed=copy.deepcopy(model).float()
            with torch.no_grad():receipt=diagnostics(deployed(**arrays),target)
            if adequate(receipt):break
        model=model.float()
    receipt=dict(receipt,steps=step,ordinary_gate_passed=adequate(receipt),width=width,
        lbfgs_max_iterations_executed=lbfgs_iterations,wall_seconds=time.perf_counter()-start,
        milestones=milestones,manual_GT_or_task_metric_selection=False)
    return model,receipt


def run():
    guard=generation_guard();torch.set_num_threads(2)
    plan=json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    assert file_sha(TEACHER_PLAN)==plan['teacher_plan_sha256']
    data=json.loads((TEACHER/'responses.json').read_text())
    assert len(data['responses'])==432
    cache=use_verified_local_features();reports={};started=time.perf_counter()
    for pair,video in sorted({(c['pair'],c['video']) for c in data['cases']}):
        arrays,target,source,mean,std=load_video(data,pair,video)
        fits={}
        for name in ARMS:
            width=128 if name=='ORDINARY_JOINT_KD' else 64
            attempts=[]
            while True:
                model,receipt=fit(name,arrays,target,width);attempts.append(receipt)
                if name!='ORDINARY_JOINT_KD' or adequate(receipt):break
                if width==512:raise RuntimeError('Ordinary KD optimization incomplete; do not compare this baseline.')
                width*=2
            fits[name]=dict(receipt,capacity_attempts=attempts)
            folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,receipt=receipt),folder/(name+'.pt'))
            print(json.dumps(dict(pair=pair,arm=name,**{k:receipt[k] for k in ('joint_KL','exactly_one_conditional_KL','ordinary_gate_passed','wall_seconds')})),flush=True)
        reports[pair]=dict(video=video,source=source,fits=fits)
        atomic_json(OUT/'progress.json',dict(completed=reports,GT_open_attempts=0))
    assert not guard
    report=dict(status='PASS_REAL_CROSS_CANDIDATE_TEACHER_RISK_FITTING_ONLY',targets=reports,
        wall_seconds=time.perf_counter()-started, no_GPU_no_new_teacher_no_manual_GT=True,
        no_RMOT_predictions_or_HOTA_or_RSI_gain_claim=True,
        ordinary_same_joint_signal_adequacy_required=True,
        not_final_all_paid_information_RMOT_baseline=True,local_features_verified_files=cache['files'])
    atomic_json(OUT/'receipt.json',report)
    atomic_json(PUBLIC/'cross_candidate_fit_pilot_receipt.json',report)
    print(json.dumps(dict(status=report['status'],wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('prepare','run'),required=True)
    args=parser.parse_args();globals()[args.mode]()
