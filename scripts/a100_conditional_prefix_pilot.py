"""Real first-prefix training handoff for conditional-observation primitives.

No full-video predictions or HOTA are generated. This tests optimization and
dataflow against equally informed ordinary heads, not scientific superiority.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.anchor_state_inputs import first_prefix
from scripts.a100_anchor_pcd_collect import STATE as SOURCE_STATE
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'conditional_prefix_pilot_20261123';STATE=OUT/'supervisor/state.json'
REGISTRATION=PUBLIC/'conditional_prefix_pilot_registration_v1.json'
SUCCESS='COMPLETE_REAL_PREFIX_CONDITIONAL_LEARNING_PILOT_NO_GT'
METHODS=('FLAT_ORDINARY','DIRECT_CONTEXT','DIRECT_CONTEXT_RELATIVE','CONDITIONAL_JOINT','CONDITIONAL_JOINT_RELATIVE')


def prepare():
    guard=generation_guard();data=first_prefix()
    atomic_json(OUT/'input_handoff.json',dict(status='PASS_REAL_FRESH_PREFIX_INPUT_HANDOFF',**data['report'],
        manual_GT_open_attempts=len(guard)))
    assert not guard
    return data


def worker():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.active.fit_capacity import continue_capacity
    from flex_uda.active.conditional_observation import BagContextHead,weighted_bernoulli_loss,relative_choice_kl
    guard=generation_guard();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);start=time.perf_counter();data=prepare()
    bags=torch.as_tensor(data['bags'],device='cuda:0');present=torch.as_tensor(data['present'],device='cuda:0')
    current=torch.as_tensor(data['current'],device='cuda:0');y=torch.as_tensor(data['target'],device='cuda:0')
    weight=torch.as_tensor(data['weight'],device='cuda:0');group=torch.as_tensor(data['group'],device='cuda:0')
    yd,wd=y.double(),weight.double()
    entropy=float((wd*(-yd*torch.log(yd)-(1.-yd)*torch.log1p(-yd))).sum()/wd.sum())
    packed=[]
    for g in range(int(data['group'].max())+1):
        rows=np.flatnonzero(data['group']==g);packed.append(np.pad(rows,(0,3-len(rows)),constant_values=-1))
    packed=torch.as_tensor(np.asarray(packed),device='cuda:0');ng=len(packed);results={}
    for name in METHODS:
        destination=OUT/name;destination.mkdir(parents=True,exist_ok=True)
        if (destination/'receipt.json').is_file():
            results[name]=json.loads((destination/'receipt.json').read_text());continue
        stamp=time.perf_counter();torch.manual_seed(0)
        if name=='FLAT_ORDINARY':
            flat=data['flat'];dim=flat.shape[1]
            _,fit,checkpoint=fit_head(flat,np.zeros(len(flat)),data['target'],flat,np.zeros(len(flat)),
                np.zeros(dim),np.ones(dim),sample_weight=data['weight'],soft=True,seed=0)
            capacity=None
            if not fit['teacher_fit_within_0_005']:
                checkpoint,capacity=continue_capacity(flat,data['raw_target'],checkpoint,seed=0)
            width=checkpoint['model']['0.weight'].shape[0]
            model=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,1)).cuda()
            model.load_state_dict(checkpoint['model'])
            with torch.no_grad():logits=model(torch.as_tensor(flat,device='cuda:0')).squeeze(-1).double()
            ce=float((wd*F.binary_cross_entropy_with_logits(logits,yd,reduction='none')).sum()/wd.sum())
            metrics=dict(teacher_CE=ce,teacher_entropy=entropy,teacher_KL=ce-entropy,
                relative_choice_KL=float(relative_choice_kl(F.logsigmoid(logits),yd,group,wd)),
                objective_excess=ce-entropy)
            torch.save(checkpoint,destination/'head.pt')
            receipt=dict(method=name,metrics=metrics,ordinary_fit=fit,capacity_continuation=capacity,
                actual_feature_dimension=dim,actual_hidden_width=width,
                final_loss_fields_above_recomputed_from_actual_deployed_head=True)
        else:
            kind='conditional' if name.startswith('CONDITIONAL') else 'direct_context'
            lam=1. if name.endswith('RELATIVE') else 0.
            model=BagContextHead(data['bags'].shape[-1],width=32,kind=kind).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            rng=torch.Generator(device='cuda:0').manual_seed(0);trace=[]
            def metrics_for(module):
                with torch.no_grad():logits=module(bags,present,current).double()
                ce=float((wd*F.binary_cross_entropy_with_logits(logits,yd,reduction='none')).sum()/wd.sum())
                relative=float(relative_choice_kl(F.logsigmoid(logits),yd,group,wd))
                return dict(teacher_CE=ce,teacher_entropy=entropy,teacher_KL=ce-entropy,
                    relative_choice_KL=relative,objective_excess=ce-entropy+lam*relative)
            for step in range(1,16001):
                selected=torch.randperm(ng,generator=rng,device='cuda:0')[:512]
                rows=packed[selected].reshape(-1);rows=rows[rows>=0]
                _,local_group=torch.unique(group[rows],sorted=True,return_inverse=True)
                logits=model(bags[rows],present[rows],current[rows])
                joint=weighted_bernoulli_loss(logits,y[rows],weight[rows])
                relative=relative_choice_kl(F.logsigmoid(logits),y[rows],local_group,weight[rows])
                loss=joint+lam*relative;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if step in (400,4000) or (step>=4000 and step%100==0):
                    metrics=metrics_for(model);trace.append(dict(step=step,**metrics))
                    if step>=4000 and metrics['teacher_KL']<=.005 and metrics['objective_excess']<=.005:break
            extra=[]
            if metrics['teacher_KL']>.005 or metrics['objective_excess']>.005:
                model.double();bb,cc=bags.double(),current.double()
                optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,
                    history_size=20,tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
                def closure():
                    optimizer.zero_grad(set_to_none=True);logits=model(bb,present,cc)
                    loss=weighted_bernoulli_loss(logits,yd,wd)+lam*relative_choice_kl(F.logsigmoid(logits),yd,group,wd)
                    loss.backward();return loss
                for block in range(10):
                    optimizer.step(closure)
                    deployed=BagContextHead(data['bags'].shape[-1],32,kind).cuda()
                    deployed.load_state_dict({k:v.float() for k,v in model.state_dict().items()})
                    metrics=metrics_for(deployed);extra.append(dict(LBFGS_block=block+1,**metrics))
                    if metrics['teacher_KL']<=.005 and metrics['objective_excess']<=.005:break
                model=deployed
            metrics=metrics_for(model)
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                mean=data['mean'],std=data['std'],kind=kind,width=32),destination/'head.pt')
            receipt=dict(method=name,kind=kind,width=32,relative_weight=lam,metrics=metrics,
                AdamW_steps=step,trace=trace,float64_same_architecture_continuation=extra,
                relative_term_uses_same_clipped_targets_as_joint_loss=True,
                all_initial_task_probabilities=.5,matched_groups_sampled_with_all_own_observations=True)
        torch.cuda.synchronize();receipt.update(wall_seconds=time.perf_counter()-stamp,
            original_teacher_fit_gate_pass=metrics['teacher_KL']<=.005,
            own_objective_gate_pass=metrics['objective_excess']<=.005,
            source_backbone_and_teacher_frozen=True,manual_GT_open_attempts=len(guard),no_quality_outputs=True)
        atomic_json(destination/'receipt.json',receipt);results[name]=receipt
        atomic_json(OUT/'worker_progress.json',dict(completed=len(results),total=len(METHODS),last_method=name))
    assert not guard
    report=dict(status=SUCCESS,input=data['report'],methods=results,
        models_passing_teacher_gate=sum(r['original_teacher_fit_gate_pass'] for r in results.values()),
        models_passing_own_objective_gate=sum(r['own_objective_gate_pass'] for r in results.values()),
        worker_wall_seconds=time.perf_counter()-start,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        no_manual_GT_or_full_video_predictions_or_HOTA=True,
        no_language_scope_factor_learner_yet=True,no_module_or_RMOT_superiority_claim=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'conditional_prefix_pilot_receipt.json').write_text(json.dumps(report,separators=(',',':'))+'\n')


def main(prepare_only=False):
    reg=json.loads(REGISTRATION.read_text())
    for name,sha in reg['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    STATE.parent.mkdir(parents=True,exist_ok=True)
    if prepare_only:
        prepare();return
    atomic_json(STATE,dict(status='AWAITING_SOURCE_QUEUE_GPU_RELEASE',pid=os.getpid(),completed=0))
    while True:
        source=json.loads(SOURCE_STATE.read_text())
        if source['status'].startswith('FAILED_'):raise RuntimeError('Resolve source collection first')
        reserved={r['GPU'] for r in source.get('active',{}).values()}
        if 'pending' in source and not source['pending']:
            raw=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()}
            free=[g for g in (0,1,2,3) if g not in reserved and memory[g]<512]
            if free:break
        time.sleep(15)
    log=STATE.parent/'worker.log';gpu=free[0]
    with log.open('a',buffering=1) as stream:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        while proc.poll() is None:
            p=OUT/'worker_progress.json';progress=json.loads(p.read_text()) if p.is_file() else {}
            atomic_json(STATE,dict(status='RUNNING_REAL_PREFIX_CONDITIONAL_TRAINING_NO_GT',pid=os.getpid(),
                worker_pid=proc.pid,physical_GPU=gpu,completed=progress.get('completed',0),progress=progress))
            time.sleep(15)
    if proc.returncode or not (OUT/'completion.json').is_file():raise RuntimeError(log.read_text()[-7000:])
    result=json.loads((OUT/'completion.json').read_text())
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=len(METHODS),
        teacher_gate_passes=result['models_passing_teacher_gate'],goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker()
    else:
        try:main(args.prepare_only)
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_CONDITIONAL_PREFIX_PILOT',error=repr(error),pid=os.getpid());atomic_json(STATE,previous);raise
