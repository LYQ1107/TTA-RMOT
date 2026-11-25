"""Finite CPU measurement of the two preserved failed optimizer states.

No fitted steps, GT, teacher calls, live-source edits or alternative primary.
Gradient disagreement is descriptive, not an isolated causal attribution of
L-BFGS's success to minibatching rather than momentum or line search.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_fullscope_visual_six import OUT,PLAN,registration
from flex_uda.active.fullscope_visual_runtime import Context,teacher_table
import flex_uda.active.evidence_visual_learning as core


def analyze(case):
    folder=OUT/'units'/case['pair']/case['evaluated_video']
    context=Context(case,folder,None,-1)
    closed=torch.load(folder/'primary/closed.pt',map_location='cpu',weights_only=False)
    saved=torch.load(folder/'primary/anchor_projection.pt',map_location='cpu',weights_only=False)
    model=core.make(context.mean,context.std,'cpu');model.load_state_dict(saved['solver'])
    x,y,w=teacher_table(context,closed['draws'],'IPW_KD')
    _,_,plain=teacher_table(context,closed['draws'],'SOFT_KD');w=(w+plain)/2
    assert len(x)==12037 and np.isfinite(w).all() and (w>0).all()
    tx=torch.as_tensor(x);ty=torch.as_tensor(y,dtype=torch.float32);tw=torch.as_tensor(w,dtype=torch.float32)
    params=list(model.parameters());n=len(x)
    def flatten():return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).clone()
    def loss(indices,scale):
        z=model(tx[indices])
        return ((torch.nn.functional.softplus(z)-ty[indices]*z)*tw[indices]).sum()*scale
    model.zero_grad(set_to_none=True)
    # Original minibatches estimate SUM(w*BCE), not a unit-mass normalized
    # empirical loss. Realized LURE mass need not equal1; no weights are edited.
    for start in range(0,n,512):loss(slice(start,start+512),1.).backward()
    full=flatten();full_norm=float(full.norm());assert full_norm>0 and torch.isfinite(full).all()
    state=saved['optimizer'];group=state['param_groups'][0]
    assert len(group['params'])==len(params) and group['lr']==.0001
    beta1,beta2=group['betas'];epsilon=group['eps']
    def next_adam_delta(gradient):
        norm=float(gradient.norm());gradient=gradient*min(1.,10./(norm+1e-6))
        pieces=[];offset=0
        for p,index in zip(params,group['params']):
            g=gradient[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
            old=state['state'][index];step=int(old['step'])+1
            m=old['exp_avg']*beta1+g*(1-beta1)
            v=old['exp_avg_sq']*beta2+g.square()*(1-beta2)
            change=-group['lr']*(m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+epsilon)
            pieces.append(change.flatten())
        return torch.cat(pieces)
    full_delta=next_adam_delta(full)
    summaries=[];cursor=torch.arange(512)
    for step in range((n+511)//512):
        indices=(cursor+512*step)%n
        model.zero_grad(set_to_none=True);loss(indices,n/512.).backward()
        gradient=flatten();norm=float(gradient.norm())
        dot=float(torch.dot(full,gradient));delta=next_adam_delta(gradient)
        summaries.append(dict(batch=step,weight_mass=float(tw[indices].sum()),
            auxiliary_rows=int((indices>=len(closed['draws'])).sum()),
            gradient_norm=norm,gradient_cosine_with_full=dot/(norm*full_norm),
            gradient_dot_full=dot,full_gradient_dot_next_Adam_delta=float(torch.dot(full,delta)),
            original_clip10_would_activate=norm>10))
    norms=np.asarray([s['gradient_norm'] for s in summaries]);cosines=np.asarray([s['gradient_cosine_with_full'] for s in summaries])
    return dict(pair=case['pair'],video=case['evaluated_video'],rows=n,
        full_objective_gradient_norm=full_norm,actual_weight_mass=float(w.sum()),
        auxiliary_weight_fraction=float(w[-len(context.ay):].sum()/w.sum()),auxiliary_rows=len(context.ay),
        auxiliary_total_weight=float(w[-len(context.ay):].sum()),
        first24cyclic_batches_count=len(summaries),
        batch_weight_mass_minmax=[min(s['weight_mass'] for s in summaries),max(s['weight_mass'] for s in summaries)],
        batch_gradient_norm_quantiles=np.quantile(norms,[0,.5,1]).tolist(),
        batch_gradient_cosine_quantiles=np.quantile(cosines,[0,.5,1]).tolist(),
        batches_with_gradient_opposite_full=sum(s['gradient_dot_full']<0 for s in summaries),
        batches_whose_next_Adam_delta_is_full_objective_ascent=sum(s['full_gradient_dot_next_Adam_delta']>0 for s in summaries),
        full_gradient_dot_next_fullbatch_Adam_delta=float(torch.dot(full,full_delta)),
        full_gradient_dot_negative_full_gradient=-full_norm**2,
        actual_next_original_phase_batch_would_be_batch0=summaries[0],
        gradient_clipping_batch_count=sum(s['original_clip10_would_activate'] for s in summaries),
        batch_details=summaries,original_failed_checkpoint_sha256=file_sha(folder/'primary/anchor_projection.pt'),
        interpretation='Positive gradient-dot-parameter-delta is local first-order ascent; not a measured finite-step loss increase or proof of the entire historical failure cause. These are24overlapping cyclic512row windows, not independent samples.')


def main():
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();plan=registration();torch.set_num_threads(2);started=time.perf_counter()
    rows=[]
    for pair in ('C2','C5'):
        case=next(c for c in plan['cases'] if c['pair']==pair and c['evaluated_video']=='0019')
        row=analyze(case);rows.append(row)
        print(json.dumps(dict(status='FAILED_OPTIMIZER_GRADIENT_STATE_AUDITED',**{k:v for k,v in row.items() if k!='batch_details'})),flush=True)
    assert not guard
    receipt=dict(status='COMPLETE_TWO_PRESERVED_FAILED_OPTIMIZER_CPU_AUDITS',cases=rows,
        wall_seconds=time.perf_counter()-started,source_sha256=file_sha(__file__),
        parent_registration_sha256=file_sha(PLAN),GT_open_attempts=0,new_teacher_calls=0,
        optimizer_steps=0,live42sources_unchanged=True,not_RMOT_HOTA_or_isolated_causal_proof=True,goal_status='ACTIVE')
    destination=ROOT/'a100_runs/visual_classical_20261121/supervisor/fullscope_failed_optimizer_gradient_audit_20261126.json'
    atomic_json(destination,receipt)


if __name__=='__main__':main()
