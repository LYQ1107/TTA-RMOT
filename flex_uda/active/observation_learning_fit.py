"""Proper teacher-only risks for shared observation representations.

Warm starts preserve the previous function under prefix renormalization;
their use is identical for ordinary and structured learners. Neither target
GT nor HOTA chooses an optimizer step, width, or output threshold.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flex_uda.active.conditional_observation import BagContextHead, weighted_bernoulli_loss
from flex_uda.active.query_profile import supported_query_profile_kl


# architecture, balanced exposure, supported mixture-profile loss, hard labels
RECIPES = {
    'FRAME_SOFT':('local',False,False,False),
    'FRAME_HARD':('local',False,False,True),
    'FLAT_SOFT':('flat',False,False,False),
    'FLAT_HARD':('flat',False,False,True),
    'DIRECT_POINT':('direct_context',False,False,False),
    'DIRECT_BALANCED':('direct_context',True,False,False),
    'DIRECT_PROFILE':('direct_context',True,True,False),
    'CONDITIONAL_BALANCED':('conditional',True,False,False),
    'FULL':('conditional',True,True,False),
}


def make_model(kind, dimension, width):
    if kind in ('local','flat'):
        inputs=dimension if kind=='local' else 4*dimension+3
        model=nn.Sequential(nn.Linear(inputs,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
        return model
    return BagContextHead(dimension,width,kind)


def forward(model, kind, bags, present, current):
    if kind=='local':return model(current).squeeze(-1)
    if kind=='flat':
        return model(torch.cat((bags.flatten(1),present.to(bags.dtype),current),1)).squeeze(-1)
    return model(bags,present,current)


def renormalize(model, kind, old_mean, old_std, new_mean, new_std):
    """Change coordinates without changing outputs, including missing slots."""
    first=model[0] if kind in ('local','flat') else model.encoder[0]
    alpha=torch.as_tensor(new_std/old_std,device=first.weight.device,dtype=first.weight.dtype)
    delta=torch.as_tensor((new_mean-old_mean)/old_std,device=first.weight.device,dtype=first.weight.dtype)
    d=len(alpha)
    with torch.no_grad():
        old=first.weight.clone()
        if kind!='flat':
            first.weight.mul_(alpha);first.bias.add_(old@delta)
        else:
            for slot in range(3):
                section=slice(slot*d,(slot+1)*d)
                first.weight[:,section].mul_(alpha)
                # Missing anchors are exactly zero after normalization.
                first.weight[:,3*d+slot].add_(old[:,section]@delta)
            section=slice(3*d+3,4*d+3)
            first.weight[:,section].mul_(alpha);first.bias.add_(old[:,section]@delta)


def fit(data, recipe, *, seed, mean, std, previous=None, device='cuda:0'):
    started=time.perf_counter();kind,balanced,profile,hard=RECIPES[recipe]
    torch.manual_seed(seed);d=data['current'].shape[1]
    width=previous['width'] if previous is not None else (64 if kind in ('local','flat') else 32)
    model=make_model(kind,d,width).to(device)
    if previous is not None:
        assert previous['kind']==kind and previous['recipe']==recipe
        model.load_state_dict(previous['model'])
        renormalize(model,kind,previous['mean'],previous['std'],mean,std)
    bb=torch.as_tensor(data['bags'],device=device);mm=torch.as_tensor(data['present'],device=device)
    cc=torch.as_tensor(data['current'],device=device)
    yy=torch.as_tensor((data['raw_target']>=.5).astype('float32') if hard else data['target'],device=device)
    ww=torch.as_tensor(data['balanced_weight'] if balanced else data['confidence'],device=device)
    raw=torch.as_tensor(data['raw_target'],device=device)
    groups=torch.as_tensor(data['group'],device=device)
    yd,wd=yy.double(),ww.double()
    entropy=float((wd*(-yd*torch.log(yd.clamp_min(1e-300))-(1-yd)*torch.log((1-yd).clamp_min(1e-300)))).sum()/wd.sum())
    def relative(logits, target, weight):
        # One explicitly projected raw-view mixture is the common observation
        # protocol here. This does not assert equal reliability of raw views.
        return supported_query_profile_kl(F.logsigmoid(logits),target,raw.to(logits.dtype),groups,weight)
    def logits_for(module):
        values=[]
        with torch.no_grad():
            for a in range(0,len(yy),2048):
                b=a+2048
                values.append(forward(module,kind,bb[a:b],mm[a:b],cc[a:b]).double())
        return torch.cat(values)
    def metrics(module):
        logits=logits_for(module)
        ce=float(weighted_bernoulli_loss(logits,yd,wd));r=float(relative(logits,yd,wd)) if profile else 0.
        return dict(target_CE=ce,target_entropy=entropy,target_KL=ce-entropy,
            supported_profile_KL=r,objective_excess=ce-entropy+(r if profile else 0.))
    def passed(values):return values['target_KL']<=.005 and values['objective_excess']<=.005
    ng=int(groups.max())+1
    members=[torch.nonzero(groups==g).flatten() for g in range(ng)]
    # Whole query groups retain both candidate coverage and exposure weights.
    max_count=max(map(len,members));pack=torch.full((ng,max_count),-1,device=device,dtype=torch.long)
    for g,rows in enumerate(members):pack[g,:len(rows)]=rows
    number=max(1,min(128,round(512/max(len(yy)/ng,1))))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator(device=device).manual_seed(seed);trace=[];early=None
    minimum=4000 if previous is None else 400
    for step in range(1,16001):
        chosen=torch.randperm(ng,generator=rng,device=device)[:number]
        rows=pack[chosen].flatten();rows=rows[rows>=0]
        logits=forward(model,kind,bb[rows],mm[rows],cc[rows])
        loss=weighted_bernoulli_loss(logits,yy[rows],ww[rows])
        if profile:
            _,local=torch.unique(groups[rows],sorted=True,return_inverse=True)
            loss=loss+supported_query_profile_kl(F.logsigmoid(logits),yy[rows],raw[rows],local,ww[rows])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step==400:early=copy.deepcopy(model).eval()
        if step==400 or (step>=minimum and step%100==0):
            value=metrics(model);trace.append(dict(step=step,**value))
            if step>=minimum and passed(value):break
    attempts=[]
    if not passed(value):
        # Adequate fitting is provided to ordinary comparators too. Every
        # extension is selected only by its own registered teacher risk.
        widths=(width,128,256,512) if kind in ('local','flat') else (width,64,128)
        for extra_width in sorted(set(w for w in widths if w>=width)):
            if extra_width!=width:
                torch.manual_seed(seed+extra_width);model=make_model(kind,d,extra_width).to(device)
                width=extra_width
            model.double();b64,c64=bb.double(),cc.double()
            opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,
                history_size=20,tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True);z=forward(model,kind,b64,mm,c64)
                loss=weighted_bernoulli_loss(z,yd,wd)
                if profile:loss=loss+relative(z,yd,wd)
                loss.backward();return loss
            for block in range(10):
                opt.step(closure)
                deployed=make_model(kind,d,width).to(device)
                deployed.load_state_dict({k:v.float() for k,v in model.state_dict().items()})
                value=metrics(deployed);attempts.append(dict(width=width,LBFGS_block=block+1,**value))
                if passed(value):break
            model=deployed
            if passed(value):break
    model.eval();value=metrics(model);torch.cuda.synchronize() if str(device).startswith('cuda') else None
    receipt=dict(recipe=recipe,kind=kind,balanced_exposure=balanced,profile_weight=1. if profile else 0.,
        hard_targets=hard,width=width,steps=step,minimum_steps=minimum,prefix_warm_start=previous is not None,
        all_initial_task_probabilities=.5 if previous is None else None,metrics=value,trace=trace,
        teacher_risk_continuations=attempts,target_fit_gate_pass=passed(value),
        no_source_logit_residual=True,teacher_examples=len(yy),wall_seconds=time.perf_counter()-started,
        target_GT_or_HOTA_used=False,final_CE_recomputed_from_actual_float32_deployed_head=True)
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        kind=kind,recipe=recipe,width=width,dimension=d,mean=mean,std=std,receipt=receipt)
    return model,early,checkpoint,receipt
