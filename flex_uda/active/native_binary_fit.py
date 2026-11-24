"""Proper-risk prefix fits for the unified native-binary observation study.

Exact inputs and the model's A/B equivariance define the attainable floor.
No target GT or task accuracy enters optimization or capacity completion.
"""
import copy
import hashlib
import time
import numpy as np
import torch
from flex_uda.active.native_binary_models import make_model,renormalize,log_probabilities,PAIR_KEYS
from flex_uda.active.joint_observation_study import teacher_risk


def coalesce(arrays,recipe):
    pair=recipe['family'] in ('structured','pair')
    if pair:
        keys=PAIR_KEYS;target=arrays['pair_target'].copy();weight=np.ones(len(target))
    else:
        keys=('point_anchors','point_present','point_current') if recipe['family']=='point' else ('coarse_inputs',)
        p=arrays['point_target']
        y=(p>=.5).astype(float) if recipe.get('hard') else (np.clip(p,.02,.98) if recipe.get('confidence') else p)
        target=np.c_[y,1-y];weight=1+3*np.abs(2*p-1) if recipe.get('confidence') else np.ones(len(p))
        if recipe.get('risk')=='logit':
            clipped=np.clip(p,1e-6,1-1e-6)
            target=(np.log(clipped)-np.log1p(-clipped))[:,None]
    lookup={};owners=[];inverse=[];orient=[]
    if recipe.get('risk')!='logit':target/=target.sum(-1,keepdims=True)
    for i in range(len(target)):
        if pair:
            a=b'';b=b''
            for x,y in (('anchors_A','anchors_B'),('present_A','present_B'),('first','second')):
                a+=np.ascontiguousarray(arrays[x][i]).tobytes();b+=np.ascontiguousarray(arrays[y][i]).tobytes()
            swap=a>b
            if swap:target[i]=target[i,[1,0,2,3]]
            if a==b:target[i,:2]=target[i,:2].mean()
            signature=hashlib.sha256(min(a,b)+max(a,b)).digest()
        else:
            swap=False;signature=hashlib.sha256(b''.join(np.ascontiguousarray(arrays[k][i]).tobytes() for k in keys)).digest()
        if signature not in lookup:
            lookup[signature]=len(owners);owners.append(i);orient.append(swap)
        inverse.append(lookup[signature])
    inverse=np.asarray(inverse);mass=np.bincount(inverse,weights=weight)
    q=np.stack([np.bincount(inverse,weights=weight*target[:,j]) for j in range(target.shape[1])],-1)/mass[:,None]
    selected={k:arrays[k][owners].copy() for k in keys}
    if pair:
        bits=np.asarray(orient)
        for a,b in (('anchors_A','anchors_B'),('present_A','present_B'),('first','second')):
            left,right=selected[a][bits].copy(),selected[b][bits].copy()
            selected[a][bits]=right;selected[b][bits]=left
    selected.update(target=q,weight=mass)
    variance=float((weight*(target[:,0]-q[inverse,0])**2).sum()/weight.sum()) if recipe.get('risk')=='logit' else 0.
    return selected,dict(original_risk_rows=len(target),exact_orbit_groups=len(owners),
        architecture_equivariance_projected=pair,proper_teacher_CE_preserved=recipe.get('risk')!='logit',
        excess_teacher_risk_preserved=True,irreducible_input_MSE_logit=variance,
        identical_input_target_conflicts_are_not_fit_as_separate_labels=True)


def risk(model,data,recipe):
    if recipe.get('risk')=='logit':
        z=model(data['point_anchors'],data['point_present'],data['point_current'])
        value=((z-data['target'][:,0]).square()*data['weight']).sum()/data['weight'].sum()
        return value,torch.zeros_like(value)
    return teacher_risk(log_probabilities(model,data,recipe),data['target'],data['weight'],recipe.get('risk','joint'))


def full_risk(model,data,recipe,backward=False,batch=1024):
    """Exact weighted full-batch objective, bounded activation memory."""
    mass=data['weight'].sum();loss=torch.zeros((),device=mass.device,dtype=torch.float64)
    floor=torch.zeros_like(loss)
    for begin in range(0,len(data['target']),batch):
        part={k:v[begin:begin+batch] for k,v in data.items()}
        c,h=risk(model,part,recipe);fraction=part['weight'].sum()/mass
        value=c*fraction
        if backward:value.backward()
        loss+=value.detach();floor+=(h*fraction).detach()
    return loss,floor


def fit_prefix(arrays,recipe,mean,std,previous=None,device='cuda:0'):
    start=time.perf_counter();selected,projection=coalesce(arrays,recipe)
    data={k:torch.as_tensor(v,device=device) for k,v in selected.items()}
    ordinary=recipe['family']!='structured';kind=recipe.get('risk','joint')
    width0=previous['width'] if previous else (128 if recipe['family'] in ('point','pair') else 64)
    widths=[width0]+([w for w in (128,256,512) if w>width0] if ordinary else [])
    minimum=400 if previous else 1000;trace=[];attempts=[]
    global_best=float('inf');chosen=None
    for wi,width in enumerate(widths):
        torch.manual_seed(0);model=make_model(recipe,len(mean),width).to(device)
        if wi==0 and previous:
            assert previous['recipe']==recipe;model.load_state_dict(previous['model'])
            renormalize(model,recipe,previous['mean'],previous['std'],mean,std)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');early=None;best_state=None
        for step in range(1,4001):
            optimizer.zero_grad(set_to_none=True)
            ce,_=full_risk(model,data,recipe,backward=True)
            assert bool(torch.isfinite(ce));torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if step==400:early=copy.deepcopy(model).eval()
            if step%250==0 or step==400:
                with torch.no_grad():c,h=full_risk(model,data,recipe)
                metrics=dict(width=width,steps=step,teacher_CE=float(c),attainable_exact_orbit_entropy=float(h),
                    excess_KL=float(c-h),optimizer='Adam_lr.001')
                trace.append(metrics)
                if metrics['excess_KL']<best:
                    best=metrics['excess_KL'];best_state=copy.deepcopy(model.state_dict());best_metrics=metrics.copy()
                if step>=minimum and best<=.005:break
                if not ordinary and step>=minimum+750 and len(trace)>=4:
                    recent=[m['excess_KL'] for m in trace[-4:]]
                    if max(recent)-min(recent)<1e-5:break
        model.load_state_dict(best_state);metrics=best_metrics
        del optimizer
        if ordinary and best>.005:
            model.double();double_data={k:v if v.dtype==torch.bool else v.double() for k,v in data.items()}
            opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,history_size=10,
                tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True)
                loss,_=full_risk(model,double_data,recipe,backward=True)
                return loss
            deployed_best=copy.deepcopy(model).float()
            for block in range(5):
                opt.step(closure);deployed=copy.deepcopy(model).float()
                with torch.no_grad():c,h=full_risk(deployed,data,recipe)
                current=dict(width=width,LBFGS_blocks=block+1,teacher_CE=float(c),
                    attainable_exact_orbit_entropy=float(h),excess_KL=float(c-h),optimizer='LBFGS64_checked32')
                attempts.append(current)
                if current['excess_KL']<best:best=current['excess_KL'];deployed_best=deployed;metrics=current
                if best<=.005:break
            model=deployed_best
            del double_data,opt
        if best<global_best:
            global_best=best;chosen=model.eval(),early,metrics,width,step
        if not ordinary or global_best<=.005:break
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    model,early,metrics,width,step=chosen
    receipt=dict(recipe=recipe,width=width,metrics=metrics,trace=trace,capacity_and_LBFGS_attempts=attempts,
        minimum_Adam_steps=minimum,actual_final_Adam_steps=step,early_step=400,
        ordinary_required=ordinary,teacher_fit_gate_passed=global_best<=.005,
        structural_residual_is_not_ordinary_adequacy=not ordinary,neural_seed=0,
        prefix_warm_start=previous is not None,GT_or_task_metric_selection=False,
        **projection,wall_seconds=time.perf_counter()-start)
    receipt['activation_batch_limit']=1024
    receipt['full_batch_weighted_risk_and_gradient_preserved']=True
    receipt['risk_units']='squared native teacher logit (p clipped1e-6)' if kind=='logit' else 'proper teacher KL in nats'
    receipt['excess_risk']=float(global_best)
    if kind=='logit':
        for item in [metrics,*trace,*attempts]:
            if 'teacher_CE' in item:
                item['projected_teacher_MSE_logit']=item.pop('teacher_CE')
                item['excess_MSE_logit']=item.pop('excess_KL')
                item['projected_MSE_logit_floor']=item.pop('attainable_exact_orbit_entropy')
    if ordinary and not receipt['teacher_fit_gate_passed']:
        raise RuntimeError('Ordinary native-binary teacher fit incomplete; no quality comparison: '+str(receipt))
    checkpoint=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},width=width,
        recipe=recipe,mean=mean.copy(),std=std.copy(),receipt=receipt)
    return model,early,checkpoint,receipt
