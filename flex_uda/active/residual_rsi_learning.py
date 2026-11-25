"""A learned response forecaster and a population-risk RMOT solver.

Original teacher answers are immutable. Forecaster predictions are a
predictable control variate, not replacement semantic labels. Acquisition
uses actual pre-answer errors and current-solver gradient geometry. Qwen
itself stays frozen in this candidate; both small neural models learn.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.residual_risk import ResidualRisk,coverage_distribution,bounded_logits

ALPHA=.25
STEPS=(500,500,500,4000)


class Head(nn.Module):
    def __init__(self,mean,std,outputs=1):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.hidden=nn.Sequential(nn.Linear(len(mean),128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.output=nn.Linear(128,outputs)

    def forward(self,x):
        result=bounded_logits(self.output(self.hidden((x-self.mean)/self.std)))
        return result.flatten() if result.shape[-1]==1 else result

    def last_layer_jacobian_norm(self,x):
        h=self.hidden((x-self.mean)/self.std);raw=self.output(h)[:,0]
        return torch.sqrt(h.square().sum(1)+1)*(1-(raw/8).tanh().square())


class MeanHead(nn.Module):
    def __init__(self,forecaster):super().__init__();self.forecaster=forecaster
    def forward(self,x):return self.forecaster(x)[:,0]


def make(mean,std,device,outputs=1):
    torch.manual_seed(0)
    return Head(mean,std,outputs).to(device)


def state(model):return {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}


def save(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def tensors(value,device):return torch.as_tensor(np.asarray(value).copy(),device=device,dtype=torch.float32)


@torch.no_grad()
def logits(model,x,device):
    return np.concatenate([model(tensors(x[b:b+512],device)).cpu().numpy() for b in range(0,len(x),512)])


def fit(model,x,y,weights,steps,device,optimizer=None):
    optimizer=optimizer or torch.optim.Adam(model.parameters(),lr=.001)
    x=tensors(x,device);y=tensors(y,device);w=tensors(weights,device)
    # Same deterministic cyclic batches, rate, clipping and optimizer
    # continuity are used by the primary and all matching ordinary heads.
    cursor=torch.arange(512,device=device)
    started=time.perf_counter()
    for step in range(steps):
        i=(cursor+512*step)%len(x);optimizer.zero_grad(set_to_none=True)
        z=model(x[i]);loss=((nn.functional.softplus(z)-y[i]*z)*w[i]*len(x)).mean()
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
    torch.cuda.synchronize()
    return optimizer,dict(steps=steps,wall_seconds=time.perf_counter()-started,last_loss=float(loss.detach()))


def forecaster_fit(model,x,y,residual,steps,device,optimizer=None):
    optimizer=optimizer or torch.optim.Adam(model.parameters(),lr=.001)
    x=tensors(x,device);y=tensors(y,device);residual=tensors(residual,device)
    cursor=torch.arange(512,device=device)
    for step in range(steps):
        i=(cursor+512*step)%len(x);optimizer.zero_grad(set_to_none=True)
        z=model(x[i]);loss=nn.functional.binary_cross_entropy_with_logits(z[:,0],y[i])
        loss=loss+nn.functional.mse_loss(z[:,1].sigmoid(),residual[i])
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
    torch.cuda.synchronize();return optimizer


def data(case):
    folder=Path(case['pool_folder']);doc=json.loads((folder/'actions.json').read_text())
    features=np.load(folder/'features.npy');query=np.load(folder/'query.npy')
    with np.load(case['auxiliary_path']) as saved:
        auxiliary=saved['features'].copy();ay=np.nanmean(saved['probability'],axis=1)
    quotas={int(q):int(n) for q,n in doc['quotas'].items()}
    assert len(features)==len(doc['order']) and features.shape[1]==9352
    mean=features.mean(0,dtype='float64').astype('float32')
    std=features.std(0,dtype='float64').clip(min=.1).astype('float32')
    return features,query,doc,quotas,auxiliary,ay,mean,std


def broker(case,doc,quotas):
    from flex_uda.active.native_binary_inputs import BinaryBroker
    return BinaryBroker(case['pair'],case['evaluated_video'],doc['actions'],quotas,dict(groups=set(),probes={}))


def observe(b,uid,stage):
    if uid in b.received:return float(b.received[uid]['probability']),True
    return float(b.request([uid],'train',stage)[0]['probability']),False


def paired_training_table(features,query,draws,quotas,auxiliary,ay,mode):
    indices=np.asarray([r['index'] for r in draws]);target=np.asarray([r['probability'] for r in draws])
    if mode=='HARD_KD':target=(target>=.5).astype(float)
    counts={q:sum(r['query']==q for r in draws) for q in quotas}
    weight=np.asarray([1/(len(quotas)*counts[r['query']]) for r in draws])
    if mode=='IPW_KD':
        nq={q:int((query==q).sum()) for q in quotas}
        weight/=np.asarray([nq[r['query']]*r['proposal_probability'] for r in draws])
    return (np.r_[features[indices],auxiliary],np.r_[target,ay],
            np.r_[weight,np.full(len(ay),ALPHA/len(ay))]/(1+ALPHA))


def fit_ordinary(case,draws,mode,folder,device):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    if mode=='HARD_KD':ay=(ay>=.5).astype(float)
    path=Path(folder)/(mode+'.pt');model=make(mean,std,device)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);return model,saved['receipt']
    optimizer,_=fit(model,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device)
    receipts=[]
    for stage,steps in enumerate(STEPS):
        prefix=[r for r in draws if r['stage']<=stage]
        x,y,w=paired_training_table(features,query,prefix,quotas,auxiliary,ay,mode)
        optimizer,r=fit(model,x,y,w,steps,device,optimizer);receipts.append(r)
    def excess():
        z=logits(model,x,device).astype('float64')
        entropy=-y*np.log(np.clip(y,1e-15,1))-(1-y)*np.log(np.clip(1-y,1e-15,1))
        return float(np.sum(w*(np.logaddexp(0,z)-y*z-entropy))/w.sum())
    total=6000;error=excess()
    while error>.005 and total<16000:
        optimizer,r=fit(model,x,y,w,500,device,optimizer);receipts.append(r);total+=500;error=excess()
    assert error<=.005,(case['pair'],mode,error)
    receipt=dict(mode=mode,total_steps=total,proper_teacher_excess_CE=error,adequacy_limit=.005,
        same_bootstrap_and_phase_schedule_as_primary=True,phase_fits=receipts)
    save(path,dict(model=state(model),receipt=receipt));return model,receipt


def fit_imputed(case,primary_folder,folder,device):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    path=Path(folder)/'IMPUTED_KD.pt';model=make(mean,std,device)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);return model,saved['receipt']
    optimizer,_=fit(model,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device)
    phases=[]
    for stage,steps in enumerate(STEPS):
        with np.load(Path(primary_folder)/f'risk_stage_{stage}.npz') as saved:
            y=np.r_[saved['prequential_proxy_mean'],ay]
            w=np.r_[saved['population_weights'],np.full(len(ay),ALPHA/len(ay))]/(1+ALPHA)
        x=np.r_[features,auxiliary];optimizer,r=fit(model,x,y,w,steps,device,optimizer);phases.append(r)
    def excess():
        z=logits(model,x,device).astype('float64')
        h=-y*np.log(np.clip(y,1e-15,1))-(1-y)*np.log(np.clip(1-y,1e-15,1))
        return float(np.sum(w*(np.logaddexp(0,z)-y*z-h))/w.sum())
    total=6000;error=excess()
    while error>.005 and total<16000:
        optimizer,r=fit(model,x,y,w,500,device,optimizer);phases.append(r);total+=500;error=excess()
    assert error<=.005,(case['pair'],'IMPUTED_KD',error)
    receipt=dict(mode='IMPUTED_KD',total_steps=total,proper_teacher_excess_CE=error,
        matching_prequential_forecast_snapshots_and_phase_schedule=True,phase_fits=phases)
    save(path,dict(model=state(model),receipt=receipt));return model,receipt


def uniform_draws(case,device):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    b=broker(case,doc,quotas);rng=np.random.default_rng(0);draws=[]
    # Strong ordinary control uses distinct uniform actions, not repeated
    # draws; it gets the same quota but can reveal more unique teacher facts.
    selected={q:rng.choice(np.flatnonzero(query==q),size=n,replace=False).tolist() for q,n in quotas.items()}
    for stage in range(4):
        for q,n in sorted(quotas.items()):
            begin=sum(n//4+int(s<n%4) for s in range(stage));count=n//4+int(stage<n%4)
            for index in selected[q][begin:begin+count]:
                value,hit=observe(b,doc['order'][index],stage)
                draws.append(dict(index=int(index),uid=doc['order'][index],query=q,stage=stage,
                    probability=value,proposal_probability=None,repeated_logical_action=hit))
    assert len(draws)==sum(quotas.values()) and not any(r['repeated_logical_action'] for r in draws)
    return draws,b.receipt(complete=True)


def train_primary(case,folder,device,registration_sha256):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=make(mean,std,device);forecaster=make(mean,std,device,2)
    final=folder/'final.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False);assert saved['registration_sha256']==registration_sha256
        model.load_state_dict(saved['solver']);forecaster.load_state_dict(saved['forecaster']);return model,forecaster,saved
    checkpoint=folder/'closed.pt';b=broker(case,doc,quotas);risk=ResidualRisk(query);rng=np.random.default_rng(0)
    known=np.full(len(features),np.nan);draws=[];rounds=[];observed_residual=[]
    sopt=torch.optim.Adam(model.parameters(),lr=.001);ropt=torch.optim.Adam(forecaster.parameters(),lr=.001)
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['registration_sha256']==registration_sha256
        model.load_state_dict(saved['solver']);forecaster.load_state_dict(saved['forecaster'])
        sopt.load_state_dict(saved['solver_optimizer']);ropt.load_state_dict(saved['forecaster_optimizer'])
        risk=saved['risk'];rng.bit_generator.state=saved['rng_state'];known=saved['known'];draws=saved['draws'];rounds=saved['rounds'];observed_residual=saved['observed_residual']
        for row in draws:observe(b,row['uid'],row['stage'])
        frozen_solver=saved['frozen_solver'];frozen_forecaster=saved['frozen_forecaster']
    else:
        sopt,_=fit(model,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device,sopt)
        ropt=forecaster_fit(forecaster,auxiliary,ay,(ay-.5)**2,500,device,ropt)
        frozen_solver=state(model);frozen_forecaster=state(forecaster)
    fixed_s=make(mean,std,device);fixed_s.load_state_dict(frozen_solver)
    fixed_r=make(mean,std,device,2);fixed_r.load_state_dict(frozen_forecaster)
    initial_s={k:v.clone() for k,v in frozen_solver.items()};initial_r={k:v.clone() for k,v in frozen_forecaster.items()}
    for stage in range(len(rounds),4):
        forecast=logits(forecaster,features,device);frozen=logits(fixed_r,features,device)
        m=1/(1+np.exp(-forecast[:,0].astype('float64')));variance=1/(1+np.exp(-forecast[:,1].astype('float64')))
        frozen_variance=1/(1+np.exp(-frozen[:,1].astype('float64')))
        with torch.no_grad():
            jac=np.concatenate([model.last_layer_jacobian_norm(tensors(features[i:i+512],device)).cpu().numpy() for i in range(0,len(features),512)])
            fixed_jac=np.concatenate([fixed_s.last_layer_jacobian_norm(tensors(features[i:i+512],device)).cpu().numpy() for i in range(0,len(features),512)])
        previous=len(draws);changes_s=changes_r=0
        for q,n in sorted(quotas.items()):
            group=risk.groups[q]
            for _ in range(n//4+int(stage<n%4)):
                unknown=~np.isfinite(known[group]);priority=np.sqrt(variance[group].clip(min=1e-6))*jac[group]*unknown
                pi=coverage_distribution(priority)
                pi_s=coverage_distribution(np.sqrt(variance[group].clip(min=1e-6))*fixed_jac[group]*unknown)
                pi_r=coverage_distribution(np.sqrt(frozen_variance[group].clip(min=1e-6))*jac[group]*unknown)
                u=float(rng.random());local=min(int(np.searchsorted(np.cumsum(pi),u,side='right')),len(group)-1);index=int(group[local])
                changes_s+=local!=min(int(np.searchsorted(np.cumsum(pi_s),u,side='right')),len(group)-1)
                changes_r+=local!=min(int(np.searchsorted(np.cumsum(pi_r),u,side='right')),len(group)-1)
                proxy=m[group].copy();proxy[~unknown]=known[group[~unknown]]
                # Everything above is computed before requesting this value.
                value,hit=observe(b,doc['order'][index],stage)
                risk.record(q,proxy,index,value,float(pi[local]))
                observed_residual.append((value-float(proxy[local]))**2)
                draws.append(dict(index=index,uid=doc['order'][index],query=q,stage=stage,probability=value,
                    proposal_probability=float(pi[local]),proxy_before=float(proxy[local]),
                    repeated_logical_action=hit,full_stratum_support_size=len(group)))
                known[index]=value
        coefficient=risk.coefficients();weight=risk.population_weights()
        denom=np.asarray([risk.draws[int(q)] for q in query])
        np.savez(folder/f'risk_stage_{stage}.npz',risk_coefficients=coefficient,
                 prequential_proxy_mean=risk.proxy_sum/denom,population_weights=weight)
        sopt,sfit=fit(model,np.r_[features,auxiliary],np.r_[coefficient,ay],
            np.r_[weight,np.full(len(ay),ALPHA/len(ay))]/(1+ALPHA),STEPS[stage],device,sopt)
        indices=np.asarray([r['index'] for r in draws]);values=np.asarray([r['probability'] for r in draws])
        ropt=forecaster_fit(forecaster,np.r_[features[indices],auxiliary],np.r_[values,ay],
            np.r_[observed_residual,(ay-.5)**2],500,device,ropt)
        report=dict(stage=stage,new_logical_draws=len(draws)-previous,logical_draws=len(draws),
            unique_revealed=int(np.isfinite(known).sum()),solver_fit=sfit,
            risk_coefficient_min=float(coefficient.min()),risk_coefficient_max=float(coefficient.max()),
            local_choices_changed_with_frozen_solver=int(changes_s),local_choices_changed_with_frozen_forecaster=int(changes_r),
            counterfactual_only_one_step_same_history_not_quality_ablation=True)
        rounds.append(report)
        save(checkpoint,dict(solver=state(model),forecaster=state(forecaster),solver_optimizer=sopt.state_dict(),
            forecaster_optimizer=ropt.state_dict(),risk=risk,rng_state=rng.bit_generator.state,known=known,
            draws=draws,rounds=rounds,observed_residual=observed_residual,frozen_solver=frozen_solver,
            frozen_forecaster=frozen_forecaster,registration_sha256=registration_sha256))
        print(json.dumps(dict(status='RESIDUAL_RSI_CLOSED_ROUND',pair=case['pair'],**report)),flush=True)
    assert risk.draws==quotas and len(draws)==sum(quotas.values())
    changes=dict(solver_max_change=max(float((state(model)[k]-v).abs().max()) for k,v in initial_s.items()),
                 forecaster_max_change=max(float((state(forecaster)[k]-v).abs().max()) for k,v in initial_r.items()))
    assert min(changes.values())>0
    result=dict(solver=state(model),forecaster=state(forecaster),draws=draws,rounds=rounds,
        parameter_changes=changes,risk_coefficients=risk.coefficients(),registration_sha256=registration_sha256,
        broker=b.receipt(complete=False),logical_draws=len(draws),fixed_query_budget=quotas,
        original_teacher_responses_unchanged=True,actual_new_Qwen_calls=0,prepaid_replay_not_fresh_cost=True)
    save(final,result);atomic_json(folder/'draws.json',dict(draws=draws,quotas=quotas))
    return model,forecaster,result
