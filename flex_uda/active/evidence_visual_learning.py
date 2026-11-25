"""Evidence-anchored visual learning with a student-error observer.

The original teacher is external and immutable. Image correspondence only
regularizes a query-independent CONTENT representation; query/state scores
remain free to vary. No signed probability targets or semantic-score
smoothness. The observer forecasts actual pre-answer STUDENT error, not
its own confidence as truth. This is a new hypothesis, not a proved gain.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.residual_rsi_learning import (
    data,broker,observe,paired_training_table,state,save,tensors)
from flex_uda.active.residual_risk import coverage_distribution

LR=.0001
STEPS=(500,500,500,4000)
CURRENT=slice(7815,8583)
VISUAL_WEIGHT=.1


def bounded(raw):
    # Polynomial derivative avoids the observed float32 tanh dead region.
    return 8*raw/(8+raw.abs())


class PlainHead(nn.Module):
    def __init__(self,mean,std,width=288):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.network=nn.Sequential(nn.Linear(len(mean),width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
    def forward(self,x):return bounded(self.network((x-self.mean)/self.std)).flatten()


class VisualHead(nn.Module):
    def __init__(self,mean,std,width=128,outputs=1):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.app=nn.Sequential(nn.Linear(768,128),nn.SiLU(),nn.Linear(128,64))
        self.context=nn.Sequential(nn.Linear(len(mean),width),nn.SiLU())
        self.gate=nn.Linear(width,64)
        self.hidden=nn.Sequential(nn.Linear(width+64,width),nn.SiLU())
        self.output=nn.Linear(width,outputs)
    def representation(self,x):
        h=self.context((x-self.mean)/self.std)
        content=self.app(x[:,CURRENT])*self.gate(h).sigmoid()
        return self.hidden(torch.cat((h,content),dim=1))
    def forward(self,x):
        result=bounded(self.output(self.representation(x)))
        return result.flatten() if result.shape[-1]==1 else result
    def last_layer_jacobian_norm(self,x):
        h=self.representation(x);raw=self.output(h)[:,0]
        return torch.sqrt(h.square().sum(1)+1)*64/(8+raw.abs()).square()


class MeanHead(nn.Module):
    def __init__(self,r):super().__init__();self.r=r
    def forward(self,x):return self.r(x)[:,0]


def make(mean,std,device,kind='visual',width=128,outputs=1):
    torch.manual_seed(0)
    model=PlainHead(mean,std,width) if kind=='plain' else VisualHead(mean,std,width,outputs)
    return model.to(device)


@torch.no_grad()
def logits(model,x,device):
    return np.concatenate([model(tensors(x[i:i+512],device)).cpu().numpy() for i in range(0,len(x),512)])


def pixel_tensors(case,device):
    with np.load(case['pixel_pairs_path']) as p:
        return {k:tensors(p[k],device) for k in ('left','right','weight')}


def content_loss(model,pixel):
    """VICReg's prior variance/invariance/covariance idea, not our invention.

No negatives: different people can share the same referring attributes.
Confidence weights apply to pair agreement, not to semantic labels.
"""
    a,b=model.app(pixel['left']),model.app(pixel['right'])
    weight=pixel['weight'];invariance=((a-b).square().mean(1)*weight).sum()/weight.sum()
    variance=covariance=0
    for x in (a,b):
        variance=variance+torch.relu(1-torch.sqrt(x.var(0,unbiased=True)+1e-4)).mean()/2
        centered=x-x.mean(0);cov=centered.T@centered/(len(x)-1)
        off=cov-torch.diag_embed(cov.diagonal())
        covariance=covariance+off.square().sum()/x.shape[1]
    return (25*invariance+25*variance+covariance)/50


def excess(model,x,y,w,device):
    z=logits(model,x,device).astype('float64')
    h=-y*np.log(np.clip(y,1e-15,1))-(1-y)*np.log(np.clip(1-y,1e-15,1))
    return float(np.sum(w*(np.logaddexp(0,z)-y*z-h))/w.sum())


def fit(model,x,y,w,steps,device,optimizer=None,*,pixel=None,align=None,unlabeled=None,residual=None):
    opt=optimizer or torch.optim.Adam(model.parameters(),lr=LR)
    x,y,w=tensors(x,device),tensors(y,device),tensors(w,device)
    residual=None if residual is None else tensors(residual,device)
    assert torch.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    assert torch.isfinite(w).all() and (w>=0).all() and w.sum()>0
    cursor=torch.arange(512,device=device);started=time.perf_counter()
    if align is not None:
        ux=tensors(unlabeled,device)
        with torch.no_grad():target=align.app(ux[:,CURRENT]).detach()
    for step in range(steps):
        i=(cursor+512*step)%len(x);opt.zero_grad(set_to_none=True)
        out=model(x[i]);z=out if residual is None else out[:,0]
        loss=((nn.functional.softplus(z)-y[i]*z)*w[i]*len(x)).mean()
        if residual is not None:
            loss=loss+((out[:,1].sigmoid()-residual[i]).square()*w[i]*len(x)).mean()
        if pixel is not None:loss=loss+VISUAL_WEIGHT*content_loss(model,pixel)
        if align is not None:
            j=(cursor+512*step)%len(ux)
            loss=loss+VISUAL_WEIGHT*nn.functional.mse_loss(model.app(ux[j,CURRENT]),target[j])
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
    torch.cuda.synchronize()
    return opt,dict(steps=steps,wall_seconds=time.perf_counter()-started,last_objective=float(loss.detach()),
        immutable_teacher_targets=True,nonnegative_BCE_weights=True)


def teacher_table(features,query,draws,quotas,auxiliary,ay,mode='SOFT_KD'):
    # All modes preserve acquired teacher probabilities, except the explicit
    # standard hard-label baseline. IPW weights are positive, never targets.
    return paired_training_table(features,query,draws,quotas,auxiliary,ay,mode)


def fit_control(case,draws,mode,folder,device):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    hard=mode=='HARD_KD'
    if hard:ay=(ay>=.5).astype(float)
    is_ssl=mode=='SSL_KD'
    model=make(mean,std,device,kind='visual' if is_ssl else 'plain',width=288)
    path=Path(folder)/(mode+'.pt');partial=Path(folder)/(mode+'_partial.pt')
    if path.exists():
        done=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(done['model']);return model,done['receipt']
    pixel=pixel_tensors(case,device) if is_ssl else None
    opt=torch.optim.Adam(model.parameters(),lr=LR);receipts=[];completed=-1
    if partial.exists():
        saved=torch.load(partial,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        receipts=saved['receipts'];completed=saved['completed']
    else:
        opt,r=fit(model,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device,opt,pixel=pixel);receipts.append(r)
    for stage,steps in enumerate(STEPS):
        if stage<=completed:continue
        prefix=[r for r in draws if r['stage']<=stage]
        x,y,w=teacher_table(features,query,prefix,quotas,auxiliary,ay,'HARD_KD' if hard else ('IPW_KD' if mode=='IPW_KD' else 'SOFT_KD'))
        opt,r=fit(model,x,y,w,steps,device,opt,pixel=pixel);receipts.append(r)
        save(partial,dict(model=state(model),optimizer=opt.state_dict(),receipts=receipts,completed=stage))
    x,y,w=teacher_table(features,query,draws,quotas,auxiliary,ay,'HARD_KD' if hard else ('IPW_KD' if mode=='IPW_KD' else 'SOFT_KD'))
    total=sum(r['steps'] for r in receipts);error=excess(model,x,y,w,device)
    # A proper teacher-only projection is available equally to SSL and the
    # primary. It prevents independent feature pretext from weakening KD.
    while error>.005 and total<16000:
        opt,r=fit(model,x,y,w,500,device,opt);receipts.append(r);total+=500;error=excess(model,x,y,w,device)
        save(partial,dict(model=state(model),optimizer=opt.state_dict(),receipts=receipts,completed=3))
    assert error<=.005,(case['pair'],mode,error)
    receipt=dict(mode=mode,total_steps=total,proper_teacher_excess_CE=error,adequacy_limit=.005,
        initial_and_phase_order_matches_primary=True,Adam_learning_rate=LR,width=288,
        parameter_count=sum(p.numel() for p in model.parameters()),phase_fits=receipts,
        SSL_gets_same_pixels_and_prior_VICReg_pretext=is_ssl)
    save(path,dict(model=state(model),receipt=receipt));return model,receipt


def train_primary(case,folder,device,registration_sha256):
    features,query,doc,quotas,auxiliary,ay,mean,std=data(case)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    solver=make(mean,std,device);observer=make(mean,std,device,outputs=2)
    final=folder/'final.pt'
    if final.exists():
        saved=torch.load(final,map_location='cpu',weights_only=False);assert saved['registration_sha256']==registration_sha256
        solver.load_state_dict(saved['solver']);observer.load_state_dict(saved['observer']);return solver,observer,saved
    checkpoint=folder/'closed.pt';pixel=pixel_tensors(case,device)
    sopt=torch.optim.Adam(solver.parameters(),lr=LR);ropt=torch.optim.Adam(observer.parameters(),lr=LR)
    rng=np.random.default_rng(0);b=broker(case,doc,quotas);known=np.full(len(features),np.nan)
    draws=[];rounds=[];student_error=[];receipts=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['registration_sha256']==registration_sha256
        solver.load_state_dict(saved['solver']);observer.load_state_dict(saved['observer'])
        sopt.load_state_dict(saved['sopt']);ropt.load_state_dict(saved['ropt']);rng.bit_generator.state=saved['rng']
        draws=saved['draws'];rounds=saved['rounds'];student_error=saved['student_error'];known=saved['known'];receipts=saved['receipts']
        fixed_s,fixed_r,auxerror=saved['fixed_s'],saved['fixed_r'],saved['auxerror']
        for row in draws:observe(b,row['uid'],row['stage'])
    else:
        sopt,r=fit(solver,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device,sopt);receipts.append(r)
        p=1/(1+np.exp(-logits(solver,auxiliary,device).astype('float64')));auxerror=(p-ay)**2
        ropt,_=fit(observer,auxiliary,ay,np.full(len(ay),1/len(ay)),500,device,ropt,pixel=pixel,residual=auxerror)
        fixed_s,fixed_r=state(solver),state(observer)
    frozen_s=make(mean,std,device);frozen_s.load_state_dict(fixed_s)
    frozen_r=make(mean,std,device,outputs=2);frozen_r.load_state_dict(fixed_r)
    for stage in range(len(rounds),4):
        z=logits(solver,features,device).astype('float64');student=1/(1+np.exp(-z))
        predicted=logits(observer,features,device);frozen=logits(frozen_r,features,device)
        variance=1/(1+np.exp(-predicted[:,1].astype('float64')))
        fixed_variance=1/(1+np.exp(-frozen[:,1].astype('float64')))
        with torch.no_grad():
            jac=np.concatenate([solver.last_layer_jacobian_norm(tensors(features[i:i+512],device)).cpu().numpy() for i in range(0,len(features),512)])
            fixed_jac=np.concatenate([frozen_s.last_layer_jacobian_norm(tensors(features[i:i+512],device)).cpu().numpy() for i in range(0,len(features),512)])
        cs=cr=0;before=len(draws)
        for q,n in sorted(quotas.items()):
            group=np.flatnonzero(query==q)
            for _ in range(n//4+int(stage<n%4)):
                unseen=~np.isfinite(known[group])
                pi=coverage_distribution(np.sqrt(variance[group].clip(min=1e-6))*jac[group]*unseen)
                ps=coverage_distribution(np.sqrt(variance[group].clip(min=1e-6))*fixed_jac[group]*unseen)
                pr=coverage_distribution(np.sqrt(fixed_variance[group].clip(min=1e-6))*jac[group]*unseen)
                u=float(rng.random())
                def choice(prob):return min(int(np.searchsorted(np.cumsum(prob),u,side='right')),len(group)-1)
                local=choice(pi);cs+=local!=choice(ps);cr+=local!=choice(pr);index=int(group[local])
                value,hit=observe(b,doc['order'][index],stage)
                err=(student[index]-value)**2
                draws.append(dict(index=index,uid=doc['order'][index],query=q,stage=stage,probability=value,
                    proposal_probability=float(pi[local]),student_probability_BEFORE_answer=float(student[index]),
                    student_squared_error=float(err),repeated_logical_action=hit))
                student_error.append(float(err));known[index]=value
        x,y,w=teacher_table(features,query,draws,quotas,auxiliary,ay)
        sopt,sfit=fit(solver,x,y,w,STEPS[stage],device,sopt,align=observer,unlabeled=features);receipts.append(sfit)
        ropt,rfit=fit(observer,x,y,w,500,device,ropt,pixel=pixel,residual=np.r_[student_error,auxerror])
        row=dict(stage=stage,new_logical_draws=len(draws)-before,logical_draws=len(draws),unique_teacher_actions=int(np.isfinite(known).sum()),
            solver_fit=sfit,observer_fit=rfit,known_teacher_KL=excess(solver,x,y,w,device),
            local_choices_changed_with_frozen_solver=int(cs),local_choices_changed_with_frozen_observer=int(cr),
            one_step_same_history_counterfactual_not_quality_ablation=True)
        rounds.append(row)
        save(checkpoint,dict(solver=state(solver),observer=state(observer),sopt=sopt.state_dict(),ropt=ropt.state_dict(),
            rng=rng.bit_generator.state,known=known,draws=draws,student_error=student_error,rounds=rounds,
            receipts=receipts,fixed_s=fixed_s,fixed_r=fixed_r,auxerror=auxerror,registration_sha256=registration_sha256))
        print(json.dumps(dict(status='EVIDENCE_VISUAL_RSI_CLOSED_ROUND',pair=case['pair'],**row)),flush=True)
    assert len(draws)==sum(quotas.values())
    x,y,w=teacher_table(features,query,draws,quotas,auxiliary,ay)
    projection=folder/'anchor_projection.pt'
    if projection.exists():
        projected=torch.load(projection,map_location='cpu',weights_only=False)
        solver.load_state_dict(projected['solver']);sopt.load_state_dict(projected['optimizer']);receipts=projected['receipts']
    error=excess(solver,x,y,w,device);total=sum(r['steps'] for r in receipts)
    while error>.005 and total<16000:
        sopt,r=fit(solver,x,y,w,500,device,sopt);receipts.append(r);total+=500;error=excess(solver,x,y,w,device)
        save(projection,dict(solver=state(solver),optimizer=sopt.state_dict(),receipts=receipts,proper_teacher_KL=error))
    assert error<=.005,(case['pair'],'PRIMARY_TEACHER_ANCHOR',error)
    changes=dict(solver=max(float((state(solver)[k]-v).abs().max()) for k,v in fixed_s.items()),
        observer=max(float((state(observer)[k]-v).abs().max()) for k,v in fixed_r.items()))
    assert min(changes.values())>0
    result=dict(solver=state(solver),observer=state(observer),draws=draws,rounds=rounds,parameter_changes=changes,
        original_teacher_targets_unchanged=True,no_signed_semantic_targets=True,proper_known_teacher_KL=error,
        solver_total_steps=total,solver_fits=receipts,broker=b.receipt(complete=False),registration_sha256=registration_sha256,
        Qwen_original_frozen=True,auxiliary_error_is_after_bootstrap_not_preanswer=True,actual_new_teacher_GPU_calls=0,
        source_inputs_prepaid_not_fresh_cost=True)
    save(final,result);atomic_json(folder/'draws.json',dict(draws=draws,quotas=quotas))
    return solver,observer,result
