"""Strong KD control with prefix-only channel statistics and full teacher fit."""
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def fit_head(x,z,y,current,current_z,mean,std,*,soft=True,seed=0):
    begin=time.perf_counter();device=torch.device('cuda:0');torch.manual_seed(seed)
    x=torch.tensor((x-mean)/std,dtype=torch.float32,device=device)
    z=torch.tensor(z,dtype=torch.float32,device=device)
    y=torch.tensor(y if soft else np.asarray(y)>=.5,dtype=torch.float32,device=device)
    cx=torch.tensor((current-mean)/std,dtype=torch.float32,device=device);cz=torch.tensor(current_z,dtype=torch.float32,device=device)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Linear(64,1)).to(device)
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator(device=device).manual_seed(seed);predictions={};loss_trace=[];checkpoint_wall={}
    entropy=float((-(y*torch.log(y.clamp_min(1e-8))+(1-y)*torch.log((1-y).clamp_min(1e-8)))).mean())
    final=None
    for step in range(1,16001):
        ids=torch.randperm(len(y),generator=rng,device=device)[:512]
        loss=F.binary_cross_entropy_with_logits(z[ids]+model(x[ids]).squeeze(-1),y[ids])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step in (400,4000) or (step>=4000 and step%100==0):
            with torch.no_grad():
                final=float(F.binary_cross_entropy_with_logits(z+model(x).squeeze(-1),y))
                if step in (400,4000):
                    predictions[str(step)]=torch.sigmoid(cz+model(cx).squeeze(-1)).cpu().numpy()
                    checkpoint_wall[str(step)]=time.perf_counter()-begin
            loss_trace.append(dict(step=step,teacher_CE=final,teacher_KL=final-entropy))
            if step>=4000 and final-entropy<=.005:break
    with torch.no_grad():predictions['CONVERGED']=torch.sigmoid(cz+model(cx).squeeze(-1)).cpu().numpy()
    torch.cuda.synchronize(device)
    checkpoint_wall['CONVERGED']=time.perf_counter()-begin
    receipt=dict(status='PREFIX_CHANNEL_NORMALIZED_ACTUAL_KD',soft=soft,steps=step,seed=seed,
        teacher_examples=len(y),teacher_entropy=entropy,teacher_CE=final,teacher_KL=final-entropy,
        teacher_fit_within_0_005=final-entropy<=.005,trace=loss_trace,wall_seconds=time.perf_counter()-begin,
        checkpoint_wall_seconds=checkpoint_wall,
        no_future_normalization=True,no_target_GT_used=True,raw_feature_dimension=x.shape[1],
        architecture='D->64 GELU->1, source-logit residual',restart_from_same_seed_each_video=True)
    saved=dict(model=model.state_dict(),mean=mean,std=std,receipt=receipt)
    return predictions,receipt,saved
