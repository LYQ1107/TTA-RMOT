"""Teacher-loss-only polishing of an existing ordinary direct head.

The architecture, inputs, raw confidence weights and 0.005 fit criterion stay
fixed. This is baseline optimization, not an RMOT innovation or GT selector.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn


def polish_head(x, y, sample_weight, checkpoint, *, device='cuda:0', tolerance=.005):
    started=time.perf_counter();device=torch.device(device)
    mean,std=checkpoint['mean'],checkpoint['std']
    x=torch.as_tensor((np.asarray(x)-mean)/std,dtype=torch.float32,device=device)
    target=torch.as_tensor(y,dtype=torch.float32,device=device)
    weight=torch.as_tensor(sample_weight,dtype=torch.float32,device=device)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Linear(64,1)).to(device)
    model.load_state_dict(checkpoint['model'])
    entropy=float(((-(target*torch.log(target)+(1-target)*torch.log1p(-target)))*weight).sum()/weight.sum())
    def loss():
        values=torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1),target,reduction='none')
        return (values*weight).sum()/weight.sum()
    initial=float(loss().detach());current=initial
    linear_iterations=linear_evaluations=0;linear_accepted=False;chunks=closures=0
    if current-entropy>tolerance:
        with torch.no_grad():hidden=model[1](model[0](x)).cpu().numpy().astype(float)
        design=np.c_[hidden,np.ones(len(hidden))]
        numpy_target=target.cpu().numpy().astype(float)
        numpy_weight=weight.cpu().numpy().astype(float);numpy_weight/=numpy_weight.sum()
        start=np.r_[model[-1].weight.detach().cpu().numpy().ravel(),model[-1].bias.detach().cpu().numpy()]
        def objective(parameter):
            z=design@parameter
            return float(numpy_weight@(np.logaddexp(0.,z)-numpy_target*z)),design.T@(numpy_weight*(expit(z)-numpy_target))
        fitted=minimize(objective,start,jac=True,method='L-BFGS-B',
            options=dict(maxiter=1000,maxls=50,gtol=1e-10,ftol=1e-14,maxcor=30))
        linear_iterations=int(fitted.nit);linear_evaluations=int(fitted.nfev)
        with torch.no_grad():
            model[-1].weight.copy_(torch.as_tensor(fitted.x[:-1],dtype=torch.float32,device=device)[None])
            model[-1].bias.copy_(torch.as_tensor(fitted.x[-1:],dtype=torch.float32,device=device))
        proposed=float(loss().detach())
        if proposed<=current+1e-7:
            current=proposed;linear_accepted=True
        else:
            with torch.no_grad():
                model[-1].weight.copy_(torch.as_tensor(start[:-1],dtype=torch.float32,device=device)[None])
                model[-1].bias.copy_(torch.as_tensor(start[-1:],dtype=torch.float32,device=device))
    if current-entropy>tolerance:
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,
            history_size=20,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
        def closure():
            nonlocal closures
            optimizer.zero_grad(set_to_none=True);value=loss();value.backward();closures+=1
            return value
        for _ in range(10):
            optimizer.step(closure);chunks+=1;current=float(loss().detach())
            if current-entropy<=tolerance:break
    final=float(loss().detach())
    assert np.isfinite(final) and final<=initial+1e-6,(initial,final)
    if device.type=='cuda':torch.cuda.synchronize(device)
    receipt=dict(status='TEACHER_ONLY_CONDITIONAL_HEAD_FIT_COMPLETION',teacher_examples=len(target),
        teacher_entropy=entropy,initial_teacher_CE=initial,initial_teacher_KL=initial-entropy,
        teacher_CE=final,teacher_KL=final-entropy,teacher_fit_within_0_005=final-entropy<=tolerance,
        linear_iterations=linear_iterations,linear_function_evaluations=linear_evaluations,
        linear_step_accepted=linear_accepted,full_head_LBFGS_100_iteration_blocks=chunks,
        full_head_loss_evaluations=closures,architecture='Same input->64GELU->1 direct head',
        optimization_selected_only_by_prefix_teacher_loss=True,no_target_GT_or_future_data=True,
        wall_seconds=time.perf_counter()-started)
    saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        mean=mean,std=std,receipt=receipt)
    return saved,receipt
