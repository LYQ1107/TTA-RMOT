"""Teacher-supported, source-anchored MAP fast calibration; no GT inputs."""
import numpy as np
import torch
import torch.nn.functional as F


def minimize_map(parameters, objective):
    optimizer=torch.optim.LBFGS(parameters,max_iter=50,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad()
        data,prior=objective();loss=data+prior
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source-MAP objective')
        loss.backward();return loss
    before=tuple(float(x.detach()) for x in objective())
    optimizer.step(closure)
    after=tuple(float(x.detach()) for x in objective())
    if not np.isfinite(after).all():raise FloatingPointError('Nonfinite source-MAP solution')
    return dict(data_before=before[0],prior_before=before[1],objective_before=sum(before),
        data_after=after[0],prior_after=after[1],objective_after=sum(after),closure_calls=calls,
        optimizer_iterations=int(optimizer.state[parameters[0]].get('n_iter',0)))


def source_map_fast_step(state,presence_logits,*,source_intercepts,temperature=1.,bias=0.):
    pseudo=state.pseudo_labels();seen=np.zeros_like(pseudo,dtype=bool)
    for event in state.events:seen[event['query'],event['track']]=True
    agrees=((pseudo==1)&(state.evidence>0))|((pseudo==0)&(state.evidence<0))
    valid=(pseudo>=0)&seen&agrees
    positive=int(((pseudo==1)&valid).sum());negative=int(((pseudo==0)&valid).sum())
    present=state.confirmed_present&~state.absent
    absent=state.absent&~state.confirmed_present
    confirmed=present|absent
    identity=dict(status='SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES',positive=positive,negative=negative,
        unqueried_confident_excluded=int(((pseudo>=0)&~seen).sum()),
        queried_confident_without_agreeing_evidence=int(((pseudo>=0)&seen&~agrees).sum()))
    existence=dict(status='SKIP_INSUFFICIENT_CONFIRMED_CLASSES',positive=int(present.sum()),negative=int(absent.sum()),
        conflicting_queries=int((state.confirmed_present&state.absent).sum()))
    if positive and negative:
        indices=np.flatnonzero(valid.any(1));mask=torch.tensor(valid[indices])
        a=torch.tensor(state.a[indices],dtype=torch.float64,requires_grad=True)
        anchor=torch.tensor(np.asarray(source_intercepts)[indices],dtype=torch.float64)
        u=torch.tensor(np.log(state.slope),dtype=torch.float64,requires_grad=True)
        z=torch.tensor(state.z[indices],dtype=torch.float64)
        target=torch.tensor(pseudo[indices].clip(0),dtype=torch.float64)
        def objective():
            b=u.exp()
            data=F.binary_cross_entropy_with_logits((a[:,None]+b*z)[mask],target[mask],reduction='sum')
            prior=.5*((a-anchor).square().sum()+(b-1).square())
            return data,prior
        identity.update(minimize_map([a,u],objective),status='MAP_UPDATED',queries=indices.tolist())
        intercept=state.a.copy();intercept[indices]=a.detach().numpy()
        state.replace_prior(intercept=intercept,slope=float(u.detach().exp()))
    if present.any() and absent.any():
        log_t=torch.tensor(np.log(temperature),dtype=torch.float64,requires_grad=True)
        beta=torch.tensor(bias,dtype=torch.float64,requires_grad=True)
        value=torch.tensor(np.asarray(presence_logits).max(axis=1)[confirmed],dtype=torch.float64)
        target=torch.tensor(present[confirmed],dtype=torch.float64)
        def objective():
            data=F.binary_cross_entropy_with_logits(value/log_t.exp()+beta,target,reduction='sum')
            return data,.5*(log_t.square()+beta.square())
        existence.update(minimize_map([log_t,beta],objective),status='MAP_UPDATED')
        temperature=float(log_t.detach().exp());bias=float(beta.detach())
    updated=identity['status']=='MAP_UPDATED' or existence['status']=='MAP_UPDATED'
    return dict(status='SOURCE_MAP_UPDATED' if updated else 'SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES',
        identity=identity,presence=existence,temperature=temperature,bias=bias,slope=state.slope,
        source_prior_variance=1.,source_slope_anchor=1.,source_temperature_anchor=1.,source_bias_anchor=0.,
        objective='sum BCE + fixed unit source Gaussian priors',teacher_evidence_not_reapplied=True,
        support_uses_target_GT=False)
