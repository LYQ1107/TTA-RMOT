"""Proper joint/marginal teacher fitting, with no GT-based model selection."""
import time
import torch
from torch.nn import functional as F


def probabilities(model,data):
    return F.log_softmax(model(data['anchors'],data['present'],data['first'],data['second']).double(),-1)


def risks(log_p,target,kind):
    target = target.double()
    target = target/target.sum(-1,keepdim=True)
    if kind == 'marginal':
        target = torch.stack((torch.stack((target[:,0]+target[:,2],target[:,1]+target[:,3]),-1),
            torch.stack((target[:,1]+target[:,2],target[:,0]+target[:,3]),-1)),-2)
        log_p = torch.stack((torch.stack((torch.logsumexp(log_p[:,(0,2)],-1),torch.logsumexp(log_p[:,(1,3)],-1)),-1),
            torch.stack((torch.logsumexp(log_p[:,(1,2)],-1),torch.logsumexp(log_p[:,(0,3)],-1)),-1)),-2)
    else:
        assert kind == 'joint'
    cross_entropy = -(target*log_p).sum(-1).mean()
    floor = -torch.xlogy(target,target).sum(-1).mean()
    return cross_entropy,floor


def change_normalization(model,old_mean,old_std,new_mean,new_std):
    layer = model.encoder[0]
    with torch.no_grad():
        old_mean,old_std,new_mean,new_std = [torch.as_tensor(v,dtype=layer.weight.dtype,device=layer.weight.device)
            for v in (old_mean,old_std,new_mean,new_std)]
        assert torch.all(old_std > 0) and torch.all(new_std > 0)
        original = layer.weight.clone()
        layer.weight.mul_(new_std/old_std)
        layer.bias.add_(original@((new_mean-old_mean)/old_std))


def fit(model,data,*,minimum_steps=4000,maximum_steps=16000,kind='joint',gate=.005,require_gate=True):
    """A failed ordinary adequacy gate is an error, never an accepted baseline.

    Structural misspecification is possible. Callers must distinguish that
    from failed numerical optimization rather than claiming semantic learning.
    """
    assert 0 < minimum_steps <= maximum_steps and kind in ('joint','marginal')
    device = next(model.parameters()).device
    assert torch.all(data['target'] >= 0) and torch.isfinite(data['target']).all()
    if device.type == 'cuda':torch.cuda.synchronize(device)
    start = time.perf_counter();optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    early = None;step = 0;milestones = [];limit = minimum_steps
    while step < maximum_steps:
        model.train()
        while step < limit:
            optimizer.zero_grad(set_to_none=True)
            log_p = probabilities(model,data)
            loss,_ = risks(log_p,data['target'],kind)
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();step += 1
            if step == max(1,minimum_steps//4):
                early = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            ce,floor = risks(probabilities(model,data),data['target'],kind)
        excess = float(ce-floor)
        milestones.append(dict(steps=step,teacher_cross_entropy=float(ce),teacher_entropy_floor=float(floor),excess_KL=excess))
        if excess <= gate or step == maximum_steps:break
        limit = min(maximum_steps,limit+minimum_steps)
    if device.type == 'cuda':torch.cuda.synchronize(device)
    receipt = dict(kind=kind,steps=step,target_rows=len(data['target']),minimum_steps=minimum_steps,maximum_steps=maximum_steps,
        teacher_excess_gate=gate,teacher_fit_gate_passed=excess <= gate,milestones=milestones,
        wall_seconds=time.perf_counter()-start,manual_GT_or_quality_checkpoint_selection=False,
        risk_units='nats per binary endpoint' if kind == 'marginal' else 'nats per joint pair')
    if require_gate and not receipt['teacher_fit_gate_passed']:
        raise RuntimeError('Complete ordinary capacity/optimization before quality comparison: '+str(receipt))
    return early,receipt
