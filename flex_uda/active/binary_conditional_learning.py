"""Conditional learning from native binary teacher replies, no new labels.

The product below is an explicit independence MODEL of two paid binary
opinions, not an observed joint teacher answer. Ordinary binary KD must keep
the original probabilities as a strong control in any later quality study.
"""
import torch
from torch.nn import functional as F


def factorized_pair_target(first_probability, second_probability):
    a,b=first_probability.double(),second_probability.double()
    assert bool(((a>=0)&(a<=1)&(b>=0)&(b<=1)).all())
    return torch.stack((a*(1-b),(1-a)*b,a*b,(1-a)*(1-b)),-1)


def conditional_pair_risk(first_utility, second_utility, first_probability, second_probability):
    q=factorized_pair_target(first_probability,second_probability)
    one=q[...,:2].sum(-1)
    relative=q[...,:2]/one.unsqueeze(-1).clamp_min(1e-300)
    log_relative=F.log_softmax(torch.stack((first_utility,second_utility),-1).double(),-1)
    # Zero exactly-one teacher mass gives zero conditional learning signal.
    return (-(relative*log_relative).sum(-1)*one).mean()


def count_and_conditional_risks(log_probability, teacher_pair):
    q=teacher_pair.double();q=q/q.sum(-1,keepdim=True)
    logp=F.log_softmax(log_probability.double(),-1)
    count_target=torch.stack((q[...,3],q[...,:2].sum(-1),q[...,2]),-1)
    log_count=torch.stack((logp[...,3],torch.logsumexp(logp[...,:2],-1),logp[...,2]),-1)
    count=-(count_target*log_count).sum(-1).mean()
    conditional=-(q[...,:2]*F.log_softmax(logp[...,:2],-1)).sum(-1).mean()
    return count,conditional
