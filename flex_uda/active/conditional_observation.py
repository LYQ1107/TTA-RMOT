"""Conditional bag support and relative-observation learning primitives.

A provided TrackID is a bag of observations, not a certified physical ID.
If A means this bag contains a match and Y_t means an observation matches,
Y_t implies A. The chain P(Y_t)=P(A)P(Y_t|A) does not identify P(A) from
sparse instance answers. No per-factor pseudo labels are manufactured.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def joint_task_logit(bag_logit, conditional_logit):
    """logit(sigmoid(u)*sigmoid(v)), without rounded probability endpoints."""
    u, v = torch.broadcast_tensors(bag_logit, conditional_logit)
    return u+v-torch.logsumexp(torch.stack((torch.zeros_like(u), u, v)), dim=0)


def joint_log_probability(bag_logit, conditional_logit):
    return F.logsigmoid(joint_task_logit(bag_logit, conditional_logit))


def weighted_bernoulli_loss(task_logit, target, weight):
    return (weight*F.binary_cross_entropy_with_logits(task_logit, target, reduction='none')).sum()/weight.sum()


def relative_choice_kl(log_probability, target, group, weight):
    """Conditional choice KL within matched query/bag/actual-view groups.

    Teacher choice mass is w_j*p_j. Predicted choice mass is w_j*q_j.
    Multiplying every q in a group by the same positive bag-support term
    cancels. Singleton and all-zero teacher groups contribute zero. This is
    a relative target from existing answers, not a new true instance label.
    """
    assert log_probability.ndim == target.ndim == group.ndim == weight.ndim == 1
    assert log_probability.shape == target.shape == group.shape == weight.shape
    n = int(group.max())+1
    teacher_mass = weight*target
    group_teacher = torch.zeros(n, dtype=target.dtype, device=target.device).scatter_add(0, group, teacher_mass)
    log_mass = torch.log(weight)+log_probability
    maximum = torch.full((n,), -torch.inf, dtype=log_probability.dtype, device=log_probability.device)
    maximum = maximum.scatter_reduce(0, group, log_mass, reduce='amax', include_self=True)
    total = torch.zeros_like(maximum).scatter_add(0, group, torch.exp(log_mass-maximum[group]))
    log_normalizer = maximum+torch.log(total)
    tiny = torch.finfo(target.dtype).tiny
    log_target = torch.log(teacher_mass.clamp_min(tiny))-torch.log(group_teacher[group].clamp_min(tiny))
    return (teacher_mass*(log_target-(log_mass-log_normalizer[group]))).sum()/weight.sum()


class BagContextHead(nn.Module):
    """Learned shared bag context and a current-observation selector.

    Inputs already include the same available PCD/geometry/scope features
    for every comparator. Source networks remain external and frozen.
    The latent bag score is not claimed a uniquely identified semantic
    posterior; only the joint task prediction is evaluated as output.
    """
    def __init__(self, feature_dimension, width=32, kind='conditional'):
        super().__init__()
        assert kind in ('conditional', 'direct_context', 'local_two_factor')
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(feature_dimension, width), nn.GELU())
        self.local_head = nn.Sequential(nn.Linear(3*width, 2*width), nn.GELU(), nn.Linear(2*width, 1))
        self.bag_head = nn.Linear(2*width if kind == 'conditional' else 3*width, 1)
        nn.init.zeros_(self.local_head[-1].weight); nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.bag_head.weight); nn.init.zeros_(self.bag_head.bias)
        # Every architecture starts with the same task probability 0.5.
        if kind != 'direct_context':
            bias = math.log(math.sqrt(.5)/(1.-math.sqrt(.5)))
            nn.init.constant_(self.local_head[-1].bias, bias)
            nn.init.constant_(self.bag_head.bias, bias)

    def forward(self, observations, present, current):
        assert observations.ndim == 3 and present.shape == observations.shape[:2]
        assert bool(present.any(1).all())
        encoded = self.encoder(observations); mask = present.unsqueeze(-1)
        mean = encoded.masked_fill(~mask, 0).sum(1)/present.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~mask, -torch.inf).max(1).values
        context = torch.cat((mean, maximum), 1)
        local = self.encoder(current); joined = torch.cat((local, context), 1)
        conditional_logit = self.local_head(joined).squeeze(-1)
        if self.kind == 'direct_context':
            return conditional_logit
        bag_logit = self.bag_head(context if self.kind == 'conditional' else joined).squeeze(-1)
        return joint_task_logit(bag_logit, conditional_logit)
