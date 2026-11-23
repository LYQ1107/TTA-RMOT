"""Candidate joint observation learner, not an accepted RMOT method.

Four events are A-only, B-only, both and neither. Shared object evidence and
local changes parameterize a pair distribution; a signed interaction permits
negative as well as positive association. These are model potentials, not
identified true identity/existence posteriors. No target annotations enter.
"""
import torch
from torch import nn
from torch.nn import functional as F

SWAP = (1, 0, 2, 3)


def pair_log_probabilities(a, b, interaction):
    scores = torch.stack((a, b, a+b+interaction, torch.zeros_like(a)), -1)
    return F.log_softmax(scores, -1)


def marginal_logits(log_probability):
    a = torch.logsumexp(log_probability[..., (0, 2)], -1)-torch.logsumexp(log_probability[..., (1, 3)], -1)
    b = torch.logsumexp(log_probability[..., (1, 2)], -1)-torch.logsumexp(log_probability[..., (0, 3)], -1)
    return torch.stack((a, b), -1)


def count_probabilities(probability):
    return torch.stack((probability[..., 3], probability[..., 0]+probability[..., 1], probability[..., 2]), -1)


def categorical_KL(target, log_probability):
    cross = torch.where(target > 0, target*log_probability, torch.zeros_like(target))
    return (torch.xlogy(target, target)-cross).sum(-1)


def count_conditional_KL(target, log_probability):
    """Exact chain decomposition; reweighting it is not extra supervision."""
    count_target = count_probabilities(target)
    count_log_p = torch.stack((log_probability[..., 3],
        torch.logsumexp(log_probability[..., :2], -1), log_probability[..., 2]), -1)
    count = categorical_KL(count_target, count_log_p)
    mass = target[..., :2].sum(-1)
    conditional_target = target[..., :2]/mass.unsqueeze(-1).clamp_min(torch.finfo(target.dtype).tiny)
    conditional_log_p = F.log_softmax(log_probability[..., :2], -1)
    contrast = mass*categorical_KL(conditional_target, conditional_log_p)
    return count, contrast


class JointObservationHead(nn.Module):
    """Small shared encoder with structured or adequate generic pair outputs.

    Both learners see the same original observations and unordered anchor set.
    Generic four-class KD explicitly receives the full joint target. Its
    swapped log-probability mixture enforces the same A/B equivariance.
    """
    def __init__(self, dimension, width=64, *, generic=False, shared=True, coupled=True):
        super().__init__()
        self.generic, self.shared, self.coupled = generic, shared, coupled
        self.encoder = nn.Sequential(nn.Linear(dimension, width), nn.Tanh())
        if generic:
            # Preserve every encoded anchor and its mask, rather than giving
            # the generic control only lossy mean/max summaries while the
            # structured model also uses anchor-wise local functions.
            self.output = nn.Sequential(nn.Linear(7*width+3, 2*width), nn.Tanh(), nn.Linear(2*width, 4))
        else:
            self.object = nn.Sequential(nn.Linear(2*width, width), nn.Tanh(), nn.Linear(width, 1))
            self.local = nn.Sequential(nn.Linear(3*width, width), nn.Tanh(), nn.Linear(width, 1))
            self.interaction = nn.Sequential(nn.Linear(4*width, width), nn.Tanh(), nn.Linear(width, 1))
        # All heads initially predict the same uniform joint opinion. Only the
        # ordinary seed0 initialization is used; no random-seed sweep.
        for module in self.modules():
            if isinstance(module, nn.Sequential) and isinstance(module[-1], nn.Linear):
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)

    def encode(self, anchors, present):
        assert torch.all(present.any(-1)), 'Each observation group needs a real anchor.'
        encoded = self.encoder(anchors)
        weight = present.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded*weight).sum(-2)/weight.sum(-2)
        maximum = encoded.masked_fill(~present.unsqueeze(-1), -torch.inf).max(-2).values
        return encoded, torch.cat((mean, maximum), -1)

    def forward(self, anchors, present, first, second):
        encoded, context = self.encode(anchors, present)
        a, b = self.encoder(first), self.encoder(second)
        if self.generic:
            assert encoded.shape[-2] == 3
            complete = encoded*present.unsqueeze(-1)
            # The penultimate input coordinate is video-normalized time. Sort
            # the full anchors canonically instead of an expensive six-order
            # ensemble. Missing slots sort last and retain their exact mask.
            order = anchors[..., -2].masked_fill(~present, torch.inf).argsort(dim=-1, stable=True)
            ordered = complete.gather(1,order.unsqueeze(-1).expand_as(complete))
            mask = present.gather(1,order).to(encoded.dtype)
            full_context = torch.cat((context,ordered.flatten(1),mask),-1)
            ab = F.log_softmax(self.output(torch.cat((full_context,a,b),-1)),-1)
            ba = F.log_softmax(self.output(torch.cat((full_context,b,a),-1)),-1)[...,SWAP]
            return torch.logaddexp(ab,ba)-0.6931471805599453
        local_a = self.local(torch.cat((context, a), -1)).squeeze(-1)
        local_b = self.local(torch.cat((context, b), -1)).squeeze(-1)
        if self.shared:
            anchor_context = context.unsqueeze(-2).expand(*encoded.shape[:-1], context.shape[-1])
            local_anchors = self.local(torch.cat((anchor_context, encoded), -1)).squeeze(-1)
            reference = (local_anchors*present).sum(-1)/present.sum(-1)
            base = self.object(context).squeeze(-1)
            local_a, local_b = base+local_a-reference, base+local_b-reference
        coupling = self.interaction(torch.cat((context, torch.abs(a-b), a*b), -1)).squeeze(-1)
        if not self.coupled:
            coupling = torch.zeros_like(coupling)
        return pair_log_probabilities(local_a, local_b, coupling)
