"""Candidate-identity-preserving conditional estimation for RMOT.

The classical idea is conditional likelihood: a common pair intercept
cancels given exactly one referent. Candidate-specific stable identity does
NOT cancel. Both/neither and signed dependence are retained in the full
distribution. This is an inductive-bias hypothesis, not an accuracy theorem.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CrossCandidateHead(nn.Module):
    def __init__(self, dimension, width=64, *, conditional_estimation=True,
                 object_reference=True, stable_identity=True):
        super().__init__()
        self.conditional_estimation = conditional_estimation
        self.object_reference = object_reference
        self.stable_identity = stable_identity
        self.utility_encoder = nn.Sequential(nn.Linear(dimension, width), nn.Tanh())
        self.count_encoder = nn.Sequential(nn.Linear(dimension, width), nn.Tanh())
        self.identity = nn.Sequential(nn.Linear(2*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.state = nn.Sequential(nn.Linear(3*width, width), nn.Tanh(), nn.Linear(width, 1))
        # Symmetric count inputs: context mean and absolute context difference.
        self.base = nn.Sequential(nn.Linear(4*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.interaction = nn.Sequential(nn.Linear(6*width, width), nn.Tanh(), nn.Linear(width, 1))
        for module in (self.identity, self.state, self.base, self.interaction):
            nn.init.zeros_(module[-1].weight); nn.init.zeros_(module[-1].bias)

    @staticmethod
    def context(encoded, present):
        assert bool(present.any(-1).all())
        weight = present.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded*weight).sum(-2)/weight.sum(-2)
        maximum = encoded.masked_fill(~present.unsqueeze(-1), -torch.inf).max(-2).values
        return torch.cat((mean, maximum), -1)

    def utility(self, anchors, present, observation):
        encoded = self.utility_encoder(anchors)
        context = self.context(encoded, present)
        local = self.utility_encoder(observation)
        identity = self.identity(context).squeeze(-1)
        if not self.stable_identity:
            identity = torch.zeros_like(identity)
        if self.object_reference:
            value = self.state(torch.cat((context, local), -1)).squeeze(-1)
            repeated = context.unsqueeze(-2).expand(*encoded.shape[:-1], context.shape[-1])
            reference = self.state(torch.cat((repeated, encoded), -1)).squeeze(-1)
            offset = (reference*present).sum(-1)/present.sum(-1)
            return identity+value-offset
        return identity+self.state(torch.cat((local, local, local), -1)).squeeze(-1)

    def forward(self, anchors_A, present_A, anchors_B, present_B, first, second):
        ua = self.utility(anchors_A, present_A, first)
        ub = self.utility(anchors_B, present_B, second)
        ca = self.context(self.count_encoder(anchors_A), present_A)
        cb = self.context(self.count_encoder(anchors_B), present_B)
        symmetric = torch.cat(((ca+cb)/2, torch.abs(ca-cb)), -1)
        a, b = self.count_encoder(first), self.count_encoder(second)
        common = self.base(symmetric).squeeze(-1)
        interaction = self.interaction(torch.cat((symmetric, torch.abs(a-b), a*b), -1)).squeeze(-1)
        va, vb = (ua.detach(), ub.detach()) if self.conditional_estimation else (ua, ub)
        count = F.log_softmax(torch.stack((torch.zeros_like(common),
            common+torch.logaddexp(va, vb), 2*common+va+vb+interaction), -1), -1)
        relative = F.log_softmax(torch.stack((ua, ub), -1), -1)
        return torch.stack((count[..., 1]+relative[..., 0], count[..., 1]+relative[..., 1],
                            count[..., 2], count[..., 0]), -1)

    def renormalize(self, old_mean, old_std, new_mean, new_std):
        for encoder in (self.utility_encoder, self.count_encoder):
            layer = encoder[0]
            with torch.no_grad():
                om, os, nm, ns = [torch.as_tensor(x, device=layer.weight.device, dtype=layer.weight.dtype)
                                   for x in (old_mean, old_std, new_mean, new_std)]
                assert bool((os > 0).all() and (ns > 0).all())
                before = layer.weight.clone(); layer.weight.mul_(ns/os)
                layer.bias.add_(before@((nm-om)/os))


class OrdinaryCrossCandidateKD(nn.Module):
    """Unrestricted same-input, swap-equivariant categorical teacher KD.

    Each candidate has its own pooled context and current observation.
    All six visual inputs and presence flags are available. Learned pair
    scores can depend on every input, without conditional gradient splitting.
    """
    def __init__(self, dimension, width=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, width), nn.Tanh())
        self.network = nn.Sequential(nn.Linear(6*width, width), nn.Tanh(), nn.Linear(width, 4))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)

    def forward(self, anchors_A, present_A, anchors_B, present_B, first, second):
        ca = CrossCandidateHead.context(self.encoder(anchors_A), present_A)
        cb = CrossCandidateHead.context(self.encoder(anchors_B), present_B)
        a, b = self.encoder(first), self.encoder(second)
        direct = self.network(torch.cat((ca, a, cb, b), -1))
        reverse = self.network(torch.cat((cb, b, ca, a), -1))[..., [1, 0, 2, 3]]
        return F.log_softmax((direct+reverse)/2, -1)


def teacher_risk(log_probability, target):
    target = target.to(log_probability.dtype)
    return -(target*log_probability).sum(-1).mean()
