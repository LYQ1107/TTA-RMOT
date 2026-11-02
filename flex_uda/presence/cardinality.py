"""Source-calibrated independent Bernoulli cardinality, no identity changes."""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'bernoulli_cardinality_v4'


class BernoulliCardinalityHead(nn.Module):
    def __init__(self, alpha_neg=1., alpha_pos=1.):
        super().__init__()
        self.slopes = nn.Parameter(torch.tensor([alpha_neg, alpha_pos], dtype=torch.float64))

    def candidate_logits(self, margin):
        if not bool((self.slopes > 0).all()): raise ValueError('Calibration slopes must remain positive')
        return margin.double() * self.slopes[(margin >= 0).long()]

    def forward(self, margin, padding_mask=None, **unused):
        single = margin.ndim == 1
        if single:
            margin = margin[None]
            padding_mask = padding_mask[None] if padding_mask is not None else None
        if padding_mask is None: padding_mask = torch.zeros_like(margin, dtype=torch.bool)
        scaled = self.candidate_logits(margin)
        p = scaled.sigmoid().masked_fill(padding_mask, 0)
        b, n = margin.shape
        distribution = p.new_zeros((b, n+1)); distribution[:, 0] = 1
        for i in range(n):
            yes = p[:, i, None]
            shifted = torch.cat([p.new_zeros((b, 1)), distribution[:, :-1]], 1)
            distribution = distribution*(1-yes) + shifted*yes
        lengths = (~padding_mask).sum(1)
        impossible = torch.arange(n+1, device=margin.device)[None] > lengths[:, None]
        logits = distribution.clamp_min(torch.finfo(torch.float64).tiny).log().masked_fill(impossible, -1e9)
        log_empty = F.logsigmoid(-scaled).masked_fill(padding_mask, 0).sum(1)
        presence = (-torch.expm1(log_empty)).clamp_min(torch.finfo(torch.float64).tiny).log() - log_empty
        result = dict(logits=logits, joint=distribution, presence=presence, candidate_probabilities=p,
                      expected_count=p.sum(1), log_empty=log_empty)
        return {k:v[0] for k,v in result.items()} if single else result
