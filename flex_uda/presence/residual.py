"""Source-supervised count distribution on an immutable identity ordering.

The learned residual determines a cardinality, never an identity reordering.
At initialization the MAP cardinality exactly reproduces margin > 0.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'residual_prefix_v3'
KL_WEIGHT = .1


class ResidualCountHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcd = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 64), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(5, 32), nn.GELU())
        self.residual = nn.Sequential(nn.Linear(288, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, pcd, margin, box, sent=None, padding_mask=None, track_id=None):
        single = pcd.ndim == 2
        if single:
            pcd, margin, box = pcd[None], margin[None], box[None]
            padding_mask = padding_mask[None] if padding_mask is not None else None
            track_id = track_id[None] if track_id is not None else None
        b, n = margin.shape
        if padding_mask is None:
            padding_mask = torch.zeros_like(margin, dtype=torch.bool)
        valid = ~padding_mask
        local = torch.cat([self.pcd(pcd), self.geometry(torch.cat([margin[..., None], box], -1))], -1)
        mean = (local * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        maximum = (local.masked_fill(padding_mask[..., None], -1e9).amax(1)
                   if n else local.new_zeros((b, 96)))
        maximum = torch.where(valid.any(1)[:, None], maximum, torch.zeros_like(maximum))
        context = torch.cat([local, mean[:, None].expand(-1, n, -1), maximum[:, None].expand(-1, n, -1)], -1)
        delta = self.residual(context).squeeze(-1).masked_fill(padding_mask, 0)
        # Stable sort by track ID, then by decreasing IMMUTABLE margin.
        ids = track_id if track_id is not None else torch.arange(n, device=margin.device).expand(b, -1)
        ids = ids.masked_fill(padding_mask, torch.iinfo(ids.dtype).max)
        order = torch.argsort(ids, dim=1, stable=True)
        score = margin.masked_fill(padding_mask, -torch.inf)
        ranked = torch.argsort(score.gather(1, order), dim=1, descending=True, stable=True)
        order = order.gather(1, ranked)
        base = margin.masked_fill(padding_mask, 0).gather(1, order)
        delta = delta.gather(1, order)
        # Double precision only for short prefix sums. Otherwise a tiny
        # positive boundary increment can round away behind a large prefix,
        # breaking exact native initialization near margin=0.
        zero = margin.new_zeros((b, 1), dtype=torch.float64)
        logits = torch.cat([zero, (base.double() + delta.double()).cumsum(1)], 1)
        baseline = torch.cat([zero, base.double().cumsum(1)], 1)
        impossible = torch.arange(n+1, device=margin.device)[None] > valid.sum(1)[:, None]
        logits = logits.masked_fill(impossible, -1e9)
        baseline = baseline.masked_fill(impossible, -1e9)
        # Marginal presence, not a second independently fitted output.
        positive = torch.logsumexp(logits[:, 1:], 1) if n else margin.new_full((b,), -1e9)
        presence = positive - logits[:, 0]
        result = dict(presence=presence, logits=logits, baseline=baseline, residual=delta)
        return {k: v[0] for k, v in result.items()} if single else result


def residual_loss(output, count):
    ce = F.cross_entropy(output['logits'], count)
    prior = output['baseline'].detach().softmax(-1)
    kl = (prior * (output['baseline'].detach().log_softmax(-1) - output['logits'].log_softmax(-1))).sum(-1).mean()
    return ce + KL_WEIGHT * kl, ce, kl


def head_from_checkpoint(state):
    if state.get('architecture') == ARCHITECTURE:
        model = ResidualCountHead()
    else:
        from flex_uda.presence.head import PresenceHead
        model = PresenceHead()
    model.load_state_dict(state['head'], strict=True)
    return model
