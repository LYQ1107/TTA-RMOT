"""Numerically stable candidate-set losses for online referring adaptation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _flatten_mask(mask: torch.Tensor | None, shape: torch.Size, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool).reshape(shape)


def masked_weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None,
                         mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    values = values.float()
    flat = values.reshape(-1)
    keep = _flatten_mask(mask, flat.shape, flat.device)
    if not bool(keep.any()):
        return _zero(values)
    if weights is None:
        weight = torch.ones_like(flat)
    else:
        weight = weights.float().reshape(-1).to(flat.device)
    weight = torch.where(keep, weight.clamp_min(0.0), torch.zeros_like(weight))
    denom = weight.sum()
    if float(denom.detach()) <= eps:
        return _zero(values)
    return (flat * weight).sum() / denom.clamp_min(eps)


def bernoulli_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if logits.numel() == 0:
        return _zero(logits)
    probability = logits.float().sigmoid().clamp(eps, 1.0 - eps)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())


def weighted_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                             weights: torch.Tensor | None = None,
                             mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.numel() == 0:
        return _zero(logits)
    values = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return masked_weighted_mean(values, weights=weights, mask=mask)


def bernoulli_kl(target_probability: torch.Tensor, predicted_probability: torch.Tensor,
                 mask: torch.Tensor | None = None, weights: torch.Tensor | None = None,
                 eps: float = 1e-6) -> torch.Tensor:
    target = target_probability.float().clamp(eps, 1.0 - eps)
    predicted = predicted_probability.float().clamp(eps, 1.0 - eps)
    if target.numel() == 0:
        return _zero(target)
    values = target * (target / predicted).log() + (1.0 - target) * ((1.0 - target) / (1.0 - predicted)).log()
    return masked_weighted_mean(values, weights=weights, mask=mask)


def equivalent_query_consistency(p0: torch.Tensor, peq: torch.Tensor,
                                 mask: torch.Tensor | None = None,
                                 weights: torch.Tensor | None = None) -> torch.Tensor:
    if p0.numel() == 0:
        return _zero(p0)
    return 0.5 * (bernoulli_kl(p0, peq, mask=mask, weights=weights) +
                  bernoulli_kl(peq, p0, mask=mask, weights=weights))


def mutual_exclusion_penalty(p0: torch.Tensor, pcf: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             weights: torch.Tensor | None = None) -> torch.Tensor:
    if p0.numel() == 0:
        return _zero(p0)
    values = torch.relu(p0.float() + pcf.float() - 1.0).square()
    return masked_weighted_mean(values, weights=weights, mask=mask)


def source_anchor_loss(source_logits: torch.Tensor, adapted_logits: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
    if source_logits.numel() == 0:
        return _zero(source_logits)
    source = source_logits.float().sigmoid().detach()
    adapted = adapted_logits.float().sigmoid()
    return bernoulli_kl(source, adapted, mask=mask)
