"""Candidate-level Bernoulli entropy control used as an explicit baseline."""
from __future__ import annotations

import torch

from .losses import bernoulli_entropy_from_logits, masked_weighted_mean


def entropy_loss(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_weighted_mean(bernoulli_entropy_from_logits(logits), mask=mask)
