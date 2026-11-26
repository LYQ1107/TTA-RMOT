"""External referring-only adapters used on top of frozen TempRMOT output."""
from __future__ import annotations

import copy

import torch
from torch import nn


class ResidualReferringAdapter(nn.Module):
    """Compute ``source_logit + bounded residual(feature)``.

    The last layer is zero initialized, so a newly-created adapter is exactly
    the frozen source model.  The adapter is kept FP32 even when the source
    model is run in another dtype.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 32, max_abs_residual: float = 6.0):
        super().__init__()
        if input_dim != 256:
            raise ValueError("TempRMOT snapshot features are expected to be 256-dimensional")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_abs_residual = float(max_abs_residual)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        if features.numel() == 0:
            return features.new_empty(features.shape[:-1])
        raw = self.mlp(features.float()).squeeze(-1)
        return self.max_abs_residual * torch.tanh(raw / self.max_abs_residual)

    def forward(self, source_logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if source_logits.numel() == 0:
            return source_logits.float().clone()
        return source_logits.float() + self.delta(features)

    def predict_probs(self, source_logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self(source_logits, features).sigmoid()

    def parameter_report(self) -> dict[str, int]:
        return {
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }


class Head257(nn.Module):
    """Minimal capacity control: an external copy of the source linear head."""

    def __init__(self, source_weight: torch.Tensor, source_bias: torch.Tensor):
        super().__init__()
        weight = source_weight.detach().float().reshape(1, -1)
        bias = source_bias.detach().float().reshape(1)
        if tuple(weight.shape) != (1, 256):
            raise ValueError(f"expected source head shape (1,256), got {tuple(weight.shape)}")
        self.linear = nn.Linear(256, 1)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.float()).squeeze(-1)

    def clone(self) -> "Head257":
        return copy.deepcopy(self)
