"""Parameter-efficient adapters attached to the released FlexHook PCD."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """Zero-output residual with a trainable scalar gate.

    The gate starts at zero, so the source checkpoint is exactly preserved at
    episode reset.  The internal projection is initialized normally so the
    gate has a nonzero first derivative and the adapter can actually learn.
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.reset_projection()

    def reset_projection(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        with torch.no_grad():
            self.alpha.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.act(self.down(x)))


class LoRALinear(nn.Module):
    """Frozen linear projection plus a zero-output low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        if base.bias is not None:
            raise ValueError("FlexHook PCD projections are expected to be bias-free")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(alpha) / float(rank)
        self.reset_lora()

    def reset_lora(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(x))


class PCDResidualTTA(nn.Module):
    """Residual adapters on the last one or two output PCD blocks."""

    def __init__(self, model: nn.Module, layer_indices: Iterable[int] = (-2, -1), bottleneck: int = 64):
        super().__init__()
        blocks = list(model.output_ca_qkv)
        self.layer_indices = tuple(index if index >= 0 else len(blocks) + index for index in layer_indices)
        dim = int(model.text_dim)
        self.adapters = nn.ModuleList([ResidualAdapter(dim, bottleneck) for _ in self.layer_indices])
        self.handles = []
        for block, adapter in zip((blocks[index] for index in self.layer_indices), self.adapters):
            self.handles.append(block.register_forward_hook(lambda _module, _inputs, output, a=adapter: a(output)))

    def reset_episode(self) -> None:
        for adapter in self.adapters:
            adapter.reset_projection()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PCDLoRATTA(nn.Module):
    """LoRA on Q/V projections of the last one or two output PCD blocks."""

    def __init__(self, model: nn.Module, rank: int = 4, layer_indices: Iterable[int] = (-2, -1),
                 projections=("wq", "wv"), alpha: float = 1.0):
        super().__init__()
        blocks = list(model.output_ca_qkv)
        if layer_indices == "all":
            layer_indices = tuple(range(len(blocks)))
        self.layer_indices = tuple(index if index >= 0 else len(blocks) + index for index in layer_indices)
        if any(name not in ("wq", "wk", "wv", "wo") for name in projections):
            raise ValueError("only PCD wq/wk/wv/wo projections can receive LoRA")
        self.projections = nn.ModuleList()
        for index in self.layer_indices:
            block = blocks[index]
            for name in projections:
                replacement = LoRALinear(getattr(block.attention, name), rank, alpha)
                setattr(block.attention, name, replacement)
                self.projections.append(replacement)

    def reset_episode(self) -> None:
        for projection in self.projections:
            projection.reset_lora()

    def close(self) -> None:
        return None


def trainable_parameters(adapter: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("adapter has no trainable parameters")
    return parameters


def parameter_drift(adapter: nn.Module) -> float:
    return float(sum(parameter.detach().float().abs().sum().item() for parameter in adapter.parameters() if parameter.requires_grad))
