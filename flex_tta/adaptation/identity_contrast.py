"""Causal track-ID self-supervised InfoNCE for existing FlexHook PCD features.

Positive here means the SAME MOT TRACK at two past/present times; it does NOT
mean a positive language-track match. Uses only the frozen MOT's physical IDs,
never GT identities, target annotations, extra checkpoints or future frames.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IdentityContrastConfig:
    max_gap: int = 2
    temperature: float = 0.15
    min_reliable_tracks: int = 2
    max_entries: int = 256
    weight: float = 0.15

    def __post_init__(self):
        if self.max_gap < 1 or self.temperature <= 0 or self.max_entries < 2:
            raise ValueError("invalid identity contrast config")
        if self.min_reliable_tracks < 2 or self.weight < 0:
            raise ValueError("need at least two history IDs and nonnegative weight")


class PCDFeatureTap:
    """Read last PCD output, currently shape [track_batch, query_count, C]."""

    def __init__(self, model: nn.Module):
        if not len(model.output_ca_qkv):
            raise ValueError("FlexHook has no PCD blocks")
        self.value: torch.Tensor | None = None
        self.handle = model.output_ca_qkv[-1].register_forward_hook(self._tap)

    def _tap(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[1] != 1:
            raise ValueError("expected last PCD output [batch, one query, channels]")
        self.value = output[:, 0, :]

    def clear(self):
        self.value = None

    def take(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("PCD feature tap has no output for this forward")
        value = self.value
        self.value = None
        return value

    def close(self):
        self.handle.remove()


class CausalIDMemory:
    """One detached prior feature per track, with expiration and reset."""

    def __init__(self, config: IdentityContrastConfig = IdentityContrastConfig()):
        self.cfg = config
        self.rows: dict[int, tuple[int, torch.Tensor]] = {}
        self.last_frame = -1

    def reset(self) -> None:
        self.rows.clear()
        self.last_frame = -1

    def eligible(self, frame_id: int, track_ids: list[int]) -> bool:
        available = sum(int(t in self.rows and 0 < frame_id - self.rows[t][0] <= self.cfg.max_gap) for t in track_ids)
        valid_gallery = sum(int(0 < frame_id - old_t <= self.cfg.max_gap) for old_t, _ in self.rows.values())
        return available >= self.cfg.min_reliable_tracks and valid_gallery >= self.cfg.min_reliable_tracks

    def loss(self, frame_id: int, features: torch.Tensor, track_ids: list[int]):
        if features.ndim != 2 or features.shape[0] != len(track_ids):
            raise ValueError("features must be [candidate count, C] and match IDs")
        ids = sorted(i for i, (t, _) in self.rows.items() if 0 < frame_id - t <= self.cfg.max_gap)
        if len(ids) < self.cfg.min_reliable_tracks:
            return features.sum() * 0, 0
        selected = [i for i, tid in enumerate(track_ids) if tid in ids]
        if len(selected) < self.cfg.min_reliable_tracks:
            return features.sum() * 0, 0
        keys = torch.stack([self.rows[tid][1].to(device=features.device,dtype=features.dtype) for tid in ids])
        queries = F.normalize(features[selected], dim=-1)
        keys = F.normalize(keys.detach(), dim=-1)
        logits = queries @ keys.T / self.cfg.temperature
        targets = torch.tensor([ids.index(track_ids[i]) for i in selected], device=features.device)
        return F.cross_entropy(logits, targets), len(selected)

    def observe(self, frame_id: int, features: torch.Tensor, track_ids: list[int]):
        if frame_id <= self.last_frame:
            raise ValueError("noncausal identity memory observation")
        if features.ndim != 2 or features.shape[0] != len(track_ids):
            raise ValueError("feature/track count mismatch")
        self.rows = {tid: (t, vec) for tid, (t, vec) in self.rows.items()
                     if frame_id - t <= self.cfg.max_gap}
        for tid, feature in zip(track_ids, features.detach()):
            self.rows[int(tid)] = (frame_id, feature.detach().float().cpu().clone())
        self.last_frame = frame_id
        if len(self.rows) > self.cfg.max_entries:
            oldest = sorted(self.rows, key=lambda tid: (self.rows[tid][0], tid))
            for tid in oldest[:len(self.rows) - self.cfg.max_entries]:
                del self.rows[tid]
