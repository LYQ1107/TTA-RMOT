"""Positive-label-free second-stage adaptation primitives for FlexHook.

Only score sequences and pre-existing tracker IDs are accepted. No target GT,
future frames, external models, target positive-count prior or pseudo class
labels are used. These are research *hypotheses*, not demonstrated improvements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from statistics import median
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RankPair:
    higher: int
    lower: int
    weight: float
    supporting_frames: int


@dataclass(frozen=True)
class RankConfig:
    history: int = 5
    min_history: int = 3
    max_gap: int = 2
    min_separation: float = 0.6
    max_pairs: int = 20
    rank_margin: float = 0.5
    rank_weight: float = 0.15
    anchor_weight: float = 0.05
    retention_weight: float = 0.08
    rank_temperature: float = 1.0

    def __post_init__(self):
        if self.history < self.min_history or self.min_history < 2:
            raise ValueError("min_history must lie between 2 and history")
        if self.max_gap < 1 or self.min_separation <= 0 or self.max_pairs < 1:
            raise ValueError("invalid rank-evidence parameters")
        if min(self.rank_weight, self.anchor_weight, self.retention_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.rank_temperature <= 0:
            raise ValueError("rank_temperature must be positive")


class CausalRankMemory:
    """Store sealed PRE-UPDATE score frames; never predict from future data.

    Historical rank pairs merely indicate STABILITY of a source preference;
    they are **not** positive/negative semantic labels. Pair filtering is
    deliberately conservative so poorly ranked domains may yield no updates.
    """

    def __init__(self, config: RankConfig = RankConfig()):
        self.cfg = config
        self.frames: deque[tuple[int, dict[int, float]]] = deque(maxlen=config.history)
        self.latest_frame = -1

    def reset(self) -> None:
        self.frames.clear()
        self.latest_frame = -1

    def observe(self, frame_id: int, scores: Mapping[int, float]) -> None:
        frame_id = int(frame_id)
        if frame_id <= self.latest_frame:
            raise ValueError("causal score frames must be strictly increasing")
        copied = {int(k): float(v) for k, v in scores.items()}
        if not all(isfinite(v) for v in copied.values()):
            raise ValueError("non-finite score")
        self.frames.append((frame_id, copied))
        self.latest_frame = frame_id

    def stable_pairs(self, frame_id: int, track_ids: Sequence[int]) -> list[RankPair]:
        if frame_id <= self.latest_frame:
            raise ValueError("pair evidence must be requested before observing this frame")
        known = set(map(int, track_ids))
        valid = [
            (past_t, values) for past_t, values in self.frames
            if 0 < frame_id - past_t <= self.cfg.history * self.cfg.max_gap
        ]
        if not valid:
            return []
        out: list[RankPair] = []
        for a, b in combinations(sorted(known), 2):
            aligned = [(t, rows[a] - rows[b]) for t, rows in valid if a in rows and b in rows]
            if len(aligned) < self.cfg.min_history:
                continue
            aligned = aligned[-self.cfg.min_history:]
            times = [t for t, _ in aligned]
            if frame_id - times[-1] > self.cfg.max_gap or any(
                right - left > self.cfg.max_gap for left, right in zip(times, times[1:])
            ):
                continue
            diffs = [d for _, d in aligned]
            if any(abs(d) < self.cfg.min_separation for d in diffs):
                continue
            if all(d > 0 for d in diffs):
                higher, lower = a, b
            elif all(d < 0 for d in diffs):
                higher, lower = b, a
            else:
                continue
            strength = min(1.0, median(abs(d) for d in diffs) / (2.0 * self.cfg.min_separation))
            out.append(RankPair(higher, lower, strength, len(aligned)))
        out.sort(key=lambda p: (-p.weight, p.higher, p.lower))
        return out[:self.cfg.max_pairs]


@dataclass(frozen=True)
class RelativeConfig:
    minimum_candidates: int = 3
    min_spread: float = 0.7
    median_mad_multiplier: float = 1.2
    min_history: int = 2
    max_gap: int = 2
    preserve_source_positives: bool = True

    def __post_init__(self):
        if self.minimum_candidates < 2 or self.min_history < 1 or self.max_gap < 1:
            raise ValueError("invalid relative-selection parameters")
        if self.min_spread < 0 or self.median_mad_multiplier < 0:
            raise ValueError("invalid robust-score selection parameters")


class CausalRelativeSelector:
    """Causal per-query relative scoring, with explicit empty-set behavior.

    Never assumes a positive exists. Historical agreement is only a heuristic;
    report this separately from learnable TTA, and evaluate false positives.
    """

    def __init__(self, config: RelativeConfig = RelativeConfig()):
        self.cfg = config
        self.frames: deque[tuple[int, dict[int, float]]] = deque(maxlen=max(12, 2 * config.min_history))
        self.last_frame = -1

    def reset(self) -> None:
        self.frames.clear()
        self.last_frame = -1

    def step(self, frame_id: int, margins: Mapping[int, float]) -> set[int]:
        frame_id = int(frame_id)
        if frame_id <= self.last_frame:
            raise ValueError("relative scoring requires ordered frames")
        rows = {int(k): float(v) for k, v in margins.items()}
        if not all(isfinite(v) for v in rows.values()):
            raise ValueError("non-finite score")
        result = {k for k, v in rows.items() if v > 0} if self.cfg.preserve_source_positives else set()
        if len(rows) >= self.cfg.minimum_candidates:
            values = list(rows.values())
            med = median(values)
            mad = median(abs(v - med) for v in values)
            spread = max(values) - min(values)
            if spread >= self.cfg.min_spread and mad > 1e-6:
                gate = med + self.cfg.median_mad_multiplier * 1.4826 * mad
                for track_id, value in rows.items():
                    if value <= gate:
                        continue
                    votes = 0
                    for old_t, past in reversed(self.frames):
                        if frame_id - old_t > self.cfg.max_gap * self.cfg.min_history:
                            break
                        if track_id not in past or len(past) < self.cfg.minimum_candidates:
                            continue
                        pm = median(past.values())
                        pd = median(abs(v - pm) for v in past.values())
                        if past[track_id] > pm + self.cfg.median_mad_multiplier * 1.4826 * pd:
                            votes += 1
                        if votes >= self.cfg.min_history:
                            result.add(track_id)
                            break
        self.frames.append((frame_id, rows))
        self.last_frame = frame_id
        return result


def positive_free_rank_loss(
    student: torch.Tensor,
    source: torch.Tensor,
    track_ids: Sequence[int],
    pairs: Sequence[RankPair],
    config: RankConfig = RankConfig(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute stable-relative-rank loss, NOT supervised positive mining.

    The scalar loss is differentiable through student logits only. There are
    no binary pseudo labels; if no stable pairs exist the caller MUST skip
    optimizer.step() (anchor-only updates are not meaningful adaptation).
    """
    if student.ndim != 1 or student.shape != source.shape or len(track_ids) != student.numel():
        raise ValueError("student, source and track IDs must describe one candidate set")
    indices = {int(t): i for i, t in enumerate(track_ids)}
    valid_pairs = [p for p in pairs if p.higher in indices and p.lower in indices]
    center_student = student - student.mean() if student.numel() else student
    center_source = source.detach() - source.detach().mean() if source.numel() else source.detach()
    anchor = F.smooth_l1_loss(center_student, center_source) if student.numel() else student.sum() * 0
    positive = source.detach() > 0
    retention = F.relu(source.detach()[positive] - student[positive]).mean() if positive.any() else student.sum() * 0
    if valid_pairs:
        hi = torch.tensor([indices[p.higher] for p in valid_pairs], device=student.device)
        lo = torch.tensor([indices[p.lower] for p in valid_pairs], device=student.device)
        weight = torch.tensor([p.weight for p in valid_pairs], device=student.device, dtype=student.dtype)
        gaps = (student[hi] - student[lo]) / config.rank_temperature
        rank = (F.softplus(config.rank_margin - gaps) * weight).sum() / weight.sum().clamp_min(1e-6)
    else:
        rank = student.sum() * 0
    total = config.rank_weight * rank + config.anchor_weight * anchor + config.retention_weight * retention
    return total, {
        "stable_pairs": len(valid_pairs),
        "rank_loss": float(rank.detach()),
        "anchor_loss": float(anchor.detach()),
        "retention_loss": float(retention.detach()),
    }
