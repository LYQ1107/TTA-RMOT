"""Bounded causal replay storage."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ReplaySample:
    episode_id: str
    expression_id: str
    pred_track_id: int
    timestamp: int
    feature: torch.Tensor
    source_logit: float
    soft_target: float
    weight: float
    evidence_source: str
    query_type: str
    semantic_state_version: int
    probe_signature: str
    alignment_quality: float


class ReplayBuffer:
    def __init__(self, capacity: int = 512, per_track_capacity: int = 16):
        self.capacity = int(capacity)
        self.per_track_capacity = int(per_track_capacity)
        self._by_track: dict[tuple[str, str, int], deque[ReplaySample]] = defaultdict(
            lambda: deque(maxlen=self.per_track_capacity)
        )

    def clear(self) -> None:
        self._by_track.clear()

    def __len__(self) -> int:
        return sum(len(values) for values in self._by_track.values())

    def add(self, sample: ReplaySample) -> None:
        if sample.pred_track_id < 0:
            return
        key = (sample.episode_id, sample.expression_id, int(sample.pred_track_id))
        copied = ReplaySample(**{**sample.__dict__, "feature": sample.feature.detach().float().cpu().clone()})
        values = self._by_track[key]
        if values and values[-1].timestamp == copied.timestamp:
            values[-1] = copied
        else:
            values.append(copied)
        while len(self) > self.capacity:
            oldest_key = min(self._by_track, key=lambda value: self._by_track[value][0].timestamp)
            self._by_track[oldest_key].popleft()
            if not self._by_track[oldest_key]:
                del self._by_track[oldest_key]

    def samples(self) -> list[ReplaySample]:
        values = [sample for bucket in self._by_track.values() for sample in bucket]
        return sorted(values, key=lambda sample: (sample.timestamp, sample.pred_track_id))

    def recent_distinct_track_time(self, minimum: int = 2) -> bool:
        values = self.samples()
        return len({(sample.pred_track_id, sample.timestamp) for sample in values}) >= int(minimum)

    def batch(self, batch_size: int, device: torch.device) -> list[ReplaySample]:
        values = self.samples()[-int(batch_size):]
        return [ReplaySample(**{**sample.__dict__, "feature": sample.feature.to(device=device)}) for sample in values]
