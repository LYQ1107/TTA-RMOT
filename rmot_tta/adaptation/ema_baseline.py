"""A small high-confidence EMA referring baseline, not the QCS method."""
from __future__ import annotations

import torch


class EmaReferringBaseline:
    def __init__(self, momentum: float = 0.9, threshold: float = 0.8):
        self.momentum = float(momentum)
        self.threshold = float(threshold)
        self.targets: dict[int, float] = {}

    def reset(self) -> None:
        self.targets.clear()

    def observe(self, track_ids: torch.Tensor, probabilities: torch.Tensor) -> None:
        for track_id, probability in zip(track_ids.detach().cpu().tolist(), probabilities.detach().cpu().tolist()):
            if float(probability) >= self.threshold:
                previous = self.targets.get(int(track_id), float(probability))
                self.targets[int(track_id)] = self.momentum * previous + (1.0 - self.momentum) * float(probability)

    def adjust(self, track_ids: torch.Tensor, source_probabilities: torch.Tensor) -> torch.Tensor:
        values = source_probabilities.float().clone()
        for index, track_id in enumerate(track_ids.detach().cpu().tolist()):
            if int(track_id) in self.targets:
                values[index] = self.targets[int(track_id)]
        return values
