"""The reusable referring-plugin shell around a frozen RMOT model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from flex_tta.adaptation.score import (
    CausalStabilityTTA,
    FrozenScoreAdapter,
    QueryTrackletCalibration,
    SourcePriorCalibration,
)
from flex_tta.types import OnlineStep, TrackerCandidate


class ScoreAdapter(Protocol):
    def reset_episode(self) -> None: ...

    def step(self, frame_id: int, margins: dict[int, float]) -> OnlineStep: ...


@dataclass(frozen=True)
class FlexHookPluginConfig:
    """Explicit plugin boundary; tracker/model settings stay outside TTA."""

    score_threshold: float = 0.0
    adapter: str = "frozen"


class FlexHookReferringPlugin:
    """Score tracker candidates without changing their geometry or IDs."""

    def __init__(self, config: FlexHookPluginConfig):
        self.config = config
        if config.adapter == "frozen":
            self.adapter: ScoreAdapter = FrozenScoreAdapter()
        elif config.adapter == "causal_stability":
            self.adapter = CausalStabilityTTA()
        elif config.adapter == "source_prior":
            self.adapter = SourcePriorCalibration(source_mean=-8.394860413379647, source_std=10.33014307587037)
        elif config.adapter == "query_tracklet":
            self.adapter = QueryTrackletCalibration()
        else:
            raise ValueError(f"unknown adapter: {config.adapter}")

    def reset_episode(self) -> None:
        self.adapter.reset_episode()

    def step(self, frame_id: int, candidates: list[TrackerCandidate]) -> tuple[OnlineStep, list[TrackerCandidate]]:
        margins = {candidate.track_id: candidate.source_margin for candidate in candidates}
        output = self.adapter.step(frame_id, margins)
        selected = [
            candidate for candidate in candidates
            if output.adapted_margins.get(candidate.track_id, float("-inf")) > self.config.score_threshold
        ]
        return output, selected


def selected_track_ids(step: OnlineStep) -> set[int]:
    """Convenience helper for downstream evaluators and diagnostics."""

    return {track_id for track_id, margin in step.adapted_margins.items() if margin > 0.0}
