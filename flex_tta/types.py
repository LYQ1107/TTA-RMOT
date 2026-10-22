"""Small contracts shared by tracker adapters and the online TTA layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackRow:
    """One MOT-format tracker row, retaining the original numeric columns."""

    frame_id: int
    track_id: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class TrackerCandidate:
    """A tracker candidate presented to a referring plugin.

    The candidate bank is immutable for a run.  A referring plugin can change
    only the score/selection decision; it never creates boxes or track IDs.
    """

    video_id: str
    frame_id: int
    track_id: int
    row: TrackRow
    source_margin: float


@dataclass(frozen=True)
class OnlineStep:
    """Output of one causal score-adapter step."""

    frame_id: int
    adapted_margins: dict[int, float]
    pseudo_updates: int
    scale: float
    bias: float
