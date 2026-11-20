"""Semantic state transitions without changing physical track IDs."""
from __future__ import annotations

from dataclasses import dataclass

from rmot_tta.evidence.records import EvidenceLabel


@dataclass
class StateTransition:
    track_id: int
    old_version: int
    new_version: int
    reason: str
    timestamp: int


def transition_for_evidence(current_version: int, label: EvidenceLabel, timestamp: int,
                            conflict_with_previous: bool) -> StateTransition | None:
    if label not in {EvidenceLabel.SUPPORTED_POS, EvidenceLabel.SUPPORTED_NEG}:
        return None
    if not conflict_with_previous:
        return None
    return StateTransition(-1, current_version, current_version + 1,
                           f"reliable_{label.value.lower()}_conflicts_with_active_support", int(timestamp))
