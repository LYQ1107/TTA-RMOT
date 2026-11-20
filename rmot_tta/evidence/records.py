"""Auditable set-valued evidence records."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceLabel(str, Enum):
    SUPPORTED_POS = "SUPPORTED_POS"
    SUPPORTED_NEG = "SUPPORTED_NEG"
    UNKNOWN = "UNKNOWN"
    CONTRADICTORY_PROBE = "CONTRADICTORY_PROBE"


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_uid: str
    label: EvidenceLabel
    soft_target: float | None
    weight: float
    evidence_source: str
    query_type: str
    alignment_quality: float
    explanation: str
    frame_index: int
    pred_track_id: int
    # Frozen-probe probabilities retained for the actual query-relation loss.
    # They are optional because confidence-only baselines do not run q_eq/q_cf.
    probe_q0: float | None = None
    probe_eq: float | None = None
    probe_cf: float | None = None
