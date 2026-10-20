"""State-aware evidence memory keyed by episode/expression/ID/version."""
from __future__ import annotations

from dataclasses import dataclass

from rmot_tta.evidence.records import EvidenceLabel
from .replay_buffer import ReplayBuffer, ReplaySample


@dataclass
class ActiveSupport:
    episode_id: str
    expression_id: str
    pred_track_id: int
    state_version: int = 0
    label: EvidenceLabel = EvidenceLabel.UNKNOWN
    last_timestamp: int = -1
    active: bool = False


class SemanticMemory:
    def __init__(self, capacity: int = 512, per_track_capacity: int = 16):
        self.replay = ReplayBuffer(capacity, per_track_capacity)
        self.supports: dict[tuple[str, str, int], ActiveSupport] = {}

    def clear(self) -> None:
        self.replay.clear()
        self.supports.clear()

    def state(self, episode_id: str, expression_id: str, track_id: int) -> ActiveSupport:
        key = (episode_id, expression_id, int(track_id))
        if key not in self.supports:
            self.supports[key] = ActiveSupport(episode_id, expression_id, int(track_id))
        return self.supports[key]

    def observe(self, episode_id: str, expression_id: str, track_id: int, label: EvidenceLabel,
                timestamp: int, feature, source_logit: float, soft_target: float | None,
                weight: float, evidence_source: str, query_type: str,
                probe_signature: str, alignment_quality: float) -> ActiveSupport:
        support = self.state(episode_id, expression_id, track_id)
        if label in {EvidenceLabel.SUPPORTED_POS, EvidenceLabel.SUPPORTED_NEG}:
            if support.active and support.label != label:
                support.state_version += 1
                support.active = False
            support.label = label
            support.last_timestamp = int(timestamp)
            support.active = True
            if soft_target is not None and weight > 0:
                self.replay.add(ReplaySample(
                    episode_id, expression_id, int(track_id), int(timestamp), feature.detach(), float(source_logit),
                    float(soft_target), float(weight), evidence_source, query_type, support.state_version,
                    probe_signature, float(alignment_quality),
                ))
        return support

    def active_label(self, episode_id: str, expression_id: str, track_id: int) -> EvidenceLabel:
        support = self.supports.get((episode_id, expression_id, int(track_id)))
        return support.label if support and support.active else EvidenceLabel.UNKNOWN
