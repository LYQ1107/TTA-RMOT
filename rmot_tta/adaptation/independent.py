"""Independent source-free adapters for the full ShiftRMOT protocol.

These modules deliberately do not alias the historical EMA/QCS controller.
They consume only the frozen model snapshot and causal evidence available at
the current frame.  The source TempRMOT parameters are never optimized.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.types import FrameSnapshot


@dataclass
class IndependentUpdateResult:
    frame_index: int
    stepped: bool
    reason: str
    sample_count: int
    distinct_track_time_samples: int
    loss: float | None
    grad_norm: float | None
    parameter_delta: float | None
    elapsed_sec: float
    loss_ref: float | None = None
    loss_set: float | None = None


@dataclass
class _ReplayItem:
    episode_id: str
    frame_index: int
    track_id: int
    feature: torch.Tensor
    query: torch.Tensor
    source_logit: float
    objectness: float
    disappear_time: float
    area_ratio: float
    target: float
    weight: float


class _ZeroInitializedScalarAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_abs_residual: float):
        super().__init__()
        self.max_abs_residual = float(max_abs_residual)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        raw = self.network(values.float()).squeeze(-1)
        return self.max_abs_residual * torch.tanh(raw / self.max_abs_residual)


class VisualCandidateAdapter(_ZeroInitializedScalarAdapter):
    """Adapt a candidate using visual track features and visual reliability."""

    def __init__(self, hidden_dim: int, max_abs_residual: float):
        super().__init__(259, hidden_dim, max_abs_residual)

    def delta(self, features, objectness, disappear_time, area_ratio):
        context = torch.cat((features.float(), objectness[:, None], disappear_time[:, None], area_ratio[:, None]), dim=-1)
        return self(context)


class LanguageProjectionAdapter(_ZeroInitializedScalarAdapter):
    """Adapt the text-side projection using a frozen RoBERTa query vector."""

    def __init__(self, hidden_dim: int, max_abs_residual: float):
        super().__init__(257, hidden_dim, max_abs_residual)

    def delta(self, source_logits, query_embedding):
        source = source_logits.float().reshape(-1, 1)
        query = query_embedding.float()
        if query.ndim == 1:
            query = query.reshape(1, -1).expand(source.shape[0], -1)
        elif query.ndim != 2 or query.shape[0] != source.shape[0]:
            raise ValueError(f"query batch shape {tuple(query.shape)} does not match source batch {source.shape[0]}")
        return self(torch.cat((query, source), dim=-1))


class FusionCandidateAdapter(_ZeroInitializedScalarAdapter):
    """Candidate-level visual-language adapter with explicit cross products."""

    def __init__(self, hidden_dim: int, max_abs_residual: float):
        super().__init__(770, hidden_dim, max_abs_residual)

    def delta(self, features, query_embedding, source_logits, objectness):
        query = query_embedding.float()
        if query.ndim == 1:
            query = query.reshape(1, -1).expand(features.shape[0], -1)
        elif query.ndim != 2 or query.shape[0] != features.shape[0]:
            raise ValueError(f"query batch shape {tuple(query.shape)} does not match feature batch {features.shape[0]}")
        cross = features.float() * query
        return self(self._context(features, query, cross, source_logits, objectness))

    @staticmethod
    def _context(features, query, cross, source_logits, objectness):
        return torch.cat((features.float(), query, cross,
                          source_logits.float()[:, None], objectness.float()[:, None]), dim=-1)


class _CompositeAdapter(nn.Module):
    def __init__(self, kinds: tuple[str, ...], hidden_dim: int, max_abs_residual: float):
        super().__init__()
        self.kinds = kinds
        self.visual = VisualCandidateAdapter(hidden_dim, max_abs_residual) if "visual" in kinds else None
        self.language = LanguageProjectionAdapter(hidden_dim, max_abs_residual) if "language" in kinds else None
        self.fusion = FusionCandidateAdapter(hidden_dim, max_abs_residual) if "fusion" in kinds else None

    def delta(self, *, features, query, source, objectness, disappear, area_ratio):
        values = features.new_zeros(features.shape[0], dtype=torch.float32)
        if self.visual is not None:
            values = values + self.visual.delta(features, objectness, disappear, area_ratio)
        if self.language is not None:
            values = values + self.language.delta(source, query)
        if self.fusion is not None:
            values = values + self.fusion.delta(features, query, source, objectness)
        return values


def _adapter_kinds(method: str) -> tuple[str, ...]:
    mapping = {
        "VISUAL_TTA": ("visual",),
        "LANGUAGE_TTA": ("language",),
        "FUSION_TTA": ("fusion",),
        "VISUAL_LANGUAGE_TTA": ("visual", "language"),
        "VISUAL_FUSION_TTA": ("visual", "fusion"),
        "LANGUAGE_FUSION_TTA": ("language", "fusion"),
        "ALL_TTA": ("visual", "language", "fusion"),
    }
    if method not in mapping:
        raise ValueError(f"unknown independent adapter method: {method}")
    return mapping[method]


class IndependentAdaptationController:
    """Causal reliability-filtered optimizer for visual/language/fusion arms."""

    def __init__(self, method: str, *, lr: float = 1e-3, update_every_n_frames: int = 4,
                 batch_size: int = 32, minimum_distinct_track_time_samples: int = 2,
                 grad_clip: float = 1.0, hidden_dim: int = 32, max_abs_residual: float = 6.0,
                 weight_decay: float = 0.0, replay_capacity: int = 512):
        self.method = method
        self.update_every_n_frames = int(update_every_n_frames)
        self.batch_size = int(batch_size)
        self.minimum_distinct_track_time_samples = int(minimum_distinct_track_time_samples)
        self.grad_clip = float(grad_clip)
        self.replay_capacity = int(replay_capacity)
        self.adapter = _CompositeAdapter(_adapter_kinds(method), int(hidden_dim), float(max_abs_residual))
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.replay: list[_ReplayItem] = []
        self.update_steps = 0
        self.nonzero_gradient_steps = 0
        self.parameter_delta_sum = 0.0
        self.evidence_pos = 0
        self.evidence_neg = 0
        self.evidence_unknown = 0
        self.last_update = IndependentUpdateResult(-1, False, "not_started", 0, 0, None, None, None, 0.0)

    def to(self, device: torch.device | str):
        self.adapter.to(device)
        return self

    def predict_logits(self, snapshot: FrameSnapshot) -> torch.Tensor:
        features, query = self._inputs(snapshot)
        source = snapshot.source_logits.float()
        objectness = snapshot.objectness.float()
        disappear = snapshot.disappear_time.float().clamp(0, 32) / 32.0
        height, width = snapshot.image_size
        area_ratio = ((snapshot.boxes_xyxy[:, 2] - snapshot.boxes_xyxy[:, 0]).clamp_min(0) *
                      (snapshot.boxes_xyxy[:, 3] - snapshot.boxes_xyxy[:, 1]).clamp_min(0) /
                      float(max(1, height * width))).float()
        return source + self.adapter.delta(features=features, query=query, source=source,
                                            objectness=objectness, disappear=disappear, area_ratio=area_ratio)

    @staticmethod
    def _inputs(snapshot: FrameSnapshot) -> tuple[torch.Tensor, torch.Tensor]:
        features = snapshot.features.float()
        if snapshot.query_embedding is None:
            query = features.new_zeros(256).float()
        else:
            query = snapshot.query_embedding.float().reshape(-1)
            if query.numel() != 256:
                raise ValueError(f"expected 256-dimensional query embedding, got {query.numel()}")
        return features, query

    def observe(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence]) -> None:
        index_by_uid = {uid: index for index, uid in enumerate(snapshot.candidate_uid)}
        height, width = snapshot.image_size
        for record in evidence:
            if record.label == EvidenceLabel.UNKNOWN or record.label == EvidenceLabel.CONTRADICTORY_PROBE:
                self.evidence_unknown += 1
                continue
            index = index_by_uid.get(record.candidate_uid)
            if index is None or record.soft_target is None or record.weight <= 0:
                self.evidence_unknown += 1
                continue
            if record.label == EvidenceLabel.SUPPORTED_POS:
                self.evidence_pos += 1
            elif record.label == EvidenceLabel.SUPPORTED_NEG:
                self.evidence_neg += 1
            else:
                self.evidence_unknown += 1
                continue
            box = snapshot.boxes_xyxy[index].detach().float()
            area_ratio = float(((box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)) / max(1, height * width))
            query = (snapshot.query_embedding.detach().float().cpu().clone()
                     if snapshot.query_embedding is not None else torch.zeros(256))
            self.replay.append(_ReplayItem(
                snapshot.episode_id, int(snapshot.frame_index), int(snapshot.pred_track_id[index]),
                snapshot.features[index].detach().float().cpu().clone(), query.cpu(),
                float(snapshot.source_logits[index]), float(snapshot.objectness[index]),
                float(snapshot.disappear_time[index].clamp(0, 32) / 32.0), area_ratio,
                float(record.soft_target), float(record.weight),
            ))
        if len(self.replay) > self.replay_capacity:
            del self.replay[:-self.replay_capacity]

    def maybe_adapt(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence]) -> IndependentUpdateResult:
        started = time.perf_counter()
        if snapshot.frame_index % self.update_every_n_frames != 0:
            return self._finish(snapshot.frame_index, False, "update_schedule", 0, 0, started)
        if not self.replay:
            return self._finish(snapshot.frame_index, False, "no_reliable_pseudo_labels", 0, 0, started)
        samples = self.replay[-self.batch_size:]
        distinct = len({(item.track_id, item.frame_index) for item in samples})
        if distinct < self.minimum_distinct_track_time_samples:
            return self._finish(snapshot.frame_index, False, "insufficient_distinct_track_time_samples",
                                len(samples), distinct, started)
        device = next(self.adapter.parameters()).device
        features = torch.stack([item.feature for item in samples]).to(device=device, dtype=torch.float32)
        query = torch.stack([item.query for item in samples]).to(device=device, dtype=torch.float32)
        source = torch.tensor([item.source_logit for item in samples], device=device, dtype=torch.float32)
        objectness = torch.tensor([item.objectness for item in samples], device=device, dtype=torch.float32)
        disappear = torch.tensor([item.disappear_time for item in samples], device=device, dtype=torch.float32)
        area_ratio = torch.tensor([item.area_ratio for item in samples], device=device, dtype=torch.float32)
        targets = torch.tensor([item.target for item in samples], device=device, dtype=torch.float32)
        weights = torch.tensor([item.weight for item in samples], device=device, dtype=torch.float32)
        adapted = source + self.adapter.delta(features=features, query=query, source=source,
                                              objectness=objectness, disappear=disappear, area_ratio=area_ratio)
        loss_ref = (F.binary_cross_entropy_with_logits(adapted, targets, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-6)
        delta = adapted - source
        loss_anchor = 0.01 * delta.square().mean()
        loss_set = adapted.sum() * 0.0
        if "fusion" in self.adapter.kinds:
            grouped: dict[str, list[int]] = {}
            for index, item in enumerate(samples):
                grouped.setdefault(f"{item.episode_id}:{item.frame_index}", []).append(index)
            terms = []
            for indices in grouped.values():
                pos = [index for index in indices if float(targets[index]) >= 0.5]
                neg = [index for index in indices if float(targets[index]) < 0.5]
                if pos and neg:
                    terms.append(F.relu(0.2 - adapted[pos].max() + adapted[neg].mean()))
            if terms:
                loss_set = torch.stack(terms).mean()
        total = loss_ref + loss_anchor + 0.1 * loss_set
        trainable = list(self.adapter.parameters())
        before = [parameter.detach().clone() for parameter in trainable]
        self.optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(total)):
            return self._finish(snapshot.frame_index, False, "nonfinite_loss", len(samples), distinct, started)
        total.backward()
        has_nonzero_grad = any(parameter.grad is not None and bool(torch.any(parameter.grad.detach() != 0)) for parameter in trainable)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip))
        if not math.isfinite(grad_norm) or not has_nonzero_grad:
            self.optimizer.zero_grad(set_to_none=True)
            return self._finish(snapshot.frame_index, False, "zero_or_nonfinite_gradient", len(samples), distinct, started,
                                loss=float(total.detach()), grad_norm=grad_norm,
                                loss_ref=float(loss_ref.detach()), loss_set=float(loss_set.detach()))
        self.optimizer.step()
        parameter_delta = math.sqrt(sum(float((after.detach() - old).square().sum())
                                       for after, old in zip(trainable, before)))
        self.update_steps += 1
        self.nonzero_gradient_steps += 1
        self.parameter_delta_sum += parameter_delta
        return self._finish(snapshot.frame_index, True, "independent_adapter_updated", len(samples), distinct, started,
                            loss=float(total.detach()), grad_norm=grad_norm, parameter_delta=parameter_delta,
                            loss_ref=float(loss_ref.detach()), loss_set=float(loss_set.detach()))

    def _finish(self, frame_index: int, stepped: bool, reason: str, sample_count: int,
                distinct: int, started: float, *, loss=None, grad_norm=None, parameter_delta=None,
                loss_ref=None, loss_set=None) -> IndependentUpdateResult:
        self.last_update = IndependentUpdateResult(frame_index, stepped, reason, sample_count, distinct,
                                                   loss, grad_norm, parameter_delta,
                                                   time.perf_counter() - started, loss_ref, loss_set)
        return self.last_update


@dataclass
class _CacheEntry:
    feature: torch.Tensor
    probability: float
    frame_index: int
    track_id: int
    weight: float


class CacheTDARetriever:
    """Non-parametric TDA cache with positive/negative reliability gates."""

    def __init__(self, *, capacity: int = 512, similarity_threshold: float = 0.92,
                 momentum: float = 0.8):
        self.capacity = int(capacity)
        self.similarity_threshold = float(similarity_threshold)
        self.momentum = float(momentum)
        self.entries: dict[int, _CacheEntry] = {}
        self.cache_hits = 0
        self.cache_updates = 0
        self.parameter_delta_sum = 0.0
        self.evidence_pos = 0
        self.evidence_neg = 0
        self.evidence_unknown = 0
        self.last_update = IndependentUpdateResult(-1, False, "not_started", 0, 0, None, None, 0.0, 0.0)

    def predict_logits(self, snapshot: FrameSnapshot) -> torch.Tensor:
        output = snapshot.source_logits.float().clone()
        valid_entries = list(self.entries.values())
        for index, track_id in enumerate(snapshot.pred_track_id.detach().cpu().tolist()):
            if int(track_id) < 0:
                continue
            entry = self.entries.get(int(track_id))
            if entry is None and valid_entries:
                feature = F.normalize(snapshot.features[index].float().cpu(), dim=0)
                similarities = [float(torch.dot(feature, F.normalize(item.feature, dim=0))) for item in valid_entries]
                best = max(range(len(similarities)), key=similarities.__getitem__)
                if similarities[best] >= self.similarity_threshold:
                    entry = valid_entries[best]
            if entry is not None:
                self.cache_hits += 1
                source_probability = float(snapshot.source_probs[index])
                probability = 0.75 * float(entry.probability) + 0.25 * source_probability
                output[index] = torch.as_tensor(probability, device=output.device).clamp(1e-5, 1 - 1e-5).logit()
        return output

    def observe(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence]) -> None:
        index_by_uid = {uid: index for index, uid in enumerate(snapshot.candidate_uid)}
        for record in evidence:
            if record.label not in {EvidenceLabel.SUPPORTED_POS, EvidenceLabel.SUPPORTED_NEG} or record.soft_target is None:
                self.evidence_unknown += 1
                continue
            index = index_by_uid.get(record.candidate_uid)
            if index is None or int(snapshot.pred_track_id[index]) < 0 or record.weight <= 0:
                self.evidence_unknown += 1
                continue
            if record.label == EvidenceLabel.SUPPORTED_POS:
                self.evidence_pos += 1
            else:
                self.evidence_neg += 1
            track_id = int(snapshot.pred_track_id[index])
            target = float(record.soft_target)
            previous = self.entries.get(track_id)
            probability = target if previous is None else self.momentum * previous.probability + (1.0 - self.momentum) * target
            self.entries[track_id] = _CacheEntry(snapshot.features[index].detach().float().cpu().clone(), probability,
                                                 int(snapshot.frame_index), track_id, float(record.weight))
            self.cache_updates += 1
        if len(self.entries) > self.capacity:
            oldest = sorted(self.entries, key=lambda key: self.entries[key].frame_index)[:-self.capacity]
            for key in oldest:
                self.entries.pop(key, None)

    def maybe_adapt(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence]) -> IndependentUpdateResult:
        self.last_update = IndependentUpdateResult(snapshot.frame_index, False, "cache_update_no_optimizer",
                                                   len(evidence), len(self.entries), 0.0, None, 0.0, 0.0)
        return self.last_update


def is_independent_method(method: str) -> bool:
    return method == "TDA_RMOT" or method in {
        "VISUAL_TTA", "LANGUAGE_TTA", "FUSION_TTA", "VISUAL_LANGUAGE_TTA",
        "VISUAL_FUSION_TTA", "LANGUAGE_FUSION_TTA", "ALL_TTA",
    }
