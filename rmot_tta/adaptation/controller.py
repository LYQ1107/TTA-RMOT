"""Causal optimizer controller for the external referring adapter."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from rmot_tta.adaptation.losses import (
    bernoulli_entropy_from_logits,
    equivalent_query_consistency,
    masked_weighted_mean,
    mutual_exclusion_penalty,
    source_anchor_loss,
    weighted_bce_with_logits,
)
from rmot_tta.adaptation.residual_head import Head257, ResidualReferringAdapter
from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.memory.semantic_memory import SemanticMemory
from rmot_tta.types import FrameSnapshot


@dataclass
class UpdateResult:
    frame_index: int
    stepped: bool
    reason: str
    sample_count: int
    distinct_track_time_samples: int
    loss: float | None
    loss_ref: float | None
    loss_eq: float | None
    loss_excl: float | None
    loss_anchor: float | None
    loss_delta: float | None
    grad_norm: float | None
    parameter_delta: float | None
    elapsed_sec: float


class AdaptationController:
    def __init__(self, method: str, source_head_weight: torch.Tensor, source_head_bias: torch.Tensor,
                 *, lr: float = 1e-3, update_every_n_frames: int = 4, batch_size: int = 32,
                 minimum_distinct_track_time_samples: int = 2, grad_clip: float = 1.0,
                 use_logic: bool = True, use_state: bool = True, hidden_dim: int = 32,
                 memory_capacity: int = 512, per_track_capacity: int = 16,
                 fixed_ttl_frames: int = 16, weight_decay: float = 0.0):
        self.method = method
        self.update_every_n_frames = int(update_every_n_frames)
        self.batch_size = int(batch_size)
        self.minimum_distinct_track_time_samples = int(minimum_distinct_track_time_samples)
        self.grad_clip = float(grad_clip)
        self.use_logic = bool(use_logic)
        self.use_state = bool(use_state)
        self.fixed_ttl_frames = int(fixed_ttl_frames)
        self.weight_decay = float(weight_decay)
        self.adapter = ResidualReferringAdapter(hidden_dim=hidden_dim)
        self.head257 = Head257(source_head_weight, source_head_bias)
        self.memory = SemanticMemory(memory_capacity, per_track_capacity)
        self.optimizer = torch.optim.AdamW(self._trainable_parameters(), lr=float(lr), weight_decay=self.weight_decay)
        self.cache_probabilities: dict[tuple[str, str, int], float] = {}
        self.logic_observations: list[tuple[torch.Tensor, float, float | None, float | None, float, int]] = []
        self.last_update = UpdateResult(-1, False, "not_started", 0, 0, None, None, None, None, None, None, None, None, 0.0)
        self.update_steps = 0

    def reset(self) -> None:
        device = next(self._trainable_parameters()).device
        self.adapter = ResidualReferringAdapter(hidden_dim=self.adapter.hidden_dim, max_abs_residual=self.adapter.max_abs_residual)
        self.adapter.to(device)
        self.head257.to(device)
        self.optimizer = torch.optim.AdamW(self._trainable_parameters(), lr=self.optimizer.param_groups[0]["lr"], weight_decay=self.weight_decay)
        self.memory.clear()
        self.cache_probabilities.clear()
        self.logic_observations.clear()
        self.update_steps = 0
        self.last_update = UpdateResult(-1, False, "reset", 0, 0, None, None, None, None, None, None, None, None, 0.0)

    def predict_logits(self, snapshot: FrameSnapshot) -> torch.Tensor:
        if self.method == "FROZEN":
            return snapshot.source_logits.float().clone()
        if self.method == "HEAD257":
            return self.head257(snapshot.features).detach()
        if self.method == "PROBE_CACHE":
            values = snapshot.source_logits.float().clone()
            for index, track_id in enumerate(snapshot.pred_track_id.detach().cpu().tolist()):
                cached = self.cache_probabilities.get((snapshot.episode_id, snapshot.expression_id, int(track_id)))
                if cached is not None:
                    values[index] = torch.as_tensor(cached, device=values.device).clamp(1e-5, 1 - 1e-5).logit()
            return values
        return self.adapter(snapshot.source_logits, snapshot.features).detach()

    def observe(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence],
                *, eq_logits: torch.Tensor | None = None, cf_logits: torch.Tensor | None = None) -> None:
        for record in evidence:
            index = next((i for i, uid in enumerate(snapshot.candidate_uid) if uid == record.candidate_uid), None)
            if index is None or record.label not in {EvidenceLabel.SUPPORTED_POS, EvidenceLabel.SUPPORTED_NEG}:
                continue
            if self.use_logic and (record.probe_eq is not None or record.probe_cf is not None):
                # Keep only detached CPU features so the bounded logic cache does
                # not retain the current CUDA graph or consume model memory.
                self.logic_observations.append((
                    snapshot.features[index].detach().float().cpu().clone(),
                    float(snapshot.source_logits[index].detach().cpu()),
                    record.probe_eq,
                    record.probe_cf,
                    float(record.weight),
                    int(snapshot.frame_index),
                ))
                if len(self.logic_observations) > 256:
                    del self.logic_observations[:-256]
            if self.method == "PROBE_CACHE":
                if record.soft_target is not None:
                    self.cache_probabilities[(snapshot.episode_id, snapshot.expression_id, record.pred_track_id)] = float(record.soft_target)
                continue
            if self.use_state:
                self.memory.observe(
                    snapshot.episode_id, snapshot.expression_id, record.pred_track_id, record.label,
                    snapshot.frame_index, snapshot.features[index], float(snapshot.source_logits[index]),
                    record.soft_target, record.weight, record.evidence_source, record.query_type,
                    f"{record.evidence_source}:{snapshot.frame_index}", record.alignment_quality,
                )
            elif record.soft_target is not None and record.weight > 0:
                # Fixed-TTL control: preserve evidence in the same replay shape,
                # but do not advance semantic state versions.
                self.memory.replay.add(__import__("rmot_tta.memory.replay_buffer", fromlist=["ReplaySample"]).ReplaySample(
                    snapshot.episode_id, snapshot.expression_id, record.pred_track_id, snapshot.frame_index,
                    snapshot.features[index], float(snapshot.source_logits[index]), float(record.soft_target),
                    float(record.weight), record.evidence_source, record.query_type, 0,
                    f"{record.evidence_source}:{snapshot.frame_index}", record.alignment_quality,
                ))

    def maybe_adapt(self, snapshot: FrameSnapshot, evidence: list[CandidateEvidence],
                    *, eq_probs: torch.Tensor | None = None, cf_probs: torch.Tensor | None = None) -> UpdateResult:
        started = time.perf_counter()
        if self.method in {"FROZEN", "PROBE_CACHE"}:
            self.last_update = UpdateResult(snapshot.frame_index, False, "method_has_no_optimizer", 0, 0, None, None, None, None, None, None, None, None, time.perf_counter() - started)
            return self.last_update
        if snapshot.frame_index % self.update_every_n_frames != 0:
            self.last_update = UpdateResult(snapshot.frame_index, False, "update_schedule", 0, 0, None, None, None, None, None, None, None, None, time.perf_counter() - started)
            return self.last_update
        if self.method == "ENTROPY_HEAD":
            valid = snapshot.valid_mask
            distinct = int(valid.sum())
            if not bool(valid.any()) or distinct < self.minimum_distinct_track_time_samples:
                self.last_update = UpdateResult(snapshot.frame_index, False, "insufficient_entropy_candidates", distinct, distinct,
                                                None, None, None, None, None, None, None, None,
                                                time.perf_counter() - started)
                return self.last_update
            features = snapshot.features[valid].detach().float()
            source = snapshot.source_logits[valid].detach().float()
            trainable = list(self._trainable_parameters())
            before = [parameter.detach().clone() for parameter in trainable]
            self.optimizer.zero_grad(set_to_none=True)
            adapted = self.adapter(source, features)
            loss_ref = masked_weighted_mean(bernoulli_entropy_from_logits(adapted))
            loss_eq = adapted.sum() * 0.0
            loss_excl = adapted.sum() * 0.0
            loss_anchor = adapted.sum() * 0.0
            loss_delta = self.adapter.delta(features).square().mean()
            total = loss_ref + 0.001 * loss_delta
            if not bool(torch.isfinite(total)):
                self.optimizer.zero_grad(set_to_none=True)
                self.last_update = UpdateResult(snapshot.frame_index, False, "nonfinite_loss", distinct, distinct,
                                                None, None, None, None, None, None, None, None,
                                                time.perf_counter() - started)
                return self.last_update
            total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip))
            if not math.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                self.last_update = UpdateResult(snapshot.frame_index, False, "nonfinite_gradient", distinct, distinct,
                                                float(total.detach()), float(loss_ref.detach()), 0.0, 0.0, 0.0,
                                                float(loss_delta.detach()), grad_norm, None,
                                                time.perf_counter() - started)
                return self.last_update
            self.optimizer.step()
            delta = math.sqrt(sum(float((after.detach() - old).square().sum()) for after, old in zip(trainable, before)))
            self.update_steps += 1
            self.last_update = UpdateResult(snapshot.frame_index, True, "entropy_updated", distinct, distinct,
                                            float(total.detach()), float(loss_ref.detach()), 0.0, 0.0, 0.0,
                                            float(loss_delta.detach()), grad_norm, delta,
                                            time.perf_counter() - started)
            return self.last_update
        samples = self.memory.replay.batch(self.batch_size, next(self.adapter.parameters()).device)
        if not self.use_state and self.fixed_ttl_frames >= 0:
            samples = [sample for sample in samples
                       if snapshot.frame_index - sample.timestamp <= self.fixed_ttl_frames]
        distinct = len({(sample.pred_track_id, sample.timestamp) for sample in samples})
        if not samples or distinct < self.minimum_distinct_track_time_samples:
            self.last_update = UpdateResult(snapshot.frame_index, False, "insufficient_supervision", len(samples), distinct, None, None, None, None, None, None, None, None, time.perf_counter() - started)
            return self.last_update
        features = torch.stack([sample.feature for sample in samples]).float()
        source = torch.tensor([sample.source_logit for sample in samples], dtype=torch.float32, device=features.device)
        targets = torch.tensor([sample.soft_target for sample in samples], dtype=torch.float32, device=features.device)
        weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32, device=features.device)
        trainable = list(self._trainable_parameters())
        before = [parameter.detach().clone() for parameter in trainable]
        self.optimizer.zero_grad(set_to_none=True)
        adapted = self.head257(features) if self.method == "HEAD257" else self.adapter(source, features)
        loss_ref = weighted_bce_with_logits(adapted, targets, weights=weights)
        # Apply query-relation constraints to the same aligned candidates that
        # produced the replay evidence.  q_eq/q_cf are frozen-probe outputs;
        # only q0 is differentiated through the external adapter.
        logic = [item for item in self.logic_observations
                 if item[5] <= snapshot.frame_index and
                 (self.use_state or snapshot.frame_index - item[5] <= self.fixed_ttl_frames)]
        if logic and self.use_logic:
            logic_features = torch.stack([item[0] for item in logic]).to(features.device)
            logic_source = torch.tensor([item[1] for item in logic], dtype=torch.float32, device=features.device)
            logic_weights = torch.tensor([item[4] for item in logic], dtype=torch.float32, device=features.device)
            logic_adapted = self.head257(logic_features) if self.method == "HEAD257" else self.adapter(logic_source, logic_features)
            logic_p0 = logic_adapted.sigmoid()
            eq_mask = torch.tensor([item[2] is not None for item in logic], dtype=torch.bool, device=features.device)
            cf_mask = torch.tensor([item[3] is not None for item in logic], dtype=torch.bool, device=features.device)
            eq_probs = torch.tensor([0.5 if item[2] is None else float(item[2]) for item in logic],
                                     dtype=torch.float32, device=features.device)
            cf_probs = torch.tensor([0.5 if item[3] is None else float(item[3]) for item in logic],
                                     dtype=torch.float32, device=features.device)
            loss_eq = equivalent_query_consistency(logic_p0, eq_probs, mask=eq_mask, weights=logic_weights)
            loss_excl = mutual_exclusion_penalty(logic_p0, cf_probs, mask=cf_mask, weights=logic_weights)
        else:
            loss_eq = adapted.sum() * 0.0
            loss_excl = adapted.sum() * 0.0
        source_probability = source.sigmoid().detach()
        anchor_mask = torch.tensor(
            [sample.evidence_source != "CONTRAST_SELF_PROBE" and
             abs(float(sample.soft_target) - float(source_probability[index])) <= 0.25
             for index, sample in enumerate(samples)], dtype=torch.bool, device=source.device)
        loss_anchor = source_anchor_loss(source, adapted, mask=anchor_mask)
        loss_delta = self.adapter.delta(features).square().mean() if len(features) and self.method != "HEAD257" else adapted.sum() * 0.0
        total = loss_ref + 0.2 * loss_eq + 0.1 * loss_excl + 0.1 * loss_anchor + 0.001 * loss_delta
        if not bool(torch.isfinite(total)):
            self.optimizer.zero_grad(set_to_none=True)
            self.last_update = UpdateResult(snapshot.frame_index, False, "nonfinite_loss", len(samples), distinct, None, None, None, None, None, None, None, None, time.perf_counter() - started)
            return self.last_update
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.last_update = UpdateResult(snapshot.frame_index, False, "nonfinite_gradient", len(samples), distinct, float(total.detach()), float(loss_ref.detach()), float(loss_eq.detach()), float(loss_excl.detach()), float(loss_anchor.detach()), float(loss_delta.detach()), grad_norm, None, time.perf_counter() - started)
            return self.last_update
        self.optimizer.step()
        delta = math.sqrt(sum(float((after.detach() - old).square().sum()) for after, old in zip(trainable, before)))
        self.update_steps += 1
        self.last_update = UpdateResult(snapshot.frame_index, True, "updated", len(samples), distinct, float(total.detach()), float(loss_ref.detach()), float(loss_eq.detach()), float(loss_excl.detach()), float(loss_anchor.detach()), float(loss_delta.detach()), grad_norm, delta, time.perf_counter() - started)
        return self.last_update

    def _trainable_parameters(self):
        return self.head257.parameters() if self.method == "HEAD257" else self.adapter.parameters()
