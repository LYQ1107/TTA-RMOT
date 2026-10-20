"""Strict causal online runner for frozen TempRMOT plus external adapters."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml

from rmot_tta.adaptation.controller import AdaptationController
from rmot_tta.adaptation.ema_baseline import EmaReferringBaseline
from rmot_tta.baseline.temprmot_wrapper import TempRMOTConfig, TempRMOTWrapper
from rmot_tta.data.shift_loader import load_compact_runtime_rows, load_runtime_image
from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.evidence.selector import select_candidate_evidence
from rmot_tta.probes.alignment import align_probe_results
from rmot_tta.probes.frozen_query_probe import FrozenQueryProbe
from rmot_tta.query.variants import generate_variants
from rmot_tta.runtime.state import RawFrameRing


ROOT = Path(__file__).resolve().parents[2]
PROBE_METHODS = {"PROBE_CACHE", "PROBE_ST", "QCS_TTA", "NO_STATE"}
METHOD_IMPLEMENTATIONS = {
    # These names are protocol labels for the first ShiftRMOT release.  They
    # map to the repository's already-audited adapters and are reported as
    # inspired/adapted, not as exact reproductions of a different backbone.
    "TDA_RMOT": "EMA_ST",
    "SAR_VISUAL_INSPIRED": "ENTROPY_HEAD",
    "TPT_LANGUAGE_INSPIRED": "QCS_TTA",
    "TCR_FUSION_INSPIRED": "QCS_TTA",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _preprocess(path: Path, row: dict[str, Any] | None = None) -> tuple[torch.Tensor, tuple[int, int]]:
    if row is not None and "visual_shift" in row:
        image = load_runtime_image(row)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 800.0 / min(height, width)
    if max(height, width) * scale > 1536.0:
        scale = 1536.0 / max(height, width)
    target_h, target_w = int(height * scale), int(width * scale)
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std, (height, width)


def _logit(probability: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(value / (1.0 - value))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(float(value) for value in values)
    index = (len(values) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _prediction_keep_mask(ids: torch.Tensor, objectness: torch.Tensor, boxes: torch.Tensor,
                          referring_logits: torch.Tensor, inference_config: dict[str, Any]) -> torch.Tensor:
    """Select candidates without using target labels or future frames.

    The default is the original calibrated Bernoulli gate.  ``relative_margin``
    is an explicit domain-shift control: it keeps candidates whose current
    referring logit is within a fixed margin of the best valid candidate in
    this frame, allowing a set of objects rather than forcing top-1.  The cap
    only limits pathological tracker frames and is applied after ranking.
    """
    keep = (ids >= 0) & (objectness > float(inference_config["objectness_threshold"]))
    wh = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    keep &= wh > float(inference_config["area_threshold"])
    selection = inference_config.get("selection", {})
    mode = str(selection.get("mode", "absolute"))
    if mode == "absolute":
        keep &= referring_logits.sigmoid() > float(inference_config["referring_threshold"])
        return keep
    if mode != "relative_margin":
        raise ValueError(f"unsupported inference selection mode: {mode}")

    valid_indices = torch.where(keep)[0]
    if valid_indices.numel() == 0:
        return keep
    margin = float(selection.get("relative_logit_margin", 0.75))
    cutoff = referring_logits[valid_indices].max() - margin
    ranked = valid_indices[referring_logits[valid_indices] >= cutoff]
    max_candidates = int(selection.get("max_candidates_per_frame", 16))
    if max_candidates > 0 and ranked.numel() > max_candidates:
        order = torch.argsort(referring_logits[ranked], descending=True)
        ranked = ranked[order[:max_candidates]]
    relative_keep = torch.zeros_like(keep)
    relative_keep[ranked] = True
    return relative_keep


def _gpu_uuid() -> str | None:
    try:
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"
        ], text=True).strip().splitlines()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        index = int(visible.split(",")[0]) if visible and visible.split(",")[0].isdigit() else 0
        for row in rows:
            parts = [value.strip() for value in row.split(",")]
            if parts and int(parts[0]) == index:
                return parts[1]
    except Exception:
        return None
    return None


class OnlineEpisodeRunner:
    def __init__(self, config_path: Path, runtime_manifest: Path, output_dir: Path,
                 *, methods: list[str] | None = None, max_expressions: int = 0,
                 max_frames: int = 0, episodes: list[str] | None = None):
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.runtime_manifest = runtime_manifest.resolve()
        self.output_dir = output_dir.resolve()
        self.methods = methods or list(self.config.get("methods", []))
        self.max_expressions = int(max_expressions)
        self.max_frames = int(max_frames)
        self.episodes = {str(value) for value in (episodes or [])}
        self.model_config = self.config["model"]
        self.inference_config = self.config["inference"]
        self.probe_config = self.config["probe"]
        self.evidence_config = self.config["evidence"]
        self.adaptation_config = self.config["adaptation"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.wrapper: TempRMOTWrapper | None = None

    @staticmethod
    def _implementation(method: str) -> str:
        return METHOD_IMPLEMENTATIONS.get(method, method)

    def _load_rows(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        if bool(self.config.get("data", {}).get("compact_shift_manifest", False)):
            grouped = load_compact_runtime_rows(self.runtime_manifest, episodes=self.episodes or None)
            if not grouped:
                raise KeyError(f"no compact runtime conditions matched episodes: {sorted(self.episodes)}")
            return grouped
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for line in self.runtime_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {"label_path", "expression_path", "gt_box", "gt_id", "label", "target"}
            if forbidden.intersection(row):
                raise AssertionError(f"runtime manifest contains evaluator field: {row}")
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])
            grouped[(str(row["video_id"]), str(row["expression_id"]))].append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: int(row["frame_index"]))
        keys = sorted(grouped)
        if self.episodes:
            keys = [key for key in keys if f"{key[0]}/{key[1]}" in self.episodes or f"{key[0]}+{key[1]}" in self.episodes]
            requested = {value.replace("+", "/") for value in self.episodes}
            found = {f"{key[0]}/{key[1]}" for key in keys}
            missing = sorted(requested - found)
            if missing:
                raise KeyError(f"requested episodes missing from runtime manifest: {missing}")
        if self.max_expressions > 0:
            keys = keys[: self.max_expressions]
        return {key: grouped[key] for key in keys}

    def _make_wrapper(self) -> TempRMOTWrapper:
        return TempRMOTWrapper(TempRMOTConfig(
            source_root=Path(self.model_config["source_root"]),
            checkpoint=Path(self.model_config["checkpoint"]),
            text_root=Path(self.model_config["text_root"]),
            device=str(self.device), hist_len=4,
        ))

    def _controllers(self, wrapper: TempRMOTWrapper) -> tuple[dict[str, AdaptationController], dict[str, EmaReferringBaseline]]:
        weight, bias = wrapper.source_head
        controllers: dict[str, AdaptationController] = {}
        emas: dict[str, EmaReferringBaseline] = {}
        for method in self.methods:
            implementation = self._implementation(method)
            if implementation == "FROZEN":
                continue
            if implementation == "EMA_ST":
                emas[method] = EmaReferringBaseline()
                continue
            controller = AdaptationController(
                implementation, weight, bias, lr=float(self.adaptation_config["lr"]),
                update_every_n_frames=int(self.adaptation_config["update_every_n_frames"]),
                batch_size=int(self.adaptation_config["batch_size"]),
                minimum_distinct_track_time_samples=int(self.adaptation_config["minimum_distinct_track_time_samples"]),
                grad_clip=float(self.adaptation_config["grad_clip"]),
                use_logic=implementation in {"QCS_TTA", "NO_STATE", "HEAD257", "PROBE_CACHE"},
                use_state=implementation not in {"PROBE_ST", "ENTROPY_HEAD"},
                hidden_dim=int(self.adaptation_config["hidden_dim"]),
                memory_capacity=int(self.config.get("memory", {}).get("capacity", 512)),
                per_track_capacity=int(self.config.get("memory", {}).get("per_track_capacity", 16)),
                fixed_ttl_frames=int(self.config.get("memory", {}).get("fixed_ttl_frames", 16)),
                weight_decay=float(self.adaptation_config.get("weight_decay", 0.0)),
            )
            controller.adapter.to(self.device)
            controller.head257.to(self.device)
            controllers[method] = controller
        return controllers, emas

    def _prediction(self, method: str, result, controller: AdaptationController | None,
                    ema: EmaReferringBaseline | None):
        instances = result.track_instances
        snapshot = result.snapshot
        # Source post-processing may keep different fields on CPU/GPU.  Move
        # the small method-specific score calculation to the snapshot device
        # before applying a joint keep mask.
        score_device = snapshot.source_logits.device
        ids = instances.obj_idxes.detach().long().to(score_device)
        boxes = instances.boxes.detach().float().to(score_device)
        objectness = instances.scores.detach().float().to(score_device)
        source_probs = instances.refers.detach().float().to(score_device)
        by_id = {}
        for index, track_id in enumerate(snapshot.pred_track_id.detach().cpu().tolist()):
            if int(track_id) >= 0 and int(track_id) not in by_id:
                by_id[int(track_id)] = index
        referring_logits = torch.tensor(
            [_logit(value) for value in source_probs.detach().cpu().tolist()],
            dtype=torch.float32,
            device=source_probs.device,
        )
        if controller is not None:
            valid_indices = [index for index, track_id in enumerate(ids.cpu().tolist()) if int(track_id) in by_id]
            if valid_indices:
                snap_indices = [by_id[int(ids[index])] for index in valid_indices]
                adapted = controller.predict_logits(
                    snapshot,  # predict the full snapshot once below; this keeps UID alignment explicit
                )
                adapted_by_id = {int(snapshot.pred_track_id[i]): adapted[i] for i in range(snapshot.count) if int(snapshot.pred_track_id[i]) >= 0}
                for index in valid_indices:
                    referring_logits[index] = adapted_by_id[int(ids[index])].detach().cpu()
        elif ema is not None:
            source = source_probs.clone()
            referring_probs = ema.adjust(ids, source)
            referring_logits = referring_probs.clamp(1e-6, 1 - 1e-6).logit()
        referring_probs = referring_logits.sigmoid()
        h, w = snapshot.image_size
        keep = _prediction_keep_mask(ids, objectness, boxes, referring_logits, self.inference_config)
        return {
            "boxes": boxes[keep].cpu(), "ids": ids[keep].cpu(), "objectness": objectness[keep].cpu(),
            "source_probs": source_probs[keep].cpu(), "referring_probs": referring_probs[keep].cpu(),
            "referring_logits": referring_logits[keep].cpu(), "image_size": (h, w),
        }

    @staticmethod
    def _write_prediction(path: Path, frame_index: int, prediction: dict[str, Any], frame_offset: int = 1) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("a", encoding="utf-8") as handle:
            for box, track_id, probability in zip(prediction["boxes"].tolist(), prediction["ids"].tolist(), prediction["referring_probs"].tolist()):
                x1, y1, x2, y2 = [float(value) for value in box]
                handle.write(f"{int(frame_index) + int(frame_offset)},{int(track_id)},{x1:.6f},{y1:.6f},{x2 - x1:.6f},{y2 - y1:.6f},{float(probability):.8f},1,1,1\n")
                count += 1
        return count

    def _confidence_evidence(self, snapshot) -> list[CandidateEvidence]:
        records: list[CandidateEvidence] = []
        for index, (uid, track_id, probability, valid) in enumerate(zip(
            snapshot.candidate_uid, snapshot.pred_track_id.tolist(), snapshot.source_probs.tolist(), snapshot.valid_mask.tolist())):
            if not valid:
                continue
            if probability >= 0.8:
                records.append(CandidateEvidence(uid, EvidenceLabel.SUPPORTED_POS, float(probability), 1.0, "HIGH_CONFIDENCE", "q0", 1.0,
                                                  "source_probability_above_self_training_threshold", snapshot.frame_index, int(track_id)))
            elif probability <= 0.2:
                records.append(CandidateEvidence(uid, EvidenceLabel.SUPPORTED_NEG, 0.0, 1.0, "HIGH_CONFIDENCE", "q0", 1.0,
                                                  "source_probability_below_self_training_threshold", snapshot.frame_index, int(track_id)))
        return records

    def _method_evidence(self, method: str, snapshot, packet) -> list[CandidateEvidence]:
        implementation = self._implementation(method)
        if implementation == "FROZEN":
            return []
        if implementation in {"ENTROPY_HEAD", "EMA_ST"}:
            return self._confidence_evidence(snapshot)
        if implementation == "PROBE_ST":
            if packet is None or not packet.results:
                return []
            q0 = next((item for item in packet.results if item.query_type == "q0"), None)
            if q0 is None:
                return []
            by_box = {candidate.track_id: candidate for candidate in q0.candidates if candidate.track_id >= 0}
            records = []
            for index, (uid, track_id, valid) in enumerate(zip(snapshot.candidate_uid, snapshot.pred_track_id.tolist(), snapshot.valid_mask.tolist())):
                candidate = by_box.get(int(track_id))
                if valid and candidate is not None and candidate.probability >= 0.8:
                    records.append(CandidateEvidence(uid, EvidenceLabel.SUPPORTED_POS, candidate.probability, 1.0, "PROBE_HIGH_CONFIDENCE", "q0", 1.0,
                                                      "q0_probe_high_confidence_without_counterfactual", snapshot.frame_index, int(track_id)))
            return records
        if packet is None or not packet.results:
            return []
        aligned = align_probe_results(snapshot, packet.results,
                                      min_iou=float(self.probe_config["min_match_iou"]),
                                      ambiguity_iou_margin=float(self.probe_config["ambiguity_iou_margin"]))
        return select_candidate_evidence(snapshot, aligned,
                                         tau_pos=float(self.evidence_config["positive_threshold"]),
                                         tau_neg=float(self.evidence_config["negative_threshold"]),
                                         equiv_tolerance=float(self.evidence_config["equivalent_tolerance"]),
                                         counterfactual_margin=float(self.evidence_config["contrast_margin"]))

    def _method_summary(self, stats: dict[str, Any], frame_count: int) -> dict[str, Any]:
        elapsed = max(float(stats["elapsed_sec"]), 1e-9)
        return {
            **stats,
            "frame_query_count": int(frame_count),
            "end_to_end_fps_shared_source": float(frame_count / elapsed),
            "p50_latency_sec": _percentile(stats["latencies"], 0.50),
            "p95_latency_sec": _percentile(stats["latencies"], 0.95),
            "peak_memory_allocated_bytes": int(stats.get("peak_memory_allocated_bytes", 0)),
            "peak_memory_reserved_bytes": int(stats.get("peak_memory_reserved_bytes", 0)),
            "optimizer_steps": int(stats.get("optimizer_steps", 0)),
            "probe_calls": int(stats.get("probe_calls", 0)),
        }

    def run(self) -> dict[str, Any]:
        grouped = self._load_rows()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prediction_root = self.output_dir / "predictions"
        if prediction_root.exists() and any(prediction_root.rglob("predict.txt")):
            raise FileExistsError(
                f"prediction files already exist under {prediction_root}; use a fresh run directory "
                "or restart the selected episode from its beginning"
            )
        trace_path = self.output_dir / "update_trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        candidate_diagnostics_path = self.output_dir / "candidate_diagnostics.jsonl"
        if candidate_diagnostics_path.exists():
            candidate_diagnostics_path.unlink()
        started_all = time.perf_counter()
        self.wrapper = self._make_wrapper()
        method_stats = {
            method: {"elapsed_sec": 0.0, "latencies": [], "selected_rows": 0, "probe_calls": 0,
                     "optimizer_steps": 0, "supported_candidates": 0, "candidate_total": 0,
                     "evidence_pos": 0, "evidence_neg": 0, "evidence_unknown": 0,
                     "shared_source_forward": True, "ring_peak_bytes": 0,
                     "peak_memory_allocated_bytes": 0, "peak_memory_reserved_bytes": 0}
            for method in self.methods
        }
        # Controllers and the shared frozen-query probe are reset at every
        # (video, expression) episode.  Keep run-level counters separately so
        # the final metadata reports the complete suite rather than only the
        # last episode.
        probe_calls_total = 0
        expression_count = 0
        frame_count = 0
        with (trace_path.open("a", encoding="utf-8") as trace,
              candidate_diagnostics_path.open("a", encoding="utf-8") as candidate_diagnostics):
            for (video_id, expression_id), rows in grouped.items():
                expression_count += 1
                if expression_count > self.max_expressions > 0:
                    break
                self.wrapper.reset_runtime()
                sentence = str(rows[0]["query_text"])
                if any(str(row["query_text"]) != sentence for row in rows):
                    raise AssertionError(f"query text changed inside episode {video_id}/{expression_id}")
                spec, variants = generate_variants(sentence)
                controllers, emas = self._controllers(self.wrapper)
                probe = FrozenQueryProbe(
                    lambda _query_type: self.wrapper.clone(),
                    every_n_frames=int(self.probe_config["every_n_frames"]),
                    dynamic_history_frames=int(self.probe_config["dynamic_history_frames"]),
                    max_queries=int(self.probe_config["max_query_count"]),
                )
                ring = RawFrameRing(max_length=int(self.probe_config["dynamic_history_frames"]))
                episode_uses_probe = bool(self.probe_config["enabled"]) and any(
                    self._implementation(method) in PROBE_METHODS for method in self.methods
                )
                track_instances = None
                start_episode = time.perf_counter()
                for local_index, row in enumerate(rows):
                    if self.max_frames > 0 and local_index >= self.max_frames:
                        break
                    image, image_size = _preprocess(Path(row["image_path"]), row)
                    frame_index = int(row["frame_index"])
                    frame_started = time.perf_counter()
                    result = self.wrapper.infer_frame(image, sentence, image_size, track_instances, frame_index,
                                                      f"{video_id}:{expression_id}", expression_id, video_id)
                    track_instances = result.track_instances
                    snapshot = result.snapshot
                    frame_count += 1
                    ring.append(frame_index, image, image_size)
                    ring_peak = ring.bytes_stored
                    packet = None
                    probe_due = bool(episode_uses_probe and probe.due(frame_index))
                    if probe_due:
                        is_dynamic = "dynamic" in spec.constraint_types
                        required = int(self.probe_config["dynamic_history_frames"]) if is_dynamic else 1
                        if len(ring.items) >= required:
                            packet = probe.run(ring.as_list(), spec, variants, snapshot)
                        else:
                            packet = None
                    if packet is not None:
                        probe_calls_total += int(packet.calls)
                    valid_candidate_rows = []
                    for uid, track_id, valid, objectness, source_probability, box in zip(
                        snapshot.candidate_uid,
                        snapshot.pred_track_id.detach().cpu().tolist(),
                        snapshot.valid_mask.detach().cpu().tolist(),
                        snapshot.objectness.detach().cpu().tolist(),
                        snapshot.source_probs.detach().cpu().tolist(),
                        snapshot.boxes_xyxy.detach().cpu().tolist(),
                    ):
                        if not valid:
                            continue
                        width = max(0.0, float(box[2]) - float(box[0]))
                        height = max(0.0, float(box[3]) - float(box[1]))
                        valid_candidate_rows.append({
                            "candidate_uid": uid,
                            "track_id": int(track_id),
                            "objectness": float(objectness),
                            "source_probability": float(source_probability),
                            "area_pixels": width * height,
                            "passes_objectness_area": bool(
                                float(objectness) > float(self.inference_config["objectness_threshold"])
                                and width * height > float(self.inference_config["area_threshold"])
                            ),
                            "box_xyxy": [float(value) for value in box],
                        })
                    candidate_diagnostics.write(json.dumps({
                        "video_id": video_id, "expression_id": expression_id,
                        "frame_index": frame_index, "image_size": list(snapshot.image_size),
                        "candidate_count": snapshot.count,
                        "valid_candidate_count": int(snapshot.valid_mask.sum()),
                        "candidates": valid_candidate_rows,
                    }, ensure_ascii=False) + "\n")
                    candidate_diagnostics.flush()
                    evidence_by_method: dict[str, list[dict[str, Any]]] = {}
                    for method in self.methods:
                        controller = controllers.get(method)
                        ema = emas.get(method)
                        prediction = self._prediction(method, result, controller, ema)
                        prediction_path = self.output_dir / "predictions" / method / video_id / expression_id / "predict.txt"
                        selected = self._write_prediction(
                            prediction_path, frame_index, prediction,
                            frame_offset=int(row.get("trackeval_frame_offset", 1)),
                        )
                        method_stats[method]["selected_rows"] += selected
                        method_stats[method]["candidate_total"] += int(snapshot.valid_mask.sum())
                        method_stats[method]["ring_peak_bytes"] = max(method_stats[method]["ring_peak_bytes"], ring_peak)
                        if ema is not None:
                            valid = snapshot.valid_mask
                            ema.observe(snapshot.pred_track_id[valid], snapshot.source_probs[valid])
                        evidence = self._method_evidence(method, snapshot, packet)
                        evidence_by_method[method] = [{
                            "candidate_uid": record.candidate_uid,
                            "label": record.label.value,
                            "soft_target": record.soft_target,
                            "weight": record.weight,
                            "evidence_source": record.evidence_source,
                            "query_type": record.query_type,
                            "alignment_quality": record.alignment_quality,
                            "explanation": record.explanation,
                            "frame_index": record.frame_index,
                            "pred_track_id": record.pred_track_id,
                            "probe_q0": record.probe_q0,
                            "probe_eq": record.probe_eq,
                            "probe_cf": record.probe_cf,
                        } for record in evidence]
                        for record in evidence:
                            method_stats[method]["evidence_pos"] += int(record.label == EvidenceLabel.SUPPORTED_POS)
                            method_stats[method]["evidence_neg"] += int(record.label == EvidenceLabel.SUPPORTED_NEG)
                            method_stats[method]["evidence_unknown"] += int(record.label == EvidenceLabel.UNKNOWN)
                        update = None
                        if controller is not None:
                            controller.observe(snapshot, evidence)
                            update = controller.maybe_adapt(snapshot, evidence)
                            method_stats[method]["optimizer_steps"] += int(update is not None and update.stepped)
                            method_stats[method]["supported_candidates"] += sum(record.label == EvidenceLabel.SUPPORTED_POS for record in evidence)
                        method_stats[method]["probe_calls"] = (
                            probe_calls_total if self._implementation(method) in PROBE_METHODS else 0
                        )
                        method_stats[method]["peak_memory_allocated_bytes"] = max(method_stats[method]["peak_memory_allocated_bytes"], int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0)
                        method_stats[method]["peak_memory_reserved_bytes"] = max(method_stats[method]["peak_memory_reserved_bytes"], int(torch.cuda.max_memory_reserved(self.device)) if self.device.type == "cuda" else 0)
                        method_stats[method]["latencies"].append(time.perf_counter() - frame_started)
                    trace.write(json.dumps({
                        "video_id": video_id, "expression_id": expression_id, "frame_index": frame_index,
                        "query_text": sentence, "query_spec": {"constraint_types": spec.constraint_types, "unsupported_reason": spec.unsupported_reason},
                        "probe_due": probe_due, "probe_calls_total": probe.calls,
                        "probe_history_frames": list(packet.results[0].history_frames) if packet and packet.results else [],
                        "probe_elapsed_sec": packet.elapsed_sec if packet else 0.0,
                        "ring_bytes": ring.bytes_stored, "snapshot_candidates": snapshot.count,
                        "valid_candidates": int(snapshot.valid_mask.sum()),
                        "methods": {method: {
                            "selected_rows": int(method_stats[method]["selected_rows"]),
                            "optimizer_steps": int(method_stats[method]["optimizer_steps"]),
                            "last_update": controllers[method].last_update.__dict__ if method in controllers else None,
                            "evidence": evidence_by_method.get(method, []),
                        } for method in self.methods},
                    }, ensure_ascii=False, default=str) + "\n")
                    trace.flush()
                episode_elapsed = time.perf_counter() - start_episode
                for method in self.methods:
                    method_stats[method]["elapsed_sec"] += episode_elapsed
                probe.reset()
                ring.clear()
        for method in self.methods:
            method_stats[method] = self._method_summary(method_stats[method], frame_count)
            method_stats[method].pop("latencies", None)
        payload = {
            "format": "locatemot-qcs-online-run-v1", "status": "PREDICTION_COMPLETE",
            "command": " ".join([sys.executable, *sys.argv]), "cwd": str(Path.cwd().resolve()),
            "config": str(self.config_path), "config_sha256": _sha256(self.config_path),
            "runtime_manifest": str(self.runtime_manifest), "runtime_manifest_sha256": _sha256(self.runtime_manifest),
            "source": self.wrapper.provenance if self.wrapper else {}, "gpu_uuid": _gpu_uuid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "methods": self.methods,
            "method_implementations": {method: self._implementation(method) for method in self.methods},
            "episodes": [f"{video}/{expression}" for video, expression in grouped],
            "video_expression_count": expression_count, "frame_query_count": frame_count,
            "wall_seconds": time.perf_counter() - started_all, "shared_source_forward": True,
            "gt_used_by_runner": False, "future_frames_used": False, "evaluation_manifest_opened": False,
            "method_stats": method_stats, "trace": str(trace_path.resolve()),
            "candidate_diagnostics": str(candidate_diagnostics_path.resolve()),
        }
        _json_dump(self.output_dir / "run_metadata.json", payload)
        return payload
