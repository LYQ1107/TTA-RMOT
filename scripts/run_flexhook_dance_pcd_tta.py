#!/usr/bin/env python3
"""Run parameter-efficient, source-free online TTA on Refer-Dance.

The released Refer-KITTI-V2 FlexHook checkpoint is the only model checkpoint
loaded here.  At prediction time the runner reads images, expression text,
and the fixed Dance DETR-NeuralSORT rows.  It never opens ``labels.json`` or
``gt_template``.  Each frame is predicted and sealed before one optimizer
step is made, so the adaptation state is causal.

This runner intentionally keeps the tracker immutable: it only changes the
second-stage score used to select an existing tracker row.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flex_tta.adaptation.pcd import (  # noqa: E402
    PCDLoRATTA,
    PCDResidualTTA,
    parameter_drift,
    trainable_parameters,
)
from flex_tta.dance_io import (  # noqa: E402
    dance_videos,
    load_expressions,
    load_tracker_lines,
    score_frames,
)
from flex_tta.evidence import TrackletLanguageEvidence  # noqa: E402
from flex_tta.runtime.dance_episode import DanceEpisode  # noqa: E402


METHODS = {
    "PCD_RESIDUAL": "residual",
    "PCD_LORA": "lora",
}


def _config_for_model(vendor: Path, checkpoint: Path, data_root: Path, tracker_root: Path,
                      output_root: Path, val_batch_size: int):
    """Construct the released model config without starting DDP."""

    os.chdir(vendor)
    # The released YACS config assumes the torch.distributed launcher always
    # exported LOCAL_RANK.  This runner is deliberately a single-process
    # episode loop, so provide the equivalent local rank explicitly.
    os.environ.setdefault("LOCAL_RANK", "0")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from config import get_config

    cli = Namespace(
        cfg=str(vendor / "configs/infer/kitti2.yaml"),
        opts=["data_root", str(data_root)],
        batch_size=8,
        val_batch_size=val_batch_size,
        visual="rope-swin-tiny",
        text="roberta",
        dataset="dance",
        freeze_text=False,
        freeze_visual=False,
        track_root=str(tracker_root),
        data_path=None,
        zip=False,
        cache_mode="part",
        pretrained=None,
        resume=str(checkpoint),
        accumulation_steps=None,
        use_checkpoint=False,
        disable_amp=False,
        amp_opt_level=None,
        output=str(output_root),
        tag=None,
        eval=True,
        throughput=False,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        lre=None,
    )
    config = get_config(cli)
    config.defrost()
    config.DATA.DATASET = "dance"
    config.DATA.VAL_BATCH_SIZE = int(val_batch_size)
    config.MODEL.PRETRAINED = str(checkpoint)
    config.MODEL.RESUME = str(checkpoint)
    config.OUTPUT = str(output_root)
    # A single expression per model call is equivalent to the official
    # expression-isolated path and avoids cross-expression adapter coupling.
    config.sample_expression_num = 1
    config.val_sample_expression_num = 1
    config.sample_frame_num = 4
    config.sample_frame_len = 4
    config.sample_frame_stride = 1
    config.val_sample_frame_stride = 1
    config.freeze()
    return config


def _load_model(vendor: Path, checkpoint: Path, data_root: Path, tracker_root: Path,
                output_root: Path, device: torch.device, val_batch_size: int):
    config = _config_for_model(vendor, checkpoint, data_root, tracker_root, output_root, val_batch_size)
    from logger import create_logger
    from models import build_model
    from utils import load_ckpt_my

    output_root.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(output_root), dist_rank=0, name="pcd_tta_model")
    model = build_model(config)
    model = model.to(device)
    load_ckpt_my(config, model, logger)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, model, logger


def _make_adapter(model: torch.nn.Module, method: str, device: torch.device, rank: int, bottleneck: int):
    if method == "PCD_RESIDUAL":
        adapter = PCDResidualTTA(model, layer_indices=(-2, -1), bottleneck=bottleneck)
    elif method == "PCD_LORA":
        adapter = PCDLoRATTA(model, rank=rank, layer_indices=(-2, -1))
    else:
        raise ValueError(f"unsupported PCD method: {method}")
    adapter.to(device)
    return adapter


def _token_batch(tokenizer, sentence: str, batch_size: int, device: torch.device):
    encoded = tokenizer(
        [sentence] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=25,
        return_tensors="pt",
    )
    return encoded["input_ids"].unsqueeze(1).to(device), encoded["attention_mask"].unsqueeze(1).to(device)


def _model_margin(model, episode: DanceEpisode, candidates, sentence: str,
                  tokenizer, device: torch.device):
    images, positions, boxes, track_ids = episode.batch_inputs(candidates, device)
    expid, expma = _token_batch(tokenizer, sentence, len(track_ids), device)
    output, _regular = model(images, positions, boxes, expid, expma)
    if output.ndim == 4:
        output = output[:, -1]
    if output.ndim != 3 or output.shape[1] != 1 or output.shape[-1] != 2:
        raise RuntimeError(f"unexpected FlexHook output shape: {tuple(output.shape)}")
    return output[:, 0, 1] - output[:, 0, 0], track_ids


def _entropy(margin: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(margin)
    return -(probability * torch.log(probability.clamp_min(1e-6))
             + (1.0 - probability) * torch.log((1.0 - probability).clamp_min(1e-6))).mean()


def _adaptation_loss(margin: torch.Tensor, source_margin: torch.Tensor,
                     track_ids: list[int], previous: dict[int, float],
                     entropy_weight: float, anchor_weight: float,
                     temporal_weight: float, pseudo_weight: float,
                     evidence_labels: list[int] | None = None,
                     evidence_weight: float = 0.0):
    terms: dict[str, torch.Tensor] = {
        "entropy": _entropy(margin),
        "anchor": F.smooth_l1_loss(margin, source_margin.detach(), beta=2.0),
    }
    shared = [index for index, track_id in enumerate(track_ids) if track_id in previous]
    if shared:
        previous_margin = torch.tensor(
            [previous[track_ids[index]] for index in shared],
            dtype=margin.dtype,
            device=margin.device,
        )
        current = margin[shared]
        terms["temporal"] = F.smooth_l1_loss(current, previous_margin, beta=1.0)
        # A label-free hard pseudo target is used only after a track has a
        # causal, high-confidence previous prediction.  It is never derived
        # from a target annotation.
        confident = previous_margin.abs() >= 1.0
        if confident.any():
                terms["pseudo"] = F.binary_cross_entropy_with_logits(
                    current[confident], (previous_margin[confident] > 0).float()
                )
    if evidence_labels is not None and evidence_weight > 0.0:
        evidence = torch.tensor(evidence_labels, dtype=torch.long, device=margin.device)
        reliable = evidence != 0
        if reliable.any():
            terms["independent_evidence"] = F.binary_cross_entropy_with_logits(
                margin[reliable], (evidence[reliable] > 0).float()
            )
    zero = margin.sum() * 0.0
    loss = (
        entropy_weight * terms["entropy"]
        + anchor_weight * terms["anchor"]
        + temporal_weight * terms.get("temporal", zero)
        + pseudo_weight * terms.get("pseudo", zero)
        + evidence_weight * terms.get("independent_evidence", zero)
    )
    return loss, {name: float(value.detach().item()) for name, value in terms.items()}


def _write_prediction(handle, rows: dict[int, str], track_ids: list[int], margin: torch.Tensor) -> int:
    values = margin.detach().cpu().tolist()
    selected = 0
    for track_id, value in zip(track_ids, values):
        if float(value) > 0.0:
            handle.write(rows[int(track_id)] + "\n")
            selected += 1
    return selected


def _run_episode(model, adapter, tokenizer, dataset_root: Path, tracker_root: Path,
                 source: dict[str, Any], video: str, expression_id: str,
                 sentence: str, method_root: Path, device: torch.device,
                 batch_size: int, lr: float, weight_decay: float,
                 entropy_weight: float, anchor_weight: float,
                 temporal_weight: float, pseudo_weight: float,
                 max_frames: int = 0, update_interval: int = 1,
                 evidence_engine: TrackletLanguageEvidence | None = None,
                 evidence_weight: float = 0.0) -> dict[str, Any]:
    episode = DanceEpisode(dataset_root, tracker_root, video)
    tracker_by_frame = load_tracker_lines(tracker_root, video)
    source_by_frame = score_frames(source, video, sentence)
    frames = sorted(set(source_by_frame) & set(tracker_by_frame))
    if max_frames > 0:
        frames = frames[:max_frames]

    adapter.reset_episode()
    parameters = trainable_parameters(adapter)
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    prediction = method_root / "results" / video / expression_id / "predict.txt"
    prediction.parent.mkdir(parents=True, exist_ok=True)

    previous: dict[int, float] = {}
    stats = {
        "video": video,
        "expression_id": expression_id,
        "frames": 0,
        "candidate_rows": 0,
        "selected_rows": 0,
        "optimizer_steps": 0,
        "pseudo_positive": 0,
        "pseudo_negative": 0,
        "temporal_rows": 0,
        "loss_sum": 0.0,
        "nonzero_gradient_steps": 0,
        "elapsed_seconds": 0.0,
        "parameter_drift": 0.0,
        "parameter_update_l2": 0.0,
        "parameter_update_max_abs": 0.0,
        "initial_model_source_mae": 0.0,
        "initial_model_source_max_abs": 0.0,
        "initial_model_source_sign_mismatch": 0,
        "initial_model_source_count": 0,
        "independent_evidence_positive": 0,
        "independent_evidence_negative": 0,
        "independent_evidence_unknown": 0,
    }
    loss_records = []
    started = time.time()

    def valid_candidates(frame_id: int):
        return [
            (track_id, history)
            for track_id, history in episode.current_candidates(frame_id)
            if track_id in source_by_frame[frame_id]
        ]

    def record_initial_alignment(margin: torch.Tensor, source_batch: torch.Tensor) -> None:
        if stats["frames"] != 0:
            return
        difference = (margin.detach() - source_batch.detach()).abs()
        stats["initial_model_source_mae"] += float(difference.sum().item())
        stats["initial_model_source_max_abs"] = max(
            stats["initial_model_source_max_abs"], float(difference.max().item())
        )
        stats["initial_model_source_sign_mismatch"] += int(
            (margin.detach() > 0).ne(source_batch.detach() > 0).sum().item()
        )
        stats["initial_model_source_count"] += int(margin.numel())

    with prediction.open("w", encoding="utf-8") as handle:
        frame_index = 0
        while frame_index < len(frames):
            frame_id = frames[frame_index]
            adapt_this_frame = (frame_index % update_interval) == 0
            if adapt_this_frame:
                valid = valid_candidates(frame_id)
                if not valid:
                    frame_index += 1
                    continue
                batches = [valid[start:start + batch_size] for start in range(0, len(valid), batch_size)]
                optimizer.zero_grad(set_to_none=True)
                frame_losses = []
                current_margins: dict[int, float] = {}
                selected = 0
                for batch in batches:
                    with torch.enable_grad():
                        margin, track_ids = _model_margin(model, episode, batch, sentence, tokenizer, device)
                        source_batch = torch.tensor(
                            [source_by_frame[frame_id][track_id] for track_id in track_ids],
                            dtype=margin.dtype,
                            device=device,
                        )
                        selected += _write_prediction(handle, tracker_by_frame[frame_id], track_ids, margin)
                        current_values = margin.detach().float().cpu().tolist()
                        current_margins.update({int(track_id): float(value) for track_id, value in zip(track_ids, current_values)})
                        record_initial_alignment(margin, source_batch)
                        shared = [track_id for track_id in track_ids if track_id in previous]
                        stats["temporal_rows"] += len(shared)
                        evidence_labels = None
                        if evidence_engine is not None:
                            batch_candidates = [(int(track_id), history) for track_id, history in batch]
                            _scores, _reliability, evidence_labels = evidence_engine.batch(
                                episode, frame_id, batch_candidates, sentence
                            )
                            stats["independent_evidence_positive"] += sum(label > 0 for label in evidence_labels)
                            stats["independent_evidence_negative"] += sum(label < 0 for label in evidence_labels)
                            stats["independent_evidence_unknown"] += sum(label == 0 for label in evidence_labels)
                        loss, terms = _adaptation_loss(
                            margin, source_batch, track_ids, previous,
                            entropy_weight, anchor_weight, temporal_weight, pseudo_weight,
                            evidence_labels=evidence_labels, evidence_weight=evidence_weight,
                        )
                        (loss / len(batches)).backward()
                        frame_losses.append({"loss": float(loss.detach().item()), **terms})
                        for track_id in shared:
                            if abs(previous[track_id]) >= 1.0:
                                if previous[track_id] > 0:
                                    stats["pseudo_positive"] += 1
                                else:
                                    stats["pseudo_negative"] += 1
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                has_gradient = any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                   and float(parameter.grad.detach().abs().sum()) > 0.0 for parameter in parameters)
                optimizer.step()
                stats["optimizer_steps"] += 1
                stats["nonzero_gradient_steps"] += int(has_gradient)
                stats["frames"] += 1
                stats["candidate_rows"] += len(valid)
                stats["selected_rows"] += selected
                stats["loss_sum"] += float(np.mean([record["loss"] for record in frame_losses]))
                if len(loss_records) < 100:
                    loss_records.append({"frame_id": frame_id, "terms": frame_losses})
                # The next frame sees the current frame's sealed predictions;
                # it cannot see the post-update logits from this frame.
                previous = current_margins
                frame_index += 1
                continue

            # Between two causal updates, the adapter is fixed.  Evaluate a
            # whole interval as one no-grad batch to reduce model-launch
            # overhead without changing the prediction/update order.
            end_index = frame_index
            while end_index < len(frames) and (end_index % update_interval) != 0:
                end_index += 1
            group_frames = frames[frame_index:end_index]
            frame_candidates = {group_frame: valid_candidates(group_frame) for group_frame in group_frames}
            flat = [(group_frame, track_id, history)
                    for group_frame in group_frames
                    for track_id, history in frame_candidates[group_frame]]
            frame_outputs: dict[int, dict[int, float]] = {group_frame: {} for group_frame in group_frames}
            group_batch_size = max(batch_size, batch_size * update_interval)
            for start in range(0, len(flat), group_batch_size):
                batch_meta = flat[start:start + group_batch_size]
                with torch.no_grad():
                    margin, _track_ids = _model_margin(
                        model, episode, [(track_id, history) for _frame, track_id, history in batch_meta],
                        sentence, tokenizer, device,
                    )
                for (_frame, track_id, _history), value in zip(batch_meta, margin.detach().float().cpu().tolist()):
                    frame_outputs[_frame][int(track_id)] = float(value)
            for group_frame in group_frames:
                current_margins = frame_outputs[group_frame]
                selected = 0
                for track_id, value in current_margins.items():
                    if value > 0.0:
                        handle.write(tracker_by_frame[group_frame][track_id] + "\n")
                        selected += 1
                shared = [track_id for track_id in current_margins if track_id in previous]
                stats["temporal_rows"] += len(shared)
                stats["frames"] += 1
                stats["candidate_rows"] += len(frame_candidates[group_frame])
                stats["selected_rows"] += selected
                previous = current_margins
            frame_index = end_index

    stats["elapsed_seconds"] = time.time() - started
    stats["parameter_drift"] = parameter_drift(adapter)
    update_sq = 0.0
    update_max = 0.0
    for parameter, initial in zip(parameters, initial_parameters):
        difference = (parameter.detach() - initial).float()
        update_sq += float(torch.sum(difference * difference).item())
        update_max = max(update_max, float(difference.abs().max().item()))
    stats["parameter_update_l2"] = update_sq ** 0.5
    stats["parameter_update_max_abs"] = update_max
    if stats["initial_model_source_count"]:
        stats["initial_model_source_mae"] /= stats["initial_model_source_count"]
    stats["mean_loss"] = stats["loss_sum"] / max(stats["frames"], 1)
    return {"stats": stats, "loss_records": loss_records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=ROOT / "vendor/FlexHook")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="PCD_RESIDUAL")
    parser.add_argument("--videos", nargs="*", default=[])
    parser.add_argument("--max-expressions", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--update-interval", type=int, default=1,
                        help="causal frame interval for adapter updates; skipped frames are forward-only")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--pseudo-weight", type=float, default=0.10)
    parser.add_argument("--evidence", choices=["NONE", "TRACKLET_LANGUAGE"], default="NONE")
    parser.add_argument("--evidence-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.update_interval < 1:
        raise ValueError("--update-interval must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PCD TTA requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    method_root = args.output_root / args.method
    if method_root.exists():
        raise FileExistsError(f"refusing to overwrite {method_root}")
    method_root.mkdir(parents=True)

    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    videos = args.videos or dance_videos(args.tracker_root)
    all_videos = set(dance_videos(args.tracker_root))
    unknown = sorted(set(videos) - all_videos)
    if unknown:
        raise ValueError(f"unknown Dance videos: {unknown}")

    config, model, logger = _load_model(
        args.vendor.resolve(), args.checkpoint.resolve(), args.dataset_root.resolve(),
        args.tracker_root.resolve(), method_root, device, args.val_batch_size,
    )
    from transformers import RobertaTokenizerFast
    tokenizer = RobertaTokenizerFast.from_pretrained(str(args.vendor.resolve() / "pretrained/roberta-base"))
    adapter = _make_adapter(model, args.method, device, args.rank, args.bottleneck)
    evidence_engine = TrackletLanguageEvidence() if args.evidence == "TRACKLET_LANGUAGE" else None
    logger.info("Loaded source checkpoint %s; method=%s", args.checkpoint, args.method)
    logger.info("trainable_adapter_parameters=%d", sum(parameter.numel() for parameter in trainable_parameters(adapter)))

    suite = []
    for video in videos:
        expressions = load_expressions(args.dataset_root, video)
        if args.max_expressions > 0:
            expressions = expressions[:args.max_expressions]
        for expression_id, sentence in expressions:
            record = _run_episode(
                model, adapter, tokenizer, args.dataset_root.resolve(), args.tracker_root.resolve(),
                source, video, expression_id, sentence, method_root, device,
                args.batch_size, args.lr, args.weight_decay, args.entropy_weight,
                args.anchor_weight, args.temporal_weight, args.pseudo_weight, args.max_frames,
                args.update_interval, evidence_engine, args.evidence_weight,
            )
            suite.append({
                "video": video,
                "expression_id": expression_id,
                "sentence": sentence,
                **record,
            })
            logger.info("completed %s/%s: %s", video, expression_id, record["stats"])

    metadata = {
        "method": args.method,
        "protocol": "source-free causal PCD TTA; prediction is sealed before the frame update",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_result": str(args.source_result.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "tracker_root": str(args.tracker_root.resolve()),
        "model_config": "vendor/FlexHook/configs/infer/kitti2.yaml with sample_expression_num=1",
        "source_free_prediction": True,
        "target_gt_opened": False,
        "target_labels_opened": False,
        "future_frames_used": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
        "videos": videos,
        "sequences": len(suite),
        "hyperparameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "entropy_weight": args.entropy_weight,
            "anchor_weight": args.anchor_weight,
            "temporal_weight": args.temporal_weight,
            "pseudo_weight": args.pseudo_weight,
            "rank": args.rank,
            "bottleneck": args.bottleneck,
            "update_interval": args.update_interval,
            "evidence": args.evidence,
            "evidence_weight": args.evidence_weight,
        },
        "episodes": suite,
    }
    (method_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(method_root.resolve()), "method": args.method, "sequences": len(suite)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
