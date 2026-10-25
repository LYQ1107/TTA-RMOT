#!/usr/bin/env python3
"""FlexHook V2 -> Dance, fixed-track positive-label-free experiments.

Default invocation is DRY RUN. `--scope smoke` is capped to <= 1 expression and
12 frames; full 25x17 requires BOTH `--scope full` and an explicit authorization
environment variable. The code NEVER runs source pre-training or uses target GT.

Methods:
  RELATIVE_ONLY       Frozen score replay with per-query causal relative tail
  ZERO_UPDATE         Reconstructed FlexHook PCD forward, optimizer_steps=0
  PF_RANK             LoRA update with stable past-only relative rank constraints
  PF_RANK_GUARD       PF_RANK plus transparent frozen-positive retention ablation
  PF_IDENTITY         Track-ID self-supervised feature InfoNCE, no referring pseudo positives
  PF_HYBRID           Conservative rank constraints + track-ID self-supervision

No method constructs binary pseudo-positive or pseudo-negative labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import time
import subprocess
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flex_tta.adaptation.identity_contrast import (
    CausalIDMemory, IdentityContrastConfig, PCDFeatureTap,
)
from flex_tta.adaptation.positive_free import (
    CausalRankMemory, CausalRelativeSelector, RankConfig, RelativeConfig,
    positive_free_rank_loss,
)
from flex_tta.runtime.pf_receipts import EpisodeStore, atomic_json, file_sha

METHODS = ("RELATIVE_ONLY", "ZERO_UPDATE", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID")
SOURCE_SHA256 = "71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613"
FROZEN_SHA256 = "fbf498a24b73d6b2db6d9c757d313f660c0d5b09a7dbfec5c2635b8ca517ee69"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("plan", "smoke", "full"), default="plan")
    p.add_argument("--method", choices=METHODS, default="RELATIVE_ONLY")
    p.add_argument("--vendor", type=Path, default=Path("/data2/user/FlexHook-TTA/vendor/FlexHook"))
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--checkpoint", type=Path, default=Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth"))
    p.add_argument("--source-result", type=Path, default=Path("/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json"))
    p.add_argument("--source-supplement", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=[])
    p.add_argument("--max-expressions", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--update-interval", type=int, default=4)
    p.add_argument("--min-history", type=int, default=3)
    p.add_argument("--nce-weight", type=float, default=0.15)
    p.add_argument("--min-separation", type=float, default=0.6)
    p.add_argument("--max-adapter-delta-norm", type=float, default=2.0)
    p.add_argument("--strict-initial-sign", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frozen-cache", action="store_true", help="reuse equivalent frozen encoders, FP32 only")
    p.add_argument("--feature-cache-mib", type=int, default=16384)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.scope == "full":
        if os.environ.get("FLEXHOOK_FULL_RUN_AUTHORIZED") != "YES":
            p.error("Full inference/adaptation is LOCKED. Require explicit user approval and FLEXHOOK_FULL_RUN_AUTHORIZED=YES")
        if args.max_expressions != 0 or args.max_frames != 0:
            p.error("full runs require --max-expressions 0 --max-frames 0")
    if args.scope == "smoke" and (args.max_expressions != 1 or not 1 <= args.max_frames <= 12):
        p.error("smoke is limited to exactly one expression and at most 12 frames")
    if args.batch_size < 1 or args.update_interval < 1 or args.rank < 1 or args.lr <= 0 or args.nce_weight < 0:
        p.error("invalid optimizer/batching arguments")
    if args.checkpoint.name != "refer-kitti-v2-best.pth":
        p.error("only the pinned V2 source checkpoint is permitted")
    return args


def plan_receipt(args) -> dict[str, Any]:
    return {
        "scope": args.scope,
        "method": args.method,
        "full_run_authorized": args.scope == "full",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_expected_sha256": SOURCE_SHA256,
        "dataset_root": str(args.dataset_root),
        "tracker_root": str(args.tracker_root),
        "source_result": str(args.source_result),
        "source_supplement": str(args.source_supplement) if args.source_supplement else None,
        "output_root": str(args.output_root),
        "videos": args.videos or "all_25_if_full_else_first",
        "max_expressions": args.max_expressions,
        "max_frames": args.max_frames,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "lr": args.lr,
        "update_interval": args.update_interval,
        "seed": args.seed,
        "frozen_cache": args.frozen_cache,
        "feature_cache_mib": args.feature_cache_mib,
        "experiment_commit": getattr(args, "experiment_commit", None),
        "gpu_uuid": os.environ.get("PF_GPU_UUID"),
        "physical_gpu": os.environ.get("PF_PHYSICAL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES")),
        "gpu_memory_budget_mib": os.environ.get("PF_GPU_MEMORY_BUDGET_MIB"),
        "nce_weight": args.nce_weight,
        "no_target_gt": True,
        "source_free_prediction": True,
        "target_gt_opened": False,
        "target_labels_opened": False,
        "future_frames_used": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
        "no_external_model": True,
        "no_hard_class_pseudolabels": True,
        "fixed_mot": "DETR-NeuralSORT-Dance",
    }


def _assets(args) -> None:
    for path in [args.dataset_root, args.tracker_root, args.source_result]:
        if not path.exists():
            raise FileNotFoundError(f"missing required inference asset: {path}")
    if args.method != "RELATIVE_ONLY":
        if not args.vendor.is_dir() or not args.checkpoint.is_file():
            raise FileNotFoundError("missing FlexHook vendor or pinned source checkpoint")
        import hashlib
        h = hashlib.sha256()
        with args.checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != SOURCE_SHA256:
            raise RuntimeError("source checkpoint SHA256 mismatch")
    if args.scope == "full" and file_sha(args.source_result) != FROZEN_SHA256:
        raise RuntimeError("frozen score SHA256 mismatch")
    if args.source_supplement is not None:
        recorded = json.loads((args.source_supplement.parent / "completion_receipt.json").read_text())
        if (recorded["original_sha256"] != FROZEN_SHA256 or recorded["checkpoint_sha256"] != SOURCE_SHA256 or
                recorded["supplement_sha256"] != file_sha(args.source_supplement)):
            raise ValueError("Frozen supplement receipt does not match pinned source assets")


def episode_config(args):
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
            if key not in {"output_root", "videos", "resume", "strict_initial_sign", "experiment_commit"}}


def validate_frames(args, tracks, scores):
    if args.scope == "full" and (set(tracks) != set(scores) or
            any(set(tracks[t]) != set(scores[t]) for t in tracks)):
        raise ValueError("Frozen score stream does not cover every fixed MOT candidate/frame")
    frames = sorted(set(tracks) & set(scores))
    return frames[:args.max_frames] if args.scope == "smoke" else frames


def restore_counters(stats, record):
    for key, value in record.get("counter_delta", {}).items():
        if key.startswith("max_") or key == "source_max_abs_error":
            stats[key] = max(stats.get(key, 0), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            prior = stats.get(key)
            stats[key] = (0 if prior is None else prior) + value
    stats["episodes"] += 1
    stats["episode_records"].append(record)


def record_delta(stats, before):
    return {key: (value if key.startswith("max_") or key == "source_max_abs_error" else value-before.get(key, 0))
            for key, value in stats.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
            and key != "episodes"}


def runtime_metadata(out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("completed_episodes.jsonl", "failed_episodes.jsonl"):
        (out / name).touch(exist_ok=True)


def deny_target_gt(args):
    """Enforce the separation, including any accidental vendor opens."""
    root = args.dataset_root.resolve()
    def audit(event, values):
        if event != "open" or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(root) and ("gt_template" in path.parts or path.name in {"labels.json", "gt.txt"}):
            raise PermissionError(f"target GT is not an adaptation input: {path}")
    sys.addaudithook(audit)


def _episodic_replay(args, source: dict, videos: list[str]):
    from flex_tta.dance_io import load_expressions, load_tracker_lines, score_frames

    out = args.output_root / args.method
    root = out / "results"
    if out.exists() and not args.resume:
        raise FileExistsError(f"run root must be new: {out}")
    root.mkdir(parents=True, exist_ok=args.resume)
    runtime_metadata(out)
    selector = CausalRelativeSelector(RelativeConfig())
    stats = {"episodes": 0, "frames": 0, "candidates": 0, "selected": 0, "optimizer_steps": 0}
    stats["episode_records"] = []
    for video in videos:
        tracker = load_tracker_lines(args.tracker_root, video)
        exprs = load_expressions(args.dataset_root, video)
        for expr_id, sentence in (exprs if args.scope == "full" else exprs[:1]):
            store = EpisodeStore(out, video, expr_id, episode_config(args), resume=args.resume)
            sealed = store.completed()
            if sealed is not None:
                restore_counters(stats, sealed)
                continue
            started = time.perf_counter()
            counters_before = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
            before = {key: stats[key] for key in ("frames", "candidates", "selected")}
            selector.reset()
            scores = score_frames(source, video, sentence)
            frames = validate_frames(args, tracker, scores)
            with store.attempt() as (handle, margin_handle):
                for t in frames:
                    rows = tracker.get(t, {})
                    margins = {i: value for i, value in scores[t].items() if i in rows}
                    selected = selector.step(t, margins)
                    margin_handle.write(json.dumps(dict(frame=t, ids=list(margins),
                        adapted=list(margins.values()), frozen=list(margins.values())))+"\n")
                    for i in sorted(selected):
                        handle.write(rows[i] + "\n")
                    stats["frames"] += 1
                    stats["candidates"] += len(margins)
                    stats["selected"] += len(selected)
            stats["episodes"] += 1
            record = {"video": video, "expression_id": expr_id, "sentence": sentence,
                "wall_seconds": time.perf_counter()-started, "full_frame_coverage": len(frames) == len(tracker),
                "counter_delta": record_delta(stats, counters_before),
                "stats": {"frames": stats["frames"] - before["frames"],
                          "candidate_rows": stats["candidates"] - before["candidates"],
                          "selected_rows": stats["selected"] - before["selected"], "optimizer_steps": 0}}
            stats["episode_records"].append(store.finish(record))
            print(json.dumps(dict(event="episode_complete", method=args.method, video=video,
                                  expression=expr_id, completed=stats["episodes"], seconds=record["wall_seconds"])), flush=True)
    return stats


def _model_run(args, source: dict, videos: list[str]):
    if not torch.cuda.is_available():
        raise RuntimeError("real FlexHook PCD inference requires a CUDA GPU; use plan/RELATIVE_ONLY on CPU")
    # Compatibility hook: construct RoBERTa from released vocab/merges before
    # any tokenizer or FlexHook model is initialized.
    import compat.sitecustomize  # noqa: F401
    from scripts.run_flexhook_dance_pcd_tta import (
        _load_model, _make_adapter, _model_margin,
    )
    from flex_tta.runtime.dance_episode import DanceEpisode
    from flex_tta.dance_io import load_expressions, load_tracker_lines, score_frames
    from flex_tta.adaptation.pcd import trainable_parameters
    from transformers import RobertaTokenizerFast

    device = torch.device("cuda:0")
    if os.environ.get("PF_GPU_MEMORY_BUDGET_MIB"):
        budget_bytes = int(os.environ["PF_GPU_MEMORY_BUDGET_MIB"]) << 20
        torch.cuda.set_per_process_memory_fraction(
            budget_bytes / torch.cuda.get_device_properties(device).total_memory, device)
    out = args.output_root / args.method
    if out.exists() and not args.resume:
        raise FileExistsError(f"run root must be new: {out}")
    out.mkdir(parents=True, exist_ok=args.resume)
    runtime_metadata(out)
    _cfg, model, _logger = _load_model(args.vendor.resolve(), args.checkpoint.resolve(),
                                      args.dataset_root.resolve(), args.tracker_root.resolve(),
                                      out, device, args.batch_size, strict_checkpoint=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(args.vendor.resolve() / "pretrained/roberta-base"))
    if len(tokenizer.encode("person", add_special_tokens=False)) > 2:
        raise RuntimeError("RoBERTa tokenizer regression: 'person' encoded incorrectly")
    adapter = None if args.method == "ZERO_UPDATE" else _make_adapter(model, "PCD_LORA", device, args.rank, 64)
    rank_cfg = RankConfig(min_history=args.min_history, min_separation=args.min_separation)
    identity_cfg = IdentityContrastConfig(weight=args.nce_weight)
    uses_rank = args.method in ("PF_RANK", "PF_RANK_GUARD", "PF_HYBRID")
    uses_identity = args.method in ("PF_IDENTITY", "PF_HYBRID")
    tap = PCDFeatureTap(model) if uses_identity else None
    identity_memory = CausalIDMemory(identity_cfg) if uses_identity else None
    memory = CausalRankMemory(rank_cfg)
    stats = {"episodes": 0, "frames": 0, "candidates": 0, "selected": 0,
             "source_selected": 0, "optimizer_steps": 0, "skipped_no_pairs": 0,
             "rank_pair_observations": 0, "identity_pairs": 0, "sign_mismatches_before_updates": 0,
             "source_comparison_rows": 0, "max_adapter_update_l2": 0.0, "rollback_updates": 0,
             "gradient_attempts": 0, "finite_nonzero_gradient_steps": 0,
             "nonfinite_gradient_steps": 0, "zero_gradient_steps": 0,
             "source_abs_error_sum": 0.0, "source_max_abs_error": 0.0,
             "rollback_restored": True, "prediction_source_selection_mismatches": 0,
             "checkpoint_strict_load": True,
             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
             "pcd_channels": int(model.text_dim), "feature_channels": None,
             "smoke_frames": [], "episode_resets": [], "episode_records": []}
    stats["lora_projections"] = ([{"in_features": p.base.in_features,
                                    "out_features": p.base.out_features,
                                    "rank": p.lora_a.out_features}
                                   for p in adapter.projections] if adapter is not None else [])
    for video in videos:
        tracks = load_tracker_lines(args.tracker_root, video)
        episode = DanceEpisode(args.dataset_root, args.tracker_root, video)
        from flex_tta.runtime.frozen_features import FrozenFeatureCache
        cache = (FrozenFeatureCache(model, episode, tokenizer, device,
                                    max_bytes=args.feature_cache_mib << 20) if args.frozen_cache else None)
        if cache is not None:
            stats.setdefault("frozen_cache", []).append({"video": video, "stats": cache.stats})
        exprs = load_expressions(args.dataset_root, video)
        for expr_id, sentence in (exprs if args.scope == "full" else exprs[:1]):
            store = EpisodeStore(out, video, expr_id, episode_config(args), resume=args.resume)
            sealed = store.completed()
            if sealed is not None:
                restore_counters(stats, sealed)
                continue
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            counters_before = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
            before = {key: stats[key] for key in ("frames", "candidates", "selected", "optimizer_steps", "rollback_updates")}
            memory.reset()
            episode_optimizer_steps = 0
            if identity_memory is not None:
                identity_memory.reset()
            if adapter is not None:
                # Same registered seed at every independent episode makes
                # resume/shard assignment unable to change LoRA initialization.
                torch.manual_seed(args.seed)
                adapter.reset_episode()
                trainables = trainable_parameters(adapter)
                originals = [p.detach().clone() for p in trainables]
                optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.0)
            else:
                trainables, originals, optimizer = [], [], None
            reset_ok = (not memory.frames and memory.latest_frame == -1 and
                        (identity_memory is None or not identity_memory.rows) and
                        (adapter is None or all(torch.count_nonzero(p.lora_b.weight).item() == 0
                                                for p in adapter.projections)))
            if not reset_ok:
                raise RuntimeError("episode memory/adapter reset failed")
            stats["episode_resets"].append({"video": video, "expression_id": expr_id, "passed": reset_ok})
            source_frames = score_frames(source, video, sentence)
            frames = validate_frames(args, tracks, source_frames)
            with store.attempt() as (handle, margin_handle):
                for frame_n, t in enumerate(frames):
                    current = [(i, h) for i, h in episode.current_candidates(t) if i in source_frames[t]]
                    if not current:
                        memory.observe(t, {})
                        stats["frames"] += 1
                        continue
                    source_scores = {i: float(source_frames[t][i]) for i, _ in current}
                    max_history_frame = max(h.frame_id for _, history in current for h in history)
                    if max_history_frame > t:
                        raise RuntimeError("future image entered the causal episode")
                    evidence_pairs = memory.stable_pairs(t, sorted(source_scores)) if uses_rank else []
                    candidate_ids = [i for i, _ in current]
                    has_identity = identity_memory.eligible(t, candidate_ids) if uses_identity else False
                    if args.method != "ZERO_UPDATE":
                        opportunity = (frame_n+1) % args.update_interval == 0
                        key = "update_opportunities" if opportunity else "skipped_interval_frames"
                        stats[key] = stats.get(key, 0)+1
                        if opportunity and not evidence_pairs and not has_identity:
                            stats["skipped_no_evidence"] = stats.get("skipped_no_evidence", 0)+1
                    will_update = (
                        args.method != "ZERO_UPDATE" and
                        (frame_n + 1) % args.update_interval == 0 and
                        ((uses_rank and bool(evidence_pairs)) or has_identity)
                    )
                    active_batches = [current[k:k + args.batch_size] for k in range(0, len(current), args.batch_size)]
                    # For a differentiable candidate SET both losses need the
                    # complete set; long frames are evaluated but not updated.
                    if will_update and len(active_batches) != 1:
                        stats["skipped_multi_batch"] = stats.get("skipped_multi_batch", 0)+1
                        will_update = False
                    if will_update:
                        optimizer.zero_grad(set_to_none=True)
                    prediction_margins = {}
                    graph_logits, graph_source, feature_chunks, graph_ids = [], [], [], []
                    for batch in active_batches:
                        with torch.set_grad_enabled(will_update):
                            if tap is not None:
                                tap.clear()
                            output, ids = (cache.margin(batch, sentence, t) if cache is not None else
                                           _model_margin(model, episode, batch, sentence, tokenizer, device))
                            frozen = torch.tensor([source_scores[i] for i in ids], device=device, dtype=output.dtype)
                            if tap is not None:
                                features = tap.take()
                                if features.shape != (len(ids), int(model.text_dim)):
                                    raise RuntimeError(f"PCD feature/track dimensions disagree: {features.shape}")
                                stats["feature_channels"] = int(features.shape[1])
                                feature_chunks.append(features if will_update else features.detach())
                            graph_ids.extend([int(i) for i in ids])
                            if will_update:
                                graph_logits.append(output)
                                graph_source.append(frozen)
                            for i, val, src in zip(ids, output.detach().tolist(), frozen.detach().tolist()):
                                prediction_margins[int(i)] = float(val)
                                if args.method == "ZERO_UPDATE" or episode_optimizer_steps == 0:
                                    stats["sign_mismatches_before_updates"] += int((val > 0) != (src > 0))
                                    stats["source_comparison_rows"] += 1
                                    stats["source_abs_error_sum"] += abs(val - src)
                                    stats["source_max_abs_error"] = max(stats["source_max_abs_error"], abs(val - src))
                    allowed_update = False
                    support = 0
                    loss_value = None
                    gradient_norm = None
                    update_status = "SKIP"
                    if will_update:
                        terms = []
                        if uses_rank and evidence_pairs:
                            rank_loss, diag = positive_free_rank_loss(
                                torch.cat(graph_logits), torch.cat(graph_source), graph_ids,
                                evidence_pairs, rank_cfg,
                            )
                            if diag["stable_pairs"] > 0:
                                terms.append(rank_loss)
                        if uses_identity and has_identity:
                            nce_loss, support = identity_memory.loss(
                                t, torch.cat(feature_chunks), graph_ids,
                            )
                            stats["identity_pairs"] += support
                            if support >= identity_cfg.min_reliable_tracks:
                                terms.append(identity_cfg.weight * nce_loss)
                        if terms:
                            loss = sum(terms)
                            loss_value = float(loss.detach())
                            loss.backward()
                            allowed_update = True
                    selected = {
                        i for i, val in prediction_margins.items()
                        if val > 0 or (args.method == "PF_RANK_GUARD" and source_scores[i] > 0)
                    }
                    if args.strict_initial_sign and stats["optimizer_steps"] == 0 and stats["sign_mismatches_before_updates"]:
                        raise RuntimeError("zero-update source sign mismatch detected; stop before adapting")
                    for i in sorted(selected):
                        handle.write(tracks[t][i] + "\n")
                    handle.flush()
                    margin_handle.write(json.dumps(dict(frame=t, ids=list(prediction_margins),
                        adapted=list(prediction_margins.values()),
                        frozen=[source_scores[i] for i in prediction_margins]))+"\n")
                    stats["margin_abs_change_sum"] = stats.get("margin_abs_change_sum", 0.) + sum(
                        abs(v-source_scores[i]) for i, v in prediction_margins.items())
                    stats["new_selected_from_negative"] = stats.get("new_selected_from_negative", 0) + sum(source_scores[i] <= 0 for i in selected)
                    stats["lost_frozen_positive"] = stats.get("lost_frozen_positive", 0) + sum(v > 0 and i not in selected for i, v in source_scores.items())
                    if args.method == "ZERO_UPDATE":
                        stats["prediction_source_selection_mismatches"] += len(
                            selected ^ {i for i, val in source_scores.items() if val > 0})
                    stats["selected"] += len(selected)
                    stats["source_selected"] += sum(val > 0 for val in source_scores.values())
                    stats["frames"] += 1
                    stats["candidates"] += len(current)
                    stats["rank_pair_observations"] += len(evidence_pairs)
                    if allowed_update:
                        stats["gradient_attempts"] += 1
                        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainables, 1.0))
                        has_grad = any(p.grad is not None and float(p.grad.detach().abs().sum()) > 0 for p in trainables)
                        all_finite = all(torch.isfinite(p.grad).all() for p in trainables if p.grad is not None)
                        stats["nonfinite_gradient_steps"] += int(not all_finite)
                        stats["zero_gradient_steps"] += int(not has_grad)
                        if has_grad and all_finite:
                            stats["finite_nonzero_gradient_steps"] += 1
                            backup = [p.detach().clone() for p in trainables]
                            optimizer.step()
                            change = sum(float((p.detach() - start).pow(2).sum()) for p, start in zip(trainables, originals)) ** 0.5
                            if change > args.max_adapter_delta_norm:
                                with torch.no_grad():
                                    for p, prior in zip(trainables, backup):
                                        p.copy_(prior)
                                # Also discard the rejected Adam state; otherwise
                                # its momentum can silently poison later updates.
                                optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.0)
                                stats["rollback_updates"] += 1
                                stats["rollback_restored"] &= all(torch.equal(p, prior) for p, prior in zip(trainables, backup))
                                update_status = "ROLLBACK"
                            else:
                                stats["optimizer_steps"] += 1
                                episode_optimizer_steps += 1
                                stats["max_adapter_update_l2"] = max(stats["max_adapter_update_l2"], change)
                                update_status = "ACCEPT"
                    elif args.method != "ZERO_UPDATE":
                        stats["skipped_no_pairs"] += 1
                    if args.scope == "smoke":
                        stats["smoke_frames"].append({
                            "frame_id": t, "max_image_history_frame": max_history_frame,
                            "rank_history_last_frame": memory.latest_frame,
                            "identity_history_last_frame": identity_memory.last_frame if identity_memory else None,
                            "rank_pairs": len(evidence_pairs), "identity_support": support,
                            "selected": len(selected), "loss": loss_value,
                            "gradient_norm": gradient_norm, "update_status": update_status,
                        })
                    if identity_memory is not None and feature_chunks:
                        identity_memory.observe(
                            t, torch.cat([f.detach() for f in feature_chunks], dim=0), graph_ids,
                        )
                    # Crucial: only frozen source scores are retained in the
                    # evidence bank. No student prediction self-reinforcement.
                    memory.observe(t, source_scores)
            stats["episodes"] += 1
            torch.cuda.synchronize(device)
            record = {"video": video, "expression_id": expr_id, "sentence": sentence,
                "wall_seconds": time.perf_counter()-started, "full_frame_coverage": len(frames) == len(tracks),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "experiment_commit": args.experiment_commit,
                "gpu_uuid": os.environ.get("PF_GPU_UUID"),
                "counter_delta": record_delta(stats, counters_before),
                "stats": {"frames": stats["frames"] - before["frames"],
                          "candidate_rows": stats["candidates"] - before["candidates"],
                          "selected_rows": stats["selected"] - before["selected"],
                          "optimizer_steps": stats["optimizer_steps"] - before["optimizer_steps"],
                          "rollback_updates": stats["rollback_updates"] - before["rollback_updates"]}}
            stats["episode_records"].append(store.finish(record))
            print(json.dumps(dict(event="episode_complete", method=args.method, video=video,
                                  expression=expr_id, completed=stats["episodes"], seconds=record["wall_seconds"])), flush=True)
    stats["source_mae_before_updates"] = stats["source_abs_error_sum"] / max(stats["source_comparison_rows"], 1)
    stats["frozen_parameters_have_grad"] = any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    stats["gpu_peak_allocated_bytes"] = max(torch.cuda.max_memory_allocated(device),
        max((e.get("gpu_peak_allocated_bytes", 0) for e in stats["episode_records"]), default=0))
    if args.scope == "smoke":
        memory.reset()
        if identity_memory is not None:
            identity_memory.reset()
        if adapter is not None:
            adapter.reset_episode()
        stats["post_episode_reset_pass"] = (not memory.frames and
            (identity_memory is None or not identity_memory.rows) and
            (adapter is None or all(torch.count_nonzero(p.lora_b.weight).item() == 0
                                    for p in adapter.projections)))
    if tap is not None:
        tap.close()
    return stats


def run(args):
    # FlexHook's config builder changes cwd to the vendor repository.
    # Resolve all caller paths before that compatibility boundary.
    for name in ("vendor", "dataset_root", "tracker_root", "checkpoint", "source_result", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.source_supplement is not None:
        args.source_supplement = args.source_supplement.resolve()
    args.experiment_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = plan_receipt(args)
    if args.scope == "plan":
        return receipt
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _assets(args)
    deny_target_gt(args)
    from flex_tta.dance_io import dance_videos
    videos = args.videos or dance_videos(args.tracker_root)
    if args.scope == "smoke":
        videos = videos[:1]
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    if args.source_supplement is not None:
        from flex_tta.dance_io import merge_source_supplement
        merge_source_supplement(source, json.loads(args.source_supplement.read_text()))
        args.source_supplement_sha256 = file_sha(args.source_supplement)
        receipt["source_supplement_sha256"] = args.source_supplement_sha256
    previous_metadata = args.output_root / args.method / "run_metadata.json"
    if args.resume and previous_metadata.is_file():
        previous = json.loads(previous_metadata.read_text())
        from flex_tta.dance_io import load_expressions
        expected = {(v, e) for v in videos for e, _ in
                    (load_expressions(args.dataset_root, v) if args.scope == "full" else load_expressions(args.dataset_root, v)[:1])}
        finished = {(e["video"], e["expression_id"]) for e in previous.get("episodes", [])}
        if previous.get("status") == "complete" and finished == expected:
            for video, expression in sorted(expected):
                if EpisodeStore(args.output_root / args.method, video, expression,
                                episode_config(args), resume=True).completed() is None:
                    raise ValueError("complete metadata lacks a valid sealed episode")
            print(json.dumps(dict(event="reuse_complete_shard", method=args.method, episodes=len(expected))), flush=True)
            return previous
    stats = (_episodic_replay(args, source, videos) if args.method == "RELATIVE_ONLY"
             else _model_run(args, source, videos))
    receipt["episodes"] = stats.pop("episode_records")
    receipt["sequences"] = len(receipt["episodes"])
    receipt["videos"] = videos
    receipt["hyperparameters"] = {name: getattr(args, name) for name in (
        "batch_size", "rank", "lr", "update_interval", "min_history", "nce_weight",
        "min_separation", "max_adapter_delta_norm", "seed")}
    receipt["stats"] = stats
    receipt["status"] = "complete"
    out = args.output_root / args.method
    atomic_json(out / "run_metadata.json", receipt)
    return receipt


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
