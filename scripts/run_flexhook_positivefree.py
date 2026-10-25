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

METHODS = ("RELATIVE_ONLY", "ZERO_UPDATE", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID")
SOURCE_SHA256 = "71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=("plan", "smoke", "full"), default="plan")
    p.add_argument("--method", choices=METHODS, default="RELATIVE_ONLY")
    p.add_argument("--vendor", type=Path, default=Path("/data2/user/FlexHook-TTA/vendor/FlexHook"))
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--checkpoint", type=Path, default=Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth"))
    p.add_argument("--source-result", type=Path, default=Path("/data2/user/FlexHook-TTA/results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json"))
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
        "output_root": str(args.output_root),
        "videos": args.videos or "all_25_if_full_else_first",
        "max_expressions": args.max_expressions,
        "max_frames": args.max_frames,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "lr": args.lr,
        "update_interval": args.update_interval,
        "nce_weight": args.nce_weight,
        "no_target_gt": True,
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


def _episodic_replay(args, source: dict, videos: list[str]):
    from flex_tta.dance_io import load_expressions, load_tracker_lines, score_frames

    out = args.output_root / args.method
    root = out / "results"
    if out.exists():
        raise FileExistsError(f"run root must be new: {out}")
    root.mkdir(parents=True)
    selector = CausalRelativeSelector(RelativeConfig())
    stats = {"episodes": 0, "frames": 0, "candidates": 0, "selected": 0, "optimizer_steps": 0}
    for video in videos:
        tracker = load_tracker_lines(args.tracker_root, video)
        exprs = load_expressions(args.dataset_root, video)
        for expr_id, sentence in (exprs if args.scope == "full" else exprs[:1]):
            selector.reset()
            scores = score_frames(source, video, sentence)
            output = root / video / expr_id / "predict.txt"
            output.parent.mkdir(parents=True, exist_ok=True)
            frames = sorted(scores) if args.scope == "full" else sorted(scores)[: args.max_frames]
            with output.open("w", encoding="utf-8") as handle:
                for t in frames:
                    rows = tracker.get(t, {})
                    margins = {i: value for i, value in scores[t].items() if i in rows}
                    selected = selector.step(t, margins)
                    for i in sorted(selected):
                        handle.write(rows[i] + "\n")
                    stats["frames"] += 1
                    stats["candidates"] += len(margins)
                    stats["selected"] += len(selected)
            stats["episodes"] += 1
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
    out = args.output_root / args.method
    if out.exists():
        raise FileExistsError(f"run root must be new: {out}")
    out.mkdir(parents=True)
    _cfg, model, _logger = _load_model(args.vendor.resolve(), args.checkpoint.resolve(),
                                      args.dataset_root.resolve(), args.tracker_root.resolve(),
                                      out, device, args.batch_size)
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
             "source_comparison_rows": 0, "max_adapter_update_l2": 0.0, "rollback_updates": 0}
    for video in videos:
        tracks = load_tracker_lines(args.tracker_root, video)
        episode = DanceEpisode(args.dataset_root, args.tracker_root, video)
        exprs = load_expressions(args.dataset_root, video)
        for expr_id, sentence in (exprs if args.scope == "full" else exprs[:1]):
            memory.reset()
            episode_optimizer_steps = 0
            if identity_memory is not None:
                identity_memory.reset()
            if adapter is not None:
                adapter.reset_episode()
                trainables = trainable_parameters(adapter)
                originals = [p.detach().clone() for p in trainables]
                optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.0)
            else:
                trainables, originals, optimizer = [], [], None
            source_frames = score_frames(source, video, sentence)
            frames = sorted(set(source_frames) & set(tracks))
            if args.scope == "smoke":
                frames = frames[:args.max_frames]
            prediction = out / "results" / video / expr_id / "predict.txt"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            with prediction.open("w", encoding="utf-8") as handle:
                for frame_n, t in enumerate(frames):
                    current = [(i, h) for i, h in episode.current_candidates(t) if i in source_frames[t]]
                    if not current:
                        memory.observe(t, {})
                        stats["frames"] += 1
                        continue
                    source_scores = {i: float(source_frames[t][i]) for i, _ in current}
                    evidence_pairs = memory.stable_pairs(t, sorted(source_scores)) if uses_rank else []
                    candidate_ids = [i for i, _ in current]
                    has_identity = identity_memory.eligible(t, candidate_ids) if uses_identity else False
                    will_update = (
                        args.method != "ZERO_UPDATE" and
                        (frame_n + 1) % args.update_interval == 0 and
                        ((uses_rank and bool(evidence_pairs)) or has_identity)
                    )
                    active_batches = [current[k:k + args.batch_size] for k in range(0, len(current), args.batch_size)]
                    # For a differentiable candidate SET both losses need the
                    # complete set; long frames are evaluated but not updated.
                    if will_update and len(active_batches) != 1:
                        will_update = False
                    if will_update:
                        optimizer.zero_grad(set_to_none=True)
                    prediction_margins = {}
                    graph_logits, graph_source, feature_chunks, graph_ids = [], [], [], []
                    for batch in active_batches:
                        with torch.set_grad_enabled(will_update):
                            if tap is not None:
                                tap.clear()
                            output, ids = _model_margin(model, episode, batch, sentence, tokenizer, device)
                            frozen = torch.tensor([source_scores[i] for i in ids], device=device, dtype=output.dtype)
                            if tap is not None:
                                feature_chunks.append(tap.take() if will_update else tap.take().detach())
                            graph_ids.extend([int(i) for i in ids])
                            if will_update:
                                graph_logits.append(output)
                                graph_source.append(frozen)
                            for i, val, src in zip(ids, output.detach().tolist(), frozen.detach().tolist()):
                                prediction_margins[int(i)] = float(val)
                                if args.method == "ZERO_UPDATE" or episode_optimizer_steps == 0:
                                    stats["sign_mismatches_before_updates"] += int((val > 0) != (src > 0))
                                    stats["source_comparison_rows"] += 1
                    allowed_update = False
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
                            sum(terms).backward()
                            allowed_update = True
                    selected = {
                        i for i, val in prediction_margins.items()
                        if val > 0 or (args.method == "PF_RANK_GUARD" and source_scores[i] > 0)
                    }
                    if args.strict_initial_sign and stats["optimizer_steps"] == 0 and stats["sign_mismatches_before_updates"]:
                        raise RuntimeError("zero-update source sign mismatch detected; stop before adapting")
                    for i in sorted(selected):
                        handle.write(tracks[t][i] + "\n")
                    stats["selected"] += len(selected)
                    stats["source_selected"] += sum(val > 0 for val in source_scores.values())
                    stats["frames"] += 1
                    stats["candidates"] += len(current)
                    stats["rank_pair_observations"] += len(evidence_pairs)
                    if allowed_update:
                        torch.nn.utils.clip_grad_norm_(trainables, 1.0)
                        has_grad = any(p.grad is not None and float(p.grad.detach().abs().sum()) > 0 for p in trainables)
                        all_finite = all(torch.isfinite(p.grad).all() for p in trainables if p.grad is not None)
                        if has_grad and all_finite:
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
                            else:
                                stats["optimizer_steps"] += 1
                                episode_optimizer_steps += 1
                                stats["max_adapter_update_l2"] = max(stats["max_adapter_update_l2"], change)
                    elif args.method != "ZERO_UPDATE":
                        stats["skipped_no_pairs"] += 1
                    if identity_memory is not None and feature_chunks:
                        identity_memory.observe(
                            t, torch.cat([f.detach() for f in feature_chunks], dim=0), graph_ids,
                        )
                    # Crucial: only frozen source scores are retained in the
                    # evidence bank. No student prediction self-reinforcement.
                    memory.observe(t, source_scores)
            stats["episodes"] += 1
    return stats


def run(args):
    receipt = plan_receipt(args)
    if args.scope == "plan":
        return receipt
    _assets(args)
    from flex_tta.dance_io import dance_videos
    videos = args.videos or dance_videos(args.tracker_root)
    if args.scope == "smoke":
        videos = videos[:1]
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    stats = (_episodic_replay(args, source, videos) if args.method == "RELATIVE_ONLY"
             else _model_run(args, source, videos))
    receipt["stats"] = stats
    out = args.output_root / args.method
    (out / "run_metadata.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
