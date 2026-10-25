#!/usr/bin/env python3
"""Compare original vs cached FP32 decode on real, complete candidate sets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compat.sitecustomize  # noqa
import torch
from transformers import RobertaTokenizerFast
from scripts.run_flexhook_dance_pcd_tta import _load_model, _model_margin, _make_adapter
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.adaptation.identity_contrast import PCDFeatureTap
from flex_tta.dance_io import load_expressions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    args = p.parse_args()
    args.output = args.output.resolve()
    vendor = ROOT / "vendor/FlexHook"
    device = torch.device("cuda:0")
    _, model, _ = _load_model(vendor, Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth"),
                              args.dataset_root, args.tracker_root, args.output.parent / "model_log", device, 24,
                              strict_checkpoint=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(vendor / "pretrained/roberta-base"))
    tap = PCDFeatureTap(model)
    logits = []
    head_hook = model.head[-1].register_forward_hook(lambda _m, _i, out: logits.append(out.detach().clone()))
    # Select workload extremes from fixed tracker rows only, never target GT.
    counts = []
    for path in sorted(args.tracker_root.glob("*.txt")):
        by_frame = {}
        for row in path.read_text().splitlines():
            t = int(float(row.split(",")[0])); by_frame[t] = by_frame.get(t, 0) + 1
        frame = max(by_frame, key=by_frame.get)
        counts.append((by_frame[frame], path.stem, frame))
    probes = [counts[0], max(counts)]
    records = []
    for _, video, busy_frame in probes:
        episode = DanceEpisode(args.dataset_root, args.tracker_root, video)
        cache = FrozenFeatureCache(model, episode, tokenizer, device)
        sentence = load_expressions(args.dataset_root, video)[0][1]
        for t in sorted({episode.frames[0], busy_frame, min(episode.frames[-1], busy_frame + 1)}):
            candidates = list(episode.current_candidates(t))
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.no_grad():
                original, ids = _model_margin(model, episode, candidates, sentence, tokenizer, device)
                orig_features = tap.take().detach().clone()
                original_logits = logits.pop()
            torch.cuda.synchronize(); original_seconds = time.perf_counter() - start
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.no_grad():
                optimized, cached_ids = cache.margin(candidates, sentence, t)
                cached_features = tap.take().detach().clone()
                cached_logits = logits.pop()
            torch.cuda.synchronize(); cached_seconds = time.perf_counter() - start
            error = float((original - optimized).abs().max())
            signs = int(((original > 0) != (optimized > 0)).sum())
            records.append(dict(video=video, frame=t, candidates=len(ids), max_abs_error=error,
                                sign_mismatches=signs, ids_equal=ids == cached_ids,
                                feature_max_abs_error=float((orig_features-cached_features).abs().max()),
                                logit_max_abs_error=float((original_logits-cached_logits).abs().max()),
                                original_logits=original_logits.cpu().tolist(),
                                optimized_logits=cached_logits.cpu().tolist(),
                                selected_track_ids=[i for i, v in zip(ids, original.tolist()) if v > 0],
                                original_seconds=original_seconds, cached_seconds=cached_seconds))
            if signs or ids != cached_ids or error > 1e-3:
                raise RuntimeError(f"cached encoder equivalence FAILED: {records[-1]}")
    # Check actual LoRA gradients, not only frozen/no-grad forward values.
    tap.close()
    head_hook.remove()
    adapter = _make_adapter(model, "PCD_LORA", device, 4, 64)
    torch.manual_seed(0); adapter.reset_episode()
    params = [x for x in adapter.parameters() if x.requires_grad]
    margins, _ = _model_margin(model, episode, candidates, sentence, tokenizer, device)
    margins.sum().backward()
    grads = [x.grad.detach().clone() for x in params]
    for x in params: x.grad = None
    cached_margins, _ = cache.margin(candidates, sentence, t)
    cached_margins.sum().backward()
    grad_error = max(float((x.grad-g).abs().max()) for x, g in zip(params, grads))
    grad_scale = max(float(g.abs().max()) for g in grads)
    if not grad_scale or grad_error > 1e-3:
        raise RuntimeError(f"cached LoRA gradient equivalence FAILED: {grad_error}, {grad_scale}")
    result = dict(pass_=True, precision="FP32", probes=records, gradient_max_abs_error=grad_error,
                  gradient_max_abs_value=grad_scale, cache_stats=cache.stats,
                  target_gt_opened=False, future_frames_used=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
