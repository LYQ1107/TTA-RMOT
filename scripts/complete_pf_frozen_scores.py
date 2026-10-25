#!/usr/bin/env python3
"""Supplement ONLY omitted fixed-MOT observations with the same frozen V2 model.

User approved complete-bank protocol on 2026-10-25. Original score file is
immutable; existing scores can never be replaced. No target GT is opened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compat.sitecustomize  # noqa
import torch
from transformers import RobertaTokenizerFast
from flex_tta.dance_io import dance_videos, load_expressions, load_tracker_lines
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.run_flexhook_dance_pcd_tta import _load_model, _model_margin
from scripts.run_flexhook_positivefree import SOURCE_SHA256, FROZEN_SHA256, deny_target_gt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--source-result", type=Path, default=ROOT / "results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json")
    args = p.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    deny_target_gt(args)
    checkpoint = Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth")
    if file_sha(args.source_result) != FROZEN_SHA256 or file_sha(checkpoint) != SOURCE_SHA256:
        raise ValueError("pinned input hash mismatch")
    source = json.loads(args.source_result.read_text())
    vendor = ROOT / "vendor/FlexHook"
    device = torch.device("cuda:0")
    _, model, _ = _load_model(vendor, checkpoint, args.dataset_root, args.tracker_root,
                              args.output_root / "model_log", device, 24, strict_checkpoint=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(vendor / "pretrained/roberta-base"))
    supplement, records, comparisons = {}, [], []
    for video in dance_videos(args.tracker_root):
        rows = load_tracker_lines(args.tracker_root, video)
        missing = {(t, i) for t, frame in rows.items() for i in frame
                   if str(t) not in source[video].get(str(i), {})}
        path = args.output_root / f"{video}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            generated = payload["scores"]
        else:
            episode = DanceEpisode(args.dataset_root, args.tracker_root, video)
            cache = FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=256 << 20)
            generated = {}
            for t in sorted({t for t, _ in missing}):
                candidates = list(episode.current_candidates(t))
                for expression_index, (_, sentence) in enumerate(load_expressions(args.dataset_root, video)):
                    with torch.no_grad():
                        logits, ids = cache.logits(candidates, sentence, t)
                    if expression_index == 0 and len(comparisons) < 8:
                        with torch.no_grad():
                            original, original_ids = _model_margin(model, episode, candidates, sentence, tokenizer, device)
                        optimized = logits[:, 1]-logits[:, 0]
                        error = float((original-optimized).abs().max())
                        mismatch = int(((original > 0) != (optimized > 0)).sum())
                        comparisons.append(dict(video=video, frame=t, candidates=len(ids), max_margin_error=error,
                                                sign_mismatches=mismatch, selected=sum(v > 0 for v in original.tolist())))
                        if error > 1e-3 or mismatch or ids != original_ids:
                            raise RuntimeError("supplement cached/original equivalence failed")
                    for tid, values in zip(ids, logits.cpu().tolist()):
                        if (t, tid) in missing:
                            generated.setdefault(str(tid), {}).setdefault(str(t), {})[sentence] = values
            payload = dict(video=video, scores=generated, cache_stats=cache.stats)
            atomic_json(path, payload)
        generated_keys = {(int(t), int(i)) for i, track in generated.items() for t in track}
        if generated_keys != missing or any(len(expr) != 17 for track in generated.values() for expr in track.values()):
            raise ValueError(f"supplement coverage invalid: {video}")
        supplement[video] = generated
        positive = sum(values[1] > values[0] for track in generated.values() for expr in track.values() for values in expr.values())
        records.append(dict(video=video, supplemented_observations=len(missing), new_source_positive_rows=positive))
        print(json.dumps(records[-1]), flush=True)
    output = args.output_root / "source_supplement.json"
    atomic_json(output, supplement)
    receipt = dict(protocol="user-authorized complete fixed-MOT bank; original Frozen remains immutable",
                   original_source_result=str(args.source_result), original_sha256=FROZEN_SHA256,
                   checkpoint=str(checkpoint), checkpoint_sha256=SOURCE_SHA256,
                   supplement=str(output), supplement_sha256=file_sha(output), records=records,
                   supplemented_observations=sum(r["supplemented_observations"] for r in records),
                   supplemented_expression_rows=17*sum(r["supplemented_observations"] for r in records),
                   new_source_positive_rows=sum(r["new_source_positive_rows"] for r in records),
                   original_cached_comparisons=comparisons, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   target_gt_opened=False, future_frames_used=False,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    atomic_json(args.output_root / "completion_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
