#!/usr/bin/env python3
"""GT-free FP32 diagnosis of sealed ZERO/Frozen decision-boundary differences."""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compat.sitecustomize  # noqa: F401
import torch
from transformers import RobertaTokenizerFast
from flex_tta.dance_io import load_expressions
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.run_flexhook_dance_pcd_tta import _load_model, _model_margin
from scripts.run_flexhook_positivefree import deny_target_gt


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    parser.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args = parser.parse_args()
    args.output = args.output.resolve()
    deny_target_gt(args)
    cases = []
    compared, maximum = 0, 0.
    for path in sorted(args.shard_root.glob("ZERO_UPDATE/results/*/*/candidate_margins.jsonl.gz")):
        for line in gzip.open(path, "rt"):
            record = json.loads(line)
            for tid, adapted, frozen in zip(record["ids"], record["adapted"], record["frozen"]):
                compared += 1
                maximum = max(maximum, abs(adapted-frozen))
                if (adapted > 0) != (frozen > 0):
                    cases.append(dict(video=path.parts[-3], expression=path.parts[-2],
                                      frame=record["frame"], track_id=tid,
                                      sealed_margin=adapted, pinned_margin=frozen))
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction((8192 << 20) / torch.cuda.get_device_properties(device).total_memory, device)
    vendor = ROOT / "vendor/FlexHook"
    _, model, _ = _load_model(vendor, Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth"),
                             args.dataset_root, args.tracker_root, args.output.parent / "model_log", device, 24,
                             strict_checkpoint=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(vendor / "pretrained/roberta-base"))
    for case in cases:
        episode = DanceEpisode(args.dataset_root, args.tracker_root, case["video"])
        sentence = dict(load_expressions(args.dataset_root, case["video"]))[case["expression"]]
        candidates = list(episode.current_candidates(case["frame"]))
        cache = FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=256 << 20)
        with torch.no_grad():
            original, ids = _model_margin(model, episode, candidates, sentence, tokenizer, device)
            optimized, cached_ids = cache.margin(candidates, sentence, case["frame"])
        if ids != cached_ids:
            raise ValueError("original/cached candidate identities differ")
        idx = ids.index(case["track_id"])
        case.update(original_uncached_margin=float(original[idx]), cached_margin=float(optimized[idx]),
                    candidates=len(ids), original_cached_max_error=float((original-optimized).abs().max()),
                    original_cached_sign_mismatches=int(((original > 0) != (optimized > 0)).sum()))
        cache = None
    result = dict(compared=compared, pinned_current_max_abs_error=maximum,
                  mismatches=cases, target_gt_opened=False, optimizer_steps=0,
                  precision="FP32", unchanged_threshold=0., peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  wall_seconds=time.perf_counter()-started, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
