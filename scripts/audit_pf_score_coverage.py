#!/usr/bin/env python3
"""Audit pinned Frozen coverage, then probe missing rows without target GT."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.dance_io import dance_videos, load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-result", type=Path, default=ROOT / "results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json")
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu-probes", type=int, default=0)
    args = p.parse_args()
    args.output = args.output.resolve()
    source = json.loads(args.source_result.read_text())
    report = dict(source_result=str(args.source_result), source_sha256=file_sha(args.source_result),
                  raw_tracker_rows=0, frozen_score_rows=0, missing_rows=[], extra_score_rows=[],
                  missing_are_last_track_observation=True, missing_positive_witnesses=[], probes=[],
                  target_gt_opened=False, future_frames_used=False)
    priorities = []
    for video in dance_videos(args.tracker_root):
        rows = load_tracker_lines(args.tracker_root, video)
        keys = {(t, i) for t, frame in rows.items() for i in frame}
        score_keys = {(int(t), int(i)) for i, track in source[video].items() for t in track}
        report["raw_tracker_rows"] += len(keys)
        report["frozen_score_rows"] += len(score_keys)
        tail = {}
        for t, i in keys:
            tail[i] = max(tail.get(i, -1), t)
        for t, i in sorted(keys-score_keys):
            report["missing_rows"].append(dict(video=video, frame=t, track_id=i, is_last_observation=t == tail[i]))
            report["missing_are_last_track_observation"] &= t == tail[i]
            track = source[video].get(str(i), {})
            previous = [int(k) for k in track if int(k) < t]
            if previous:
                last = max(previous)
                sentence, logits = max(track[str(last)].items(), key=lambda kv: kv[1][1]-kv[1][0])
                priorities.append((float(logits[1]-logits[0]), video, t, i, sentence))
        report["extra_score_rows"].extend(dict(video=video, frame=t, track_id=i) for t, i in sorted(score_keys-keys))
    report["missing_row_count"] = len(report["missing_rows"])
    atomic_json(args.output, report)
    if args.gpu_probes:
        import compat.sitecustomize  # noqa
        import torch
        from transformers import RobertaTokenizerFast
        from scripts.run_flexhook_dance_pcd_tta import _load_model
        from scripts.run_flexhook_positivefree import deny_target_gt
        from flex_tta.runtime.dance_episode import DanceEpisode
        from flex_tta.runtime.frozen_features import FrozenFeatureCache
        deny_target_gt(args)
        checkpoint = Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth")
        if file_sha(checkpoint) != "71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613":
            raise ValueError("pinned checkpoint mismatch")
        vendor = ROOT / "vendor/FlexHook"
        device = torch.device("cuda:0")
        _, model, _ = _load_model(vendor, checkpoint, args.dataset_root, args.tracker_root,
                                  args.output.parent / "coverage_probe_model_log", device, 24, strict_checkpoint=True)
        tokenizer = RobertaTokenizerFast.from_pretrained(str(vendor / "pretrained/roberta-base"))
        loaded = {}
        for prior, video, t, tid, sentence in sorted(priorities, reverse=True)[:args.gpu_probes]:
            if video not in loaded:
                episode = DanceEpisode(args.dataset_root, args.tracker_root, video)
                loaded[video] = (episode, FrozenFeatureCache(model, episode, tokenizer, device, max_bytes=256 << 20))
            episode, cache = loaded[video]
            with torch.no_grad():
                margin, ids = cache.margin(list(episode.current_candidates(t)), sentence, t)
            value = float(margin[ids.index(tid)])
            item = dict(video=video, frame=t, track_id=tid, sentence=sentence,
                        previous_frozen_margin=prior, missing_candidate_source_margin=value,
                        complete_current_candidate_count=len(ids), source_selected=value > 0,
                        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
            report["probes"].append(item)
            if value > 0:
                report["missing_positive_witnesses"].append(item)
            atomic_json(args.output, report)
        report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["exact_old_frozen_row_reproduction_conflicts_with_complete_bank"] = bool(report["missing_positive_witnesses"])
    report["vendor_root_cause"] = "mydataloader.py sample_end=obj_last-frame_num with exclusive range omits final observation of each track"
    atomic_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k != "missing_rows"}, indent=2))


if __name__ == "__main__": main()
