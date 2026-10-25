#!/usr/bin/env python3
"""Merge whole-episode PCD shard predictions without opening target GT."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_tta.runtime.pf_receipts import file_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        metadata = args.output_root / args.method / "run_metadata.json"
        if args.resume and metadata.is_file():
            previous = json.loads(metadata.read_text())
            if previous.get("sequences") == 425 and previous.get("shards_root") == str(args.shards_root.resolve()):
                print(json.dumps({"status": "already_merged", "output": str(args.output_root)}))
                return 0
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    shard_methods = sorted(args.shards_root.glob(f"shard*/{args.method}"))
    if not shard_methods:
        raise FileNotFoundError(f"no {args.method} shard directories under {args.shards_root}")
    output_method = args.output_root / args.method
    output_results = output_method / "results"
    output_results.mkdir(parents=True)
    shard_metadata = []
    episodes = []
    videos = set()
    for shard_method in shard_methods:
        metadata_path = shard_method / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("scope") == "full" and metadata.get("status") != "complete":
            raise ValueError(f"shard not complete: {shard_method}")
        shard_metadata.append(metadata)
        episodes.extend(metadata.get("episodes", []))
        for source in sorted((shard_method / "results").glob("*/*/predict.txt")):
            relative = source.relative_to(shard_method / "results")
            target = output_results / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"duplicate shard prediction {target}")
            target.symlink_to(source.resolve())
            if metadata.get("scope") == "full":
                receipt_path = source.parent / "completion.json"
                receipt = json.loads(receipt_path.read_text())
                margins = source.parent / "candidate_margins.jsonl.gz"
                if (receipt["status"] != "complete" or not receipt["full_frame_coverage"] or
                    receipt["prediction_sha256"] != file_sha(source) or receipt["margins_sha256"] != file_sha(margins)):
                    raise ValueError(f"invalid sealed episode: {source.parent}")
                (target.parent / "completion.json").symlink_to(receipt_path.resolve())
                (target.parent / "candidate_margins.jsonl.gz").symlink_to(margins.resolve())
            videos.add(relative.parts[0])
    if len(videos) != 25:
        raise ValueError(f"expected 25 merged videos, found {len(videos)}")
    if len(episodes) != 425:
        raise ValueError(f"expected 425 merged episodes, found {len(episodes)}")
    keys = [(e["video"], e["expression_id"]) for e in episodes]
    paths = [(p.parts[-3], p.parts[-2]) for p in output_results.glob("*/*/predict.txt")]
    if len(set(keys)) != 425 or set(keys) != set(paths):
        raise ValueError("duplicate or missing episode predictions/metadata")
    stats = {}
    for episode in episodes:
        for key, value in episode.get("counter_delta", episode.get("stats", {})).items():
            if isinstance(value, (int, float)) and key not in {"video", "expression_id"}:
                stats[key] = (max(stats.get(key, 0), value) if key.startswith("max_") or key == "source_max_abs_error"
                              else stats.get(key, 0) + value)
    stats["videos"] = len(videos)
    stats["expressions"] = len(episodes)
    metadata = {
        "method": args.method,
        "status": "complete",
        "protocol": "merged independent whole-episode source-free PCD TTA shards",
        "source_free_prediction": True,
        "target_gt_opened": False,
        "target_labels_opened": False,
        "future_frames_used": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
        "shards_root": str(args.shards_root.resolve()),
        "videos": sorted(videos),
        "sequences": len(episodes),
        "aggregate_stats": stats,
        "shards": [
            {
                "source_checkpoint": item.get("source_checkpoint"),
                "source_result": item.get("source_result"),
                "videos": item.get("videos"),
                "sequences": item.get("sequences"),
                "hyperparameters": item.get("hyperparameters"),
                "experiment_commit": item.get("experiment_commit"),
                "gpu_uuid": item.get("gpu_uuid"),
                "source_supplement": item.get("source_supplement"),
                "source_supplement_sha256": item.get("source_supplement_sha256"),
                "matched_frozen_root": item.get("matched_frozen_root"),
                "matched_frozen_receipt_sha256": item.get("matched_frozen_receipt_sha256"),
                "stats": item.get("stats"),
            }
            for item in shard_metadata
        ],
        "episodes": episodes,
    }
    (output_method / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_method.resolve()), "videos": len(videos), "sequences": len(episodes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
