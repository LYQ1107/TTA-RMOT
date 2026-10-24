#!/usr/bin/env python3
"""Merge whole-episode PCD shard predictions without opening target GT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
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
        shard_metadata.append(metadata)
        episodes.extend(metadata.get("episodes", []))
        for source in sorted((shard_method / "results").glob("*/*/predict.txt")):
            relative = source.relative_to(shard_method / "results")
            target = output_results / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"duplicate shard prediction {target}")
            target.symlink_to(source.resolve())
            videos.add(relative.parts[0])
    if len(videos) != 25:
        raise ValueError(f"expected 25 merged videos, found {len(videos)}")
    if len(episodes) != 425:
        raise ValueError(f"expected 425 merged episodes, found {len(episodes)}")
    stats = {}
    for episode in episodes:
        for key, value in episode.get("stats", {}).items():
            if isinstance(value, (int, float)) and key not in {"video", "expression_id"}:
                stats[key] = stats.get(key, 0) + value
    stats["videos"] = len(videos)
    stats["expressions"] = len(episodes)
    metadata = {
        "method": args.method,
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
