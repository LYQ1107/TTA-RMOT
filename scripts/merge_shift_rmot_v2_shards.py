#!/usr/bin/env python3
"""Merge disjoint ShiftRMOT online-run shards before post-hoc evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_metadata(path: Path) -> dict:
    metadata_path = path / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PREDICTION_COMPLETE":
        raise RuntimeError(f"shard is not complete: {metadata_path}")
    if metadata.get("gt_used_by_runner") or metadata.get("future_frames_used") or metadata.get("evaluation_manifest_opened"):
        raise RuntimeError(f"shard violates GT-free protocol: {metadata_path}")
    return metadata


def _merge_jsonl(shards: list[Path], name: str, output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for shard in shards:
            source = shard / name
            if source.is_file():
                handle.write(source.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=0)
    args = parser.parse_args()
    shards = [path.resolve() for path in args.shard_dir]
    if len({str(path) for path in shards}) != len(shards):
        raise ValueError("duplicate shard directory")
    metadata = [_read_metadata(path) for path in shards]
    if len(metadata) == 1 and "shard_index" not in metadata[0]:
        # Accept a pre-sharding one-process run as a one-shard regression
        # artifact; multi-shard full runs must record explicit indices.
        metadata[0]["shard_index"] = 0
    shard_ids = sorted((int(item.get("shard_index", -1)), index) for index, item in enumerate(metadata))
    expected_ids = list(range(len(shards)))
    if [item[0] for item in shard_ids] != expected_ids:
        raise ValueError(f"shards must cover indices {expected_ids}, got {[item[0] for item in shard_ids]}")
    ordered = [shards[index] for _, index in shard_ids]
    ordered_metadata = [metadata[index] for _, index in shard_ids]
    config_hashes = {item.get("config_sha256") for item in ordered_metadata}
    manifest_hashes = {item.get("runtime_manifest_sha256") for item in ordered_metadata}
    if len(config_hashes) != 1 or len(manifest_hashes) != 1:
        raise ValueError("shards do not share one config/runtime manifest")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_root = output / "predictions"
    for shard in ordered:
        source_root = shard / "predictions"
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        for source in source_root.rglob("predict.txt"):
            relative = source.relative_to(source_root)
            target = prediction_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            target.symlink_to(source.resolve())

    _merge_jsonl(ordered, "update_trace.jsonl", output / "update_trace.jsonl")
    _merge_jsonl(ordered, "candidate_diagnostics.jsonl", output / "candidate_diagnostics.jsonl")
    episodes = [episode for item in ordered_metadata for episode in item.get("episodes", [])]
    if len(episodes) != len(set(episodes)):
        raise ValueError("shards contain overlapping episodes")
    # Use the explicit episode list rather than a legacy counter from smoke
    # runs created before the max-expressions off-by-one fix.
    episode_count = sum(len(item.get("episodes", [])) for item in ordered_metadata)
    if args.expected_episodes and episode_count != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episodes, found {episode_count}")
    payload = {
        **ordered_metadata[0],
        "format": "locatemot-qcs-online-run-sharded-v1",
        "status": "PREDICTION_COMPLETE",
        "output_dir": str(output),
        "episodes": sorted(episodes),
        "shard_index": None,
        "num_shards": len(ordered),
        "shards": [str(path) for path in ordered],
        "video_expression_count": episode_count,
        "frame_query_count": sum(int(item.get("frame_query_count", 0)) for item in ordered_metadata),
        "wall_seconds": max(float(item.get("wall_seconds", 0.0)) for item in ordered_metadata),
        "trace": str((output / "update_trace.jsonl").resolve()),
        "candidate_diagnostics": str((output / "candidate_diagnostics.jsonl").resolve()),
        "gt_used_by_runner": False,
        "future_frames_used": False,
        "evaluation_manifest_opened": False,
    }
    (output / "run_metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output_dir": str(output), "episode_count": episode_count, "frame_query_count": payload["frame_query_count"], "num_shards": len(ordered)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
