#!/usr/bin/env python3
"""Repair run-level counters after an episode-reset online suite.

Older runner outputs assigned controller/probe counters after every episode,
so the final metadata contained only the last expression's values.  This
utility derives cumulative values from the sealed online trace and updates
the local result summaries transparently, retaining the previous values in a
correction record.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROBE_METHODS = {"PROBE_CACHE", "PROBE_ST", "QCS_TTA", "NO_STATE"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()

    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    methods = list(metadata["methods"])
    optimizer_steps = {method: 0 for method in methods}
    episode_probe_calls: dict[tuple[str, str], int] = {}
    with (run_dir / "update_trace.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            trace = json.loads(raw)
            episode = (str(trace["video_id"]), str(trace["expression_id"]))
            episode_probe_calls[episode] = max(
                episode_probe_calls.get(episode, 0), int(trace.get("probe_calls_total", 0))
            )
            for method in methods:
                update = trace.get("methods", {}).get(method, {}).get("last_update")
                if update and bool(update.get("stepped")):
                    optimizer_steps[method] += 1
    probe_calls = sum(episode_probe_calls.values())

    previous = {
        method: {
            "optimizer_steps": metadata.get("method_stats", {}).get(method, {}).get("optimizer_steps", 0),
            "probe_calls": metadata.get("method_stats", {}).get(method, {}).get("probe_calls", 0),
        }
        for method in methods
    }
    for method in methods:
        stats = metadata.setdefault("method_stats", {}).setdefault(method, {})
        stats["optimizer_steps"] = optimizer_steps[method]
        stats["probe_calls"] = probe_calls if method in PROBE_METHODS else 0
    metadata["counter_correction"] = {
        "source": "update_trace.jsonl",
        "episode_count": len(episode_probe_calls),
        "shared_probe_calls": probe_calls,
        "previous_final_episode_values": previous,
        "note": "The original runner reset controller/probe counters at each expression; these values are cumulative over the complete suite.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = out_dir / "summary.csv"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0]) if rows else []
        for row in rows:
            method = row["method"]
            row["optimizer_steps"] = str(optimizer_steps[method])
            row["probe_calls"] = str(probe_calls if method in PROBE_METHODS else 0)
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    timing_path = out_dir / "timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        for method in methods:
            stats = timing.setdefault("method_stats", {}).setdefault(method, {})
            stats["optimizer_steps"] = optimizer_steps[method]
            stats["probe_calls"] = probe_calls if method in PROBE_METHODS else 0
        timing["counter_correction"] = metadata["counter_correction"]
        timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    probe_path = out_dir / "probe_coverage.json"
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        for method in methods:
            stats = probe.setdefault("by_method", {}).setdefault(method, {})
            stats["probe_calls"] = probe_calls if method in PROBE_METHODS else 0
            stats["probe_calls_scope"] = "shared_probe_suite" if method in PROBE_METHODS else "not_used"
        probe["counter_correction"] = metadata["counter_correction"]
        probe_path.write_text(json.dumps(probe, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"optimizer_steps": optimizer_steps, "shared_probe_calls": probe_calls, "episodes": len(episode_probe_calls)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
