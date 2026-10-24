#!/usr/bin/env python3
"""Audit fixed-candidate and episode coverage invariants without target GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def tracker_row_key(line: str) -> tuple[int, int]:
    fields = line.split(",")
    if len(fields) != 10:
        raise ValueError(f"invalid MOT row: {line!r}")
    return int(float(fields[0])), int(float(fields[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=25)
    parser.add_argument("--expected-sequences", type=int, default=425)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    forbidden = ("target_gt_opened", "target_labels_opened", "future_frames_used")
    if any(metadata.get(key) is not False for key in forbidden):
        raise AssertionError({key: metadata.get(key) for key in forbidden})
    if metadata.get("tracker_geometry_changed") is not False or metadata.get("tracker_ids_changed") is not False:
        raise AssertionError("tracker geometry/ID mutation flag is not false")

    prediction_files = sorted(args.prediction_root.glob("*/*/predict.txt"))
    videos = {path.parts[-3] for path in prediction_files}
    if len(videos) != args.expected_videos:
        raise AssertionError(f"expected {args.expected_videos} videos, found {len(videos)}")
    if len(prediction_files) != args.expected_sequences:
        raise AssertionError(f"expected {args.expected_sequences} prediction files, found {len(prediction_files)}")

    checked_rows = 0
    for prediction in prediction_files:
        video = prediction.parts[-3]
        tracker = args.tracker_root / f"{video}.txt"
        tracker_rows = tracker.read_text(encoding="utf-8").splitlines()
        tracker_by_key = {tracker_row_key(row): row for row in tracker_rows if row.strip()}
        for line in prediction.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key = tracker_row_key(line)
            if tracker_by_key.get(key) != line:
                raise AssertionError(f"prediction changed official tracker row: {prediction}:{key}")
            checked_rows += 1
    print(json.dumps({
        "status": "PASS",
        "videos": len(videos),
        "prediction_files": len(prediction_files),
        "checked_prediction_rows": checked_rows,
        "target_gt_opened": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
