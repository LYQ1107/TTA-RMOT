"""Label-free I/O for the fixed official Refer-Dance candidate bank.

This module intentionally reads only expression JSON files, tracker rows, and
the already-computed FlexHook score stream.  It never opens ``labels.json`` or
``gt_template``; the evaluator owns those paths in a separate process.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def dance_videos(tracker_root: Path) -> list[str]:
    videos = sorted(path.stem for path in tracker_root.glob("*.txt"))
    if len(videos) != 25:
        raise ValueError(f"expected 25 official Dance tracker files, found {len(videos)}")
    return videos


def load_expressions(dataset_root: Path, video: str) -> list[tuple[str, str]]:
    records = []
    for path in sorted((dataset_root / "expression" / video).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append((path.stem, str(payload["sentence"])))
    if len(records) != 17:
        raise ValueError(f"{video}: expected 17 validation expressions, found {len(records)}")
    return records


def load_tracker_lines(tracker_root: Path, video: str) -> dict[int, dict[int, str]]:
    by_frame: dict[int, dict[int, str]] = defaultdict(dict)
    path = tracker_root / f"{video}.txt"
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != 10:
            raise ValueError(f"invalid Dance MOT row {path}:{line_number}")
        frame_id = int(float(fields[0]))
        track_id = int(float(fields[1]))
        if track_id in by_frame[frame_id]:
            raise ValueError(f"duplicate tracker key {video}/{frame_id}/{track_id}")
        by_frame[frame_id][track_id] = line
    return {frame: dict(sorted(rows.items())) for frame, rows in sorted(by_frame.items())}


def score_frames(source: dict[str, Any], video: str, sentence: str) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = defaultdict(dict)
    for object_id, frames in source[video].items():
        track_id = int(object_id)
        for frame_id, expressions in frames.items():
            try:
                score = expressions[sentence]
            except KeyError as exc:
                raise KeyError(f"score stream missing {video}/{track_id}/{frame_id}/{sentence}") from exc
            if len(score) != 2:
                raise ValueError(f"expected two-class score at {video}/{track_id}/{frame_id}")
            result[int(frame_id)][track_id] = float(score[1]) - float(score[0])
    return {frame: dict(sorted(rows.items())) for frame, rows in sorted(result.items())}


def source_score_rows(source: dict[str, Any], video: str, sentence: str):
    """Yield ``(frame, track_id, margin)`` without opening target labels."""

    for frame_id, rows in score_frames(source, video, sentence).items():
        for track_id, margin in rows.items():
            yield frame_id, track_id, margin
