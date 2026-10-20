"""Expand compact ShiftRMOT condition rows at the runtime/evaluator boundary."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2

from .visual_corruptions import apply_visual_shift


RUNTIME_FORBIDDEN = {"label_path", "expression_path", "gt_box", "gt_id", "label", "target"}


def _read_compact(path: Path) -> Iterable[dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            yield json.loads(raw)


def _matches(row: dict[str, Any], episodes: set[str] | None) -> bool:
    if not episodes:
        return True
    candidates = {
        f"{row['video_id']}/{row['base_expression_id']}",
        f"{row['video_id']}/{row['runtime_expression_id']}",
        f"{row['video_id']}/{row['condition_id']}",
        f"{row['video_id']}+{row['runtime_expression_id']}",
    }
    return bool(candidates.intersection(episodes))


def _expand(condition: dict[str, Any], *, evaluator: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_root = Path(condition["image_root"])
    suffix = str(condition.get("image_suffix", ".png"))
    for frame_index in condition["frame_indices"]:
        frame_index = int(frame_index)
        image_path = image_root / f"{frame_index:06d}{suffix}"
        row = {
            "dataset": "shift_rmot_v1",
            "video_id": str(condition["video_id"]),
            "expression_id": str(condition["runtime_expression_id"]),
            "base_expression_id": str(condition["base_expression_id"]),
            "condition_id": str(condition["condition_id"]),
            "condition_family": str(condition["condition_family"]),
            "query_text": str(condition["query_text"]),
            "original_query_text": str(condition["original_query_text"]),
            "frame_index": frame_index,
            "trackeval_frame_offset": int(condition.get("trackeval_frame_offset", 1)),
            "image_path": str(image_path),
            "visual_shift": str(condition.get("visual_shift", "identity")),
            "visual_severity": str(condition.get("visual_severity", "identity")),
            "shift_seed_basis": str(condition.get("shift_seed_basis", "")),
        }
        if evaluator:
            row.update({
                "expression_path": str(condition["expression_path"]),
                "label_path": str(Path(condition["label_root"]) / f"{frame_index:06d}.txt"),
                "label_format": str(condition.get("label_format", "cxcywh")),
            })
        elif RUNTIME_FORBIDDEN.intersection(row):
            raise AssertionError(f"compact runtime row unexpectedly contains GT fields: {row}")
        rows.append(row)
    return rows


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["video_id"]), str(row["expression_id"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["frame_index"]))
    return dict(grouped)


def load_compact_runtime_rows(path: Path, *, episodes: set[str] | None = None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand only selected compact conditions and validate image paths."""
    rows: list[dict[str, Any]] = []
    for condition in _read_compact(path):
        if not _matches(condition, episodes):
            continue
        for row in _expand(condition, evaluator=False):
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])
            rows.append(row)
    return _group(rows)


def load_compact_evaluation_rows(path: Path, *, episodes: set[str] | None = None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand evaluator-only rows after prediction is complete."""
    rows: list[dict[str, Any]] = []
    for condition in _read_compact(path):
        if not _matches(condition, episodes):
            continue
        rows.extend(_expand(condition, evaluator=True))
    return _group(rows)


def load_runtime_image(row: dict[str, Any]):
    """Read one base image and apply the row's deterministic visual shift."""
    path = Path(row["image_path"])
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return apply_visual_shift(
        image,
        shift_id=str(row.get("visual_shift", "identity")),
        severity=str(row.get("visual_severity", "identity")),
        seed_basis=str(row.get("shift_seed_basis", "")),
    )
