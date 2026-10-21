"""Expand compact ShiftRMOT condition rows at the runtime/evaluator boundary."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

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
    if evaluator:
        frame_indices = condition.get("evaluation_frame_indices", condition["frame_indices"])
        frame_role = "evaluation_label_range"
    else:
        frame_indices = condition.get("runtime_frame_indices", condition["frame_indices"])
        frame_role = "runtime_video_axis"
    for frame_index in frame_indices:
        frame_index = int(frame_index)
        image_path = image_root / f"{frame_index:06d}{suffix}"
        seed_basis = str(condition.get("shift_seed_basis", ""))
        # The frame index is appended at the runtime boundary so visual noise
        # is independent of expression/language condition IDs.
        row_seed_basis = f"{seed_basis}|frame={frame_index}"
        row = {
            "dataset": str(condition.get("dataset", "shift_rmot_v1")),
            "video_id": str(condition["video_id"]),
            "expression_id": str(condition["runtime_expression_id"]),
            "base_expression_id": str(condition["base_expression_id"]),
            "condition_id": str(condition["condition_id"]),
            "condition_family": str(condition["condition_family"]),
            "query_text": str(condition["query_text"]),
            "original_query_text": str(condition["original_query_text"]),
            "frame_index": frame_index,
            "frame_role": frame_role,
            "trackeval_frame_offset": int(condition.get("trackeval_frame_offset", 1)),
            "image_path": str(image_path),
            "visual_shift": str(condition.get("visual_shift", "identity")),
            "visual_severity": str(condition.get("visual_severity", "identity")),
            "shift_seed_basis": row_seed_basis,
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


def iter_compact_runtime_groups(path: Path, *, episodes: set[str] | None = None) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """Yield one compact condition at a time for large online runs."""
    verified_images: set[str] = set()
    for condition in _read_compact(path):
        if not _matches(condition, episodes):
            continue
        rows = _expand(condition, evaluator=False)
        for row in rows:
            image_path = str(row["image_path"])
            if image_path in verified_images:
                continue
            if not Path(image_path).is_file():
                raise FileNotFoundError(row["image_path"])
            verified_images.add(image_path)
        yield (str(condition["video_id"]), str(condition["runtime_expression_id"])), rows


def iter_compact_evaluation_groups(path: Path, *, episodes: set[str] | None = None) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    """Yield evaluator-only rows one compact condition at a time."""
    for condition in _read_compact(path):
        if not _matches(condition, episodes):
            continue
        yield (str(condition["video_id"]), str(condition["runtime_expression_id"])), _expand(condition, evaluator=True)


def load_compact_runtime_rows(path: Path, *, episodes: set[str] | None = None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand only selected compact conditions and validate image paths."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in iter_compact_runtime_groups(path, episodes=episodes):
        grouped[key] = rows
    return grouped


def load_compact_evaluation_rows(path: Path, *, episodes: set[str] | None = None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Expand evaluator-only rows after prediction is complete."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, rows in iter_compact_evaluation_groups(path, episodes=episodes):
        grouped[key] = rows
    return grouped


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
