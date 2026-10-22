"""Read-only adapters for tracker output formats used by the benchmark.

Temp-NeuralSORT uses separate ``car`` and ``pedestrian`` files in the
Refer-KITTI release.  StrongSORT and OCSORT usually emit one MOT CSV.  All
adapters normalize to the same ``frame_id, track_id, original row`` contract;
the referring plugin is therefore independent of the association backend.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from flex_tta.types import TrackRow, TrackerCandidate


def _parse_line(line: str, path: Path, line_number: int) -> TrackRow | None:
    raw = line.strip()
    if not raw:
        return None
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
        if len(values) < 6:
            raise ValueError("MOT row has fewer than six columns")
        return TrackRow(frame_id=int(values[0]), track_id=int(values[1]), values=values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MOT row {path}:{line_number}: {raw!r}") from exc


def load_mot_csv(path: Path) -> list[TrackRow]:
    """Load one standard MOT CSV while preserving all numeric columns."""

    rows: list[TrackRow] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = _parse_line(line, path, number)
        if row is not None:
            rows.append(row)
    return rows


def load_kitti_class_split(path: Path, *, id_offset: int = 0) -> list[TrackRow]:
    """Load a Temp-NeuralSORT class file and optionally offset track IDs."""

    rows = load_mot_csv(path)
    if not id_offset:
        return rows
    # Keep the normalized identity and the serialized MOT row in sync.  The
    # official FlexHook merger adds max(car_id) to every pedestrian ID before
    # writing final predictions; changing only the in-memory key would emit
    # duplicate IDs when car and pedestrian files both contain ID=1.
    return [
        TrackRow(
            row.frame_id,
            row.track_id + id_offset,
            (row.values[0], row.values[1] + id_offset, *row.values[2:]),
        )
        for row in rows
    ]


def merge_kitti_class_splits(car_path: Path, pedestrian_path: Path) -> list[TrackRow]:
    """Merge KITTI car/pedestrian files with the official ID offset rule."""

    cars = load_kitti_class_split(car_path)
    max_car_id = max((row.track_id for row in cars), default=0)
    pedestrians = load_kitti_class_split(pedestrian_path, id_offset=max_car_id)
    return sorted([*cars, *pedestrians], key=lambda row: (row.frame_id, row.track_id))


def rows_by_frame(rows: Iterable[TrackRow]) -> dict[int, dict[int, TrackRow]]:
    result: dict[int, dict[int, TrackRow]] = defaultdict(dict)
    for row in rows:
        result[row.frame_id][row.track_id] = row
    return {frame: dict(sorted(items.items())) for frame, items in sorted(result.items())}


def candidates_from_rows(video_id: str, rows: Iterable[TrackRow], margins: dict[tuple[int, int], float]) -> list[TrackerCandidate]:
    """Attach source referring margins to already parsed tracker rows."""

    candidates: list[TrackerCandidate] = []
    for row in rows:
        key = (row.frame_id, row.track_id)
        if key in margins:
            candidates.append(TrackerCandidate(video_id, row.frame_id, row.track_id, row, float(margins[key])))
    return candidates
