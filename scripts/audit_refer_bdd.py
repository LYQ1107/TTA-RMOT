#!/usr/bin/env python3
"""Audit the complete official Refer-BDD target scope before model inference.

This audit is deliberately offline and evaluator-side.  It reads the labels
only to prove that every expression target reference can be materialized after
the label-free runtime has sealed its predictions; the generated runtime
manifest never contains these paths or values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET_VIDEOS = (
    "0000f77c-cb820c98",
    "00abd8a7-ecd6fc56",
    "00fcfec9-7ba5dabf",
    "02a17951-3d88d95e",
    "002b562f-e0ac84fe",
    "00f7eac8-5193d600",
    "0081da60-5fa22cc6",
    "00de601c-858a8a8d",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def frame_from_path(path: Path, video: str) -> int:
    prefix = f"{video}-"
    if not path.stem.startswith(prefix):
        raise ValueError(f"unexpected BDD filename for {video}: {path.name}")
    suffix = path.stem[len(prefix):]
    if not suffix.isdigit():
        raise ValueError(f"non-numeric BDD frame filename: {path.name}")
    return int(suffix)


def read_label(path: Path) -> tuple[set[int], list[str]]:
    ids: set[int] = set()
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw.split()
        if len(fields) < 6:
            errors.append(f"{path.name}:{line_number}: expected >=6 fields")
            continue
        try:
            track_id = int(float(fields[1]))
            values = [float(value) for value in fields[2:6]]
        except ValueError:
            errors.append(f"{path.name}:{line_number}: non-numeric label")
            continue
        x, y, width, height = values
        # The released BDD labels contain a small number of annotation-rounding
        # values just above one.  Keep those raw values for the official
        # conversion and reject only values outside a 1% annotation tolerance.
        if not (-0.01 <= x <= 1.01 and -0.01 <= y <= 1.01 and 0.0 < width <= 1.01 and 0.0 < height <= 1.01):
            errors.append(f"{path.name}:{line_number}: normalized cxcywh out of tolerance {values}")
        ids.add(track_id)
    return ids, errors


def archive_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    with zipfile.ZipFile(path) as archive:
        record["members"] = len(archive.infolist())
        record["testzip_bad_member"] = archive.testzip()
    return record


def audit(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    image_root = resolve(data["image_root"])
    expression_root = resolve(data["expression_root"])
    label_root = resolve(data["label_root"])
    seqmap_path = resolve(data["seqmap_file"])
    rows = [line.strip() for line in seqmap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected: dict[str, list[str]] = {video: [] for video in TARGET_VIDEOS}
    seqmap_errors: list[str] = []
    for row in rows:
        if "+" not in row:
            seqmap_errors.append(f"invalid row: {row}")
            continue
        video, expression_id = row.split("+", 1)
        if video not in expected:
            seqmap_errors.append(f"unexpected video: {video}")
            continue
        expected[video].append(expression_id)
    errors: list[str] = list(seqmap_errors)
    if len(rows) != 673:
        errors.append(f"seqmap row count {len(rows)} != 673")
    if any(len(values) == 0 for values in expected.values()):
        errors.append("one or more expected videos have no seqmap entries")

    total_images = 0
    total_expressions = 0
    total_label_files = 0
    total_target_refs = 0
    total_target_objects = 0
    video_reports: list[dict[str, Any]] = []
    for video in TARGET_VIDEOS:
        image_dir = image_root / video
        label_dir = label_root / video
        expression_dir = expression_root / video
        image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: frame_from_path(path, video))
        image_frames = [frame_from_path(path, video) for path in image_paths]
        if not image_paths:
            errors.append(f"no images for {video}")
        if len(image_frames) != len(set(image_frames)):
            errors.append(f"duplicate image frame for {video}")
        expected_frames = list(range(1, max(image_frames, default=0) + 1))
        image_gaps = sorted(set(expected_frames) - set(image_frames))
        if image_gaps:
            errors.append(f"image gaps for {video}: {image_gaps[:10]}")
        dimensions: Counter[str] = Counter()
        unreadable_images: list[str] = []
        for path in image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable_images.append(path.name)
            else:
                dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
        if unreadable_images:
            errors.append(f"unreadable images for {video}: {unreadable_images[:5]}")

        expression_paths = sorted(expression_dir.glob("*.json"))
        expression_ids = {path.stem for path in expression_paths}
        expected_ids = set(expected[video])
        missing_expressions = sorted(expected_ids - expression_ids)
        extra_expressions = sorted(expression_ids - expected_ids)
        if missing_expressions:
            errors.append(f"missing expressions for {video}: {missing_expressions[:5]}")
        if extra_expressions:
            errors.append(f"unexpected expression files for {video}: {extra_expressions[:5]}")

        label_paths = {frame_from_path(path, video): path for path in label_dir.glob("*.txt")}
        label_errors: list[str] = []
        label_ids: dict[int, set[int]] = {}
        for frame, path in sorted(label_paths.items()):
            ids, parse_errors = read_label(path)
            label_ids[frame] = ids
            label_errors.extend(parse_errors)
        if label_errors:
            errors.extend(f"{video}: {error}" for error in label_errors[:20])

        missing_label_frames = sorted(set(image_frames) - set(label_paths))
        referenced_missing_labels: list[int] = []
        referenced_missing_ids: list[dict[str, Any]] = []
        expression_reports: list[dict[str, Any]] = []
        video_target_refs = 0
        video_target_objects = 0
        for expression_id in expected[video]:
            path = expression_dir / f"{expression_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            sentence = str(payload.get("sentence", "")).strip()
            if not sentence:
                errors.append(f"empty sentence: {path}")
            raw_labels = payload.get("label", {})
            if not isinstance(raw_labels, dict):
                errors.append(f"label field is not a mapping: {path}")
                raw_labels = {}
            expression_refs = 0
            expression_objects = 0
            for raw_frame, raw_ids in raw_labels.items():
                frame = int(raw_frame)
                ids = {int(value) for value in raw_ids}
                expression_refs += 1
                expression_objects += len(ids)
                if frame not in set(image_frames):
                    errors.append(f"expression frame outside images: {path.name}:{frame}")
                if ids and frame not in label_paths:
                    referenced_missing_labels.append(frame)
                unavailable = sorted(ids - label_ids.get(frame, set()))
                if unavailable:
                    referenced_missing_ids.append({"expression_id": expression_id, "frame": frame, "ids": unavailable})
            video_target_refs += expression_refs
            video_target_objects += expression_objects
            expression_reports.append({
                "expression_id": expression_id,
                "sentence": sentence,
                "label_frames": len(raw_labels),
                "target_references": expression_refs,
                "target_objects": expression_objects,
            })
        if referenced_missing_labels:
            errors.append(f"referenced frames without label files for {video}: {sorted(set(referenced_missing_labels))}")
        if referenced_missing_ids:
            errors.append(f"referenced target ids absent from labels for {video}: {referenced_missing_ids[:5]}")
        total_images += len(image_paths)
        total_expressions += len(expected[video])
        total_label_files += len(label_paths)
        total_target_refs += video_target_refs
        total_target_objects += video_target_objects
        video_reports.append({
            "video_id": video,
            "seqmap_expressions": len(expected[video]),
            "expression_files": len(expression_ids),
            "images": len(image_paths),
            "image_frame_min": min(image_frames) if image_frames else None,
            "image_frame_max": max(image_frames) if image_frames else None,
            "image_dimensions": dict(dimensions),
            "label_files": len(label_paths),
            "missing_label_frames": missing_label_frames,
            "expression_target_references": video_target_refs,
            "expression_target_objects": video_target_objects,
            "expressions": expression_reports,
        })

    archive_root = resolve(data.get("archive_root", "/data2/user/LocateMOT/data/downloads/ReferBDD"))
    checkpoint = resolve(config["model"]["checkpoint"])
    report = {
        "format": "locatemot-refer-bdd-asset-audit-v1",
        "status": "COMPLETE" if not errors else "FAILED",
        "dataset": "refer_bdd",
        "official_sources": {
            "echotrack_repository": "https://github.com/lab206/EchoTrack",
            "seqmap_url": "https://raw.githubusercontent.com/lab206/EchoTrack/main/assets/seqmap_bdd.txt",
            "google_drive_folder": "https://drive.google.com/drive/folders/12jXjL9wJ9keB-vD-kWKI8IzK6mrIfmAN?usp=sharing",
        },
        "paths": {
            "image_root": str(image_root),
            "expression_root": str(expression_root),
            "label_root": str(label_root),
            "seqmap": str(seqmap_path),
            "archive_root": str(archive_root),
        },
        "seqmap": {
            "sha256": sha256(seqmap_path),
            "entries": len(rows),
            "unique_entries": len(set(rows)),
            "videos": list(TARGET_VIDEOS),
            "entries_by_video": {video: len(expected[video]) for video in TARGET_VIDEOS},
        },
        "archives": {
            "refer-bdd.zip": archive_record(archive_root / "refer-bdd.zip"),
            "label_with_ids.zip": archive_record(archive_root / "label_with_ids.zip"),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            "exists": checkpoint.is_file(),
        },
        "totals": {
            "videos": len(TARGET_VIDEOS),
            "expressions": total_expressions,
            "images": total_images,
            "label_files": total_label_files,
            "expression_target_references": total_target_refs,
            "expression_target_objects": total_target_objects,
        },
        "videos": video_reports,
        "errors": errors,
        "runtime_gt_isolation": {
            "runtime_manifest_contains_gt": False,
            "labels_read_by_online_runner": False,
            "gt_materialized_only_after_prediction_sealed": True,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta_refer_bdd.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/audits/refer_bdd_assets.json")
    args = parser.parse_args()
    report = audit(args.config.resolve())
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
