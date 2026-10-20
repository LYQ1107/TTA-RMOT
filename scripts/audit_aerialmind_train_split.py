#!/usr/bin/env python3
"""Audit the official AerialMind training split without opening test GT.

This is a data-contract check for a future supervised source checkpoint.  It
does not train, read target-test expressions, or select any TTA parameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_aerialmind import UAVDT_TEST, VISDRONE_TEST

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(dataset_root: Path, split_file: Path) -> dict[str, Any]:
    image_root = dataset_root / "image_02"
    label_root = dataset_root / "labels_with_ids"
    expression_root = dataset_root / "expression"
    if not all(path.is_dir() for path in (image_root, label_root, expression_root)):
        raise FileNotFoundError(f"incomplete AerialMind root: {dataset_root}")

    rows = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    videos: set[str] = set()
    missing_images: list[str] = []
    missing_labels: list[str] = []
    for row in rows:
        parts = Path(row).parts
        if len(parts) < 3 or parts[0] != "training" or parts[1] != "image_02":
            raise ValueError(f"unexpected official split row: {row}")
        video, image_name = parts[-2], parts[-1]
        videos.add(video)
        image_path = image_root / video / image_name
        label_path = label_root / video / f"{Path(image_name).stem}.txt"
        if not image_path.is_file() and len(missing_images) < 20:
            missing_images.append(str(image_path))
        if not label_path.is_file() and len(missing_labels) < 20:
            missing_labels.append(str(label_path))

    expressions = 0
    invalid_json: list[str] = []
    missing_expression_dirs: list[str] = []
    for video in sorted(videos):
        expression_dir = expression_root / video
        if not expression_dir.is_dir():
            missing_expression_dirs.append(video)
            continue
        for path in sorted(expression_dir.glob("*.json")):
            expressions += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                invalid_json.append(str(path))
                continue
            if not path.stem.strip() or not isinstance(payload.get("label"), dict):
                invalid_json.append(str(path))

    test_videos = set(VISDRONE_TEST + UAVDT_TEST)
    overlap = sorted(videos & test_videos)
    report: dict[str, Any] = {
        "format": "locatemot-aerialmind-training-split-audit-v1",
        "dataset_root": str(dataset_root.resolve()),
        "official_split_file": str(split_file.resolve()),
        "official_split_sha256": sha256(split_file),
        "image_rows": len(rows),
        "train_video_count": len(videos),
        "train_videos": sorted(videos),
        "expression_count": expressions,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "missing_expression_dirs": missing_expression_dirs,
        "invalid_json_or_query_files": invalid_json,
        "test_video_overlap": overlap,
        "complete": not (missing_images or missing_labels or missing_expression_dirs or invalid_json or overlap),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/AerialMind")
    parser.add_argument("--split", type=Path,
                        default=Path("/data2/user/reference_envs/aerialmind/datasets/data_path/refer-uav.train"))
    parser.add_argument("--out", type=Path, default=ROOT / "results/audits/aerialmind_training_split_audit.json")
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.split.resolve())
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "complete", "image_rows", "train_video_count", "expression_count",
        "missing_images", "missing_labels", "missing_expression_dirs",
        "invalid_json_or_query_files", "test_video_overlap", "official_split_sha256",
    )} | {"out": str(output)}, indent=2, ensure_ascii=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
