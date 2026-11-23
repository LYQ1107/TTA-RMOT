#!/usr/bin/env python3
"""Audit a complete AerialMind release before any model evaluation.

The audit follows the official release semantics: every JSON file is one
expression, its filename is the query text, and UAV/VisDrone labels use
top-left ``xywh`` while UAVDT labels use center ``cxcywh``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

VISDRONE_TEST = [
    "uav0000009_03358_v", "uav0000073_00600_v", "uav0000073_04464_v",
    "uav0000077_00720_v", "uav0000088_00290_v", "uav0000119_02301_v",
    "uav0000120_04775_v", "uav0000161_00000_v", "uav0000188_00000_v",
    "uav0000201_00000_v", "uav0000249_00001_v", "uav0000249_02688_v",
    "uav0000297_00000_v", "uav0000297_02761_v", "uav0000306_00230_v",
    "uav0000355_00001_v", "uav0000370_00001_v",
]
UAVDT_TEST = ["M0203", "M0205", "M0209", "M0403", "M0701", "M0801", "M1001",
              "M1004", "M1007", "M1101", "M1301", "M1302", "M1401"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _frame_files(directory: Path, suffixes: set[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.suffix.lower() not in suffixes:
            continue
        try:
            frame = int(path.stem)
        except ValueError:
            continue
        if frame in result:
            raise ValueError(f"duplicate frame number in {directory}: {result[frame]} and {path}")
        result[frame] = path
    return result


def _format_for(video: str) -> str:
    return "xywh_topleft" if video.startswith("uav") else "cxcywh"


def _read_label_ids(path: Path) -> tuple[set[int], list[str]]:
    ids: set[int] = set()
    errors: list[str] = []
    if not path.is_file():
        return ids, ["missing"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if len(fields) < 6:
            errors.append(f"line {line_number}: fewer than 6 fields")
            continue
        try:
            track_id = int(float(fields[1]))
            values = [float(value) for value in fields[2:6]]
        except ValueError:
            errors.append(f"line {line_number}: non-numeric fields")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: non-finite box")
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"line {line_number}: normalized box outside [0,1]")
        if values[2] <= 0 or values[3] <= 0:
            errors.append(f"line {line_number}: non-positive box size")
        ids.add(track_id)
    return ids, errors


def audit(root: Path) -> dict[str, Any]:
    image_root, label_root, expression_root = (root / name for name in ("image_02", "labels_with_ids", "expression"))
    for path in (image_root, label_root, expression_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    roots = {
        name: {path.name for path in path_root.iterdir() if path.is_dir()}
        for name, path_root in (("image", image_root), ("label", label_root), ("expression", expression_root))
    }
    videos = sorted(roots["image"] | roots["label"] | roots["expression"])
    report: dict[str, Any] = {
        "format": "locatemot-aerialmind-audit-v1",
        "root": str(root.resolve()),
        "video_count": len(videos),
        "expected_video_count": 93,
        "test_partition": {"visdrone": VISDRONE_TEST, "uavdt": UAVDT_TEST},
        "videos": {},
        "errors": [],
        "root_video_sets_match": roots["image"] == roots["label"] == roots["expression"],
        "complete_for_referenced_frames": True,
        "all_image_frames_have_labels": True,
    }
    if not report["root_video_sets_match"]:
        report["errors"].append({"type": "root_video_set_mismatch", "image_only": sorted(roots["image"] - roots["label"] - roots["expression"]),
                                 "label_only": sorted(roots["label"] - roots["image"] - roots["expression"]),
                                 "expression_only": sorted(roots["expression"] - roots["image"] - roots["label"])})
    label_cache: dict[Path, tuple[set[int], list[str]]] = {}
    total = Counter()
    for video in videos:
        image_dir, label_dir, expression_dir = (root / name / video for name in ("image_02", "labels_with_ids", "expression"))
        image_frames = _frame_files(image_dir, IMAGE_SUFFIXES) if image_dir.is_dir() else {}
        label_frames = _frame_files(label_dir, {".txt"}) if label_dir.is_dir() else {}
        expression_paths = sorted(expression_dir.glob("*.json")) if expression_dir.is_dir() else []
        missing_labels = sorted(set(image_frames) - set(label_frames))
        if missing_labels:
            report["all_image_frames_have_labels"] = False
        video_report: dict[str, Any] = {
            "label_format": _format_for(video),
            "image_frames": len(image_frames),
            "label_frames": len(label_frames),
            "missing_labels_for_images": missing_labels[:20],
            "missing_labels_for_images_count": len(missing_labels),
            "expression_count": len(expression_paths),
            "expression_frame_references": 0,
            "target_references": 0,
            "missing_images_for_references": [],
            "missing_labels_for_references": [],
            "missing_target_ids": [],
            "label_format_errors": [],
        }
        for label_path in sorted(label_frames.values()):
            if label_path not in label_cache:
                label_cache[label_path] = _read_label_ids(label_path)
            errors = label_cache[label_path][1]
            if errors and len(video_report["label_format_errors"]) < 20:
                video_report["label_format_errors"].extend(f"{label_path.name}: {error}" for error in errors[:20])
        for expression_path in expression_paths:
            try:
                payload = json.loads(expression_path.read_text(encoding="utf-8"))
            except Exception as exc:
                report["errors"].append(f"{expression_path}: invalid JSON: {exc}")
                continue
            labels = payload.get("label", {})
            if not isinstance(labels, dict):
                report["errors"].append(f"{expression_path}: label is not an object")
                continue
            if not str(expression_path.stem).strip():
                report["errors"].append(f"{expression_path}: empty filename query")
            for frame_key, target_values in labels.items():
                try:
                    frame = int(frame_key)
                    target_ids = {int(value) for value in target_values}
                except (TypeError, ValueError):
                    report["errors"].append(f"{expression_path}: invalid frame/target entry {frame_key!r}")
                    continue
                video_report["expression_frame_references"] += 1
                video_report["target_references"] += len(target_ids)
                if frame not in image_frames and len(video_report["missing_images_for_references"]) < 20:
                    video_report["missing_images_for_references"].append(frame)
                label_path = label_frames.get(frame)
                if label_path is None:
                    if len(video_report["missing_labels_for_references"]) < 20:
                        video_report["missing_labels_for_references"].append(frame)
                    continue
                if label_path not in label_cache:
                    label_cache[label_path] = _read_label_ids(label_path)
                available_ids = label_cache[label_path][0]
                missing_ids = sorted(target_ids - available_ids)
                if missing_ids and len(video_report["missing_target_ids"]) < 20:
                    video_report["missing_target_ids"].append({"frame": frame, "ids": missing_ids})
        # An expression may keep a persistent target ID while that object is
        # occluded or outside the frame; absence of that ID in a frame label
        # is not a missing dataset file.  Missing image/label files are hard
        # completeness failures and are checked separately.
        if video_report["missing_images_for_references"] or video_report["missing_labels_for_references"]:
            report["complete_for_referenced_frames"] = False
        total["image_frames"] += video_report["image_frames"]
        total["label_frames"] += video_report["label_frames"]
        total["expressions"] += video_report["expression_count"]
        total["expression_frame_references"] += video_report["expression_frame_references"]
        total["target_references"] += video_report["target_references"]
        report["videos"][video] = video_report
    report["totals"] = dict(total)
    report["test_partition_present"] = all(video in report["videos"] for video in VISDRONE_TEST + UAVDT_TEST)
    report["error_count"] = len(report["errors"])
    report["complete"] = bool(
        report["video_count"] == report["expected_video_count"]
        and report["test_partition_present"]
        and report["root_video_sets_match"]
        and report["complete_for_referenced_frames"]
        and report["error_count"] == 0
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/AerialMind")
    parser.add_argument("--out", type=Path, default=ROOT / "results/audits/aerialmind_dataset_audit.json")
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete" if report["complete"] else "failed",
        "out": str(args.out.resolve()), "video_count": report["video_count"],
        "totals": report["totals"], "complete_for_referenced_frames": report["complete_for_referenced_frames"],
        "all_image_frames_have_labels": report["all_image_frames_have_labels"],
        "error_count": report["error_count"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
