#!/usr/bin/env python3
"""Audit ShiftRMOT-Bench-v1 metadata before any model is run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.data.language_rewrites import generate_safe_rewrites  # noqa: E402
from rmot_tta.data.visual_corruptions import apply_visual_shift, visual_condition_registry  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/ShiftRMOT-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/audits/shift_rmot_v1_audit.json")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_root = dataset_root / "manifests"
    runtime_path = manifest_root / "paired_conditions.jsonl"
    evaluation_path = manifest_root / "paired_conditions_evaluation.jsonl"
    base_frames_path = manifest_root / "base_frames.jsonl"
    base_episodes_path = manifest_root / "base_episodes.jsonl"
    language_path = manifest_root / "language_variants.jsonl"
    for path in (runtime_path, evaluation_path, base_frames_path, base_episodes_path, language_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    runtime = _read_jsonl(runtime_path)
    evaluation = _read_jsonl(evaluation_path)
    base_frames = _read_jsonl(base_frames_path)
    base_episodes = _read_jsonl(base_episodes_path)
    language = _read_jsonl(language_path)
    runtime_ids = {str(row["condition_id"]) for row in runtime}
    evaluation_ids = {str(row["condition_id"]) for row in evaluation}
    if len(runtime_ids) != len(runtime) or len(evaluation_ids) != len(evaluation):
        raise AssertionError("condition IDs are not unique")
    if runtime_ids != evaluation_ids:
        raise AssertionError("runtime/evaluation condition IDs differ")

    runtime_forbidden = {"label_path", "expression_path", "gt_box", "gt_id", "label", "target"}
    forbidden_rows = sorted(runtime_forbidden.intersection(runtime[0])) if runtime else []
    if forbidden_rows:
        raise AssertionError(f"runtime compact manifest contains GT fields: {forbidden_rows}")
    missing_images: list[str] = []
    for row in runtime:
        image_root = Path(row["image_root"])
        if not image_root.is_symlink():
            raise AssertionError(f"image root is not a symlink: {image_root}")
        for frame_index in row["frame_indices"]:
            image_path = image_root / f"{int(frame_index):06d}{row['image_suffix']}"
            if not image_path.is_file():
                missing_images.append(str(image_path))
    if missing_images:
        raise FileNotFoundError(f"missing compact-manifest image paths: {missing_images[:5]}")

    first_frame = base_frames[0]
    image = cv2.imread(first_frame["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(first_frame["image_path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shift_checks = []
    for condition in visual_condition_registry()["conditions"]:
        first = apply_visual_shift(image, condition["shift_id"], condition["severity"], "audit-seed")
        second = apply_visual_shift(image, condition["shift_id"], condition["severity"], "audit-seed")
        if first.shape != image.shape or first.dtype != image.dtype or not (first == second).all():
            raise AssertionError(f"visual shift is not shape-preserving/deterministic: {condition}")
        shift_checks.append({**condition, "shape": list(first.shape), "deterministic": True})

    unsafe_language = []
    for row in language:
        if row["original_token_hash"] == row["rewrite_token_hash"]:
            raise AssertionError(f"rewrite did not change token hash: {row}")
        generated = generate_safe_rewrites(row["original_query_text"])
        if not any(candidate["query_text"] == row["query_text"] for candidate in generated):
            raise AssertionError(f"language row is not reproducible: {row}")
        for token in ("car", "vehicle", "moving", "parked", "not", "without"):
            if token in row["original_query_text"].lower() or token in row["query_text"].lower():
                unsafe_language.append(row["variant_id"])

    link_roots = sorted({str(Path(row["image_root"])) for row in runtime})
    report = {
        "format": "shift-rmot-v1-audit", "status": "PASS", "dataset_root": str(dataset_root),
        "base_frame_count": len(base_frames), "base_episode_count": len(base_episodes),
        "language_variant_count": len(language), "condition_count": len(runtime),
        "condition_counts": {family: sum(row["condition_family"] == family for row in runtime)
                             for family in ("A0", "A1", "A2", "A3")},
        "runtime_evaluation_condition_ids_equal": True,
        "runtime_contains_ground_truth_fields": bool(forbidden_rows),
        "raw_image_copy_check": {"passed": True, "image_roots_are_symlinks": link_roots},
        "visual_shift_checks": shift_checks, "unsafe_language_rows": unsafe_language,
        "complete_source_video_manifest": sorted({str(row["video_id"]) for row in base_episodes}),
        "protocol_note": "A0/A1/A2/A3 are compact condition rows; images are shared symlink targets and shifts are generated online.",
    }
    if unsafe_language:
        raise AssertionError(f"unsafe language rewrite rows: {unsafe_language[:5]}")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
