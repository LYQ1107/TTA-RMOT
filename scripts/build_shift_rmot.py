#!/usr/bin/env python3
"""Build the compact, zero-duplicate ShiftRMOT-Bench-v1 release.

The source Refer-KITTI-V2 files stay in the external data2 asset directory.
This command creates only symlinked image roots and compact JSON/JSONL metadata
inside the TTA-RMOT workspace; visual variants are generated online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.data.language_rewrites import generate_safe_rewrites  # noqa: E402
from rmot_tta.data.visual_corruptions import visual_condition_registry  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_VIDEOS = ("0005", "0011", "0013", "0019")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _split(video: str) -> tuple[str, str]:
    if video == "0005":
        return "dev", "official_test_video; historical-use-overlap-audit-required"
    return "confirmation_overlap_audit", "official_test_video; historical-use-overlap-audit-required"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _ensure_image_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"existing image symlink points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace non-symlink image root: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data2/user/LocateMOT/data"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/ShiftRMOT-v1")
    parser.add_argument("--videos", nargs="*", default=list(DEFAULT_VIDEOS))
    parser.add_argument("--severity", choices=("light", "medium"), default="medium")
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    output_root = _resolve(args.output_root)
    image_source_root = data_root / "kitti_tracking/training/image_02"
    expression_root = data_root / "refer_kitti_v2/expression"
    label_root = data_root / "refer_kitti_v2/labels_with_ids/image_02"
    for required in (image_source_root, expression_root, label_root):
        if not required.is_dir():
            raise FileNotFoundError(required)

    image_link_root = output_root / "images"
    manifest_root = output_root / "manifests"
    metadata_root = output_root / "metadata"
    annotation_root = output_root / "annotations"
    base_frame_rows: list[dict[str, Any]] = []
    base_episode_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    runtime_conditions: list[dict[str, Any]] = []
    evaluation_conditions: list[dict[str, Any]] = []
    source_expression_hashes: dict[str, str] = {}
    source_label_hashes: dict[str, str] = {}
    source_counts: dict[str, Any] = {}

    registry = visual_condition_registry()
    selected_visuals = [item for item in registry["conditions"]
                        if item["shift_id"] == "identity" or item["severity"] == args.severity]

    for video in [str(value) for value in args.videos]:
        source_images = image_source_root / video
        expression_dir = expression_root / video
        source_labels = label_root / video
        if not source_images.is_dir() or not expression_dir.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"missing V2 asset directory for video {video}")
        image_paths = sorted((path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
                             key=lambda path: (int(path.stem), path.name))
        if not image_paths:
            raise FileNotFoundError(f"no images for {video}")
        suffixes = {path.suffix.lower() for path in image_paths}
        if len(suffixes) != 1:
            raise ValueError(f"mixed image suffixes in {video}: {suffixes}")
        image_suffix = next(iter(suffixes))
        image_link = image_link_root / video
        _ensure_image_symlink(image_link, source_images)
        split, overlap_status = _split(video)
        for image_path in image_paths:
            frame_index = int(image_path.stem)
            base_frame_rows.append({
                "dataset": "shift_rmot_v1", "video_id": video, "frame_index": frame_index,
                "frame_name": image_path.name, "image_path": str(image_link / image_path.name),
                "image_source_path": str(image_path.resolve()), "image_suffix": image_suffix,
            })
        expression_paths = sorted(expression_dir.glob("*.json"), key=lambda path: path.name)
        expression_count = 0
        frame_reference_count = 0
        for expression_path in expression_paths:
            payload = json.loads(expression_path.read_text(encoding="utf-8"))
            sentence = str(payload.get("sentence", "")).strip()
            if not sentence:
                raise ValueError(f"empty sentence: {expression_path}")
            frame_indices = sorted(int(value) for value in payload.get("label", {}).keys())
            image_by_frame = {int(path.stem): path for path in image_paths}
            missing_images = sorted(set(frame_indices) - set(image_by_frame))
            if missing_images:
                raise FileNotFoundError(f"expression references missing images in {expression_path}: {missing_images[:5]}")
            for frame_index in frame_indices:
                label_path = source_labels / f"{frame_index:06d}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"expression references missing label file: {label_path}")
                source_label_hashes[str(label_path.resolve())] = sha256_file(label_path)
            expression_id = expression_path.stem
            base_episode_id = f"{video}/{expression_id}"
            expression_token = f"{_safe_id(video)}__{_safe_id(expression_id)}"
            base_episode_rows.append({
                "dataset": "shift_rmot_v1", "video_id": video, "expression_id": expression_id,
                "base_episode_id": base_episode_id, "query_text": sentence,
                "original_query_text": sentence, "expression_path": str(expression_path.resolve()),
                "label_root": str(source_labels.resolve()), "frame_indices": frame_indices,
                "frame_count": len(frame_indices), "image_root": str(image_link),
                "image_source_root": str(source_images.resolve()), "image_suffix": image_suffix,
                "label_format": "cxcywh", "trackeval_frame_offset": 1,
                "split": split, "history_overlap_status": overlap_status,
                "source_video_name": str(payload.get("video_name", f"KITTI_{video}")),
            })
            source_expression_hashes[str(expression_path.resolve())] = sha256_file(expression_path)
            rewrites = generate_safe_rewrites(sentence)
            for rewrite in rewrites:
                language_rows.append({
                    "dataset": "shift_rmot_v1", "video_id": video, "expression_id": expression_id,
                    "base_episode_id": base_episode_id, "original_query_text": sentence,
                    **rewrite,
                })
            language_by_id = {str(row["variant_id"]): row for row in rewrites}
            for visual in selected_visuals:
                visual_id = str(visual["shift_id"])
                severity = str(visual["severity"])
                visual_token = visual_id if severity == "identity" else f"{visual_id}_{severity}"
                base_condition = {
                    "dataset": "shift_rmot_v1", "video_id": video, "base_expression_id": expression_id,
                    "base_episode_id": base_episode_id, "original_query_text": sentence,
                    "query_text": sentence, "frame_indices": frame_indices, "frame_count": len(frame_indices),
                    "image_root": str(image_link), "image_suffix": image_suffix,
                    "visual_shift": visual_id, "visual_severity": severity,
                    "shift_seed_basis": f"ShiftRMOT-v1|{video}|{expression_id}|{visual_token}",
                    "expression_path": str(expression_path.resolve()), "label_root": str(source_labels.resolve()),
                    "label_format": "cxcywh", "trackeval_frame_offset": 1,
                    "split": split, "history_overlap_status": overlap_status,
                }
                condition_id = f"{expression_token}__A0" if visual_id == "identity" else f"{expression_token}__A1__{_safe_id(visual_token)}"
                condition = {**base_condition, "condition_id": condition_id,
                             "runtime_expression_id": condition_id, "condition_family": "A0" if visual_id == "identity" else "A1"}
                runtime_conditions.append({key: value for key, value in condition.items()
                                           if key not in {"expression_path", "label_root", "label_format"}})
                evaluation_conditions.append(condition)
            for variant_id, rewrite in language_by_id.items():
                language_token = _safe_id(variant_id)
                condition_id = f"{expression_token}__A2__{language_token}"
                condition = {
                    "dataset": "shift_rmot_v1", "video_id": video, "base_expression_id": expression_id,
                    "base_episode_id": base_episode_id, "condition_id": condition_id,
                    "runtime_expression_id": condition_id, "condition_family": "A2",
                    "original_query_text": sentence, "query_text": rewrite["query_text"],
                    "frame_indices": frame_indices, "frame_count": len(frame_indices),
                    "image_root": str(image_link), "image_suffix": image_suffix,
                    "visual_shift": "identity", "visual_severity": "identity",
                    "shift_seed_basis": f"ShiftRMOT-v1|{video}|{expression_id}|{condition_id}",
                    "language_variant_id": variant_id, "rewrite_rule": rewrite["rewrite_rule"],
                    "expression_path": str(expression_path.resolve()), "label_root": str(source_labels.resolve()),
                    "label_format": "cxcywh", "trackeval_frame_offset": 1,
                    "split": split, "history_overlap_status": overlap_status,
                }
                runtime_conditions.append({key: value for key, value in condition.items()
                                           if key not in {"expression_path", "label_root", "label_format"}})
                evaluation_conditions.append(condition)
                for visual in selected_visuals:
                    if visual["shift_id"] == "identity":
                        continue
                    visual_id = str(visual["shift_id"])
                    severity = str(visual["severity"])
                    visual_token = f"{visual_id}_{severity}"
                    a3_id = f"{expression_token}__A3__{_safe_id(visual_token)}__{language_token}"
                    a3 = {**condition, "condition_id": a3_id, "runtime_expression_id": a3_id,
                          "condition_family": "A3", "visual_shift": visual_id,
                          "visual_severity": severity,
                          "shift_seed_basis": f"ShiftRMOT-v1|{video}|{expression_id}|{a3_id}"}
                    runtime_conditions.append({key: value for key, value in a3.items()
                                               if key not in {"expression_path", "label_root", "label_format"}})
                    evaluation_conditions.append(a3)
            expression_count += 1
            frame_reference_count += len(frame_indices)
        source_counts[video] = {
            "images": len(image_paths), "expressions": expression_count,
            "expression_frame_references": frame_reference_count,
            "split": split, "history_overlap_status": overlap_status,
        }

    base_frame_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    base_episode_rows.sort(key=lambda row: (row["video_id"], row["expression_id"]))
    language_rows.sort(key=lambda row: (row["video_id"], row["expression_id"], row["variant_id"]))
    runtime_conditions.sort(key=lambda row: (row["video_id"], row["runtime_expression_id"]))
    evaluation_conditions.sort(key=lambda row: (row["video_id"], row["runtime_expression_id"]))
    _jsonl_write(manifest_root / "base_frames.jsonl", base_frame_rows)
    _jsonl_write(manifest_root / "base_episodes.jsonl", base_episode_rows)
    _jsonl_write(manifest_root / "language_variants.jsonl", language_rows)
    _jsonl_write(manifest_root / "paired_conditions.jsonl", runtime_conditions)
    _jsonl_write(manifest_root / "paired_conditions_evaluation.jsonl", evaluation_conditions)
    _jsonl_write(annotation_root / "rewrite_audit.jsonl", language_rows)
    _json_write(metadata_root / "visual_conditions.json", registry)
    _json_write(metadata_root / "file_hashes.json", {
        "format": "shift-rmot-source-file-hashes-v1",
        "expression_json": source_expression_hashes,
        "label_txt_referenced_by_expression": source_label_hashes,
        "images_are_external_symlink_targets": True,
    })
    _json_write(metadata_root / "split_provenance.json", {
        "format": "shift-rmot-split-provenance-v1", "dataset": "refer_kitti_v2",
        "videos": source_counts,
        "dev_videos": [video for video in args.videos if str(video) == "0005"],
        "confirmation_videos": [str(video) for video in args.videos if str(video) != "0005"],
        "overlap_policy": "official V2 test videos are retained, but historical prior use is disclosed; confirmation is not claimed unseen",
        "source_root": str(data_root),
    })
    metadata = {
        "format": "shift-rmot-bench-v1",
        "dataset": "ShiftRMOT-Bench-v1",
        "source_dataset": "Refer-KITTI-V2",
        "source_root": str(data_root), "videos": [str(value) for value in args.videos],
        "source_counts": source_counts,
        "base_frame_count": len(base_frame_rows), "base_episode_count": len(base_episode_rows),
        "language_variant_count": len(language_rows),
        "condition_count": len(runtime_conditions),
        "condition_counts": {family: sum(row["condition_family"] == family for row in runtime_conditions)
                             for family in ("A0", "A1", "A2", "A3")},
        "visual_severity_selected": args.severity,
        "runtime_manifest": str((manifest_root / "paired_conditions.jsonl").resolve()),
        "evaluation_manifest": str((manifest_root / "paired_conditions_evaluation.jsonl").resolve()),
        "runtime_contains_ground_truth": False,
        "raw_images_copied": False, "image_layout": "symlink_to_external_data2",
        "visual_variants_materialized": False,
        "seed_protocol": "sha256(ShiftRMOT-v1|video|expression|condition) first 8 bytes",
    }
    _json_write(metadata_root / "dataset_version.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
