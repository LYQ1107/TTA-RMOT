#!/usr/bin/env python3
"""Build the complete Refer-KITTI-V2 controlled-shift release.

This builder keeps the 21 source videos and all 9,758 expression records in a
single compact manifest, while expanding controlled A0/A1/A2/A3 conditions only
for the official 862-expression test list.  Images and labels remain in the
external data asset directory; only image-root symlinks and metadata are
created in this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
TEST_VIDEOS = ("0005", "0011", "0013", "0019")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _token_hash(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest()


def _canonical_label_map(payload: dict[str, Any]) -> tuple[dict[str, list[int]], str]:
    labels = payload.get("label", {})
    normalized = {
        str(int(frame)): sorted(int(track_id) for track_id in values)
        for frame, values in labels.items()
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return normalized, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_train_videos(train_list: Path) -> tuple[list[str], int]:
    videos: set[str] = set()
    image_count = 0
    pattern = re.compile(r"(?:^|/)image_02/(\d{4})/")
    for raw in train_list.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = pattern.search(line)
        if not match:
            raise ValueError(f"cannot parse official train-list image path: {line}")
        videos.add(match.group(1))
        image_count += 1
    return sorted(videos), image_count


def _load_test_ids(seqmap: Path) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for raw in seqmap.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "+" not in line:
            raise ValueError(f"cannot parse official seqmap entry: {line}")
        video, expression_id = line.split("+", 1)
        result.add((video, expression_id))
    return result


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"existing image symlink points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace non-symlink image root: {link}")
    link.symlink_to(target, target_is_directory=True)


def _runtime_condition(condition: dict[str, Any]) -> dict[str, Any]:
    forbidden = {
        "expression_path", "label_root", "label_format", "language_variant_source_expression_path",
        "source_label_signature",
    }
    return {key: value for key, value in condition.items() if key not in forbidden}


def _variant_rows(base: dict[str, Any], records: list[dict[str, Any]], max_variants: int) -> list[dict[str, Any]]:
    """Select deterministic same-label-map variants, then safe local rewrites."""
    candidates = [
        record for record in records
        if record["expression_id"] != base["expression_id"]
        and record["label_signature"] == base["label_signature"]
        and record["sentence"] != base["sentence"]
    ]
    candidates.sort(key=lambda record: record["expression_id"])
    rows: list[dict[str, Any]] = []
    seen_text: set[str] = {base["sentence"]}
    for record in candidates:
        if record["sentence"] in seen_text or len(rows) >= max_variants:
            continue
        seen_text.add(record["sentence"])
        variant_id = f"same_label__{_safe_id(record['expression_id'])}"
        rows.append({
            "variant_id": variant_id,
            "query_text": record["sentence"],
            "rewrite_rule": "existing_expression_same_label_map",
            "original_token_hash": _token_hash(base["sentence"]),
            "rewrite_token_hash": _token_hash(record["sentence"]),
            "token_hash_equal": _token_hash(base["sentence"]) == _token_hash(record["sentence"]),
            "semantic_equivalence_status": "same_label_map_verified",
            "review_status": "exact_label_map_match",
            "language_variant_source": "existing_same_label_map",
            "source_expression_id": record["expression_id"],
            "source_expression_path": str(record["expression_path"]),
            "source_label_signature": record["label_signature"],
        })
    for rewrite in generate_safe_rewrites(base["sentence"]):
        if len(rows) >= max_variants or rewrite["query_text"] in seen_text:
            continue
        seen_text.add(rewrite["query_text"])
        rows.append({
            **rewrite,
            "language_variant_source": "conservative_rewrite",
            "source_expression_id": None,
            "source_expression_path": None,
            "source_label_signature": base["label_signature"],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data2/user/LocateMOT/data"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/ShiftRMOT-v2")
    parser.add_argument("--train-list", type=Path,
                        default=Path("/data2/user/reference_repos/TempRMOT/datasets/data_path/refer-kitti-v2.train"))
    parser.add_argument("--seqmap", type=Path,
                        default=Path("/data2/user/reference_repos/TempRMOT/datasets/data_path/seqmap.txt"))
    parser.add_argument("--language-variants-per-expression", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=20261021)
    parser.add_argument("--severity", nargs="+", choices=("light", "medium"), default=["light", "medium"])
    args = parser.parse_args()
    if args.language_variants_per_expression < 1:
        raise ValueError("--language-variants-per-expression must be positive")

    data_root = _resolve(args.data_root)
    output_root = _resolve(args.output_root)
    train_list = _resolve(args.train_list)
    seqmap = _resolve(args.seqmap)
    image_source_root = data_root / "kitti_tracking/training/image_02"
    expression_root = data_root / "refer_kitti_v2/expression"
    label_root = data_root / "refer_kitti_v2/labels_with_ids/image_02"
    for required in (image_source_root, expression_root, label_root, train_list, seqmap):
        if not required.exists():
            raise FileNotFoundError(required)

    train_videos, train_image_count = _load_train_videos(train_list)
    test_ids = _load_test_ids(seqmap)
    if set(train_videos) & {video for video, _ in test_ids}:
        raise AssertionError("official train videos and seqmap test videos overlap")
    if args.language_variants_per_expression < 2:
        raise ValueError("the complete benchmark requires at least two requested language variants")

    videos = sorted(path.name for path in image_source_root.iterdir() if path.is_dir())
    if len(videos) != 21:
        raise AssertionError(f"expected 21 Refer-KITTI-V2 videos, found {len(videos)}: {videos}")
    if sorted({video for video, _ in test_ids}) != list(TEST_VIDEOS):
        raise AssertionError(f"unexpected official test videos: {sorted({video for video, _ in test_ids})}")

    manifest_root = output_root / "manifests"
    metadata_root = output_root / "metadata"
    annotation_root = output_root / "annotations"
    image_link_root = output_root / "images"
    base_frame_rows: list[dict[str, Any]] = []
    base_episode_rows: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    runtime_conditions: list[dict[str, Any]] = []
    evaluation_conditions: list[dict[str, Any]] = []
    source_expression_hashes: dict[str, str] = {}
    source_label_hashes: dict[str, str] = {}
    records_by_video: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, Any] = {}

    for video in videos:
        source_images = image_source_root / video
        expression_dir = expression_root / video
        source_labels = label_root / video
        if not expression_dir.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"missing V2 asset directory for video {video}")
        image_paths = sorted(
            (path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: (int(path.stem), path.name),
        )
        if not image_paths:
            raise FileNotFoundError(f"no images for {video}")
        suffixes = {path.suffix.lower() for path in image_paths}
        if len(suffixes) != 1:
            raise ValueError(f"mixed image suffixes in {video}: {suffixes}")
        image_suffix = next(iter(suffixes))
        image_indices = [int(path.stem) for path in image_paths]
        image_index_set = set(image_indices)
        image_link = image_link_root / video
        _ensure_symlink(image_link, source_images)
        for image_path, frame_index in zip(image_paths, image_indices):
            base_frame_rows.append({
                "dataset": "shift_rmot_v2", "video_id": video, "frame_index": frame_index,
                "frame_name": image_path.name, "image_path": str(image_link / image_path.name),
                "image_source_path": str(image_path.resolve()), "image_suffix": image_suffix,
            })

        records: list[dict[str, Any]] = []
        expression_paths = sorted(expression_dir.glob("*.json"), key=lambda path: path.name)
        frame_reference_count = 0
        for expression_path in expression_paths:
            payload = json.loads(expression_path.read_text(encoding="utf-8"))
            sentence = str(payload.get("sentence", "")).strip()
            if not sentence:
                raise ValueError(f"empty sentence: {expression_path}")
            label_map, label_signature = _canonical_label_map(payload)
            evaluation_indices = sorted(int(value) for value in label_map)
            missing_images = sorted(set(evaluation_indices) - image_index_set)
            if missing_images:
                raise FileNotFoundError(f"expression references missing images in {expression_path}: {missing_images[:5]}")
            for frame_index in evaluation_indices:
                label_path = source_labels / f"{frame_index:06d}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"expression references missing label file: {label_path}")
                source_label_hashes[str(label_path.resolve())] = _sha256(label_path)
            expression_id = expression_path.stem
            record = {
                "video_id": video,
                "expression_id": expression_id,
                "sentence": sentence,
                "expression_path": expression_path.resolve(),
                "label_signature": label_signature,
                "label_map": label_map,
                "evaluation_indices": evaluation_indices,
                "runtime_indices": image_indices,
                "image_suffix": image_suffix,
                "image_root": image_link,
                "image_source_root": source_images.resolve(),
                "label_root": source_labels.resolve(),
                "payload": payload,
            }
            records.append(record)
            source_expression_hashes[str(expression_path.resolve())] = _sha256(expression_path)
            frame_reference_count += len(evaluation_indices)
        records_by_video[video] = records
        source_counts[video] = {
            "images": len(image_paths), "expressions": len(records),
            "expression_frame_references": frame_reference_count,
            "official_source_train_video": video in train_videos,
            "official_test_video": video in TEST_VIDEOS,
        }
        for record in records:
            split = "source_train_video" if video in train_videos else "target_test_video"
            is_official_test_expression = (video, record["expression_id"]) in test_ids
            if is_official_test_expression:
                split = "target_test_expression"
            base_episode_id = f"{video}/{record['expression_id']}"
            base_episode_rows.append({
                "dataset": "shift_rmot_v2", "video_id": video, "expression_id": record["expression_id"],
                "base_episode_id": base_episode_id, "query_text": record["sentence"],
                "original_query_text": record["sentence"], "expression_path": str(record["expression_path"]),
                "label_root": str(record["label_root"]),
                "frame_indices": record["evaluation_indices"],
                "runtime_frame_indices": record["runtime_indices"],
                "evaluation_frame_indices": record["evaluation_indices"],
                "frame_count": len(record["evaluation_indices"]),
                "runtime_frame_count": len(record["runtime_indices"]),
                "evaluation_frame_count": len(record["evaluation_indices"]),
                "image_root": str(record["image_root"]),
                "image_source_root": str(record["image_source_root"]),
                "image_suffix": record["image_suffix"], "label_format": "xywh_topleft",
                "trackeval_frame_offset": 1, "split": split,
                "official_source_train_video": video in train_videos,
                "official_test_expression": is_official_test_expression,
                "source_label_signature": record["label_signature"],
                "source_video_name": str(record["payload"].get("video_name", f"KITTI_{video}")),
            })

    test_records = [
        record for video in sorted(records_by_video)
        for record in records_by_video[video]
        if (video, record["expression_id"]) in test_ids
    ]
    if len(test_records) != len(test_ids) or len(test_records) != 862:
        raise AssertionError(f"official seqmap has {len(test_ids)} entries but {len(test_records)} records resolved")

    registry = visual_condition_registry()
    selected_visuals = [
        item for item in registry["conditions"]
        if item["shift_id"] == "identity" or item["severity"] in set(args.severity)
    ]
    selected_visuals.sort(key=lambda item: (item["shift_id"] != "identity", item["shift_id"], item["severity"]))

    variant_coverage: dict[str, int] = {}
    variant_source_counts = {"existing_same_label_map": 0, "conservative_rewrite": 0}
    for record in test_records:
        video = record["video_id"]
        expression_id = record["expression_id"]
        base_episode_id = f"{video}/{expression_id}"
        expression_token = f"{_safe_id(video)}__{_safe_id(expression_id)}"
        variants = _variant_rows(record, records_by_video[video], args.language_variants_per_expression)
        variant_coverage[base_episode_id] = len(variants)
        for variant in variants:
            variant_source_counts[str(variant["language_variant_source"])] += 1
            language_rows.append({
                "dataset": "shift_rmot_v2", "video_id": video, "expression_id": expression_id,
                "base_episode_id": base_episode_id, "original_query_text": record["sentence"],
                **variant,
            })

        for visual in selected_visuals:
            visual_id = str(visual["shift_id"])
            severity = str(visual["severity"])
            visual_basis = (
                f"ShiftRMOT-v2|global_seed={args.global_seed}|video={video}|"
                f"shift={visual_id}|severity={severity}"
            )
            base_condition = {
                "dataset": "shift_rmot_v2", "video_id": video,
                "base_expression_id": expression_id, "base_episode_id": base_episode_id,
                "original_query_text": record["sentence"], "query_text": record["sentence"],
                "frame_indices": record["evaluation_indices"],
                "runtime_frame_indices": record["runtime_indices"],
                "evaluation_frame_indices": record["evaluation_indices"],
                "frame_count": len(record["evaluation_indices"]),
                "runtime_frame_count": len(record["runtime_indices"]),
                "evaluation_frame_count": len(record["evaluation_indices"]),
                "image_root": str(record["image_root"]), "image_suffix": record["image_suffix"],
                "visual_shift": visual_id, "visual_severity": severity,
                "shift_seed_basis": visual_basis,
                "expression_path": str(record["expression_path"]), "label_root": str(record["label_root"]),
                "label_format": "xywh_topleft", "trackeval_frame_offset": 1,
                "split": "target_test_expression", "history_overlap_status": "official_v2_test_seqmap",
            }
            if visual_id == "identity":
                condition_id = f"{expression_token}__A0"
                condition = {**base_condition, "condition_id": condition_id,
                             "runtime_expression_id": condition_id, "condition_family": "A0"}
            else:
                condition_id = f"{expression_token}__A1__{_safe_id(visual_id)}_{_safe_id(severity)}"
                condition = {**base_condition, "condition_id": condition_id,
                             "runtime_expression_id": condition_id, "condition_family": "A1"}
            runtime_conditions.append(_runtime_condition(condition))
            evaluation_conditions.append(condition)

        for variant in variants:
            language_token = _safe_id(str(variant["variant_id"]))
            variant_common = {
                **{key: value for key, value in base_condition.items() if key not in {"visual_shift", "visual_severity", "shift_seed_basis"}},
                "query_text": variant["query_text"], "language_variant_id": variant["variant_id"],
                "language_variant_source": variant["language_variant_source"],
                "language_variant_source_expression_id": variant["source_expression_id"],
                "language_variant_source_expression_path": variant["source_expression_path"],
                "source_label_signature": variant["source_label_signature"],
                "visual_shift": "identity", "visual_severity": "identity",
                "shift_seed_basis": (
                    f"ShiftRMOT-v2|global_seed={args.global_seed}|video={video}|"
                    "shift=identity|severity=identity"
                ),
            }
            condition_id = f"{expression_token}__A2__{language_token}"
            a2 = {**variant_common, "condition_id": condition_id,
                  "runtime_expression_id": condition_id, "condition_family": "A2"}
            runtime_conditions.append(_runtime_condition(a2))
            evaluation_conditions.append(a2)
            for visual in selected_visuals:
                if visual["shift_id"] == "identity":
                    continue
                visual_id = str(visual["shift_id"])
                severity = str(visual["severity"])
                visual_basis = (
                    f"ShiftRMOT-v2|global_seed={args.global_seed}|video={video}|"
                    f"shift={visual_id}|severity={severity}"
                )
                a3_id = f"{expression_token}__A3__{_safe_id(visual_id)}_{_safe_id(severity)}__{language_token}"
                a3 = {**variant_common, "condition_id": a3_id,
                      "runtime_expression_id": a3_id, "condition_family": "A3",
                      "visual_shift": visual_id, "visual_severity": severity,
                      "shift_seed_basis": visual_basis}
                runtime_conditions.append(_runtime_condition(a3))
                evaluation_conditions.append(a3)

    base_frame_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    base_episode_rows.sort(key=lambda row: (row["video_id"], row["expression_id"]))
    language_rows.sort(key=lambda row: (row["video_id"], row["expression_id"], row["variant_id"]))
    runtime_conditions.sort(key=lambda row: (row["video_id"], row["condition_id"]))
    evaluation_conditions.sort(key=lambda row: (row["video_id"], row["condition_id"]))
    _write_jsonl(manifest_root / "base_frames.jsonl", base_frame_rows)
    _write_jsonl(manifest_root / "base_episodes.jsonl", base_episode_rows)
    _write_jsonl(manifest_root / "language_variants.jsonl", language_rows)
    _write_jsonl(manifest_root / "paired_conditions.jsonl", runtime_conditions)
    _write_jsonl(manifest_root / "paired_conditions_evaluation.jsonl", evaluation_conditions)
    _write_jsonl(annotation_root / "rewrite_audit.jsonl", language_rows)
    _write_json(metadata_root / "visual_conditions.json", {
        **registry, "selected_severities": sorted(set(args.severity)),
        "global_seed": args.global_seed,
        "seed_protocol": "sha256(video_id, frame_index, shift_id, severity, global_seed)",
    })
    _write_json(metadata_root / "official_protocol.json", {
        "format": "shift-rmot-official-refer-kitti-v2-protocol-v1",
        "train_list": str(train_list), "train_list_sha256": _sha256(train_list),
        "seqmap": str(seqmap), "seqmap_sha256": _sha256(seqmap),
        "official_train_videos": train_videos, "official_train_image_path_count": train_image_count,
        "official_test_videos": list(TEST_VIDEOS), "official_test_expression_count": len(test_ids),
        "train_test_video_disjoint": True,
    })
    _write_json(metadata_root / "file_hashes.json", {
        "format": "shift-rmot-source-file-hashes-v2",
        "expression_json": source_expression_hashes,
        "label_txt_referenced_by_expression": source_label_hashes,
        "images_are_external_symlink_targets": True,
    })
    _write_json(metadata_root / "split_provenance.json", {
        "format": "shift-rmot-split-provenance-v2", "dataset": "refer_kitti_v2",
        "videos": source_counts, "official_train_videos": train_videos,
        "official_test_videos": list(TEST_VIDEOS),
        "official_test_expression_count": len(test_ids),
        "controlled_conditions_only_for_official_seqmap": True,
        "source_root": str(data_root),
    })
    metadata = {
        "format": "shift-rmot-bench-v2", "dataset": "ShiftRMOT-Bench-v2",
        "source_dataset": "Refer-KITTI-V2", "source_root": str(data_root), "videos": videos,
        "source_counts": source_counts,
        "base_frame_count": len(base_frame_rows), "base_episode_count": len(base_episode_rows),
        "raw_expression_count": len(base_episode_rows),
        "official_test_expression_count": len(test_records),
        "language_variant_count": len(language_rows),
        "language_variant_coverage": {
            "expressions_with_at_least_one": sum(value >= 1 for value in variant_coverage.values()),
            "expressions_with_requested_count": sum(value >= args.language_variants_per_expression for value in variant_coverage.values()),
            "expressions_without_requested_count": sum(value < args.language_variants_per_expression for value in variant_coverage.values()),
            "per_expression_counts": {"min": min(variant_coverage.values()), "max": max(variant_coverage.values()),
                                      "histogram": {str(count): sum(value == count for value in variant_coverage.values())
                                                    for count in sorted(set(variant_coverage.values()))}},
            "source_counts": variant_source_counts,
        },
        "condition_count": len(runtime_conditions),
        "condition_counts": {family: sum(row["condition_family"] == family for row in runtime_conditions)
                             for family in ("A0", "A1", "A2", "A3")},
        "visual_severities_selected": sorted(set(args.severity)),
        "runtime_manifest": str((manifest_root / "paired_conditions.jsonl").resolve()),
        "evaluation_manifest": str((manifest_root / "paired_conditions_evaluation.jsonl").resolve()),
        "runtime_contains_ground_truth": False, "raw_images_copied": False,
        "image_layout": "symlink_to_external_data2", "visual_variants_materialized": False,
        "runtime_time_axis": "all_source_video_images_in_original_order",
        "evaluation_time_axis": "expression_label_keys_only_after_prediction",
        "seed_protocol": "sha256(video_id, frame_index, shift_id, severity, global_seed)",
    }
    _write_json(metadata_root / "dataset_version.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
