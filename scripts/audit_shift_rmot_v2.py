#!/usr/bin/env python3
"""Audit the complete Refer-KITTI-V2 controlled-shift release."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.data.shift_loader import _expand  # noqa: E402
from rmot_tta.data.visual_corruptions import apply_visual_shift, visual_condition_registry  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _label_signature(payload: dict[str, Any]) -> str:
    normalized = {
        str(int(frame)): sorted(int(track_id) for track_id in values)
        for frame, values in payload.get("label", {}).items()
    }
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/ShiftRMOT-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/audits/shift_rmot_v2_audit.json")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_root = dataset_root / "manifests"
    metadata_root = dataset_root / "metadata"
    runtime = _read_jsonl(manifest_root / "paired_conditions.jsonl")
    evaluation = _read_jsonl(manifest_root / "paired_conditions_evaluation.jsonl")
    base_frames = _read_jsonl(manifest_root / "base_frames.jsonl")
    base_episodes = _read_jsonl(manifest_root / "base_episodes.jsonl")
    language = _read_jsonl(manifest_root / "language_variants.jsonl")
    metadata = json.loads((metadata_root / "dataset_version.json").read_text(encoding="utf-8"))
    protocol = json.loads((metadata_root / "official_protocol.json").read_text(encoding="utf-8"))

    runtime_ids = {str(row["condition_id"]) for row in runtime}
    evaluation_ids = {str(row["condition_id"]) for row in evaluation}
    if len(runtime_ids) != len(runtime) or len(evaluation_ids) != len(evaluation):
        raise AssertionError("condition IDs are not unique")
    if runtime_ids != evaluation_ids:
        raise AssertionError("runtime/evaluation condition IDs differ")

    runtime_forbidden = {
        "label_path", "expression_path", "gt_box", "gt_id", "label", "target", "label_root",
        "language_variant_source_expression_path", "source_label_signature",
    }
    runtime_keys = set().union(*(set(row) for row in runtime)) if runtime else set()
    forbidden_rows = sorted(runtime_forbidden.intersection(runtime_keys))
    if forbidden_rows:
        raise AssertionError(f"runtime compact manifest contains GT/provenance fields: {forbidden_rows}")

    frames_by_video: dict[str, list[int]] = defaultdict(list)
    image_roots: set[str] = set()
    image_source_paths: set[str] = set()
    for row in base_frames:
        frames_by_video[str(row["video_id"])].append(int(row["frame_index"]))
        image_root = Path(row["image_path"]).parent
        image_roots.add(str(image_root))
        image_source_paths.add(str(row["image_source_path"]))
        if not image_root.is_symlink():
            raise AssertionError(f"image root is not a symlink: {image_root}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    for values in frames_by_video.values():
        values.sort()
    if len(image_source_paths) != len(base_frames):
        raise AssertionError("base frame manifest contains duplicate image source paths")
    image_root_entries = list((dataset_root / "images").iterdir())
    if any(not path.is_symlink() for path in image_root_entries):
        raise AssertionError("image roots contain materialized entries instead of symlinks")

    base_by_episode = {(str(row["video_id"]), str(row["expression_id"])): row for row in base_episodes}
    target_episodes = {
        (str(row["video_id"]), str(row["expression_id"])): row
        for row in base_episodes if row.get("official_test_expression")
    }
    if len(base_episodes) != 9758 or len(target_episodes) != 862:
        raise AssertionError(f"unexpected expression counts: all={len(base_episodes)}, target={len(target_episodes)}")
    for row in runtime:
        key = (str(row["video_id"]), str(row["base_expression_id"]))
        expected_runtime = frames_by_video[str(row["video_id"])]
        if [int(value) for value in row["runtime_frame_indices"]] != expected_runtime:
            raise AssertionError(f"runtime frame axis is not complete for {row['condition_id']}")
        expected_eval = base_by_episode[key]["evaluation_frame_indices"]
        if [int(value) for value in row["evaluation_frame_indices"]] != [int(value) for value in expected_eval]:
            raise AssertionError(f"evaluation frame axis differs from source label keys for {row['condition_id']}")
        if not set(row["evaluation_frame_indices"]).issubset(set(expected_runtime)):
            raise AssertionError(f"evaluation frame is outside source video for {row['condition_id']}")

    language_by_base: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in language:
        key = (str(row["video_id"]), str(row["expression_id"]))
        language_by_base[key][str(row["variant_id"])] = row
        if row["query_text"] == row["original_query_text"]:
            raise AssertionError(f"language variant did not change text: {row}")
        if row["language_variant_source"] == "existing_same_label_map":
            source_path = Path(str(row["source_expression_path"]))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            if _label_signature(source_payload) != str(row["source_label_signature"]):
                raise AssertionError(f"natural language source signature changed: {row['variant_id']}")

    condition_by_id = {str(row["condition_id"]): row for row in evaluation}
    a0 = {(str(row["video_id"]), str(row["base_expression_id"])): row
          for row in evaluation if row["condition_family"] == "A0"}
    a1 = {(str(row["video_id"]), str(row["base_expression_id"]), str(row["visual_shift"]), str(row["visual_severity"])): row
          for row in evaluation if row["condition_family"] == "A1"}
    a2 = {(str(row["video_id"]), str(row["base_expression_id"]), str(row["language_variant_id"])): row
          for row in evaluation if row["condition_family"] == "A2"}
    a3 = {(str(row["video_id"]), str(row["base_expression_id"]), str(row["visual_shift"]),
           str(row["visual_severity"]), str(row["language_variant_id"])): row
          for row in evaluation if row["condition_family"] == "A3"}
    if len(a0) != 862 or len(a1) != 862 * 8 or len(a2) != len(language) or len(a3) != len(language) * 8:
        raise AssertionError(f"unexpected condition family cardinality: A0={len(a0)}, A1={len(a1)}, A2={len(a2)}, A3={len(a3)}")

    pair_checks = {"a1_a3_visual_seed_equal": True, "a0_a2_identity_seed_equal": True,
                   "a2_a3_query_equal": True, "sampled_pixels_equal": True}
    sample_image_by_video: dict[str, Any] = {}
    for key, row in a1.items():
        video, expression_id, visual, severity = key
        for variant_id, variant in language_by_base[(video, expression_id)].items():
            joint = a3[(video, expression_id, visual, severity, variant_id)]
            if row["shift_seed_basis"] != joint["shift_seed_basis"]:
                raise AssertionError(f"A1/A3 visual seed differs: {key}, {variant_id}")
            if a2[(video, expression_id, variant_id)]["query_text"] != joint["query_text"]:
                raise AssertionError(f"A2/A3 query differs: {key}, {variant_id}")
        if video not in sample_image_by_video:
            first_frame = frames_by_video[video][0]
            source = next(item for item in base_frames if item["video_id"] == video and int(item["frame_index"]) == first_frame)
            image = cv2.imread(str(source["image_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(source["image_path"])
            sample_image_by_video[video] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = sample_image_by_video[video]
        seed = f"{row['shift_seed_basis']}|frame={frames_by_video[video][0]}"
        left = apply_visual_shift(image, visual, severity, seed)
        right = apply_visual_shift(image, visual, severity, seed)
        if not (left.shape == image.shape and left.dtype == image.dtype and (left == right).all()):
            raise AssertionError(f"visual transform is not deterministic: {key}")
    for key, row in a0.items():
        video, expression_id = key
        for variant_id, variant in language_by_base[key].items():
            if row["shift_seed_basis"] != a2[(video, expression_id, variant_id)]["shift_seed_basis"]:
                raise AssertionError(f"A0/A2 identity seed differs: {key}, {variant_id}")

    # Target-set statistics are computed from the official expression JSON, not
    # used by the runtime or router.
    empty = multi = dynamic = 0
    target_track_ids: dict[str, set[int]] = defaultdict(set)
    for (video, expression_id), episode in target_episodes.items():
        payload = json.loads(Path(episode["expression_path"]).read_text(encoding="utf-8"))
        label = {int(frame): sorted(int(track_id) for track_id in values)
                 for frame, values in payload.get("label", {}).items()}
        sets = [set(values) for values in label.values()]
        union = set().union(*sets) if sets else set()
        target_track_ids[video].update(union)
        empty += not union
        multi += any(len(values) > 1 for values in sets)
        dynamic += len({tuple(sorted(values)) for values in sets}) > 1
    target_count = len(target_episodes)
    report = {
        "format": "shift-rmot-v2-audit",
        "status": "PASS_WITH_DISCLOSED_LANGUAGE_COVERAGE_GAP",
        "dataset_root": str(dataset_root),
        "official_protocol": protocol,
        "base_video_count": len(frames_by_video), "base_frame_count": len(base_frames),
        "base_expression_count": len(base_episodes), "official_test_expression_count": target_count,
        "language_variant_count": len(language), "condition_count": len(runtime),
        "condition_counts": {family: sum(row["condition_family"] == family for row in runtime)
                             for family in ("A0", "A1", "A2", "A3")},
        "runtime_evaluation_condition_ids_equal": True,
        "runtime_contains_ground_truth_fields": bool(forbidden_rows),
        "time_axis_check": {
            "runtime": "all_source_video_images_in_original_order",
            "evaluation": "official_expression_label_keys",
            "passed": True,
        },
        "raw_image_copy_check": {"passed": True, "image_root_count": len(image_roots),
                                  "unique_source_images": len(image_source_paths)},
        "paired_condition_checks": pair_checks,
        "language_coverage": metadata["language_variant_coverage"],
        "target_set_statistics": {
            "empty_expression_count": empty, "multi_target_expression_count": multi,
            "dynamic_target_set_expression_count": dynamic,
            "unique_gt_track_ids_by_video": {video: len(ids) for video, ids in sorted(target_track_ids.items())},
        },
        "official_test_videos": sorted({video for video, _ in target_episodes}),
        "protocol_note": "A0/A1/A2/A3 share source frames and labels; visual shifts are generated online; runtime never receives GT paths.",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
