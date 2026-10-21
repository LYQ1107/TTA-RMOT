#!/usr/bin/env python3
"""Audit the external FlexHook Refer-KITTI-V2 conversion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALL_VIDEOS = [f"{index:04d}" for index in range(21)]
TEST_VIDEOS = ["0005", "0011", "0013", "0019"]
TRAIN_VIDEOS = [
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0018", "0020",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/audits/flexhook_refer_kitti_v2_audit.json",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    metadata = _read_json(dataset_root / "ttarmot_metadata.json")
    labels = _read_json(dataset_root / "labels.json")
    expression_root = dataset_root / "expression"

    if set(labels) != set(ALL_VIDEOS):
        raise AssertionError("labels.json does not contain exactly the 21 source videos")
    if not (dataset_root / "KITTI/training/image_02").is_symlink():
        raise AssertionError("FlexHook image root is not a symlink")
    if not (dataset_root / "expression").is_symlink():
        raise AssertionError("FlexHook expression root is not a symlink")

    expression_count = 0
    source_sentences: dict[str, set[str]] = {video: set() for video in ALL_VIDEOS}
    expression_files: dict[str, list[str]] = {video: [] for video in ALL_VIDEOS}
    for video in ALL_VIDEOS:
        for path in sorted((expression_root / video).glob("*.json")):
            payload = _read_json(path)
            expression_count += 1
            expression_files[video].append(path.stem)
            source_sentences[video].add(str(payload["sentence"]))
    if expression_count != 9758:
        raise AssertionError(f"unexpected expression count: {expression_count}")

    object_count = 0
    frame_count = 0
    attached_expression_count = 0
    for video in ALL_VIDEOS:
        for track_id, frames in labels[video].items():
            object_count += 1
            for frame, record in frames.items():
                frame_count += 1
                bbox = record.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise AssertionError(f"invalid bbox at {video}/{track_id}/{frame}")
                if any(float(value) < 0 or float(value) > 1 for value in bbox):
                    raise AssertionError(f"bbox is not normalized: {video}/{track_id}/{frame}")
                if set(record) != {"bbox", "expression_raw"}:
                    raise AssertionError(f"private/generated fields leaked into labels.json: {record}")
                for sentence in record["expression_raw"]:
                    if sentence not in source_sentences[video]:
                        raise AssertionError(f"unknown expression sentence at {video}/{track_id}/{frame}")
                    attached_expression_count += 1

    # Use an explicit two-level traversal to keep the expected 862 count
    # obvious and to avoid depending on a seqmap file outside this asset.
    template_paths = sorted(
        path for video in TEST_VIDEOS
        for path in (dataset_root / "gt_template" / video).glob("*/gt.txt")
    )
    expected_templates = sum(len(expression_files[video]) for video in TEST_VIDEOS)
    if len(template_paths) != expected_templates or expected_templates != 862:
        raise AssertionError(f"unexpected GT template count: {len(template_paths)} != {expected_templates}")
    if any(not path.is_file() for path in template_paths):
        raise AssertionError("a GT template is missing")

    report = {
        "format": "flexhook-refer-kitti-v2-audit",
        "status": "PASS_WITH_DISCLOSED_SOURCE_LABEL_GAPS",
        "dataset_root": str(dataset_root),
        "box_format": metadata["box_format"],
        "source_expression_count": expression_count,
        "label_video_count": len(labels),
        "label_object_count": object_count,
        "label_frame_record_count": frame_count,
        "attached_expression_reference_count": attached_expression_count,
        "gt_template_count": len(template_paths),
        "official_flexhook_train_videos": TRAIN_VIDEOS,
        "official_flexhook_test_videos": TEST_VIDEOS,
        "train_test_video_disjoint": not (set(TRAIN_VIDEOS) & set(TEST_VIDEOS)),
        "image_root_is_symlink": True,
        "expression_root_is_symlink": True,
        "raw_images_copied": False,
        "source_expression_target_box_miss_count": metadata["expression_target_box_miss_count"],
        "source_expression_target_box_miss_examples": metadata["expression_target_box_miss_examples"],
        "protocol_note": (
            "FlexHook kitti-2 uses its pinned 16-video train list, which differs "
            "from TempRMOT's official 17-video source list; source-label misses "
            "are counted and follow the pinned TempRMOT evaluator's skip behavior."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
