#!/usr/bin/env python3
"""Prepare the official FlexHook Refer-KITTI-V2 data interface.

FlexHook expects a generated ``labels.json`` whose per-object records contain
normalized top-left ``x1,y1,w,h`` boxes and frame-level ``expression_raw``
sentences.  The public Refer-KITTI-V2 release stores the same boxes in
``labels_with_ids`` and keeps expression labels in one JSON file per sentence.
This converter joins those two official sources without copying images.

The generated directory is an external asset, not a repository artifact:

    <output>/KITTI/training -> <image-root>
    <output>/expression    -> <source-root>/expression
    <output>/labels.json
    <output>/gt_template/<video>/<expression>/gt.txt

Only the four official FlexHook kitti-2 test videos receive GT templates;
training and validation read ``labels.json`` directly.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


FLEXHOOK_TRAIN_VIDEOS = [
    "0000", "0001", "0002", "0003", "0004", "0006", "0007", "0008",
    "0009", "0010", "0012", "0014", "0015", "0016", "0018", "0020",
]
FLEXHOOK_TEST_VIDEOS = ["0005", "0011", "0013", "0019"]
ALL_VIDEOS = [f"{index:04d}" for index in range(21)]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _ensure_symlink(path: Path, target: Path) -> None:
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() != target:
            raise RuntimeError(f"existing symlink points elsewhere: {path} -> {path.resolve()}")
        return
    if path.exists():
        raise RuntimeError(f"refusing to replace existing non-symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


def _read_expressions(expression_root: Path) -> tuple[
    dict[str, dict[str, dict[int, list[int]]]],
    dict[str, list[str]],
    dict[str, dict[str, str]],
]:
    labels_by_expression: dict[str, dict[str, dict[int, list[int]]]] = {}
    sentences_by_video: dict[str, list[str]] = {}
    expression_names: dict[str, dict[str, str]] = {}
    for video in ALL_VIDEOS:
        video_root = expression_root / video
        if not video_root.is_dir():
            raise FileNotFoundError(video_root)
        labels_by_expression[video] = {}
        sentences_by_video[video] = []
        expression_names[video] = {}
        for expression_path in sorted(video_root.glob("*.json")):
            payload = json.loads(expression_path.read_text(encoding="utf-8"))
            sentence = str(payload["sentence"])
            if sentence not in sentences_by_video[video]:
                sentences_by_video[video].append(sentence)
            expression_names[video][expression_path.stem] = sentence
            labels_by_expression[video][expression_path.stem] = {
                int(frame): [int(track_id) for track_id in track_ids]
                for frame, track_ids in payload.get("label", {}).items()
            }
    return labels_by_expression, sentences_by_video, expression_names


def _read_boxes(
    label_root: Path,
    image_root: Path,
    expression_labels: dict[str, dict[str, dict[int, list[int]]]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, Any]]]], dict[str, int]]:
    labels: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    frame_counts: dict[str, int] = {}
    for video in ALL_VIDEOS:
        labels[video] = {}
        image_video_root = image_root / video
        label_video_root = label_root / video
        if not image_video_root.is_dir() or not label_video_root.is_dir():
            raise FileNotFoundError(f"missing image/label video root for {video}")

        # Read actual image dimensions once per video.  The KITTI dimensions
        # differ slightly for videos 0014--0020, so do not hard-code them.
        image_paths = sorted(image_video_root.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"no images for {video}")
        with Image.open(image_paths[0]) as image:
            width, height = image.size

        for label_path in sorted(label_video_root.glob("*.txt")):
            frame = int(label_path.stem)
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                fields = raw.split()
                if len(fields) < 6:
                    continue
                track_id = str(int(float(fields[1])))
                x, y, box_w, box_h = (float(value) for value in fields[2:6])
                frame_record = labels[video].setdefault(track_id, {}).setdefault(str(frame), {})
                if "bbox" in frame_record:
                    raise AssertionError(f"duplicate bbox for {video}/{frame}/{track_id}")
                frame_record["bbox"] = [x, y, box_w, box_h]
                frame_record["_pixel_bbox"] = [x * width, y * height, box_w * width, box_h * height]
            frame_counts[video] = frame_counts.get(video, 0) + 1

        # Empty expression lists are intentionally kept.  FlexHook's loader
        # skips those object frames, which is the official behavior for
        # objects without a referring expression at that time.
        for track_frames in labels[video].values():
            for frame_record in track_frames.values():
                frame_record["expression_raw"] = []

    return labels, frame_counts


def _attach_expressions(
    labels: dict[str, dict[str, dict[str, dict[str, Any]]]],
    expression_labels: dict[str, dict[str, dict[int, list[int]]]],
    expression_names: dict[str, dict[str, str]],
) -> tuple[int, list[dict[str, Any]]]:
    missing_count = 0
    missing_examples: list[dict[str, Any]] = []
    for video, expressions in expression_labels.items():
        for expression_id, frame_to_tracks in expressions.items():
            sentence = expression_names[video][expression_id]
            for frame, track_ids in frame_to_tracks.items():
                for track_id in track_ids:
                    record = labels[video].get(str(track_id), {}).get(str(frame))
                    if record is None:
                        # This is also how the official TempRMOT evaluator
                        # behaves: the expression label can name an object
                        # while its per-frame detector label is absent, and
                        # that frame contributes no GT box.  Keep the event
                        # counted and disclosed instead of inventing a box.
                        missing_count += 1
                        if len(missing_examples) < 20:
                            missing_examples.append({
                                "video": video, "frame": frame,
                                "track_id": int(track_id), "expression_id": expression_id,
                            })
                        continue
                    if sentence not in record["expression_raw"]:
                        record["expression_raw"].append(sentence)
    return missing_count, missing_examples


def _write_gt_templates(
    output_root: Path,
    labels: dict[str, dict[str, dict[str, dict[str, Any]]]],
    expression_labels: dict[str, dict[str, dict[int, list[int]]]],
    expression_names: dict[str, dict[str, str]],
) -> int:
    written = 0
    template_root = output_root / "gt_template"
    for video in FLEXHOOK_TEST_VIDEOS:
        for expression_id, frame_to_tracks in expression_labels[video].items():
            output = template_root / video / expression_id / "gt.txt"
            lines: list[str] = []
            for frame in sorted(frame_to_tracks):
                for track_id in sorted(frame_to_tracks[frame]):
                    record = labels[video].get(str(track_id), {}).get(str(frame))
                    if record is None:
                        continue
                    x, y, width, height = record["_pixel_bbox"]
                    # FlexHook/TrackEval uses one-based MOT frame IDs while
                    # the source expression and image files are zero-based.
                    lines.append(
                        f"{frame + 1},{track_id},{x:.6f},{y:.6f},{width:.6f},{height:.6f},1,1,1\n"
                    )
            _write_text_atomic(output, "".join(lines))
            written += 1
    return written


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path("/data2/user/LocateMOT/data/refer_kitti_v2"))
    parser.add_argument("--image-root", type=Path,
                        default=Path("/data2/user/LocateMOT/data/kitti_tracking/training/image_02"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2"))
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()
    expression_root = source_root / "expression"
    label_root = source_root / "labels_with_ids/image_02"
    if not source_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(f"missing source/image root: {source_root}, {image_root}")

    expression_labels, _, expression_names = _read_expressions(expression_root)
    labels, frame_counts = _read_boxes(label_root, image_root, expression_labels)
    missing_target_count, missing_target_examples = _attach_expressions(
        labels, expression_labels, expression_names
    )

    # Private helper keys are useful while producing pixel-space GT but must
    # not leak into FlexHook's public labels.json schema.
    json_labels = json.loads(json.dumps(labels))
    for video in json_labels.values():
        for track in video.values():
            for frame in track.values():
                frame.pop("_pixel_bbox", None)

    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(output_root / "KITTI/training/image_02", image_root)
    _ensure_symlink(output_root / "expression", expression_root)
    _write_json_atomic(output_root / "labels.json", json_labels)
    template_count = _write_gt_templates(output_root, labels, expression_labels, expression_names)
    metadata = {
        "format": "ttarmot-flexhook-refer-kitti-v2",
        "source_root": str(source_root),
        "image_root": str(image_root),
        "box_format": "normalized_x1_y1_w_h_as_used_by_official_TempRMOT_loader",
        "videos": ALL_VIDEOS,
        "flexhook_train_videos": FLEXHOOK_TRAIN_VIDEOS,
        "flexhook_test_videos": FLEXHOOK_TEST_VIDEOS,
        "image_frame_counts": frame_counts,
        "label_video_count": len(json_labels),
        "expression_file_count": sum(len(values) for values in expression_names.values()),
        "gt_template_count": template_count,
        "expression_target_box_miss_count": missing_target_count,
        "expression_target_box_miss_examples": missing_target_examples,
        "raw_images_copied": False,
    }
    _write_json_atomic(output_root / "ttarmot_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
