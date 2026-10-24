#!/usr/bin/env python3
"""Generate DanceTrack validation tracks with an image-only public tracker.

The tracker is deliberately independent of Refer-Dance labels and GT.  It is
used for the requested cross-domain protocol where the second-stage referring
weight is fixed to Refer-KITTI-V2 but the detector/tracker may change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


DEFAULT_VIDEOS = [
    "dancetrack0081", "dancetrack0063", "dancetrack0047", "dancetrack0035",
    "dancetrack0097", "dancetrack0043", "dancetrack0010", "dancetrack0007",
    "dancetrack0090", "dancetrack0058", "dancetrack0065", "dancetrack0041",
    "dancetrack0079", "dancetrack0018", "dancetrack0004", "dancetrack0094",
    "dancetrack0030", "dancetrack0077", "dancetrack0019", "dancetrack0005",
    "dancetrack0034", "dancetrack0026", "dancetrack0014", "dancetrack0025",
    "dancetrack0073",
]


def _videos(value: str | None) -> list[str]:
    return DEFAULT_VIDEOS if not value else [item.strip() for item in value.split(",") if item.strip()]


def _write_result(path: Path, results: Iterable, frame_count: int) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = 0
    frames_with_tracks = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for frame_id, result in enumerate(results, start=1):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue
            xywh = boxes.xywh.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            track_ids = boxes.id.detach().cpu().numpy().astype(int)
            if len(track_ids) == 0:
                continue
            frames_with_tracks += 1
            for (x, y, width, height), confidence, track_id in zip(xywh, confidences, track_ids):
                # MOT/DanceTrack uses 1-based frame IDs and xywh boxes.
                handle.write(
                    f"{frame_id},{int(track_id)},{float(x - width / 2):.4f},"
                    f"{float(y - height / 2):.4f},{float(width):.4f},"
                    f"{float(height):.4f},{float(confidence):.6f},-1,-1,-1\n"
                )
                rows += 1
    temporary.replace(path)
    if frame_id != frame_count:
        raise RuntimeError(f"tracker emitted {frame_id} frames, expected {frame_count} for {path.stem}")
    return {"frames": frame_count, "frames_with_tracks": frames_with_tracks, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--videos", default=None, help="comma-separated video names; default is the 25-video val split")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    image_root = args.data_root / "DanceTrack" / "training" / "image_02"
    videos = _videos(args.videos)
    args.tracker_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    video_stats: dict[str, dict[str, int]] = {}

    for video in videos:
        frame_paths = sorted(image_root.joinpath(video).glob("*.png"))
        if not frame_paths:
            raise FileNotFoundError(f"no PNG frames found for {video}: {image_root / video}")
        output_path = args.tracker_root / f"{video}.txt"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite existing tracker output: {output_path}")
        # A fresh predictor/tracker state is required for every video.
        model.predictor = None
        results = model.track(
            source=str(image_root / video),
            stream=True,
            persist=False,
            tracker=args.tracker,
            classes=[0],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        video_stats[video] = _write_result(output_path, results, len(frame_paths))

    receipt = {
        "protocol": "image-only DanceTrack val tracker generation",
        "tracker": "YOLO11x + ByteTrack",
        "model": args.model,
        "tracker_config": args.tracker,
        "data_root": str(args.data_root.resolve()),
        "tracker_root": str(args.tracker_root.resolve()),
        "videos": videos,
        "ground_truth_read": False,
        "expressions_read": False,
        "video_stats": video_stats,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
