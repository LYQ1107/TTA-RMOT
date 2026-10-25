#!/usr/bin/env python3
"""Read-only source/target inventory for future full-source/full-target study.

No optimizer, model forward, GT file, network, or GPU is accessed. Source-train
labels are permitted later; Dance target labels are NEVER inspected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

EXPECTED_SHA = "71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613"
EXPECTED_TRAIN_VIDEOS = 17
EXPECTED_DANCE_VIDEOS = 25
EXPECTED_EXPRESSIONS = 17
EXPECTED_TARGET_EPISODES = EXPECTED_DANCE_VIDEOS * EXPECTED_EXPRESSIONS


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def count_source_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"status": "MISSING", "manifest": str(path)}
    lines = [v.strip() for v in path.read_text(encoding="utf-8").splitlines() if v.strip()]
    vids = set()
    for line in lines:
        matches = re.findall(r"(?<!\d)(?:00\d\d|001\d|0020)(?!\d)", line)
        vids.update(matches)
    return {"status": "PRESENT", "manifest": str(path), "rows": len(lines),
            "detected_unique_video_ids": sorted(vids), "sha256": hash_file(path)}


def inventory(dance_root: Path, tracker_root: Path, checkpoint: Path,
              source_manifest: Path | None = None, check_checkpoint_hash: bool = False) -> dict:
    tracker_files = sorted(tracker_root.glob("*.txt")) if tracker_root.is_dir() else []
    videos = [v.stem for v in tracker_files]
    target = {}
    for video in videos:
        expr_path = dance_root / "expression" / video
        images = dance_root / "DanceTrack" / "training" / "image_02" / video
        exps = sorted(expr_path.glob("*.json"))
        target[video] = {
            "tracker_file": str(tracker_root / (video + ".txt")),
            "expression_count": len(exps),
            "image_root_exists": images.is_dir(),
            "has_frame_image": any(images.glob("*.png")) if images.is_dir() else False,
        }
    chk = {"path": str(checkpoint), "present": checkpoint.is_file()}
    if chk["present"]:
        chk["bytes"] = checkpoint.stat().st_size
        if check_checkpoint_hash:
            chk["sha256"] = hash_file(checkpoint)
            chk["sha256_valid"] = chk["sha256"] == EXPECTED_SHA
    total_exps = sum(v["expression_count"] for v in target.values())
    report = {
        "protocol": "full_official_source_train_to_complete_unlabeled_dance_tta",
        "training_source": "Refer-KITTI-V2 official source train, 17 videos; NEVER Dance target labels",
        "target": "Refer-Dance validation, all 25 videos / 425 video-expression episodes",
        "target_gt_opened": False,
        "source_train_manifest": count_source_manifest(source_manifest) if source_manifest else {"status": "NOT_SPECIFIED"},
        "checkpoint": chk,
        "target_video_count": len(videos),
        "target_expression_count": total_exps,
        "target_files": target,
    }
    report["target_inventory_pass"] = (
        len(videos) == EXPECTED_DANCE_VIDEOS
        and total_exps == EXPECTED_TARGET_EPISODES
        and all(x["expression_count"] == EXPECTED_EXPRESSIONS and x["has_frame_image"] for x in target.values())
    )
    if source_manifest and report["source_train_manifest"]["status"] == "PRESENT":
        report["source_train_count_pass"] = len(report["source_train_manifest"]["detected_unique_video_ids"]) == EXPECTED_TRAIN_VIDEOS
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dance-root", type=Path, required=True)
    p.add_argument("--tracker-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-train-manifest", type=Path)
    p.add_argument("--check-checkpoint-hash", action="store_true")
    p.add_argument("--strict", action="store_true", help="require complete 25/425 target and, if set, 17 source train videos")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = inventory(args.dance_root, args.tracker_root, args.checkpoint,
                       args.source_train_manifest, args.check_checkpoint_hash)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "target_files"}, indent=2, ensure_ascii=False))
    if args.strict and (not report["target_inventory_pass"] or
                        report.get("source_train_count_pass") is False or
                        report["checkpoint"].get("sha256_valid") is False):
        raise SystemExit("preflight FAILED: missing target videos/expressions or source-train/hash mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
