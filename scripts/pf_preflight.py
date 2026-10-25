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
import sys
import ast
from pathlib import Path

EXPECTED_SHA = "71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613"
EXPECTED_TRAIN_VIDEOS = 17
EXPECTED_DANCE_VIDEOS = 25
EXPECTED_EXPRESSIONS = 17
EXPECTED_TARGET_EPISODES = EXPECTED_DANCE_VIDEOS * EXPECTED_EXPRESSIONS
EXPECTED_SOURCE_IDS = {f"{i:04d}" for i in range(21)} - {"0005", "0011", "0013", "0019"}


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


def verify_server_assets(report: dict, dance_root: Path, tracker_root: Path,
                         source_root: Path | None, source_manifest: Path | None,
                         source_result: Path | None, vendor: Path | None,
                         tracker_receipt: Path | None = None) -> None:
    """Check actual images/score coverage and source split, without Dance GT."""
    for video, record in report["target_files"].items():
        rows = (tracker_root / f"{video}.txt").read_text().splitlines()
        frames = {int(float(row.split(",")[0])) for row in rows if row.strip()}
        images = dance_root / "DanceTrack/training/image_02" / video
        missing = [t for t in sorted(frames) if not (images / f"{t:06d}.png").is_file()]
        record.update(tracker_rows=len(rows), tracker_frames=len(frames), missing_image_frames=missing)
    report["all_tracker_images_present"] = all(not x["missing_image_frames"] for x in report["target_files"].values())
    if tracker_receipt is not None:
        receipt = json.loads(tracker_receipt.read_text())
        mismatches = [item["file"] for item in receipt["files"]
                      if hash_file(tracker_root / item["file"]) != item["sha256"]]
        report["official_tracker_receipt"] = {"path": str(tracker_receipt),
            "files": len(receipt["files"]), "hash_mismatches": mismatches,
            "pass": not mismatches and {Path(x["file"]).stem for x in receipt["files"]} == set(report["target_files"])}
    if source_manifest is not None:
        src = report["source_train_manifest"]
        ids = set(src.get("detected_unique_video_ids", []))
        report["source_train_count_pass"] = ids == EXPECTED_SOURCE_IDS
        if source_root is not None and source_manifest.is_file():
            lines = source_manifest.read_text().splitlines()
            missing = [line for line in lines if line.strip() and not (source_root / line.strip()).is_file()]
            source_files = {v: {
                "image_count": sum(1 for _ in (source_root / "KITTI/training/image_02" / v).glob("*.png")),
                "expression_count": sum(1 for _ in (source_root / "expression" / v).glob("*.json")),
            } for v in sorted(EXPECTED_SOURCE_IDS)}
            src.update(source_root=str(source_root), missing_manifest_images=missing,
                       videos=source_files, source_labels_present=(source_root / "labels.json").is_file())
            source_label_videos = (set(json.loads((source_root / "labels.json").read_text()))
                                   if src["source_labels_present"] else set())
            src["missing_label_videos"] = sorted(EXPECTED_SOURCE_IDS - source_label_videos)
            src["source_label_video_count"] = len(source_label_videos)
            report["source_train_assets_pass"] = (not missing and not src["missing_label_videos"] and src["source_labels_present"] and
                all(x["image_count"] and x["expression_count"] for x in source_files.values()))
    if source_result is not None:
        score_report = {"path": str(source_result), "present": source_result.is_file()}
        if score_report["present"]:
            scores = json.loads(source_result.read_text())
            score_report["bytes"] = source_result.stat().st_size
            score_report["sha256"] = hash_file(source_result)
            coverage = {}
            for video in report["target_files"]:
                sentences = [json.loads(p.read_text())["sentence"] for p in (dance_root / "expression" / video).glob("*.json")]
                observations = [exprs for frames in scores.get(video, {}).values() for exprs in frames.values()]
                missing = sum(any(s not in exprs or len(exprs[s]) != 2 for s in sentences) for exprs in observations)
                coverage[video] = {"observation_rows": len(observations), "missing_expression_scores": missing}
            score_report["coverage"] = coverage
            score_report["pass"] = all(x["observation_rows"] and not x["missing_expression_scores"] for x in coverage.values())
        report["frozen_scores"] = score_report
    if vendor is not None:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import compat.sitecustomize  # noqa: F401
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(str(vendor / "pretrained/roberta-base"), local_files_only=True)
        ids = tokenizer.encode("person", add_special_tokens=False)
        vocab = json.loads((vendor / "pretrained/roberta-base/vocab.json").read_text())
        report["tokenizer"] = {"path": str(vendor / "pretrained/roberta-base"), "person_ids": ids, "pass": ids == [vocab["person"]]}
        tree = ast.parse((vendor / "data/utils.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VIDEOS" for t in node.targets):
                train = ast.literal_eval(node.value)["kitti-2"]["train"]
                report["vendor_source_train_split"] = {"videos": train, "count": len(train),
                    "missing_requested_source_ids": sorted(EXPECTED_SOURCE_IDS - set(train))}
                break


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dance-root", type=Path, required=True)
    p.add_argument("--tracker-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-train-manifest", type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--source-result", type=Path)
    p.add_argument("--vendor", type=Path)
    p.add_argument("--tracker-receipt", type=Path)
    p.add_argument("--check-checkpoint-hash", action="store_true")
    p.add_argument("--strict", action="store_true", help="require complete 25/425 target and, if set, 17 source train videos")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    report = inventory(args.dance_root, args.tracker_root, args.checkpoint,
                       args.source_train_manifest, args.check_checkpoint_hash)
    verify_server_assets(report, args.dance_root, args.tracker_root, args.source_root,
                         args.source_train_manifest, args.source_result, args.vendor, args.tracker_receipt)
    report["server_assets_pass"] = (
        report["target_inventory_pass"] and report["all_tracker_images_present"] and
        report["checkpoint"]["present"] and report["checkpoint"].get("sha256_valid", True) and
        (args.source_train_manifest is None or report.get("source_train_count_pass", False)) and
        (args.source_root is None or report.get("source_train_assets_pass", False)) and
        (args.source_result is None or report["frozen_scores"].get("pass", False)) and
        (args.tracker_receipt is None or report["official_tracker_receipt"]["pass"]) and
        (args.vendor is None or report["tokenizer"]["pass"]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "target_files"}, indent=2, ensure_ascii=False))
    if args.strict and not report["server_assets_pass"]:
        raise SystemExit("preflight FAILED: missing target videos/expressions or source-train/hash mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
