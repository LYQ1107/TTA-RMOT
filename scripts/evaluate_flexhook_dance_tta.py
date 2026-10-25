#!/usr/bin/env python3
"""Attach sealed Dance GT and run official TrackEval for score-TTA outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def attach_gt(method_results: Path, dataset_root: Path, videos: list[str]) -> int:
    attached = 0
    for video in videos:
        for expression_dir in sorted((dataset_root / "gt_template" / video).iterdir()):
            if not expression_dir.is_dir():
                continue
            target = method_results / video / expression_dir.name / "gt.txt"
            if not (target.parent / "predict.txt").is_file():
                raise ValueError(f"prediction must already be sealed: {target.parent}")
            if not target.exists():
                target.symlink_to(expression_dir / "gt.txt")
                attached += 1
    return attached


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    videos = sorted(path.stem for path in (args.output_root / args.methods[0] / "results").glob("*.txt"))
    # The prediction tree has video directories, not flat files.  Derive the
    # fixed 25-video set from the tracker-bank-independent runtime outputs.
    videos = sorted(path.name for path in (args.output_root / args.methods[0] / "results").iterdir() if path.is_dir())
    if len(videos) != 25:
        raise ValueError(f"expected 25 prediction video directories, found {len(videos)}")
    reports = []
    for method in args.methods:
        method_results = (args.output_root / method / "results").resolve()
        predictions = list(method_results.glob("*/*/predict.txt"))
        if len(predictions) != 425:
            raise ValueError(f"official full evaluation needs 425 sealed predictions, found {len(predictions)}")
        attached = attach_gt(method_results, args.dataset_root, videos)
        log_path = args.output_root / method / "trackeval.log"
        command = [
            sys.executable, "./TrackEval/scripts/run_mot_challenge.py",
            "--METRICS", "HOTA", "CLEAR", "Identity",
            "--SEQMAP_FILE", "./seqmaps/dance.txt",
            "--SKIP_SPLIT_FOL", "True",
            "--GT_FOLDER", str(args.dataset_root / "DanceTrack/training/image_02"),
            "--TRACKERS_FOLDER", str(method_results),
            "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
            "--TRACKERS_TO_EVAL", str(method_results),
            "--USE_PARALLEL", "True", "--NUM_PARALLEL_CORES", "2",
            "--PLOT_CURVES", "False",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.vendor_root, stdout=handle, stderr=subprocess.STDOUT, check=False)
        summary = method_results / "pedestrian_summary.txt"
        reports.append({
            "method": method,
            "attached_gt_symlinks": attached,
            "trackeval_exit_status": completed.returncode,
            "summary": str(summary),
        })
        if completed.returncode != 0 or not summary.is_file():
            raise RuntimeError(f"TrackEval failed for {method}; see {log_path}")
    output = args.output_root / "evaluation_metadata.json"
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
