#!/usr/bin/env python3
"""Run frozen-vs-causal score TTA on an official FlexHook score stream.

This runner deliberately separates prediction from evaluation.  During the
prediction pass it reads only source scores, expression text, and tracker
rows.  Ground-truth symlinks are attached only after every prediction file is
written, then TrackEval is invoked as a post-hoc evaluator.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flex_tta.adaptation.score import CausalStabilityTTA, FrozenScoreAdapter  # noqa: E402
from flex_tta.mot_interface.trackers import merge_kitti_class_splits  # noqa: E402
from flex_tta.plugin import FlexHookPluginConfig, FlexHookReferringPlugin  # noqa: E402
from flex_tta.types import TrackerCandidate  # noqa: E402


FRAME_RANGES = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339), "0019": (0, 1058)}


def _load_expression_files(dataset_root: Path, video: str) -> list[tuple[str, str]]:
    expressions: list[tuple[str, str]] = []
    for path in sorted((dataset_root / "expression" / video).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expressions.append((path.stem, str(payload["sentence"])))
    return expressions


def _load_scores(source: dict[str, Any], video: str, sentence: str) -> dict[int, dict[int, float]]:
    """Convert FlexHook [negative, positive] logits to positive margins."""

    by_frame: dict[int, dict[int, float]] = defaultdict(dict)
    for track_id, frame_dict in source[video].items():
        track = int(track_id)
        for frame_id, expression_dict in frame_dict.items():
            if sentence not in expression_dict:
                raise KeyError(f"sentence missing from official score stream: {video}/{track}/{frame_id}/{sentence}")
            score = expression_dict[sentence]
            if len(score) < 2:
                raise ValueError(f"FlexHook score is not a two-class logit: {video}/{track}/{frame_id}")
            by_frame[int(frame_id)][track] = float(score[1]) - float(score[0])
    return {frame: dict(sorted(items.items())) for frame, items in sorted(by_frame.items())}


def _runtime_rows(tracker_root: Path, video: str) -> dict[int, dict[int, Any]]:
    rows = merge_kitti_class_splits(
        tracker_root / video / "car" / "predict.txt",
        tracker_root / video / "pedestrian" / "predict.txt",
    )
    # FlexHook's official loader converts the 1-based tracker frame to a
    # zero-based score-stream frame before matching.
    return {
        frame - 1: dict(sorted((row.track_id, row) for row in rows if row.frame_id == frame))
        for frame in sorted({row.frame_id for row in rows})
    }


def _write_row(handle, values: tuple[float, ...]) -> None:
    handle.write(",".join(f"{value:g}" for value in values) + "\n")


def _adapter(name: str):
    if name == "FROZEN":
        return FrozenScoreAdapter()
    if name == "ONLINE_STABILITY":
        return CausalStabilityTTA()
    raise ValueError(f"unsupported method: {name}")


def _attach_ground_truth(method_root: Path, dataset_root: Path, videos: list[str]) -> None:
    """Attach GT only after the prediction phase has completely finished."""

    for video in videos:
        for expression_dir in sorted((dataset_root / "gt_template" / video).iterdir()):
            if not expression_dir.is_dir():
                continue
            target = method_root / "results" / video / expression_dir.name / "gt.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(expression_dir / "gt.txt")


def _run_trackeval(method_root: Path, vendor_root: Path) -> int:
    command = [
        sys.executable,
        "./TrackEval/scripts/run_mot_challenge.py",
        "--METRICS", "HOTA",
        "--SEQMAP_FILE", "./seqmaps/kitti-2.txt",
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", "./datasets/refer-kitti/KITTI/training/image_02",
        "--TRACKERS_FOLDER", str((method_root / "results").resolve()),
        "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--TRACKERS_TO_EVAL", str((method_root / "results").resolve()),
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", "2",
        "--PLOT_CURVES", "False",
    ]
    log_path = method_root / "trackeval.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=vendor_root, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(completed.returncode)


def run_method(method: str, source: dict[str, Any], dataset_root: Path, tracker_root: Path,
               output_root: Path, vendor_root: Path, *, max_expressions: int = 0,
               run_trackeval: bool = True) -> dict[str, Any]:
    method_root = output_root / method
    result_root = method_root / "results"
    if result_root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {result_root}")
    result_root.mkdir(parents=True)
    plugin = FlexHookReferringPlugin(FlexHookPluginConfig(adapter="frozen" if method == "FROZEN" else "causal_stability"))
    videos = sorted(source)
    stats = {"videos": 0, "expressions": 0, "frames": 0, "candidate_rows": 0, "selected_rows": 0, "pseudo_updates": 0}
    state_records: list[dict[str, Any]] = []

    for video in videos:
        runtime_rows = _runtime_rows(tracker_root, video)
        expressions = _load_expression_files(dataset_root, video)
        if max_expressions > 0:
            expressions = expressions[:max_expressions]
        for expression_id, sentence in expressions:
            score_frames = _load_scores(source, video, sentence)
            plugin.reset_episode()
            prediction_path = result_root / video / expression_id / "predict.txt"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("w", encoding="utf-8") as handle:
                for frame_id, margins in score_frames.items():
                    rows = runtime_rows.get(frame_id, {})
                    candidates = [
                        TrackerCandidate(video, frame_id, track_id, rows[track_id], margin)
                        for track_id, margin in margins.items() if track_id in rows
                    ]
                    step, selected = plugin.step(frame_id, candidates)
                    for candidate in selected:
                        if FRAME_RANGES[video][0] <= candidate.row.frame_id <= FRAME_RANGES[video][1] + 1:
                            _write_row(handle, candidate.row.values)
                    stats["frames"] += 1
                    stats["candidate_rows"] += len(candidates)
                    stats["selected_rows"] += len(selected)
                    stats["pseudo_updates"] += step.pseudo_updates
            state = getattr(plugin.adapter, "state_dict", lambda: {"scale": 1.0, "bias": 0.0})()
            state_records.append({"video": video, "expression_id": expression_id, **state})
            stats["expressions"] += 1
        stats["videos"] += 1

    # This is deliberately after all prediction files have been closed.
    if max_expressions == 0:
        _attach_ground_truth(method_root, dataset_root, videos)
    eval_status = None
    if run_trackeval and max_expressions == 0:
        eval_status = _run_trackeval(method_root, vendor_root)
    receipt = {
        "method": method,
        "protocol": "FlexHook official V2 score stream, causal tracker replay",
        "source_free_prediction": True,
        "ground_truth_opened_before_prediction_complete": False,
        "future_frames_used": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
        "stats": stats,
        "evaluation_exit_status": eval_status,
        "adapter_states": state_records,
    }
    (method_root / "run_metadata.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["FROZEN", "ONLINE_STABILITY"])
    parser.add_argument("--max-expressions", type=int, default=0)
    parser.add_argument("--skip-trackeval", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for method in args.methods:
        receipts.append(run_method(
            method, source, args.dataset_root, args.tracker_root, args.output_root,
            args.vendor_root, max_expressions=args.max_expressions,
            run_trackeval=not args.skip_trackeval,
        ))
    (args.output_root / "suite_metadata.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
