#!/usr/bin/env python3
"""Run source-free score/calibration controls on the official Dance score stream.

The prediction process reads only the frozen V2 logits, expressions and the
official tracker bank.  It never opens ``labels.json`` or ``gt_template``.
TrackEval is deliberately delegated to ``evaluate_flexhook_dance_tta.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flex_tta.dance_io import dance_videos, load_expressions, load_tracker_lines, score_frames  # noqa: E402
from flex_tta.plugin import FlexHookPluginConfig, FlexHookReferringPlugin  # noqa: E402


METHODS = {
    "FROZEN": "frozen",
    "ONLINE_STABILITY": "causal_stability",
    "SOURCE_PRIOR_CALIBRATION": "source_prior",
    "QUERY_TRACKLET_CALIBRATION": "query_tracklet",
}


def run_method(method: str, source: dict, dataset_root: Path, tracker_root: Path, output_root: Path) -> dict:
    method_root = output_root / method
    results_root = method_root / "results"
    if method_root.exists():
        raise FileExistsError(f"refusing to overwrite existing score-TTA run: {method_root}")
    results_root.mkdir(parents=True)
    plugin = FlexHookReferringPlugin(FlexHookPluginConfig(adapter=METHODS[method]))
    videos = dance_videos(tracker_root)
    tracker_cache = {video: load_tracker_lines(tracker_root, video) for video in videos}
    stats = {
        "videos": 0,
        "expressions": 0,
        "frames": 0,
        "candidate_rows": 0,
        "selected_rows": 0,
        "nonempty_predictions": 0,
        "pseudo_updates": 0,
    }
    states = []
    for video in videos:
        expressions = load_expressions(dataset_root, video)
        for expression_id, sentence in expressions:
            plugin.reset_episode()
            frame_scores = score_frames(source, video, sentence)
            prediction = results_root / video / expression_id / "predict.txt"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            selected_rows = 0
            with prediction.open("w", encoding="utf-8") as handle:
                for frame_id, margins in frame_scores.items():
                    rows = tracker_cache[video].get(frame_id, {})
                    valid = {track_id: margin for track_id, margin in margins.items() if track_id in rows}
                    step, selected = plugin.step(frame_id, [
                        # The plugin contract only needs source margins; keep
                        # this local object lightweight and write original rows.
                        _Candidate(track_id, margin) for track_id, margin in valid.items()
                    ])
                    for candidate in selected:
                        handle.write(rows[candidate.track_id] + "\n")
                    selected_rows += len(selected)
                    stats["frames"] += 1
                    stats["candidate_rows"] += len(valid)
                    stats["selected_rows"] += len(selected)
                    stats["pseudo_updates"] += step.pseudo_updates
            if selected_rows:
                stats["nonempty_predictions"] += 1
            states.append({
                "video": video,
                "expression_id": expression_id,
                **getattr(plugin.adapter, "state_dict", lambda: {})(),
            })
            stats["expressions"] += 1
        stats["videos"] += 1
    metadata = {
        "method": method,
        "protocol": "source-free Dance score replay; evaluator is a separate process",
        "source_free_prediction": True,
        "target_gt_opened": False,
        "target_gt_template_opened": False,
        "future_frames_used": False,
        "tracker_geometry_changed": False,
        "tracker_ids_changed": False,
        "stats": stats,
        "adapter_states": states,
    }
    (method_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_threshold_method(threshold: float, source: dict, dataset_root: Path, tracker_root: Path, output_root: Path) -> dict:
    """Write a pre-registered threshold control without target-side tuning."""

    if threshold == 0:
        label = "THRESHOLD_0"
    else:
        label = f"THRESHOLD_{threshold:+g}".replace("+", "P").replace("-", "M").replace(".", "d")
    method_root = output_root / label
    results_root = method_root / "results"
    if method_root.exists():
        raise FileExistsError(f"refusing to overwrite threshold run: {method_root}")
    results_root.mkdir(parents=True)
    videos = dance_videos(tracker_root)
    tracker_cache = {video: load_tracker_lines(tracker_root, video) for video in videos}
    stats = {"videos": 0, "expressions": 0, "frames": 0, "candidate_rows": 0, "selected_rows": 0}
    for video in videos:
        for expression_id, sentence in load_expressions(dataset_root, video):
            prediction = results_root / video / expression_id / "predict.txt"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            selected = 0
            with prediction.open("w", encoding="utf-8") as handle:
                for frame_id, margins in score_frames(source, video, sentence).items():
                    rows = tracker_cache[video].get(frame_id, {})
                    valid = [(track_id, margin) for track_id, margin in margins.items() if track_id in rows]
                    for track_id, margin in valid:
                        if margin > threshold:
                            handle.write(rows[track_id] + "\n")
                            selected += 1
                    stats["frames"] += 1
                    stats["candidate_rows"] += len(valid)
            stats["selected_rows"] += selected
            stats["expressions"] += 1
        stats["videos"] += 1
    metadata = {
        "method": label,
        "threshold": threshold,
        "protocol": "source-free fixed threshold development sweep",
        "source_free_prediction": True,
        "target_gt_opened": False,
        "stats": stats,
    }
    (method_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


class _Candidate:
    def __init__(self, track_id: int, source_margin: float):
        self.track_id = int(track_id)
        self.source_margin = float(source_margin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=sorted(METHODS))
    parser.add_argument("--thresholds", nargs="*", type=float, default=[])
    args = parser.parse_args()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    suite = [run_method(method, source, args.dataset_root, args.tracker_root, args.output_root) for method in args.methods]
    suite.extend(run_threshold_method(threshold, source, args.dataset_root, args.tracker_root, args.output_root) for threshold in args.thresholds)
    (args.output_root / "suite_metadata.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_root.resolve()), "methods": args.methods, "sequences": 425}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
