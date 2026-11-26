#!/usr/bin/env python3
"""Run the pinned TrackEval MOT metrics with explicit sequence lengths.

The upstream command-line wrapper cannot pass a dictionary-valued ``SEQ_INFO``
argument, while the pinned MOT dataset class otherwise falls back to an old
hard-coded KITTI image directory.  This small adapter constructs the official
TrackEval objects directly and does not change the evaluator implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--trackers-folder", type=Path, required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--seqmap", type=Path, required=True)
    parser.add_argument("--seq-info-file", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "TrackEval"))
    import trackeval  # noqa: E402

    sequence_info = json.loads(args.seq_info_file.resolve().read_text(encoding="utf-8"))
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": str(args.error_log.resolve()),
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
    })
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(args.gt_folder.resolve()) + "/",
        "TRACKERS_FOLDER": str(args.trackers_folder.resolve()) + "/",
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": [str(args.tracker.resolve())],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None,
        "SEQMAP_FILE": str(args.seqmap.resolve()),
        "SEQ_INFO": {str(key): int(value) for key, value in sequence_info.items()},
        "GT_LOC_FORMAT": "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "SKIP_SPLIT_FOL": True,
    })
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metric_types = [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]
    metrics_list = [metric(metrics_config) for metric in metric_types]
    evaluator.evaluate(dataset_list, metrics_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
