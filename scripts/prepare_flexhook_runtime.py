#!/usr/bin/env python3
"""Build a GT-free runtime view for the official FlexHook evaluator.

The public FlexHook tracker archive contains both ``predict.txt`` and
``gt.txt``.  Its test loader also opens ``labels.json`` even though test mode
uses tracker predictions instead of label boxes, and the expression JSONs
contain the target track ids.  This script creates a separate runtime view
that exposes only the information needed by inference:

* images (read-only symlink);
* expression sentences, with all ``label`` fields removed;
* an empty ``labels.json`` (the official test loader still expects the file);
* tracker ``predict.txt`` files only.

The original prepared data and tracker archive remain outside this runtime
view and are used only after predictions are sealed for evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


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


def _link_once(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise RuntimeError(f"existing symlink points elsewhere: {target} -> {target.resolve()}")
        return
    if target.exists():
        raise RuntimeError(f"refusing to replace existing path: {target}")
    target.symlink_to(source, target_is_directory=source.is_dir())


def _prepare_expressions(source_root: Path, runtime_root: Path) -> int:
    source_expression = source_root / "expression"
    output_expression = runtime_root / "expression"
    count = 0
    for source_video in sorted(source_expression.iterdir()):
        if not source_video.is_dir():
            continue
        for source_file in sorted(source_video.glob("*.json")):
            payload = json.loads(source_file.read_text(encoding="utf-8"))
            if "sentence" not in payload:
                raise ValueError(f"expression has no sentence: {source_file}")
            output_file = output_expression / source_video.name / source_file.name
            _write_json_atomic(output_file, {"sentence": str(payload["sentence"])})
            count += 1
    return count


def _prepare_tracker(source_root: Path, runtime_root: Path) -> int:
    count = 0
    for source_file in sorted(source_root.rglob("predict.txt")):
        relative = source_file.relative_to(source_root)
        _link_once(source_file, runtime_root / relative)
        count += 1
    if count == 0:
        raise FileNotFoundError(f"no predict.txt files under {source_root}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2"),
    )
    parser.add_argument(
        "--tracker-root", type=Path,
        default=Path("/data3/user/TTA-RMOT-assets/flexhook/tracker_outputs/Temp-NeuralSORT-kitti2"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2-runtime"),
    )
    parser.add_argument(
        "--output-tracker-root", type=Path,
        default=Path("/data3/user/TTA-RMOT-assets/flexhook/tracker_outputs/Temp-NeuralSORT-kitti2-runtime"),
    )
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    tracker_root = args.tracker_root.resolve()
    output_root = args.output_root.resolve()
    output_tracker_root = args.output_tracker_root.resolve()

    for required in [data_root / "KITTI/training/image_02", data_root / "expression", data_root / "labels.json", tracker_root]:
        if not required.exists():
            raise FileNotFoundError(required)

    output_root.mkdir(parents=True, exist_ok=True)
    output_tracker_root.mkdir(parents=True, exist_ok=True)
    _link_once(data_root / "KITTI", output_root / "KITTI")
    expression_count = _prepare_expressions(data_root, output_root)
    _write_json_atomic(output_root / "labels.json", {})
    predict_count = _prepare_tracker(tracker_root, output_tracker_root)

    metadata = {
        "format": "ttarmot-flexhook-gt-free-runtime-v1",
        "source_data_root": str(data_root),
        "source_tracker_root": str(tracker_root),
        "runtime_data_root": str(output_root),
        "runtime_tracker_root": str(output_tracker_root),
        "expression_file_count": expression_count,
        "predict_file_count": predict_count,
        "labels_json_is_empty": True,
        "expression_label_fields_removed": True,
        "gt_template_exposed_at_runtime": False,
        "tracker_gt_files_exposed_at_runtime": False,
        "intended_use": "inference_only; evaluate only after prediction sealing",
    }
    _write_json_atomic(output_root / "ttarmot_runtime_metadata.json", metadata)
    _write_json_atomic(output_tracker_root / "ttarmot_runtime_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
