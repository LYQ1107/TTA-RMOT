#!/usr/bin/env python3
"""Seal a prediction run and evaluate it with the pinned TempRMOT TrackEval fork.

The online runner never opens this manifest.  This script is the evaluator
boundary: it materializes expression-specific GT files only after prediction
files already exist, then invokes the official evaluator shipped with the
pinned TempRMOT source checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_rows(path: Path, episodes: set[str] | None = None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = f"{row['video_id']}/{row['expression_id']}"
            if episodes and key not in episodes and key.replace("/", "+") not in episodes:
                continue
            grouped[(str(row["video_id"]), str(row["expression_id"]))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return grouped


def _expression_ids(path: Path) -> dict[str, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(frame): {int(value) for value in values}
            for frame, values in payload.get("label", {}).items()}


def _label_rows(label_path: Path, image_path: Path, ids: set[int], frame_index: int,
                label_format: str = "cxcywh", frame_offset: int = 1) -> list[str]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    lines: list[str] = []
    content = label_path.read_text(encoding="utf-8").splitlines()
    for raw in content:
        fields = raw.split()
        if len(fields) < 6:
            continue
        track_id = int(float(fields[1]))
        if track_id not in ids:
            continue
        x, y, box_w, box_h = [float(value) for value in fields[2:6]]
        if label_format in {"xywh", "xywh_topleft", "x1y1wh"}:
            pixel_x = x * width
            pixel_y = y * height
        elif label_format in {"cxcywh", "center_xywh"}:
            pixel_x = (x - box_w / 2.0) * width
            pixel_y = (y - box_h / 2.0) * height
        else:
            raise ValueError(f"unsupported label format: {label_format}")
        lines.append(
            f"{int(frame_index) + int(frame_offset)},{track_id},{pixel_x:.6f},{pixel_y:.6f},"
            f"{box_w * width:.6f},{box_h * height:.6f},1,1,1\n"
        )
    return lines


def materialize_ground_truth(rows: dict[tuple[str, str], list[dict[str, Any]]], trackeval_root: Path) -> list[str]:
    """Write official RMOT-format GT files and return the seqmap entries."""
    sequence_names: list[str] = []
    sequence_info: dict[str, int] = {}
    for (video_id, expression_id), expression_rows in sorted(rows.items()):
        sequence_name = f"{video_id}+{expression_id}"
        sequence_names.append(sequence_name)
        sequence_info[sequence_name] = len(expression_rows)
        expression_path = Path(expression_rows[0]["expression_path"])
        labels_by_frame = _expression_ids(expression_path)
        output = trackeval_root / video_id / expression_id / "gt.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered: list[str] = []
        for row in expression_rows:
            ids = labels_by_frame.get(str(int(row["frame_index"])), set())
            if ids:
                rendered.extend(_label_rows(
                    Path(row["label_path"]), Path(row["image_path"]), ids, int(row["frame_index"]),
                    label_format=str(row.get("label_format", "cxcywh")),
                    frame_offset=int(row.get("trackeval_frame_offset", 1)),
                ))
        output.write_text("".join(rendered), encoding="utf-8")
        # The pinned TempRMOT TrackEval fork falls back to a hard-coded KITTI
        # image directory unless SEQ_INFO is supplied.  Its sequence length is
        # all we need here because the GT and predictions are already
        # materialized in MOT text format.
        seqinfo = trackeval_root / sequence_name / "seqinfo.ini"
        seqinfo.parent.mkdir(parents=True, exist_ok=True)
        seqinfo.write_text(
            "[Sequence]\n"
            f"name={sequence_name}\n"
            f"seqLength={len(expression_rows)}\n"
            "imDir=.\n"
            "frameRate=30\n",
            encoding="utf-8",
        )
    seqmap = trackeval_root / "seqmap.txt"
    seqmap.write_text("".join(f"{name}\n" for name in sequence_names), encoding="utf-8")
    (trackeval_root / "seq_info.json").write_text(
        json.dumps(sequence_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sequence_names


def _read_trackeval_summary(method_root: Path) -> dict[str, float | int | None]:
    summary_path = method_root / "pedestrian_summary.txt"
    if not summary_path.is_file():
        return {"HOTA": None, "DetA": None, "AssA": None, "DetRe": None, "DetPr": None, "IDF1": None}
    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return {"HOTA": None, "DetA": None, "AssA": None, "DetRe": None, "DetPr": None, "IDF1": None}
    header, values = lines[0].split(), lines[1].split()
    parsed: dict[str, float | int | None] = {}
    for key, value in zip(header, values):
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = None
    return {key: parsed.get(key) for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1")}


def evaluate_method(*, method: str, prediction_root: Path, trackeval_root: Path,
                    source_root: Path, output_dir: Path, seqmap: Path,
                    seq_info_file: Path) -> dict[str, Any]:
    script = ROOT / "scripts" / "run_locatemot_trackeval.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    method_root = prediction_root / method
    if not method_root.is_dir():
        raise FileNotFoundError(method_root)
    log_path = output_dir / "logs" / f"{method}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(script), "--source-root", str(source_root),
        "--gt-folder", str(trackeval_root), "--trackers-folder", str(trackeval_root),
        "--tracker", str(method_root), "--seqmap", str(seqmap),
        "--seq-info-file", str(seq_info_file),
        "--error-log", str(output_dir / "logs" / f"{method}_trackeval_error.log"),
    ]
    completed = subprocess.run(command, cwd=source_root / "TrackEval", text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8")
    result = {
        "method": method, "returncode": completed.returncode,
        "command": " ".join(command), "log": str(log_path.resolve()),
        "summary": _read_trackeval_summary(method_root) if completed.returncode == 0 else {},
    }
    if completed.returncode != 0:
        result["error"] = "TrackEval failed; see log"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta.yaml")
    parser.add_argument("--evaluation-manifest", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    manifest = (args.evaluation_manifest.resolve() if args.evaluation_manifest else
                resolve(config["data"]["evaluation_manifest"]))
    metadata_path = run_dir / "run_metadata.json"
    recorded_episodes = []
    if metadata_path.is_file():
        recorded_episodes = json.loads(metadata_path.read_text(encoding="utf-8")).get("episodes", [])
    episode_values = args.episodes if args.episodes is not None else recorded_episodes
    episodes = {str(value).replace("+", "/") for value in episode_values}
    rows = _read_rows(manifest, episodes=episodes or None)
    methods = args.methods or list(config.get("methods", []))
    prediction_root = run_dir / "predictions"
    trackeval_root = run_dir / "trackeval_data"
    sequence_names = materialize_ground_truth(rows, trackeval_root)
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in methods:
        results.append(evaluate_method(
            method=method, prediction_root=prediction_root, trackeval_root=trackeval_root,
            source_root=Path(config["model"]["source_root"]).resolve(), output_dir=output_dir,
            seqmap=trackeval_root / "seqmap.txt", seq_info_file=trackeval_root / "seq_info.json",
        ))
    payload = {
        "format": "locatemot-qcs-trackeval-v1", "status": "EVALUATION_COMPLETE" if all(item["returncode"] == 0 for item in results) else "EVALUATION_FAILED",
        "run_dir": str(run_dir), "evaluation_manifest": str(manifest),
        "evaluation_manifest_sha256": sha256(manifest), "trackeval_source": str((Path(config["model"]["source_root"]).resolve() / "TrackEval")),
        "sequence_count": len(sequence_names), "sequences": sequence_names, "methods": results,
        "episodes": sorted(episodes),
        "gt_materialized_after_prediction": True, "runtime_manifest_opened": False,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if payload["status"] == "EVALUATION_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
