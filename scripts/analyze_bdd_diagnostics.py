#!/usr/bin/env python3
"""Seal full-run metrics and source-free diagnostics after prediction output exists."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBE_METHODS = {"PROBE_CACHE", "PROBE_ST", "QCS_TTA", "NO_STATE"}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(area_left + area_right - inter, 1e-9)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def rank_auc(scores: list[float], labels: list[bool]) -> float | None:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    wins = sum(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for pos in positive for neg in negative)
    return wins / (len(positive) * len(negative))


def load_manifest(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                rows[(str(row["video_id"]), str(row["expression_id"]), int(row["frame_index"]))] = row
    return rows


def load_labels(path: Path) -> list[tuple[int, list[float]]]:
    """Read one official Refer-BDD per-image label file.

    The first column in EchoTrack's ``label_with_ids`` files is a local
    placeholder (it is ``0`` for every row); the image filename, not that
    column, identifies the frame.  Evaluation already uses the manifest
    frame index for this reason.  Keep the diagnostic loader consistent with
    that official writer instead of indexing rows by the placeholder.
    """
    result: list[tuple[int, list[float]]] = []
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        track_id = int(float(fields[1]))
        values = [float(value) for value in fields[2:6]]
        del frame
        result.append((track_id, values))
    return result


def target_boxes(row: dict[str, Any], expression_labels: dict[str, Any], label_cache: dict[str, list[tuple[int, list[float]]]]) -> list[tuple[int, list[float]]]:
    frame = int(row["frame_index"])
    ids = {int(value) for value in expression_labels.get(str(frame), [])}
    if not ids:
        return []
    label_path = str(row["label_path"])
    if label_path not in label_cache:
        label_cache[label_path] = load_labels(Path(label_path))
    height, width = [int(value) for value in row.get("_image_size", [720, 1280])]
    targets = []
    for track_id, values in label_cache[label_path]:
        if track_id not in ids:
            continue
        x, y, box_width, box_height = values
        if str(row.get("label_format", "xywh_topleft")) in {"xywh", "xywh_topleft", "x1y1wh"}:
            x1, y1 = x * width, y * height
        else:
            x1, y1 = (x - box_width / 2.0) * width, (y - box_height / 2.0) * height
        targets.append((track_id, [x1, y1, x1 + box_width * width, y1 + box_height * height]))
    return targets


def read_candidate_rows(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                result[(str(row["video_id"]), str(row["expression_id"]), int(row["frame_index"]))] = row
    return result


def source_diagnostics(config: dict[str, Any], run_dir: Path, evaluation_manifest: Path) -> tuple[dict[str, Any], dict[tuple[str, str, int], list[tuple[int, list[float]]]]]:
    rows = load_manifest(evaluation_manifest)
    candidates = read_candidate_rows(run_dir / "candidate_diagnostics.jsonl")
    expression_cache: dict[str, dict[str, Any]] = {}
    label_cache: dict[str, list[tuple[int, list[float]]]] = {}
    frame_targets: dict[tuple[str, str, int], list[tuple[int, list[float]]]] = {}
    by_video: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "expressions": set(), "frames": 0, "valid_candidates": 0,
        "objectness_area_candidates": 0, "source_selected_candidates": 0,
        "target_boxes": 0, "object_recall_all": [], "object_recall_objectness_area": [],
        "object_recall_source_selected": [], "source_scores": [], "matched_scores": [],
        "unmatched_scores": [],
    })
    all_scores: list[float] = []
    all_labels: list[bool] = []
    target_count = 0
    for key, diag in candidates.items():
        video, expression_id, frame = key
        row = rows.get(key)
        if row is None:
            continue
        row = dict(row)
        row["_image_size"] = diag.get("image_size", [720, 1280])
        expression_path = str(row["expression_path"])
        if expression_path not in expression_cache:
            expression_cache[expression_path] = json.loads(Path(expression_path).read_text(encoding="utf-8"))
        targets = target_boxes(row, expression_cache[expression_path].get("label", {}), label_cache)
        frame_targets[key] = targets
        report = by_video[video]
        report["expressions"].add(expression_id)
        report["frames"] += 1
        report["valid_candidates"] += int(diag.get("valid_candidate_count", len(diag.get("candidates", []))))
        pre_gate = [item for item in diag.get("candidates", []) if item.get("passes_objectness_area")]
        report["objectness_area_candidates"] += len(pre_gate)
        selected = [item for item in pre_gate if float(item["source_probability"]) > float(config["inference"]["referring_threshold"])]
        report["source_selected_candidates"] += len(selected)
        report["target_boxes"] += len(targets)
        target_count += len(targets)
        for item in diag.get("candidates", []):
            score = float(item["source_probability"])
            max_iou = max((iou(item["box_xyxy"], box) for _, box in targets), default=0.0)
            matched = max_iou >= 0.5
            all_scores.append(score)
            all_labels.append(matched)
            if item.get("passes_objectness_area"):
                report["source_scores"].append(score)
                (report["matched_scores"] if matched else report["unmatched_scores"]).append(score)
        for _, target_box in targets:
            report["object_recall_all"].append(max((iou(item["box_xyxy"], target_box) for item in diag.get("candidates", [])), default=0.0))
            report["object_recall_objectness_area"].append(max((iou(item["box_xyxy"], target_box) for item in pre_gate), default=0.0))
            report["object_recall_source_selected"].append(max((iou(item["box_xyxy"], target_box) for item in selected), default=0.0))

    def finalize_video(report: dict[str, Any]) -> dict[str, Any]:
        result = dict(report)
        result["expressions"] = len(report["expressions"])
        for key in ("object_recall_all", "object_recall_objectness_area", "object_recall_source_selected"):
            values = report[key]
            result[key] = {
                "target_boxes": len(values),
                "matched_iou_0.5": sum(value >= 0.5 for value in values) / max(1, len(values)),
                "matched_iou_0.3": sum(value >= 0.3 for value in values) / max(1, len(values)),
                "mean_best_iou": mean(values) if values else 0.0,
            }
        for key in ("source_scores", "matched_scores", "unmatched_scores"):
            values = report[key]
            result[key] = {"count": len(values), "p05": percentile(values, .05), "p50": percentile(values, .5), "p95": percentile(values, .95), "mean": mean(values) if values else None}
        for key in ("expressions", "object_recall_all", "object_recall_objectness_area", "object_recall_source_selected", "source_scores", "matched_scores", "unmatched_scores"):
            report.pop(key, None)
        return result

    video_reports = {video: finalize_video(report) for video, report in by_video.items()}
    candidate_recall = {
        "format": "locatemot-refer-bdd-candidate-recall-v1",
        "run_dir": str(run_dir),
        "thresholds": {
            "objectness": float(config["inference"]["objectness_threshold"]),
            "area_pixels": float(config["inference"]["area_threshold"]),
            "referring": float(config["inference"]["referring_threshold"]),
            "iou": {"match": 0.5, "diagnostic": 0.3},
        },
        "scope": {"videos": len(video_reports), "expressions": len({(key[0], key[1]) for key in candidates}), "frames": len(frame_targets), "target_boxes": target_count},
        "source_score_distribution": {"count": len(all_scores), "p05": percentile(all_scores, .05), "p50": percentile(all_scores, .5), "p95": percentile(all_scores, .95), "mean": mean(all_scores) if all_scores else None, "matched_iou_0.5_auc": rank_auc(all_scores, all_labels)},
        "by_video": video_reports,
        "interpretation": {
            "D1_candidate_recall_upper_bound": "valid tracker candidates before objectness/area/referring gates; boxes are matched to sealed target GT offline at IoU 0.5",
            "D2_language_matching": "source referring scores are summarized only on valid objectness/area candidates and labeled by offline IoU matching",
            "runtime_gt_isolation": True,
        },
    }
    return candidate_recall, frame_targets


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(raw) for raw in handle if raw.strip()]


def trace_diagnostics(run_dir: Path, frame_targets: dict[tuple[str, str, int], list[tuple[int, list[float]]]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    traces = read_trace(run_dir / "update_trace.jsonl")
    candidate_rows = read_candidate_rows(run_dir / "candidate_diagnostics.jsonl")
    methods = list(config.get("methods", []))
    probe_by_method = {method: {"frames": 0, "due_frames": 0, "frames_with_probe_result": 0, "probe_calls": 0, "history_frame_counts": []} for method in methods}
    updates = {method: {"frames": 0, "optimizer_steps": 0, "step_records": [], "reason_counts": Counter(), "evidence": Counter(), "evidence_records": 0, "selected_rows_final": 0} for method in methods}
    pseudo = {method: {"records": 0, "supported_pos": 0, "supported_neg": 0, "pos_correct": 0, "neg_correct": 0, "unknown": 0, "by_source": Counter()} for method in methods}
    episode_probe_calls: dict[tuple[str, str], int] = {}
    for trace in traces:
        key = (str(trace["video_id"]), str(trace["expression_id"]), int(trace["frame_index"]))
        episode_key = (key[0], key[1])
        episode_probe_calls[episode_key] = max(
            episode_probe_calls.get(episode_key, 0), int(trace.get("probe_calls_total", 0))
        )
        targets = frame_targets.get(key, [])
        for method in methods:
            method_trace = trace["methods"].get(method, {})
            update = method_trace.get("last_update")
            if update is not None:
                stats = updates[method]
                stats["frames"] += 1
                stats["reason_counts"][str(update.get("reason"))] += 1
                if update.get("stepped"):
                    stats["optimizer_steps"] += 1
                    stats["step_records"].append({key_name: update.get(key_name) for key_name in ("frame_index", "sample_count", "distinct_track_time_samples", "loss", "grad_norm", "parameter_delta", "elapsed_sec")})
            evidence = method_trace.get("evidence", [])
            stats = updates[method]
            stats["evidence_records"] += len(evidence)
            for record in evidence:
                label = str(record.get("label"))
                stats["evidence"][label] += 1
                p = pseudo[method]
                p["records"] += 1
                p["by_source"][str(record.get("evidence_source"))] += 1
                if label not in {"SUPPORTED_POS", "SUPPORTED_NEG"}:
                    p["unknown"] += 1
                    continue
                track_id = int(record.get("pred_track_id", -1))
                candidate_key = (key[0], key[1], key[2])
                candidate = next((item for item in candidate_rows.get(candidate_key, {}).get("candidates", [])
                                  if int(item.get("track_id", -1)) == track_id), None)
                if label == "SUPPORTED_POS":
                    p["supported_pos"] += 1
                else:
                    p["supported_neg"] += 1
                matched = None
                if candidate is not None:
                    matched = max((iou(candidate["box_xyxy"], box) for _, box in targets), default=0.0) >= 0.5
                if matched is None:
                    p["unknown"] += 1
                elif label == "SUPPORTED_POS":
                    p["pos_correct"] += int(matched)
                else:
                    p["neg_correct"] += int(not matched)
        if trace.get("probe_due"):
            for method in methods:
                p = probe_by_method[method]
                p["due_frames"] += 1
                if trace.get("probe_history_frames"):
                    p["frames_with_probe_result"] += 1
                    p["history_frame_counts"].append(len(trace["probe_history_frames"]))
        for method in methods:
            probe_by_method[method]["frames"] += 1

    total_probe_calls = sum(episode_probe_calls.values())
    for method in methods:
        probe_by_method[method]["probe_calls"] = total_probe_calls if method in PROBE_METHODS else 0
        probe_by_method[method]["probe_calls_scope"] = "shared_probe_suite" if method in PROBE_METHODS else "not_used"
        updates[method]["reason_counts"] = dict(updates[method]["reason_counts"])
        updates[method]["evidence"] = dict(updates[method]["evidence"])
        for key in ("step_records",):
            values = updates[method][key]
            updates[method]["grad_norm"] = {"count": len([v for v in values if v.get("grad_norm") is not None]), "p50": percentile([float(v["grad_norm"]) for v in values if v.get("grad_norm") is not None], .5), "p95": percentile([float(v["grad_norm"]) for v in values if v.get("grad_norm") is not None], .95)}
            updates[method]["parameter_delta"] = {"count": len([v for v in values if v.get("parameter_delta") is not None]), "nonzero": sum(float(v.get("parameter_delta") or 0.0) > 0.0 for v in values), "p50": percentile([float(v["parameter_delta"]) for v in values if v.get("parameter_delta") is not None], .5), "p95": percentile([float(v["parameter_delta"]) for v in values if v.get("parameter_delta") is not None], .95)}
        updates[method].pop("step_records", None)
        pseudo[method]["by_source"] = dict(pseudo[method]["by_source"])
        pseudo[method]["positive_precision"] = pseudo[method]["pos_correct"] / max(1, pseudo[method]["supported_pos"])
        pseudo[method]["negative_precision"] = pseudo[method]["neg_correct"] / max(1, pseudo[method]["supported_neg"])
    return probe_by_method, updates, pseudo


def metric_rows(run_dir: Path, config: dict[str, Any], audit: dict[str, Any], updates: dict[str, Any], probe: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = json.loads((run_dir / "evaluation" / "evaluation.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    methods = list(config.get("methods", []))
    summaries = {item["method"]: item.get("summary", {}) for item in evaluation["methods"]}
    frozen = summaries.get("FROZEN", {})
    summary_rows = []
    expression_rows = []
    video_values: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for method in methods:
        method_root = run_dir / "predictions" / method
        detailed = method_root / "pedestrian_detailed.csv"
        if detailed.is_file():
            with detailed.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("seq") == "COMBINED":
                        continue
                    video, expression_id = row["seq"].split("+", 1)
                    item = {"method": method, "video_id": video, "expression_id": expression_id}
                    for key in ("HOTA___AUC", "HOTA(0)", "HOTA", "DetA___AUC", "AssA___AUC", "IDF1", "DetRe___AUC", "DetPr___AUC", "GT_Dets", "Dets", "IDSW"):
                        if key in row:
                            try: item[key] = float(row[key])
                            except ValueError: item[key] = None
                    item["HOTA"] = item.get("HOTA___AUC", item.get("HOTA(0)", item.get("HOTA")))
                    item["DetA"] = item.get("DetA___AUC")
                    item["AssA"] = item.get("AssA___AUC")
                    item["DetRe"] = item.get("DetRe___AUC")
                    item["DetPr"] = item.get("DetPr___AUC")
                    expression_rows.append(item)
                    video_values[(method, video)].append(item)
        stats = metadata.get("method_stats", {}).get(method, {})
        derived_updates = updates.get(method, {})
        derived_probe = probe.get(method, {})
        main = summaries.get(method, {})
        summary_rows.append({
            "run": run_dir.name, "dataset": "refer_bdd", "method": method,
            "source_checkpoint_hash": metadata.get("source", {}).get("checkpoint_sha256"),
            "video_count": audit.get("totals", {}).get("videos"),
            "video_expression_count": metadata.get("video_expression_count"),
            "frame_query_count": metadata.get("frame_query_count"),
            "HOTA": main.get("HOTA"), "DetA": main.get("DetA"), "AssA": main.get("AssA"), "IDF1": main.get("IDF1"),
            "delta_HOTA_vs_FROZEN": None if main.get("HOTA") is None or frozen.get("HOTA") is None else main["HOTA"] - frozen["HOTA"],
            "delta_IDF1_vs_FROZEN": None if main.get("IDF1") is None or frozen.get("IDF1") is None else main["IDF1"] - frozen["IDF1"],
            "optimizer_steps": derived_updates.get("optimizer_steps", stats.get("optimizer_steps", 0)),
            "probe_calls": derived_probe.get("probe_calls", stats.get("probe_calls", 0)),
            "selected_rows": stats.get("selected_rows", 0), "candidate_total": stats.get("candidate_total", 0),
            "fps": stats.get("end_to_end_fps_shared_source"), "p95_latency_sec": stats.get("p95_latency_sec"),
            "peak_memory_allocated_bytes": stats.get("peak_memory_allocated_bytes"),
            "metric_scope": "all_8_videos_673_expressions" if metadata.get("video_expression_count") == 673 else "pilot",
        })
    per_video = []
    for (method, video), values in sorted(video_values.items()):
        per_video.append({"method": method, "video_id": video, "expressions": len(values), "GT_Dets": sum(item.get("GT_Dets", 0) or 0 for item in values), **{key: mean([item[key] for item in values if item.get(key) is not None]) if any(item.get(key) is not None for item in values) else None for key in ("HOTA", "DetA", "AssA", "IDF1", "DetRe", "DetPr")}, "aggregation": "macro_expression"})
    return summary_rows, per_video, expression_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta_refer_bdd.yaml")
    parser.add_argument("--evaluation-manifest", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/audits/refer_bdd_assets.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/bdd_crossdomain")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation_manifest = (args.evaluation_manifest.resolve() if args.evaluation_manifest else resolve(config["data"]["evaluation_manifest"]))
    audit = json.loads(args.audit.resolve().read_text(encoding="utf-8"))
    candidate, frame_targets = source_diagnostics(config, run_dir, evaluation_manifest)
    probe, updates, pseudo = trace_diagnostics(run_dir, frame_targets, config)
    summary, per_video, per_expression = metric_rows(run_dir, config, audit, updates, probe)
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "candidate_recall.json", candidate)
    dump(output / "probe_coverage.json", {"format": "locatemot-refer-bdd-probe-coverage-v1", "run_dir": str(run_dir), "by_method": probe, "runtime_gt_isolation": True})
    dump(output / "update_statistics.json", {"format": "locatemot-refer-bdd-update-statistics-v1", "run_dir": str(run_dir), "by_method": updates, "runtime_gt_isolation": True})
    dump(output / "pseudo_label_quality.json", {"format": "locatemot-refer-bdd-pseudo-quality-v1", "run_dir": str(run_dir), "by_method": pseudo, "comparison": "offline IoU against sealed GT; never used by runner"})
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    timing_stats = metadata.get("method_stats", {})
    for method in methods:
        timing_stats.setdefault(method, {})["optimizer_steps"] = updates.get(method, {}).get(
            "optimizer_steps", timing_stats.get(method, {}).get("optimizer_steps", 0)
        )
        timing_stats.setdefault(method, {})["probe_calls"] = probe.get(method, {}).get(
            "probe_calls", timing_stats.get(method, {}).get("probe_calls", 0)
        )
    dump(output / "timing.json", {"format": "locatemot-refer-bdd-timing-v1", "run_dir": str(run_dir), "wall_seconds": metadata.get("wall_seconds"), "frame_query_count": metadata.get("frame_query_count"), "method_stats": timing_stats, "gpu_uuid": metadata.get("gpu_uuid"), "cuda_visible_devices": metadata.get("cuda_visible_devices")})
    for path, rows in ((output / "summary.csv", summary), (output / "per_video.csv", per_video), (output / "per_expression.csv", per_expression)):
        if not rows:
            path.write_text("\n", encoding="utf-8")
            continue
        fields = list(rows[0])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"summary": summary, "candidate_scope": candidate["scope"], "output": str(output)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
