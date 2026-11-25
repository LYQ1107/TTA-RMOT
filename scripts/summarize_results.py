#!/usr/bin/env python3
"""Aggregate sealed LocateMOT runs without inventing missing metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_rows(run_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    metadata_path = run_dir / "run_metadata.json"
    evaluation_path = run_dir / "evaluation" / "evaluation.json"
    if not metadata_path.is_file():
        return [], [], [], []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else {}
    eval_by_method = {item["method"]: item.get("summary", {}) for item in evaluation.get("methods", [])}
    base = {
        "run": run_dir.name, "dataset": metadata.get("config", "").split("/")[-1],
        "config": metadata.get("config"), "source_checkpoint_hash": metadata.get("source", {}).get("checkpoint_sha256"),
        "video_expression_count": metadata.get("video_expression_count"), "frame_query_count": metadata.get("frame_query_count"),
        "gpu_uuid": metadata.get("gpu_uuid"), "cuda_visible_devices": metadata.get("cuda_visible_devices"),
    }
    rows: list[dict] = []
    for method, stats in metadata.get("method_stats", {}).items():
        metrics = eval_by_method.get(method, {})
        row = {**base, "method": method, **metrics, "HOTA_reason": None if metrics.get("HOTA") is not None else "not_evaluated_or_evaluation_missing",
               "DetA_reason": None if metrics.get("DetA") is not None else "not_evaluated_or_evaluation_missing",
               "AssA_reason": None if metrics.get("AssA") is not None else "not_evaluated_or_evaluation_missing",
               "DetRe_reason": None if metrics.get("DetRe") is not None else "not_evaluated_or_evaluation_missing",
               "DetPr_reason": None if metrics.get("DetPr") is not None else "not_evaluated_or_evaluation_missing",
               "IDF1_reason": None if metrics.get("IDF1") is not None else "not_evaluated_or_evaluation_missing",
               "end_to_end_fps_shared_source": stats.get("end_to_end_fps_shared_source"),
               "p95_latency_sec": stats.get("p95_latency_sec"),
               "peak_memory_allocated_bytes": stats.get("peak_memory_allocated_bytes"),
               "peak_memory_reserved_bytes": stats.get("peak_memory_reserved_bytes"),
               "probe_calls": stats.get("probe_calls"), "optimizer_steps": stats.get("optimizer_steps"),
               "supported_query_ratio": None, "supported_candidate_ratio": (
                   stats.get("supported_candidates", 0) / stats.get("candidate_total", 1)
                   if stats.get("candidate_total", 0) else None),
               "delta_vs_frozen": None, "delta_vs_probe_ST": None, "delta_vs_probe_cache": None,
               "metric_scope": "sealed_run" if evaluation.get("status") == "EVALUATION_COMPLETE" else "prediction_only"}
        rows.append(row)
    frozen = next((row.get("HOTA") for row in rows if row["method"] == "FROZEN"), None)
    probe_st = next((row.get("HOTA") for row in rows if row["method"] == "PROBE_ST"), None)
    probe_cache = next((row.get("HOTA") for row in rows if row["method"] == "PROBE_CACHE"), None)
    for row in rows:
        if row.get("HOTA") is not None:
            row["delta_vs_frozen"] = row["HOTA"] - frozen if frozen is not None else None
            row["delta_vs_probe_ST"] = row["HOTA"] - probe_st if probe_st is not None else None
            row["delta_vs_probe_cache"] = row["HOTA"] - probe_cache if probe_cache is not None else None

    per_expression: list[dict] = []
    per_video: list[dict] = []
    for method in metadata.get("methods", []):
        detailed = run_dir / "predictions" / method / "pedestrian_detailed.csv"
        if not detailed.is_file():
            continue
        with detailed.open(encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                sequence = item.get("seq") or item.get("sequence")
                if not sequence or sequence == "COMBINED":
                    continue
                if "+" in sequence:
                    video, expression = sequence.split("+", 1)
                else:
                    video, expression = sequence, ""
                record = {"run": run_dir.name, "method": method, "video_id": video,
                          "expression_id": expression, **item}
                detailed_metric_keys = {
                    "HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC",
                    "DetRe": "DetRe___AUC", "DetPr": "DetPr___AUC", "IDF1": "IDF1",
                }
                for metric, source_key in detailed_metric_keys.items():
                    try:
                        record[metric] = float(item[source_key]) * 100.0
                    except (KeyError, TypeError, ValueError):
                        record[metric] = None
                per_expression.append(record)
    by_video: dict[tuple[str, str], list[dict]] = {}
    for row in per_expression:
        by_video.setdefault((row["run"], row["method"], row["video_id"]), []).append(row)
    for (run, method, video), values in sorted(by_video.items()):
        numeric: dict[str, Any] = {"run": run, "method": method, "video_id": video, "expression_count": len(values)}
        for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1"):
            numbers = []
            for value in values:
                try:
                    numbers.append(float(value[key]))
                except (KeyError, TypeError, ValueError):
                    pass
            numeric[key] = sum(numbers) / len(numbers) if numbers else None
        per_video.append(numeric)

    pseudo = []
    transitions = []
    trace_path = metadata.get("trace")
    if trace_path and Path(trace_path).is_file():
        for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            valid = max(int(item.get("valid_candidates", 0)), 1)
            for method, state in item.get("methods", {}).items():
                update = state.get("last_update") or {}
                transitions.append({"run": run_dir.name, "method": method, "video_id": item.get("video_id"),
                                    "expression_id": item.get("expression_id"), "frame_index": item.get("frame_index"),
                                    "reason": update.get("reason"), "stepped": update.get("stepped"),
                                    "loss": update.get("loss"), "parameter_delta": update.get("parameter_delta")})
                pseudo.append({"run": run_dir.name, "method": method, "video_id": item.get("video_id"),
                               "expression_id": item.get("expression_id"), "frame_index": item.get("frame_index"),
                               "evidence_pos": None, "evidence_neg": None, "evidence_unknown": None,
                               "valid_candidates": valid,
                               "quality_status": "GT_join_not_materialized_in_summary"})
    return rows, per_video, per_expression, pseudo, transitions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    runs_root = args.runs_root.resolve()
    all_rows: list[dict] = []
    all_video: list[dict] = []
    all_expression: list[dict] = []
    all_pseudo: list[dict] = []
    all_transitions: list[dict] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        rows, videos, expressions, pseudo, transitions = _run_rows(run_dir)
        all_rows.extend(rows); all_video.extend(videos); all_expression.extend(expressions)
        all_pseudo.extend(pseudo); all_transitions.extend(transitions)
    output = args.output_dir.resolve()
    fields = ["run", "dataset", "method", "source_checkpoint_hash", "video_expression_count", "frame_query_count",
              "HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1", "delta_vs_frozen", "delta_vs_probe_ST",
              "delta_vs_probe_cache", "end_to_end_fps_shared_source", "p95_latency_sec",
              "peak_memory_allocated_bytes", "peak_memory_reserved_bytes", "probe_calls", "optimizer_steps",
              "supported_query_ratio", "supported_candidate_ratio", "metric_scope", "HOTA_reason", "DetA_reason",
              "AssA_reason", "DetRe_reason", "DetPr_reason", "IDF1_reason"]
    _write_csv(output / "summary.csv", all_rows, fields)
    _write_csv(output / "per_video.csv", all_video, ["run", "method", "video_id", "expression_count", "HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1"])
    expression_fields = sorted({key for row in all_expression for key in row}) if all_expression else ["run", "method", "video_id", "expression_id"]
    _write_csv(output / "per_expression.csv", all_expression, expression_fields)
    _write_csv(output / "pseudo_label_quality.csv", all_pseudo,
               ["run", "method", "video_id", "expression_id", "frame_index", "evidence_pos", "evidence_neg", "evidence_unknown", "valid_candidates", "quality_status"])
    _write_csv(output / "state_transition_diagnostics.csv", all_transitions,
               ["run", "method", "video_id", "expression_id", "frame_index", "reason", "stepped", "loss", "parameter_delta"])
    payload = {"status": "SUMMARY_COMPLETE", "runs": len({row.get("run") for row in all_rows}),
               "summary": str((output / "summary.csv").resolve()), "note": "null metrics remain null when no sealed evaluator output exists"}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
