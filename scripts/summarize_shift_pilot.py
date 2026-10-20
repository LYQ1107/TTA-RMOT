#!/usr/bin/env python3
"""Create compact, condition-level ShiftRMOT pilot tables from sealed runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str, scale: float = 1.0) -> float | None:
    try:
        return float(row[key]) * scale
    except (KeyError, TypeError, ValueError):
        return None


def _manifest_metadata(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[str(row["condition_id"])] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ShiftRMOT-v1/manifests/paired_conditions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/shift_rmot_pilot")
    args = parser.parse_args()
    conditions = _manifest_metadata(args.manifest.resolve())
    condition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run_arg in args.run_dir:
        run_dir = run_arg.resolve()
        run_name = run_dir.name
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run_dir / "evaluation/evaluation.json").read_text(encoding="utf-8"))
        eval_by_method = {item["method"]: item.get("summary", {}) for item in evaluation.get("methods", [])}
        for method, stats in metadata["method_stats"].items():
            metrics = eval_by_method.get(method, {})
            summary_rows.append({
                "run": run_name, "structure": "relative_margin" if "relative" in run_name else "absolute_gate",
                "method": method, "HOTA": metrics.get("HOTA"), "DetA": metrics.get("DetA"),
                "AssA": metrics.get("AssA"), "DetRe": metrics.get("DetRe"),
                "DetPr": metrics.get("DetPr"), "IDF1": metrics.get("IDF1"),
                "selected_rows": stats.get("selected_rows"), "optimizer_steps": stats.get("optimizer_steps"),
                "probe_calls": stats.get("probe_calls"), "fps": stats.get("end_to_end_fps_shared_source"),
                "peak_memory_gb": round(float(stats.get("peak_memory_allocated_bytes", 0)) / 2**30, 4),
                "gpu_uuid": metadata.get("gpu_uuid"), "frame_query_count": metadata.get("frame_query_count"),
                "condition_count": metadata.get("video_expression_count"),
            })
            detailed = run_dir / "predictions" / method / "pedestrian_detailed.csv"
            if not detailed.is_file():
                continue
            with detailed.open(encoding="utf-8") as handle:
                for item in csv.DictReader(handle):
                    sequence = item.get("seq", "")
                    if not sequence or sequence == "COMBINED" or "+" not in sequence:
                        continue
                    video_id, condition_id = sequence.split("+", 1)
                    condition = conditions.get(condition_id, {})
                    condition_rows.append({
                        "run": run_name,
                        "structure": "relative_margin" if "relative" in run_name else "absolute_gate",
                        "method": method, "video_id": video_id, "condition_id": condition_id,
                        "base_expression_id": condition.get("base_expression_id"),
                        "condition_family": condition.get("condition_family"),
                        "visual_shift": condition.get("visual_shift"),
                        "visual_severity": condition.get("visual_severity"),
                        "language_variant_id": condition.get("language_variant_id", ""),
                        "HOTA_percent": _float(item, "HOTA___AUC", 100.0),
                        "DetA_percent": _float(item, "DetA___AUC", 100.0),
                        "AssA_percent": _float(item, "AssA___AUC", 100.0),
                        "DetRe_percent": _float(item, "DetRe___AUC", 100.0),
                        "DetPr_percent": _float(item, "DetPr___AUC", 100.0),
                        "IDF1_percent": _float(item, "IDF1", 100.0),
                        "Dets": item.get("Dets"), "GT_Dets": item.get("GT_Dets"),
                    })

    family_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        family_groups[(row["run"], row["method"], row["condition_family"])].append(row)
    family_rows: list[dict[str, Any]] = []
    for (run, method, family), values in sorted(family_groups.items()):
        output: dict[str, Any] = {"run": run, "structure": values[0]["structure"], "method": method,
                                  "condition_family": family, "condition_count": len(values)}
        for metric in ("HOTA_percent", "DetA_percent", "AssA_percent", "DetRe_percent", "DetPr_percent", "IDF1_percent"):
            numbers = [float(row[metric]) for row in values if row[metric] is not None]
            output[metric] = sum(numbers) / len(numbers) if numbers else None
        family_rows.append(output)

    summary_rows.sort(key=lambda row: (row["run"], row["method"]))
    condition_rows.sort(key=lambda row: (row["run"], row["video_id"], row["condition_id"], row["method"]))
    family_rows.sort(key=lambda row: (row["run"], row["condition_family"], row["method"]))
    out = args.output_dir.resolve()
    _write(out / "method_summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else [])
    _write(out / "condition_metrics.csv", condition_rows, list(condition_rows[0]) if condition_rows else [])
    _write(out / "condition_family_summary.csv", family_rows, list(family_rows[0]) if family_rows else [])

    by_structure_method = {(row["structure"], row["method"]): row for row in summary_rows}
    comparison: dict[str, Any] = {"format": "shift-rmot-pilot-analysis-v1", "runs": [str(path.resolve()) for path in args.run_dir],
                                   "condition_count_per_run": sorted({int(row["condition_count"]) for row in summary_rows}),
                                   "frame_query_count_per_run": sorted({int(row["frame_query_count"]) for row in summary_rows})}
    absolute_frozen = by_structure_method.get(("absolute_gate", "FROZEN"), {})
    relative_frozen = by_structure_method.get(("relative_margin", "FROZEN"), {})
    comparison["structure_change_frozen_delta"] = {
        metric: (relative_frozen.get(metric) - absolute_frozen.get(metric))
        if relative_frozen.get(metric) is not None and absolute_frozen.get(metric) is not None else None
        for metric in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1")
    }
    comparison["relative_method_delta_vs_frozen"] = {}
    for method in sorted({row["method"] for row in summary_rows}):
        row = by_structure_method.get(("relative_margin", method), {})
        comparison["relative_method_delta_vs_frozen"][method] = {
            metric: (row.get(metric) - relative_frozen.get(metric))
            if row.get(metric) is not None and relative_frozen.get(metric) is not None else None
            for metric in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1")
        }
    (out / "analysis.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SHIFT_PILOT_SUMMARY_COMPLETE", "output_dir": str(out),
                      "condition_rows": len(condition_rows), "family_rows": len(family_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
