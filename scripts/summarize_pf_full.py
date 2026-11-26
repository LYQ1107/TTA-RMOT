#!/usr/bin/env python3
"""Publish compact official metrics, pooled per-video metrics and actual costs."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json

METHOD_ORDER = ("ZERO_UPDATE", "RELATIVE_ONLY", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID")
METRICS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1", "IDSW", "Dets", "IDs")


def report_stats(metadata):
    """Keep additive counters, but read model dimensions once per replica."""
    stats = dict(metadata.get("aggregate_stats") or {})
    for key in ("trainable_parameters", "pcd_channels", "feature_channels"):
        values = {s.get("stats", {}).get(key) for s in metadata.get("shards", [])}
        values.discard(None)
        if len(values) > 1:
            raise ValueError(f"inconsistent per-replica {key}: {values}")
        if values:
            stats[key] = values.pop()
    return stats


def summarize(root, method, publish):
    folder = Path(root) / method
    audit = json.loads((folder / "full_audit.json").read_text())
    if audit["status"] != "PASS" or audit["valid_episodes"] != 425:
        raise ValueError("cannot summarize incomplete predictions")
    headers, values = (folder / "results/pedestrian_summary.txt").read_text().splitlines()[:2]
    official = dict(zip(headers.split(), map(float, values.split())))
    with (folder / "results/pedestrian_detailed.csv").open() as stream:
        detailed = list(csv.DictReader(stream))
    per_episode = [r for r in detailed if r["seq"] != "COMBINED"]
    if len(per_episode) != 425 or len({r["seq"] for r in per_episode}) != 425:
        raise ValueError("official TrackEval did not evaluate all 425 unique sequences")
    sys.path.insert(0, str(ROOT / "vendor/FlexHook/TrackEval"))
    import numpy as np
    from trackeval.metrics import HOTA, Identity
    hota, identity = HOTA(), Identity()
    def unpack(row, metric):
        result = {f: float(row[f]) for f in metric.float_fields+metric.integer_fields}
        for f in metric.float_array_fields+metric.integer_array_fields:
            result[f] = np.asarray([float(row[f+"___"+str(int(100*x))]) for x in metric.array_labels])
        return result
    compact = []
    for row in per_episode:
        item = {"sequence": row["seq"]}
        for key in METRICS:
            if key in {"HOTA", "DetA", "AssA", "DetRe", "DetPr"}:
                item[key] = 100*float(row[key+"___AUC"])
            elif key == "IDF1": item[key] = 100*float(row[key])
            else: item[key] = float(row[key])
        compact.append(item)
    per_video = []
    for video in sorted({r["seq"].split("+")[0] for r in per_episode}):
        group = [r for r in per_episode if r["seq"].split("+")[0] == video]
        h = hota.combine_sequences({r["seq"]: unpack(r, hota) for r in group})
        ident = identity.combine_sequences({r["seq"]: unpack(r, identity) for r in group})
        per_video.append(dict(video=video, sequences=len(group),
            **{key: float(100*np.mean(h[key])) for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")},
            IDF1=100*float(ident["IDF1"]), IDSW=sum(float(r["IDSW"]) for r in group)))
    metadata = json.loads((folder / "run_metadata.json").read_text())
    seconds = sum(float(e.get("wall_seconds", 0)) for e in metadata["episodes"])
    shards = metadata.get("shards", [])
    peaks = [s.get("stats", {}).get("gpu_peak_allocated_bytes", 0) for s in shards]
    launcher_path = Path(str(root).removesuffix("_merged")) / "launcher_status.json"
    launcher = json.loads(launcher_path.read_text()) if launcher_path.exists() else {}
    compute_seconds = sum(r["ended_at"]-r["started_at"] for r in launcher.get("completed", []) if r["gpu_uuid"] != "CPU")
    result = dict(method=method, official_metrics=official,
                  baseline="complete-bank ZERO_UPDATE, NOT legacy 8.206", metrics={k:official[k] for k in METRICS},
                  full_audit={k:v for k,v in audit.items() if k != "episodes"},
                  cost=dict(total_episode_seconds=seconds, aggregate_active_frame_fps=audit["frames"]/max(seconds,1e-12),
                            worker_active_frame_fps=audit["frames"]/max(seconds,1e-12),
                            scheduler_wall_active_frame_fps=audit["frames"]/max(launcher.get("wall_seconds",seconds),1e-12),
                            wall_seconds=launcher.get("wall_seconds", seconds), gpu_hours=compute_seconds/3600,
                            wall_definition="launcher loop, including its waits; excludes preflight, evaluation and attribution",
                            peak_vram_bytes=max(peaks, default=0),
                            peak_vram_measurement="PyTorch max_memory_allocated; excludes CUDA context/reserved memory",
                            gpu_uuids=sorted({r["gpu_uuid"] for r in launcher.get("completed", []) if r["gpu_uuid"] != "CPU"})),
                  aggregate_stats=report_stats(metadata), raw_output_root=str(Path(root).resolve()),
                  development_target=True, target_gt_used_only_for_sealed_offline_evaluation=True)
    result["structural_stats_definition"] = "per-replica constants from shard stats, not additive episode deltas"
    if metadata.get("derived_from"):
        rank_root = Path(metadata["derived_from"])
        rank_metadata = json.loads((rank_root / "PF_RANK/run_metadata.json").read_text())
        result["execution_mode"] = "CPU derivation of sealed PF_RANK margins"
        result["reused_rank_stats"] = report_stats(rank_metadata)
        result["reused_rank_output_root"] = str(rank_root)
        result["aggregate_stats"]["rank_optimizer_steps_reused"] = result["reused_rank_stats"]["optimizer_steps"]
        result["aggregate_stats"]["additional_optimizer_steps"] = 0
        result["cost"]["cost_scope"] = "incremental CPU derivation only; requires the full PF_RANK run and separate GPU equivalence check"
    else:
        result["execution_mode"] = "direct inference/adaptation"
        if method == "RELATIVE_ONLY":
            result["execution_mode"] = "CPU selection from sealed ZERO_UPDATE margins"
            result["cost"]["cost_scope"] = "incremental CPU selection only; requires the full ZERO_UPDATE source inference"
    evidence = result.get("reused_rank_stats", result["aggregate_stats"])
    opportunities = evidence.get("update_opportunities", 0)
    result["effective_evidence_coverage"] = dict(
        definition="fraction of registered update opportunities, not all frames; reused Rank for CPU Guard",
        update_opportunities=opportunities,
        evidence_eligible_fraction=(opportunities-evidence.get("skipped_no_evidence",0)-evidence.get("skipped_multi_batch",0))/opportunities if opportunities else None,
        gradient_attempt_fraction=evidence.get("gradient_attempts",0)/opportunities if opportunities else None,
        accepted_update_fraction=evidence.get("optimizer_steps",0)/opportunities if opportunities else None)
    destination = Path(publish) / method.lower()
    destination.mkdir(parents=True, exist_ok=True)
    raw_root = Path(str(root).removesuffix("_merged"))
    for name in ("manifest.json", "active_manifest.json", "launcher_status.json"):
        if (raw_root / name).is_file():
            shutil.copy2(raw_root / name, destination / name)
    for name in ("pedestrian_summary.txt", "pedestrian_detailed.csv"):
        shutil.copy2(folder / "results" / name, destination / name)
    shutil.copy2(Path(root) / "evaluation_metadata.json", destination / "evaluation_metadata.json")
    atomic_json(destination / "metrics.json", result)
    atomic_json(destination / "per_episode_metrics.json", compact)
    atomic_json(destination / "per_video_metrics.json", per_video)
    baseline_path = Path(publish) / "zero_update/metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())["metrics"]
        result["delta"] = {key: result["metrics"][key]-baseline[key] for key in METRICS}
        if method == "ZERO_UPDATE": result["interpretation"] = "MATCHED_COMPLETE_BANK_BASELINE"
        elif (method == "PF_IDENTITY" and result["metrics"]["AssA"] > baseline["AssA"]
              and result["metrics"]["DetRe"] < baseline["DetRe"] and result["metrics"]["HOTA"] <= baseline["HOTA"]
              and result["metrics"]["IDSW"] < baseline["IDSW"]):
            result["interpretation"] = "IDENTITY_CONSISTENCY_GAIN_WITH_REFERRING_RECALL_LOSS"
        elif result["metrics"]["AssA"] > baseline["AssA"] and result["metrics"]["DetRe"] < baseline["DetRe"]:
            result["interpretation"] = "ASSOCIATION_ONLY_GAIN_WITH_RECALL_LOSS"
        elif result["delta"]["HOTA"] > 0: result["interpretation"] = "HOTA_GAIN_REQUIRES_TARGET_RECOVERY_ATTRIBUTION"
        else: result["interpretation"] = "NO_OVERALL_HOTA_GAIN"
        atomic_json(destination / "metrics.json", result)
    reports = [json.loads((Path(publish)/m.lower()/"metrics.json").read_text()) for m in METHOD_ORDER
               if (Path(publish)/m.lower()/"metrics.json").exists()]
    lines = ["# Positive-Free full complete-bank results", "", "Development-target experiment; old Frozen HOTA 8.206 uses an incomplete bank and is historical only.", "",
             "| Method | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW | ΔHOTA |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        m = report["metrics"]
        lines.append("| "+report["method"]+" | "+" | ".join(f"{m[k]:.4f}" for k in METRICS[:7])+f" | {report.get('delta',{}).get('HOTA',0):+.4f} |")
    lines += ["", f"Completed methods: {len(reports)}/6. No missing method is treated as complete.", "",
              "Per-video metrics pool 17 sequences using official HOTA/Identity combine_sequences; they are not unweighted macro means."]
    (ROOT / "docs/PF_FULL_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"full_audit"}}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--publish-root", type=Path, default=ROOT / "results/positive_free/full")
    args = p.parse_args()
    summarize(args.output_root, args.method, args.publish_root)


if __name__ == "__main__": main()
