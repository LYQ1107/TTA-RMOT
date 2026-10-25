#!/usr/bin/env python3
"""Whole-video, heterogeneous scheduling from source-only ZERO timings, never GT."""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_shards import weighted_manifest, save_manifest, eligible_gpus, gpu_snapshot


def measured_rates(status, manifest):
    by_shard = {s["shard"]: s for s in manifest["shards"]}
    totals = {}
    for record in status["completed"]:
        if record["exit_status"] != 0 or record["gpu_uuid"] == "CPU":
            continue
        rate = totals.setdefault(record["gpu_uuid"], [0., 0.])
        rate[0] += by_shard[record["shard"]]["estimated_forward_calls"]
        rate[1] += record["ended_at"]-record["started_at"]
    return {uuid: frames/seconds for uuid, (frames, seconds) in totals.items() if seconds > 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--zero-root", type=Path, default=ROOT / "results/runs/pf_zero_completebank_full_20261025")
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args = p.parse_args()
    active = args.run_root / "active_manifest.json"
    if active.exists() or list(args.run_root.glob("shard*")):
        print("Retain existing episode ownership; no timing-based repartition on resume")
        return
    status = json.loads((args.zero_root / "launcher_status.json").read_text())
    source_plan = args.zero_root / "active_manifest.json"
    if not source_plan.exists():
        source_plan = args.zero_root / "manifest.json"
    source_manifest = json.loads(source_plan.read_text())
    if status["status"] != "complete" or sum(s["episodes"] for s in source_manifest["shards"]) != 425:
        raise ValueError("only the complete ZERO timings may drive the next fresh schedule")
    rates = measured_rates(status, source_manifest)
    requested = os.environ.get("PF_GPUS", "auto")
    requested = None if requested == "auto" else [int(i) for i in requested.split(",")]
    available = eligible_gpus(gpu_snapshot(), shared=os.environ.get("PF_ALLOW_SHARED_GPU") == "YES",
                              requested=requested,
                              memory_budget_mib=int(os.environ.get("PF_GPU_MEMORY_BUDGET_MIB", "24576")),
                              headroom_mib=int(os.environ.get("PF_GPU_HEADROOM_MIB", "8192")))
    if not available:
        print("No available GPU; retain normal resource-wait/launch checks")
        return
    # Real timings showed active-frame overhead dominates this <=22 candidate bank.
    # A new GPU has an explicitly estimated (median), not falsely measured rate.
    median = statistics.median(rates.values())
    capacities = [rates.get(g["uuid"], median) for g in available]
    plan = weighted_manifest(args.tracker_root, args.dataset_root, num_shards=len(available),
                             row_weight=0., frame_weight=1., capacities=capacities)
    plan["protocol"] = "whole-video LPT normalized by source-only measured per-GPU active-frame throughput"
    plan["profile_source"] = str(args.zero_root.resolve())
    plan["metrics_or_gt_used"] = False
    plan["timing_caveat"] = "shared-GPU contention can change; timings are estimates, not reserved throughput"
    for shard, gpu, rate in zip(plan["shards"], available, capacities):
        shard.update(preferred_gpu_uuid=gpu["uuid"], preferred_physical_gpu=gpu["index"],
                     estimated_active_frame_fps=rate, rate_measured=gpu["uuid"] in rates,
                     estimated_seconds=shard["cost"]/rate)
    save_manifest(active, plan)
    print(json.dumps(dict(event="profiled_schedule", source_only_rates=rates, shards=plan["shards"])), flush=True)


if __name__ == "__main__": main()
