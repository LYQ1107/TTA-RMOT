#!/usr/bin/env python3
"""At most four GPUs, explicit shared-memory budgets, resumable episodes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_shards import (weighted_manifest, save_manifest, gpu_snapshot,
                                       eligible_gpus, rebalance_unstarted, sealed_shards)
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch-full", action="store_true")
    args = p.parse_args()
    env = os.environ
    run_root = Path(env.get("PF_RUN_ROOT", str(ROOT / "results/runs/pf_rank_full"))).resolve()
    method = env.get("PF_METHOD", "PF_RANK")
    tracker = Path(env.get("PF_TRACKER_ROOT", "/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    dataset = Path(env.get("PF_DATASET_ROOT", "/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    checkpoint = env.get("PF_CHECKPOINT", "/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth")
    source = env.get("PF_SOURCE_RESULT", str(ROOT / "results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json"))
    supplement = env.get("PF_SOURCE_SUPPLEMENT", str(ROOT / "results/runs/pf_frozen_complete_20261025/source_supplement.json"))
    matched_reference = env.get("PF_MATCHED_FROZEN_ROOT")
    if args.launch_full and (env.get("PF_LAUNCH_FULL") != "YES" or env.get("FLEXHOOK_FULL_RUN_AUTHORIZED") != "YES"):
        print("BLOCKED: both explicit full-run authorization switches are required", file=sys.stderr)
        return 3
    manifest = weighted_manifest(tracker, dataset)
    for shard in manifest["shards"]:
        print(f"Logical GPU slot {shard['shard']}: {shard['videos']}; rows={shard['tracker_rows']}; episodes={shard['episodes']}", flush=True)
    if not args.launch_full:
        print("DRY_RUN_ONLY: no GPU process created; up to four actually idle GPUs at launch")
        return 0
    if method not in {"ZERO_UPDATE", "RELATIVE_ONLY", "PF_RANK", "PF_RANK_GUARD", "PF_IDENTITY", "PF_HYBRID"}:
        raise ValueError("unknown positive-free method")
    shared = env.get("PF_ALLOW_SHARED_GPU") == "YES"
    budget = int(env.get("PF_GPU_MEMORY_BUDGET_MIB", "24576"))
    headroom = int(env.get("PF_GPU_HEADROOM_MIB", "8192"))
    def eligible(snapshot, requested):
        return eligible_gpus(snapshot, shared=shared, requested=requested,
                             memory_budget_mib=budget, headroom_mib=headroom)
    run_root.mkdir(parents=True, exist_ok=True)
    save_manifest(run_root / "manifest.json", manifest)
    active_path = run_root / "active_manifest.json"
    if active_path.exists():
        manifest = json.loads(active_path.read_text())
    elif env.get("PF_REBALANCE_UNSTARTED") == "YES":
        sealed = sealed_shards(run_root, manifest, method)
        if sealed and len(sealed) < len(manifest["shards"]):
            # Never move partially started episodes to a different owner.
            for shard in manifest["shards"]:
                if shard["shard"] not in sealed and (run_root / f"shard{shard['shard']}").exists():
                    raise ValueError("cannot repartition a started shard; resume its existing episode plan")
            manifest = rebalance_unstarted(manifest, sealed)
            save_manifest(active_path, manifest)
    sealed = sealed_shards(run_root, manifest, method)
    previous_path = run_root / "launcher_status.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous:
        atomic_json(run_root / f"launcher_attempt_{time.time_ns()}.json", previous)
    elapsed_before = previous.get("wall_seconds", 0.)
    requested = env.get("PF_GPUS", "auto")
    requested = None if requested == "auto" else [int(v) for v in requested.split(",")]
    if requested is not None and (len(set(requested)) != len(requested) or len(requested) > 4):
        raise ValueError("PF_GPUS must contain at most four distinct physical indices")
    if method != "RELATIVE_ONLY":
        snapshot = gpu_snapshot()
        atomic_json(run_root / "initial_gpu_resources.json", snapshot)
        available = eligible(snapshot, requested)
        if not available:
            raise RuntimeError("no GPU with the required memory budget; do not terminate other tasks")
        print(json.dumps(dict(event="gpu_policy", shared=shared, memory_budget_mib=budget,
                              headroom_mib=headroom, selected=available)), flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/pf_preflight.py"),
                    "--dance-root", str(dataset), "--tracker-root", str(tracker),
                    "--checkpoint", checkpoint, "--check-checkpoint-hash", "--strict",
                    "--source-result", source, "--vendor", str(ROOT / "vendor/FlexHook"),
                    "--source-supplement", supplement,
                    "--tracker-receipt", "/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/official_detr_neuralsort_receipt.json",
                    "--output", str(run_root / "preflight.json")], check=True)
    pending = [s for s in manifest["shards"] if s["shard"] not in sealed]
    running = []
    finished = [dict(r, reused=True) for r in previous.get("completed", [])
                if r["shard"] in sealed and r.get("exit_status") == 0]
    for shard in manifest["shards"]:
        print(json.dumps(dict(event="active_plan", **shard, reused=shard["shard"] in sealed)), flush=True)
    failed = False
    started = time.time()
    last_notice = 0.
    locks_root = ROOT / "results/pf_four_gpu/gpu_locks"
    locks_root.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for item in list(running):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            if item["lock"] is not None:
                item["lock"].close()
            record = {k: v for k, v in item.items() if k not in {"process", "log", "lock"}}
            record.update(exit_status=code, ended_at=time.time())
            finished.append(record)
            running.remove(item)
            failed |= code != 0
            print(json.dumps(dict(event="shard_finished", **record)), flush=True)
        if failed and not running:
            atomic_json(run_root / "launcher_status.json", dict(method=method, completed=finished,
                        status="failed", wall_seconds=elapsed_before+time.time()-started))
            break
        if pending and not failed:
            if method == "RELATIVE_ONLY":
                free = [dict(index=None, uuid="CPU") for _ in range(max(0, 4-len(running)))]
            else:
                used = {r["gpu_uuid"] for r in running}
                free = [r for r in eligible(gpu_snapshot(), requested) if r["uuid"] not in used]
            for gpu in free:
                if not pending or len(running) >= 4:
                    break
                lock = None
                if gpu["index"] is not None:
                    lock = (locks_root / (gpu["uuid"] + ".lock")).open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close(); continue
                    if gpu["uuid"] not in {r["uuid"] for r in eligible(gpu_snapshot(), [gpu["index"]])}:
                        lock.close(); continue
                shard = pending.pop(0)
                shard_root = run_root / f"shard{shard['shard']}"
                shard_root.mkdir(parents=True, exist_ok=True)
                child_env = dict(env, CUDA_VISIBLE_DEVICES=("" if gpu["index"] is None else gpu["uuid"]),
                                 PF_GPU_UUID=gpu["uuid"], PF_PHYSICAL_GPU=str(gpu["index"]),
                                 PF_GPU_MEMORY_BUDGET_MIB=str(budget),
                                 OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
                command = [sys.executable, str(ROOT / "scripts/run_flexhook_positivefree.py"),
                           "--scope", "full", "--method", method, "--vendor", str(ROOT / "vendor/FlexHook"),
                           "--dataset-root", str(dataset), "--tracker-root", str(tracker),
                           "--checkpoint", checkpoint, "--source-result", source,
                           "--source-supplement", supplement,
                           "--max-expressions", "0", "--max-frames", "0", "--resume", "--frozen-cache",
                           "--videos", *shard["videos"], "--output-root", str(shard_root)]
                if matched_reference and method != "ZERO_UPDATE":
                    command += ["--matched-frozen-root", matched_reference]
                    if method != "RELATIVE_ONLY":
                        command += ["--strict-initial-sign"]
                log = (shard_root / "stdout.log").open("a")
                log.write(json.dumps(dict(event="launch_attempt", time=time.time(), command=command, gpu=gpu))+"\n")
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                running.append(dict(shard=shard["shard"], gpu=gpu["index"], gpu_uuid=gpu["uuid"], pid=process.pid,
                                    started_at=time.time(), process=process, log=log, lock=lock))
                print(json.dumps({k: v for k, v in running[-1].items() if k not in {"process", "log", "lock"}}), flush=True)
        atomic_json(run_root / "launcher_status.json", dict(method=method, started_at=started,
            wall_seconds=elapsed_before+time.time()-started, attempt_wall_seconds=time.time()-started,
            shared_gpu_authorized=shared, memory_budget_mib=budget, headroom_mib=headroom,
            reused_shards=sealed, completed=finished, pending=[s["shard"] for s in pending],
            running=[{k:v for k,v in r.items() if k not in {"process", "log", "lock"}} for r in running],
            status="failed" if failed else ("running" if pending or running else "complete")))
        if time.time()-last_notice >= 60:
            completed_episodes = sum(sum(1 for line in path.open() if line.strip())
                                     for path in run_root.glob(f"shard*/{method}/completed_episodes.jsonl"))
            print(json.dumps(dict(event="progress", method=method, completed_episodes=completed_episodes,
                                  expected_episodes=425, elapsed_seconds=time.time()-started,
                                  running_shards=[r["shard"] for r in running])), flush=True)
            last_notice = time.time()
        if running or pending and not failed:
            time.sleep(10)
    if failed:
        raise RuntimeError("one or more shards failed; inspect retained logs and resume unfinished episodes")
    print("ALL SHARDS COMPLETE; audit and merge before independent evaluation", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
