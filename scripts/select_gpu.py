#!/usr/bin/env python3
"""Select an idle-enough physical GPU for a single-card run.

Selection is read-only.  It filters by free memory first, then prefers fewer
compute processes and more free memory.  The caller must export the reported
physical index through CUDA_VISIBLE_DEVICES before importing torch.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess


def query() -> list[dict]:
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    counts: dict[str, int] = {}
    for row in csv.reader(process_rows.splitlines()):
        if row and row[0].strip():
            counts[row[0].strip()] = counts.get(row[0].strip(), 0) + 1
    result = []
    for row in csv.reader(gpu_rows.splitlines()):
        if len(row) < 6:
            continue
        index, name, total, used, free, uuid = [value.strip() for value in row[:6]]
        result.append({
            "index": int(index), "name": name, "total_mib": int(total),
            "used_mib": int(used), "free_mib": int(free), "uuid": uuid,
            "compute_processes": counts.get(uuid, 0),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=20000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    devices = query()
    eligible = [device for device in devices if device["free_mib"] >= args.min_free_mib]
    if not eligible:
        raise SystemExit(f"no GPU has {args.min_free_mib} MiB free: {json.dumps(devices)}")
    selected = sorted(eligible, key=lambda device: (device["compute_processes"], -device["free_mib"], device["index"]))[0]
    payload = {"selected": selected, "eligible": sorted(eligible, key=lambda device: device["index"]), "min_free_mib": args.min_free_mib}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(selected["index"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
