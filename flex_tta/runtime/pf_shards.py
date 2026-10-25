"""GT-free workload scheduling and read-only GPU resource inspection."""
from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from flex_tta.dance_io import dance_videos, load_expressions


def weighted_manifest(tracker_root, dataset_root, *, num_shards=4, batch_size=24,
                      row_weight=1.0, frame_weight=0.0):
    if not 1 <= num_shards <= 4:
        raise ValueError("one to four logical shards are supported")
    records = []
    for video in dance_videos(Path(tracker_root)):
        path = Path(tracker_root) / f"{video}.txt"
        counts = Counter(int(float(row.split(",")[0])) for row in path.read_text().splitlines() if row.strip())
        exprs = load_expressions(Path(dataset_root), video)
        rows = sum(counts.values())
        records.append(dict(video=video, tracker_rows=rows, frames_with_tracks=len(counts),
                            first_frame=min(counts), last_frame=max(counts),
                            max_candidates_per_frame=max(counts.values()), expressions_count=len(exprs),
                            expression_ids=[e for e, _ in exprs],
                            estimated_forward_calls=len(exprs)*sum(math.ceil(n/batch_size) for n in counts.values()),
                            cost=len(exprs)*(row_weight*rows + frame_weight*len(counts))))
    shards = [dict(shard=i, videos=[], episodes=0, tracker_rows=0, estimated_forward_calls=0, cost=0.)
              for i in range(num_shards)]
    for record in sorted(records, key=lambda x: (-x["cost"], x["video"])):
        shard = min(shards, key=lambda x: (x["cost"], x["shard"]))
        shard["videos"].append(record["video"])
        for key, value in (("episodes", record["expressions_count"]), ("tracker_rows", record["tracker_rows"]),
                           ("estimated_forward_calls", record["estimated_forward_calls"]), ("cost", record["cost"])):
            shard[key] += value
    for shard in shards:
        shard["videos"].sort()
    assigned = [v for s in shards for v in s["videos"]]
    if len(assigned) != 25 or len(set(assigned)) != 25 or sum(s["episodes"] for s in shards) != 425:
        raise ValueError("manifest must cover 25 distinct videos / 425 complete episodes")
    return dict(protocol="whole-video LPT; no target GT or metrics used", unique_videos=25, episodes=425,
                no_duplicate_episodes=True, no_missing_episodes=True, complete_tracker_frame_plan=True,
                row_weight=row_weight, frame_weight=frame_weight, batch_size=batch_size,
                tracker_root=str(Path(tracker_root).resolve()), dataset_root=str(Path(dataset_root).resolve()),
                videos=records, shards=shards)


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True)
    rows = csv.reader(io.StringIO(query(["--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu"])))
    result = [dict(index=int(r[0]), uuid=r[1].strip(), name=r[2].strip(), total_mib=int(r[3]),
                   free_mib=int(r[4]), utilization=int(r[5]), processes=[]) for r in rows]
    apps = csv.reader(io.StringIO(query(["--query-compute-apps=gpu_uuid,pid,used_gpu_memory"])))
    by_uuid = {r["uuid"]: r for r in result}
    for row in apps:
        if len(row) != 3 or row[0].strip() not in by_uuid:
            continue
        pid = int(row[1])
        owner = subprocess.run(["ps", "-p", str(pid), "-o", "user=,comm="], capture_output=True, text=True).stdout.strip()
        by_uuid[row[0].strip()]["processes"].append(dict(pid=pid, used_mib=row[2].strip(), owner_executable=owner))
    return result


def idle_gpus(snapshot, *, requested=None, minimum_free_mib=24576, limit=4):
    allowed = set(requested) if requested is not None else None
    return sorted([r for r in snapshot if (allowed is None or r["index"] in allowed) and
                   not r["processes"] and r["utilization"] <= 5 and r["free_mib"] >= minimum_free_mib],
                  key=lambda r: (-r["free_mib"], r["index"]))[:limit]


def eligible_gpus(snapshot, *, shared=False, requested=None, memory_budget_mib=24576,
                  headroom_mib=8192, limit=4):
    """Sharing is opt-in; process count takes priority over available memory."""
    if not shared:
        return idle_gpus(snapshot, requested=requested, minimum_free_mib=memory_budget_mib, limit=limit)
    allowed = set(requested) if requested is not None else None
    return sorted([r for r in snapshot if (allowed is None or r["index"] in allowed)
                   and r["free_mib"] >= memory_budget_mib + headroom_mib],
                  key=lambda r: (len(r["processes"]), -r["free_mib"], r["utilization"], r["index"]))[:limit]


def rebalance_unstarted(manifest, completed_shards, *, num_shards=4):
    """Retain sealed groups verbatim; LPT only the remaining whole videos."""
    completed = set(completed_shards)
    retained = [s for s in manifest["shards"] if s["shard"] in completed]
    done = {v for s in retained for v in s["videos"]}
    records = [r for r in manifest["videos"] if r["video"] not in done]
    next_id = max(completed, default=-1) + 1
    groups = [dict(shard=next_id+i, videos=[], episodes=0, tracker_rows=0,
                   estimated_forward_calls=0, cost=0.) for i in range(num_shards)]
    for record in sorted(records, key=lambda r: (-r["cost"], r["video"])):
        group = min(groups, key=lambda s: (s["cost"], s["shard"]))
        group["videos"].append(record["video"])
        for key, source_key in (("episodes", "expressions_count"), ("tracker_rows", "tracker_rows"),
                                ("estimated_forward_calls", "estimated_forward_calls"), ("cost", "cost")):
            group[key] += record[source_key]
    for group in groups:
        group["videos"].sort()
    result = dict(manifest, shards=retained+[g for g in groups if g["videos"]],
                  reused_shards=sorted(completed),
                  repartition_reason="user-authorized shared GPUs; rebalance only unstarted whole videos")
    assigned = [v for s in result["shards"] for v in s["videos"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {r["video"] for r in manifest["videos"]}:
        raise ValueError("rebalance lost or duplicated a video")
    if sum(s["episodes"] for s in result["shards"]) != manifest["episodes"]:
        raise ValueError("rebalance changed episode coverage")
    return result


def sealed_shards(run_root, manifest, method):
    from flex_tta.runtime.pf_receipts import EpisodeStore
    expected_by_video = {r["video"]: r["expression_ids"] for r in manifest["videos"]}
    result = []
    for shard in manifest["shards"]:
        folder = Path(run_root) / f"shard{shard['shard']}" / method
        metadata = folder / "run_metadata.json"
        if not metadata.is_file():
            continue
        data = json.loads(metadata.read_text())
        expected = {(v, e) for v in shard["videos"] for e in expected_by_video[v]}
        records = data.get("episodes", [])
        if data.get("status") != "complete" or {(r["video"], r["expression_id"]) for r in records} != expected:
            continue
        for record in records:
            if EpisodeStore(folder, record["video"], record["expression_id"],
                            record["config"], resume=True).completed() is None:
                raise ValueError("sealed shard has an unsealed episode")
        result.append(shard["shard"])
    return result


def save_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"existing manifest differs; do not overwrite {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2)+"\n")
