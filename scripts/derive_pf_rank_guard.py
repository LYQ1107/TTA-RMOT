#!/usr/bin/env python3
"""Derive the explicit source-positive union ablation from sealed PF_RANK scores."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import EpisodeStore, atomic_json


def derive(rank_root, output_root, tracker_root, *, one_episode=None):
    rank_root, output_root = Path(rank_root), Path(output_root)
    meta = json.loads((rank_root / "PF_RANK/run_metadata.json").read_text())
    audit = json.loads((rank_root / "PF_RANK/full_audit.json").read_text())
    if audit["status"] != "PASS" or audit["valid_episodes"] != 425:
        raise ValueError("PF_RANK must be complete, sealed and audited first")
    episodes = meta["episodes"]
    if one_episode is not None:
        episodes = [e for e in episodes if (e["video"], e["expression_id"]) == tuple(one_episode)]
        if len(episodes) != 1:
            raise ValueError("verification episode missing")
    out = output_root / "PF_RANK_GUARD"
    out.mkdir(parents=True, exist_ok=True)
    for file in ("completed_episodes.jsonl", "failed_episodes.jsonl"):
        (out / file).touch(exist_ok=True)
    records, tracks = [], {}
    for old in episodes:
        video, expression = old["video"], old["expression_id"]
        config = dict(old["config"], method="PF_RANK_GUARD")
        store = EpisodeStore(out, video, expression, config, resume=True)
        sealed = store.completed()
        if sealed is not None:
            records.append(sealed); continue
        started = time.perf_counter()
        if video not in tracks:
            tracks[video] = load_tracker_lines(tracker_root, video)
        source = rank_root / "PF_RANK/results" / video / expression / "candidate_margins.jsonl.gz"
        pp, mp = store.begin()
        mp.symlink_to(source.resolve())
        rows = frames = candidates = source_selected = 0
        with pp.open("w") as pred, gzip.open(source, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                t = row["frame"]
                for tid, adapted, frozen in sorted(zip(row["ids"], row["adapted"], row["frozen"])):
                    if adapted > 0 or frozen > 0:
                        pred.write(tracks[video][t][tid]+"\n"); rows += 1
                    source_selected += int(frozen > 0)
                frames += 1; candidates += len(row["ids"])
        record = dict(sentence=old["sentence"], full_frame_coverage=old["full_frame_coverage"],
                      wall_seconds=time.perf_counter()-started, gpu_forward_reused_from="PF_RANK", additional_gpu_seconds=0,
                      stats=dict(frames=frames, candidate_rows=candidates, selected_rows=rows, optimizer_steps=0),
                      counter_delta=dict(frames=frames, candidates=candidates, selected=rows, source_selected=source_selected,
                                         optimizer_steps=0, rank_optimizer_steps_reused=old["stats"]["optimizer_steps"]))
        records.append(store.seal(pp, mp, record))
    result = dict(method="PF_RANK_GUARD", status="complete", scope="equivalence" if one_episode else "full",
                  derivation_experiment_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  source_free_prediction=True, target_gt_opened=False, target_labels_opened=False,
                  future_frames_used=False, tracker_geometry_changed=False, tracker_ids_changed=False,
                  derived_from=str(rank_root), additional_gpu_forward_calls=0,
                  sequences=len(records), episodes=records, videos=sorted({r["video"] for r in records}))
    atomic_json(out / "run_metadata.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args = p.parse_args()
    result = derive(args.rank_root, args.output_root, args.tracker_root)
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__": main()
