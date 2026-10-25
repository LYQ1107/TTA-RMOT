#!/usr/bin/env python3
"""Sequential one-GPU, one-expression, <=12-frame server integration check."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(argv):
    from scripts.run_flexhook_positivefree import parse_args, run
    args = parse_args(argv)
    if args.scope != "smoke":
        raise ValueError("this entry point permits smoke only")
    target = args.dataset_root.resolve()
    audit = {"target_gt_open_attempts": [], "target_read_count": 0}

    def on_open(event, values):
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(target):
            if path.name in ("labels.json", "gt.txt") or "gt_template" in path.parts:
                audit["target_gt_open_attempts"].append(str(path))
                raise AssertionError(f"smoke attempted target GT access: {path}")
            audit["target_read_count"] += 1

    sys.addaudithook(on_open)
    args.output_root = args.output_root.resolve()
    try:
        receipt = run(args)
        print(json.dumps(receipt, indent=2))
    finally:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "file_access_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def check(root, method, expect_rollback=False, expect_no_evidence=False):
    metadata = json.loads((root / method / "run_metadata.json").read_text())
    stats = metadata["stats"]
    audit = json.loads((root / "file_access_audit.json").read_text())
    assert not audit["target_gt_open_attempts"]
    assert stats["episodes"] == 1 and 0 < stats["frames"] <= 12
    if method == "RELATIVE_ONLY":
        assert stats["optimizer_steps"] == 0
        return prediction_audit(root, method, metadata, stats, audit)
    assert stats["checkpoint_strict_load"]
    assert stats["sign_mismatches_before_updates"] == 0
    assert stats["source_max_abs_error"] < 1e-3
    assert not stats["frozen_parameters_have_grad"]
    assert stats["nonfinite_gradient_steps"] == 0 and stats["zero_gradient_steps"] == 0
    assert stats["post_episode_reset_pass"] and all(x["passed"] for x in stats["episode_resets"])
    for frame in stats["smoke_frames"]:
        assert frame["max_image_history_frame"] <= frame["frame_id"]
        if frame["update_status"] in ("ACCEPT", "ROLLBACK"):
            assert frame["rank_pairs"] or frame["identity_support"] >= 2
            assert frame["rank_history_last_frame"] < frame["frame_id"]
            assert frame["gradient_norm"] > 0 and math.isfinite(frame["gradient_norm"])
    if method == "ZERO_UPDATE":
        assert stats["optimizer_steps"] == 0 and stats["prediction_source_selection_mismatches"] == 0
    elif expect_no_evidence:
        assert stats["optimizer_steps"] == 0 and stats["gradient_attempts"] == 0
        assert stats["rank_pair_observations"] == 0
    elif expect_rollback:
        assert stats["optimizer_steps"] == 0 and stats["rollback_updates"] > 0 and stats["rollback_restored"]
    else:
        assert stats["optimizer_steps"] > 0 and stats["max_adapter_update_l2"] > 0
    return prediction_audit(root, method, metadata, stats, audit)


def prediction_audit(root, method, metadata, stats, audit):
    predictions = list((root / method / "results").glob("*/*/predict.txt"))
    assert len(predictions) == 1
    tracker = Path(metadata["tracker_root"]) / f"{predictions[0].parts[-3]}.txt"
    official_rows = set(tracker.read_text().splitlines())
    predicted_rows = predictions[0].read_text().splitlines()
    assert all(row in official_rows for row in predicted_rows)
    return {"status": "PASS", "metadata": str(root / method / "run_metadata.json"),
            "checked_prediction_rows": len(predicted_rows), "file_access_audit": audit, "stats": stats}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--run-child":
        child(sys.argv[2:])
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["ZERO_UPDATE", "PF_RANK", "PF_IDENTITY", "PF_HYBRID"],
                   choices=["ZERO_UPDATE", "PF_RANK", "PF_IDENTITY", "PF_HYBRID", "PF_RANK_GUARD", "RELATIVE_ONLY"])
    p.add_argument("--video", default="dancetrack0004")
    p.add_argument("--rollback-probe", action="store_true")
    p.add_argument("--no-evidence-probe", action="store_true")
    p.add_argument("--frozen-cache", action="store_true")
    args = p.parse_args()
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        p.error("select exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    runs = [(method, method, False, False) for method in args.methods]
    if args.rollback_probe:
        runs.append(("PF_RANK_ROLLBACK", "PF_RANK", True, False))
    if args.no_evidence_probe:
        runs.append(("PF_RANK_NO_EVIDENCE", "PF_RANK", False, True))
    for name, method, rollback, no_evidence in runs:
        run_root = root / name
        cmd = [sys.executable, str(Path(__file__).resolve()), "--run-child", "--scope", "smoke",
               "--method", method, "--videos", args.video, "--max-frames", "12", "--max-expressions", "1",
               "--strict-initial-sign", "--output-root", str(run_root)]
        if rollback:
            cmd.extend(["--max-adapter-delta-norm", "1e-12"])
        if args.frozen_cache:
            cmd.append("--frozen-cache")
        if no_evidence:
            cmd.extend(["--min-separation", "100000"])
        print(f"START {name} on GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        with (root / f"{name}.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            results[name] = {"status": "FAIL", "exit_code": proc.returncode, "log": str(root / f"{name}.log")}
        else:
            try:
                results[name] = check(run_root, method, rollback, no_evidence)
            except AssertionError as exc:
                results[name] = {"status": "FAIL", "assertion": str(exc), "log": str(root / f"{name}.log")}
        results[name]["command"] = cmd
        (root / "smoke_summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {results[name]['status']}", flush=True)
        if results[name]["status"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
