#!/usr/bin/env python3
"""One *complete*, source-positive real episode: direct GPU GUARD vs CPU derivation.

Select the episode using Frozen scores only, never target GT. This is not a
25-video performance experiment. A failed equivalence requires direct full GUARD.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.derive_pf_rank_guard import derive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--matched-frozen-root", type=Path)
    args = p.parse_args()
    started = time.perf_counter()
    root = args.output_root.resolve()
    audit = json.loads((args.rank_root / "PF_RANK/full_audit.json").read_text())
    active = [e for e in audit["episodes"] if e["diagnostics"].get("optimizer_steps", 0) > 0]
    chosen = max(active or audit["episodes"], key=lambda e: e["source_selected_rows"])
    video, expression = chosen["video"], chosen["expression"]
    from scripts.run_flexhook_positivefree import parse_args, run
    import flex_tta.dance_io as dio
    options = parse_args(["--scope", "plan", "--method", "PF_RANK_GUARD", "--videos", video,
                          "--source-supplement", str(ROOT / "results/runs/pf_frozen_complete_20261025/source_supplement.json"),
                          "--max-frames", "0", "--max-expressions", "1", "--frozen-cache", "--output-root", str(root / "direct")])
    # Reuse the unchanged model loop for one full independent episode. Its
    # public full-study path still requires all expressions; do not weaken it.
    options.scope = "equivalence"
    options.matched_frozen_root = args.matched_frozen_root
    original = dio.load_expressions
    def single_expression(dataset, v):
        return [(e, s) for e, s in original(dataset, v) if e == expression]
    dio.load_expressions = single_expression
    try:
        direct = run(options)
    finally:
        dio.load_expressions = original
    derive(args.rank_root, root / "derived", options.tracker_root, one_episode=(video, expression))
    relative = Path("PF_RANK_GUARD/results") / video / expression / "predict.txt"
    a, b = root / "direct" / relative, root / "derived" / relative
    direct_steps = direct["episodes"][0]["stats"]["optimizer_steps"]
    rank_meta = json.loads((args.rank_root / "PF_RANK/results" / video / expression / "completion.json").read_text())
    same = a.read_bytes() == b.read_bytes()
    nonempty = len(a.read_text().splitlines())
    result = dict(status="PASS" if same and nonempty and direct_steps == rank_meta["stats"]["optimizer_steps"] else "FAIL",
                  video=video, expression=expression, frames=direct["stats"]["frames"],
                  nonempty_selected_rows=nonempty, predictions_byte_identical=same,
                  direct_optimizer_steps=direct_steps, rank_optimizer_steps=rank_meta["stats"]["optimizer_steps"],
                  wall_seconds=time.perf_counter()-started, peak_allocated_bytes=direct["stats"]["gpu_peak_allocated_bytes"],
                  target_gt_opened=False, scope="one complete real episode, source-score-only episode choice")
    atomic_json(root / "equivalence.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
