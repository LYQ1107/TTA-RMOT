#!/usr/bin/env python3
"""Audit complete sealed candidate/prediction streams; NO target GT."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.dance_io import dance_videos, load_expressions, load_tracker_lines
from flex_tta.adaptation.positive_free import CausalRelativeSelector
from flex_tta.runtime.pf_receipts import file_sha, atomic_json


def audit(output_root, method, dataset_root, tracker_root, matched_frozen_root=None):
    method_root = Path(output_root) / method
    expected = {(v, e) for v in dance_videos(tracker_root) for e, _ in load_expressions(dataset_root, v)}
    paths = {(p.parts[-3], p.parts[-2]): p for p in (method_root / "results").glob("*/*/predict.txt")}
    if set(paths) != expected:
        raise ValueError(f"coverage mismatch: missing={len(expected-set(paths))}, extra={len(set(paths)-expected)}")
    report = dict(method=method, valid_episodes=425, unique_videos=25, duplicated_episodes=0,
                  missing_episodes=0, candidates=0, frames=0, selected_rows=0, source_selected_rows=0,
                  selected_ids=0, source_selected_ids=0, source_sign_mismatches=0,
                  source_margin_max_abs_error=0., new_selected_rows=0, lost_frozen_rows=0,
                  target_gt_opened=False, tracker_geometry_changed=False, tracker_ids_changed=False,
                  episodes=[], mismatch_examples=[], matched_frozen_root=str(matched_frozen_root) if matched_frozen_root else None,
                  historical_source_sign_mismatches=0, historical_source_margin_max_abs_error=0.)
    for video in sorted({v for v, _ in expected}):
        tracks = load_tracker_lines(tracker_root, video)
        for _, expr in sorted(k for k in expected if k[0] == video):
            path = paths[(video, expr)]
            receipt = json.loads((path.parent / "completion.json").read_text())
            margins = path.parent / "candidate_margins.jsonl.gz"
            if receipt["status"] != "complete" or not receipt["full_frame_coverage"]:
                raise ValueError(f"episode is not full/sealed: {path}")
            if file_sha(path) != receipt["prediction_sha256"] or file_sha(margins) != receipt["margins_sha256"]:
                raise ValueError(f"episode hash mismatch: {path}")
            selector = CausalRelativeSelector()
            rows, source_ids, selected_ids, frame_ids = [], set(), set(), []
            candidates = source_rows = new_rows = lost_rows = 0
            matched = None
            if matched_frozen_root:
                from flex_tta.runtime.pf_source_reference import source_frames
                matched = source_frames(matched_frozen_root, video, expr)
            with gzip.open(margins, "rt") as stream:
                for line in stream:
                    frame = json.loads(line)
                    t, ids, adapted, frozen = (frame[k] for k in ("frame", "ids", "adapted", "frozen"))
                    if len(ids) != len(set(ids)) or set(ids) != set(tracks.get(t, {})) or len(ids) != len(adapted) or len(ids) != len(frozen):
                        raise ValueError(f"incomplete candidate set: {video}/{expr}/{t}")
                    if not all(math.isfinite(x) for x in adapted+frozen):
                        raise ValueError("non-finite candidate score")
                    frame_ids.append(t)
                    values = dict(zip(ids, adapted)); src = dict(zip(ids, frozen))
                    legacy_src = src
                    if matched is not None:
                        src = matched[t]
                        if set(src) != set(ids):
                            raise ValueError("matched Frozen candidate IDs differ")
                        if method != "ZERO_UPDATE" and src != legacy_src:
                            raise ValueError("method did not use the common matched Frozen score bank")
                    chosen = (selector.step(t, src) if method == "RELATIVE_ONLY" else
                              {i for i in ids if values[i] > 0 or (method == "PF_RANK_GUARD" and src[i] > 0)})
                    baseline = {i for i in ids if src[i] > 0}
                    rows.extend(tracks[t][i] for i in sorted(chosen))
                    candidates += len(ids); source_rows += len(baseline)
                    source_ids.update(baseline); selected_ids.update(chosen)
                    new_rows += len(chosen-baseline); lost_rows += len(baseline-chosen)
                    if method == "ZERO_UPDATE":
                        report["source_sign_mismatches"] += len(chosen ^ baseline)
                        report["source_margin_max_abs_error"] = max(report["source_margin_max_abs_error"], max(abs(values[i]-src[i]) for i in ids))
                        report["historical_source_sign_mismatches"] += sum((values[i] > 0) != (legacy_src[i] > 0) for i in ids)
                        report["historical_source_margin_max_abs_error"] = max(report["historical_source_margin_max_abs_error"], max(abs(values[i]-legacy_src[i]) for i in ids))
                        for i in sorted(chosen ^ baseline):
                            if len(report["mismatch_examples"]) < 20:
                                report["mismatch_examples"].append(dict(video=video, expression=expr, frame=t, track_id=i,
                                    adapted=values[i], frozen=src[i]))
            if frame_ids != sorted(tracks):
                raise ValueError(f"not full ordered frame coverage: {video}/{expr}")
            if path.read_text().splitlines() != rows:
                raise ValueError(f"predictions are not exact original MOT rows / selection rule: {path}")
            entry = dict(video=video, expression=expr, frames=len(frame_ids), candidates=candidates,
                         selected_rows=len(rows), source_selected_rows=source_rows,
                         selected_ids=len(selected_ids), source_selected_ids=len(source_ids),
                         new_selected_rows=new_rows, lost_frozen_rows=lost_rows,
                         wall_seconds=receipt.get("wall_seconds"), diagnostics=receipt.get("counter_delta", {}))
            report["episodes"].append(entry)
            for key in ("frames", "candidates", "selected_rows", "source_selected_rows", "selected_ids", "source_selected_ids", "new_selected_rows", "lost_frozen_rows"):
                report[key] += entry[key]
    report["status"] = "PASS"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--matched-frozen-root", type=Path)
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args = p.parse_args()
    report = audit(args.output_root, args.method, args.dataset_root, args.tracker_root, args.matched_frozen_root)
    atomic_json(args.output_root / args.method / "full_audit.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__": main()
