#!/usr/bin/env python3
"""Post-sealing GT attribution. Never imported by the online TTA processes.

IoU-0.5 framewise assignment/retrieval diagnostics complement, not replace,
official TrackEval. No thresholds or optimizer parameters are written back.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from array import array
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes, box_from_fields


def load_eval_gt(dataset_root, video, expression):
    """Match official MOT ground-truth mark/class filtering, offline only."""
    result = defaultdict(list)
    path = Path(dataset_root) / "gt_template" / video / expression / "gt.txt"
    for row in path.read_text().splitlines():
        fields = row.strip().split(",")
        if len(fields) < 8:
            raise ValueError("official Dance GT requires mark and class columns")
        if int(float(fields[6])) != 0 and int(float(fields[7])) == 1:
            result[int(float(fields[0]))].append((int(float(fields[1])), box_from_fields(fields)))
    return dict(sorted(result.items()))


def overlaps(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a, b = np.asarray(a), np.asarray(b)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(high-low, 0).prod(axis=2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(axis=1)
    bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(axis=1)
    return inter/np.maximum(aa[:, None]+bb[None, :]-inter, 1e-12)


def matched_gt(iou, selected):
    if not selected or not iou.shape[1]: return set(), set()
    scores = iou[selected]
    # Maximize valid-match cardinality first, IoU second, with fixed 0.5 gate.
    valid = scores >= .5
    cost = valid.astype(float)*100 + scores*valid
    rows, cols = linear_sum_assignment(-cost)
    return ({int(c) for r, c in zip(rows, cols) if valid[r, c]},
            {selected[r] for r, c in zip(rows, cols) if valid[r, c]})


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    p, n = int(labels.sum()), int((~labels).sum())
    if not p or not n: return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))


def analyze(root, method, baseline_root, dataset_root, tracker_root, output):
    folder = Path(root)/method
    audit = json.loads((folder/"full_audit.json").read_text())
    baseline = Path(baseline_root)/"ZERO_UPDATE"
    base_audit = json.loads((baseline/"full_audit.json").read_text())
    if audit["status"] != "PASS" or base_audit["status"] != "PASS" or audit["valid_episodes"] != 425:
        raise ValueError("GT attribution requires all sealed/audited predictions")
    stats = {k: 0 for k in ("gt_observations", "candidate_retrievable_gt", "baseline_matched_gt", "adapted_matched_gt",
                            "recovered_gt", "lost_gt", "new_false_positive_rows", "removed_false_positive_rows",
                            "baseline_false_positive_rows", "adapted_false_positive_rows", "ranking_valid_frames")}
    stats["ranking"] = {name: {str(k):0 for k in (1,3,5)} for name in ("baseline", "adapted", "random_null")}
    labels, baseline_scores, adapted_scores = array("b"), array("d"), array("d")
    per_episode = []
    for video in sorted({e["video"] for e in audit["episodes"]}):
        boxes = load_tracker_boxes(tracker_root, video)
        for entry in [e for e in audit["episodes"] if e["video"] == video]:
            expression = entry["expression"]
            before = {k:v for k,v in stats.items() if isinstance(v, int)}
            gt = load_eval_gt(dataset_root, video, expression)
            stats["gt_observations"] += sum(len(rows) for rows in gt.values())
            relative = Path("results")/video/expression
            def chosen(path):
                return {(int(float(r.split(",")[0])), int(float(r.split(",")[1])))
                        for r in path.read_text().splitlines() if r.strip()}
            selected = chosen(folder/relative/"predict.txt")
            reference = chosen(baseline/relative/"predict.txt")
            baseline_path = baseline/relative/"candidate_margins.jsonl.gz"
            with gzip.open(folder/relative/"candidate_margins.jsonl.gz", "rt") as stream, gzip.open(baseline_path, "rt") as src_stream:
                for line, src_line in zip(stream, src_stream, strict=True):
                    row, src = json.loads(line), json.loads(src_line)
                    t, ids, values = row["frame"], row["ids"], row["adapted"]
                    if src["frame"] != t or set(src["ids"]) != set(ids):
                        raise ValueError("baseline/adapted candidate universe differs")
                    base_values = dict(zip(src["ids"], src["adapted"]))
                    previous = [base_values[i] for i in ids]
                    matrix = overlaps([boxes[t][i] for i in ids], [box for _, box in gt.get(t, [])])
                    positive = (matrix >= .5).any(axis=1) if matrix.shape[1] else np.zeros(len(ids), dtype=bool)
                    retrievable = int((matrix >= .5).any(axis=0).sum())
                    stats["candidate_retrievable_gt"] += retrievable
                    a = [j for j, i in enumerate(ids) if (t, i) in selected]
                    b = [j for j, i in enumerate(ids) if (t, i) in reference]
                    am, a_matched = matched_gt(matrix, a)
                    bm, b_matched = matched_gt(matrix, b)
                    stats["adapted_matched_gt"] += len(am); stats["baseline_matched_gt"] += len(bm)
                    stats["recovered_gt"] += len(am-bm); stats["lost_gt"] += len(bm-am)
                    stats["new_false_positive_rows"] += len((set(a)-set(b))-a_matched)
                    stats["removed_false_positive_rows"] += len((set(b)-set(a))-b_matched)
                    stats["baseline_false_positive_rows"] += len(b)-len(bm)
                    stats["adapted_false_positive_rows"] += len(a)-len(am)
                    labels.extend(map(int, positive)); baseline_scores.extend(previous); adapted_scores.extend(values)
                    p, n = int(positive.sum()), len(ids)
                    if p:
                        stats["ranking_valid_frames"] += 1
                        for name, scores in (("baseline", previous), ("adapted", values)):
                            order = sorted(range(n), key=lambda j:(-scores[j], ids[j]))
                            for k in (1,3,5): stats["ranking"][name][str(k)] += int(any(positive[j] for j in order[:k]))
                        for k in (1,3,5):
                            absent = 1.
                            for j in range(min(k,n)):
                                absent *= max(n-p-j, 0)/(n-j)
                            stats["ranking"]["random_null"][str(k)] += 1-absent
            episode_stats = {k:v-before[k] for k,v in stats.items() if isinstance(v, int)}
            per_episode.append(dict(video=video, expression=expression, **episode_stats))
    result = dict(method=method, development_only_gt_used=True, threshold_selection_from_gt=False,
                  definition="offline framewise max-cardinality IoU>=0.5 assignment; official HOTA remains authoritative",
                  gt_definition="nonzero mark and pedestrian class 1, following official MOT GT filtering",
                  added_fp_definition="newly selected rows unmatched after adaptation; existing-row matching status can also change",
                  net_gt_recovery=stats["recovered_gt"]-stats["lost_gt"],
                  net_false_positive_change=stats["adapted_false_positive_rows"]-stats["baseline_false_positive_rows"],
                  **stats, source_auroc=auroc(baseline_scores, labels), adapted_auroc=auroc(adapted_scores, labels),
                  conditional_candidate_recall=stats["candidate_retrievable_gt"]/max(stats["gt_observations"],1),
                  baseline_conditional_gate_recall=stats["baseline_matched_gt"]/max(stats["candidate_retrievable_gt"],1),
                  adapted_conditional_gate_recall=stats["adapted_matched_gt"]/max(stats["candidate_retrievable_gt"],1),
                  episodes=per_episode)
    for group in result["ranking"].values():
        for key in group: group[key] /= max(stats["ranking_valid_frames"],1)
    atomic_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--i-allow-offline-gt", action="store_true")
    args = p.parse_args()
    if not args.i_allow_offline_gt: p.error("GT is permitted only in explicitly isolated offline evaluation")
    analyze(args.output_root, args.method, args.baseline_root, args.dataset_root, args.tracker_root, args.output)


if __name__ == "__main__": main()
