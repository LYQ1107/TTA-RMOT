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


def matched_pairs(iou, selected):
    if not selected or not iou.shape[1]: return {}
    scores = iou[selected]
    # Maximize valid-match cardinality first, IoU second, with fixed 0.5 gate.
    valid = scores >= .5
    cost = valid.astype(float)*100 + scores*valid
    rows, cols = linear_sum_assignment(-cost)
    return {selected[r]:int(c) for r,c in zip(rows,cols) if valid[r,c]}


def matched_gt(iou, selected):
    pairs = matched_pairs(iou, selected)
    return set(pairs.values()), set(pairs)


def observable_id_switches(previous, current, frame):
    """Only query-GT-visible, <=2-frame transitions; not a full MOT ID audit."""
    valid = [(old_id,current[tid]) for tid,(old_t,old_id) in previous.items()
             if tid in current and old_id is not None and current[tid] is not None and 0 < frame-old_t <= 2]
    return len(valid), sum(old != now for old,now in valid)


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    p, n = int(labels.sum()), int((~labels).sum())
    if not p or not n: return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))


def query_presence_summary(episodes):
    """GT grouping exists only in this sealed-output evaluator."""
    groups = {}
    for name, present in (("query_present", True), ("query_empty", False)):
        rows = [e for e in episodes if bool(e["gt_observations"]) == present]
        fields = ("gt_observations", "baseline_matched_gt", "adapted_matched_gt",
                  "baseline_false_positive_rows", "adapted_false_positive_rows", "recovered_gt", "lost_gt",
                  "new_false_positive_rows", "removed_false_positive_rows", "baseline_selected_rows", "adapted_selected_rows")
        group = {k: sum(e[k] for e in rows) for k in fields}
        group["episodes"] = len(rows)
        for method in ("baseline", "adapted"):
            group[method+"_GT_recall"] = (group[method+"_matched_gt"]/group["gt_observations"]
                                           if group["gt_observations"] else None)
            count = sum(e[method+"_selected_rows"] > 0 for e in rows)
            group[method+"_nonempty_prediction_episodes"] = count
            group[method+"_nonempty_prediction_rate"] = count/len(rows) if rows else None
        group["net_false_positive_change"] = group["adapted_false_positive_rows"]-group["baseline_false_positive_rows"]
        groups[name] = group
    return groups


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
    recall_at_k = {name: {str(k):0. for k in (1,3,5)} for name in ("baseline", "adapted")}
    temporal = [dict(candidates=0, baseline_selected=0, adapted_selected=0) for _ in range(4)]
    id_diagnostic = dict(query_gt_visible_transitions=0, query_gt_visible_switches=0,
                         eligible_update_frames_with_visible_switch=0)
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
            prior_ids = {}
            baseline_path = baseline/relative/"candidate_margins.jsonl.gz"
            with gzip.open(folder/relative/"candidate_margins.jsonl.gz", "rt") as stream, gzip.open(baseline_path, "rt") as src_stream:
                for frame_index, (line, src_line) in enumerate(zip(stream, src_stream, strict=True)):
                    row, src = json.loads(line), json.loads(src_line)
                    t, ids, values = row["frame"], row["ids"], row["adapted"]
                    if src["frame"] != t or set(src["ids"]) != set(ids):
                        raise ValueError("baseline/adapted candidate universe differs")
                    base_values = dict(zip(src["ids"], src["adapted"]))
                    previous = [base_values[i] for i in ids]
                    matrix = overlaps([boxes[t][i] for i in ids], [box for _, box in gt.get(t, [])])
                    pairs = matched_pairs(matrix,list(range(len(ids))))
                    current_gt = {tid: None for tid in ids}
                    current_gt.update({ids[j]:gt[t][k][0] for j,k in pairs.items()})
                    transitions, switches = observable_id_switches(prior_ids,current_gt,t)
                    id_diagnostic["query_gt_visible_transitions"] += transitions
                    id_diagnostic["query_gt_visible_switches"] += switches
                    reliable = sum(tid in prior_ids and 0 < t-prior_ids[tid][0] <= 2 for tid in ids)
                    if switches and reliable >= 2 and (frame_index+1)%4 == 0:
                        id_diagnostic["eligible_update_frames_with_visible_switch"] += 1
                    prior_ids = {tid:(old_t,old_id) for tid,(old_t,old_id) in prior_ids.items() if t-old_t <= 2}
                    prior_ids.update({tid:(t,gid) for tid,gid in current_gt.items()})
                    positive = (matrix >= .5).any(axis=1) if matrix.shape[1] else np.zeros(len(ids), dtype=bool)
                    retrievable = int((matrix >= .5).any(axis=0).sum())
                    stats["candidate_retrievable_gt"] += retrievable
                    a = [j for j, i in enumerate(ids) if (t, i) in selected]
                    b = [j for j, i in enumerate(ids) if (t, i) in reference]
                    bucket = temporal[min(3, 4*frame_index//entry["frames"])]
                    bucket["candidates"] += len(ids)
                    bucket["baseline_selected"] += len(b)
                    bucket["adapted_selected"] += len(a)
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
                            for k in (1,3,5):
                                hits = int(positive[order[:k]].sum())
                                stats["ranking"][name][str(k)] += int(hits > 0)
                                recall_at_k[name][str(k)] += hits/p
                        for k in (1,3,5):
                            absent = 1.
                            for j in range(min(k,n)):
                                absent *= max(n-p-j, 0)/(n-j)
                            stats["ranking"]["random_null"][str(k)] += 1-absent
            episode_stats = {k:v-before[k] for k,v in stats.items() if isinstance(v, int)}
            per_episode.append(dict(video=video, expression=expression, baseline_selected_rows=len(reference),
                                    adapted_selected_rows=len(selected), **episode_stats))
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
    presence = query_presence_summary(per_episode)
    present, empty = presence["query_present"], presence["query_empty"]
    for group in recall_at_k.values():
        for key in group: group[key] /= max(stats["ranking_valid_frames"],1)
    result.update(query_presence=presence, query_present_episodes=present["episodes"],
                  query_empty_episodes=empty["episodes"], query_present_GT_recall=present["adapted_GT_recall"],
                  query_present_FP=present["adapted_false_positive_rows"], query_empty_FP=empty["adapted_false_positive_rows"],
                  query_empty_nonempty_prediction_rate=empty["adapted_nonempty_prediction_rate"],
                  source_Recall_at_1_3_5=recall_at_k["baseline"], adapted_Recall_at_1_3_5=recall_at_k["adapted"],
                  recall_at_k_definition="macro over target-retrievable frames: fraction of IoU>=0.5-positive candidates in top K; distinct from any-hit TopK",
                  query_group_definition="episode has any official scored GT; offline only, never provided to model",
                  identity_switch_diagnostic=dict(**id_diagnostic,
                    limitation="query-GT-observable transitions only; misses unannotated people/gaps, and counts possible evidence conflicts, not proven accepted optimizer steps; the online ID memory cannot independently detect swaps within max_gap=2"),
                  temporal_selection_quartiles=temporal,
                  temporal_definition="four equal active-frame-index parts per episode; candidate-weighted negative fractions, not a causal trend test")
    for bucket in temporal:
        for name in ("baseline", "adapted"):
            bucket[name+"_negative_fraction"] = 1-bucket[name+"_selected"]/max(bucket["candidates"],1)
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
