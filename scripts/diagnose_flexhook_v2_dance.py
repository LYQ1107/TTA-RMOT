#!/usr/bin/env python3
"""Offline D1--D3 diagnosis for the fixed V2→Dance score stream.

This script is intentionally an evaluator-side tool.  It opens Dance GT only
after the frozen score stream has already been produced and never writes a
TTA state, threshold, or pseudo-label back into a runtime.  Its oracle and
threshold results are development-only bounds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from flex_tta.dance_io import dance_videos, load_expressions, load_tracker_lines, score_frames


THRESHOLDS = (-10.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0)
COLORS = {"black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"}
CLOTHING = {"shirt", "tshirt", "pants", "shorts", "skirt", "dress", "jeans", "coat", "trousers", "tee"}
SPATIAL = {"left", "right", "front", "behind", "side", "near", "far", "middle", "center"}
MOTION = {"dancing", "dance", "standing", "walking", "moving", "running", "turning"}
RELATIONAL = {"next", "between", "beside", "behind", "front"}


def box_from_fields(fields: list[str]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(fields[index]) for index in (2, 3, 4, 5))
    return x, y, x + w, y + h


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def load_tracker_boxes(tracker_root: Path, video: str) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    result: dict[int, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
    path = tracker_root / f"{video}.txt"
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split(",")
        if len(fields) != 10:
            raise ValueError(f"invalid tracker row {path}:{number}")
        frame = int(float(fields[0]))
        track = int(float(fields[1]))
        result[frame][track] = box_from_fields(fields)
    return {frame: dict(rows) for frame, rows in sorted(result.items())}


def load_gt(dataset_root: Path, video: str, expression_id: str) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    result: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    path = dataset_root / "gt_template" / video / expression_id / "gt.txt"
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split(",")
        if len(fields) < 6:
            raise ValueError(f"invalid GT row {path}:{number}")
        result[int(float(fields[0]))].append((int(float(fields[1])), box_from_fields(fields)))
    return {frame: rows for frame, rows in sorted(result.items())}


def query_categories(sentence: str) -> list[str]:
    words = set(re.findall(r"[a-z]+", sentence.lower().replace("t-shirt", "tshirt")))
    categories = []
    if words & COLORS or words & CLOTHING:
        categories.append("appearance")
    if words & CLOTHING:
        categories.append("clothing")
    if words & SPATIAL:
        categories.append("spatial")
    if words & MOTION:
        categories.append("motion")
    if words & RELATIONAL or "with" in words:
        categories.append("relational")
    if len(categories) >= 2:
        categories.append("compositional")
    return sorted(set(categories)) or ["unknown"]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("p01", "p05", "p25", "p50", "p75", "p95", "p99", "mean", "std")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def auroc(scores: list[float], labels: list[int]) -> float | None:
    if not scores or sum(labels) == 0 or sum(labels) == len(labels):
        return None
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    ranks = np.empty(len(sorted_scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    positive = int(label_array.sum())
    negative = len(label_array) - positive
    return float((ranks[order.argsort()][label_array == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    positive = sum(labels)
    if positive == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered = np.asarray(labels, dtype=np.int8)[order]
    cumulative = np.cumsum(ordered)
    positions = np.flatnonzero(ordered)
    return float(np.mean(cumulative[positions] / (positions + 1))) if len(positions) else None


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float | None:
    if not scores:
        return None
    values = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -50, 50)))
    total = len(values)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (probabilities >= low) & ((probabilities < high) if index < bins - 1 else (probabilities <= high))
        if mask.any():
            error += float(mask.sum()) / total * abs(float(probabilities[mask].mean()) - float(labels_array[mask].mean()))
    return error


def rank_summary(ranks: list[int | None]) -> dict[str, float | int | None]:
    values = [int(item) for item in ranks if item is not None]
    if not values:
        return {"count": 0, "p50": None, "p90": None, "top1": 0, "top5": 0, "top10": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "top1": sum(item <= 1 for item in values) / len(values),
        "top5": sum(item <= 5 for item in values) / len(values),
        "top10": sum(item <= 10 for item in values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    videos = dance_videos(args.tracker_root)
    tracker_cache = {video: load_tracker_boxes(args.tracker_root, video) for video in videos}

    score_values: list[float] = []
    score_labels: list[int] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    threshold_counts = {str(threshold): {"selected": 0, "tp": 0, "fp": 0, "fn": 0} for threshold in THRESHOLDS}
    coverage_records = []
    ranking_records = []
    semantic_records = []
    category_values: dict[str, list[dict[str, float]]] = defaultdict(list)
    wrong_track_counts: Counter[tuple[str, int]] = Counter()
    top_sets: dict[str, dict[str, dict[int, set[int]]]] = defaultdict(lambda: defaultdict(dict))
    fragmentation_switches = 0
    fragmentation_links = 0
    total_gt = 0
    total_candidate_rows = 0
    total_candidate_frames = 0
    total_source_selected = 0

    for video in videos:
        expressions = load_expressions(args.dataset_root, video)
        tracker = tracker_cache[video]
        for expression_id, sentence in expressions:
            gt = load_gt(args.dataset_root, video, expression_id)
            scores = score_frames(source, video, sentence)
            gt_records = 0
            all_hits_50 = all_hits_75 = selected_hits_50 = 0
            no_candidate_frames = no_selected_frames = 0
            expr_scores: list[float] = []
            expr_labels: list[int] = []
            expr_ranks: list[int | None] = []
            source_tp = source_fp = source_selected = 0
            high_conf_false = 0
            matched_by_gt: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
            frames_with_gt = set(gt)

            for frame_id, gt_rows in gt.items():
                candidates = tracker.get(frame_id, {})
                frame_scores = scores.get(frame_id, {})
                candidate_items = [(track_id, box, frame_scores[track_id]) for track_id, box in candidates.items() if track_id in frame_scores]
                total_candidate_rows += len(candidate_items)
                if candidates:
                    total_candidate_frames += 1
                if not candidates:
                    no_candidate_frames += 1
                selected_items = [(track_id, box, margin) for track_id, box, margin in candidate_items if margin > 0]
                if selected_items:
                    total_source_selected += len(selected_items)
                for track_id, box, margin in candidate_items:
                    max_overlap = max((iou(box, target_box) for _, target_box in gt_rows), default=0.0)
                    label = int(max_overlap >= 0.5)
                    score_values.append(margin)
                    score_labels.append(label)
                    expr_scores.append(margin)
                    expr_labels.append(label)
                    if label:
                        positive_scores.append(margin)
                    else:
                        negative_scores.append(margin)
                    if margin >= 4.0 and not label:
                        high_conf_false += 1
                    for threshold in THRESHOLDS:
                        counter = threshold_counts[str(threshold)]
                        selected = margin > threshold
                        counter["selected"] += int(selected)
                        counter["tp"] += int(selected and label)
                        counter["fp"] += int(selected and not label)
                        counter["fn"] += int((not selected) and label)
                ranked = sorted(candidate_items, key=lambda item: item[2], reverse=True)
                top_sets[video][expression_id][frame_id] = {item[0] for item in ranked[:5]}
                for gt_id, target_box in gt_rows:
                    overlaps = [(iou(box, target_box), track_id) for track_id, box, _ in candidate_items]
                    best_overlap, best_track = max(overlaps, default=(0.0, None))
                    if best_overlap >= 0.5:
                        all_hits_50 += 1
                    if best_overlap >= 0.75:
                        all_hits_75 += 1
                    selected_overlap = max((iou(box, target_box) for _, box, _ in selected_items), default=0.0)
                    if selected_overlap >= 0.5:
                        selected_hits_50 += 1
                    if not selected_items:
                        no_selected_frames += 1
                    gt_records += 1
                    matched_by_gt[gt_id].append((frame_id, best_track if best_overlap >= 0.5 else None))
                    positive_candidates = [index + 1 for index, item in enumerate(ranked) if iou(item[1], target_box) >= 0.5]
                    expr_ranks.append(min(positive_candidates) if positive_candidates else None)
                source_tp += sum(int(max((iou(box, target_box) for _, target_box in gt_rows), default=0.0) >= 0.5) for _, box, margin in selected_items)
                source_fp += sum(int(max((iou(box, target_box) for _, target_box in gt_rows), default=0.0) < 0.5) for _, box, margin in selected_items)
                source_selected += len(selected_items)
                for track_id, box, margin in selected_items:
                    if max((iou(box, target_box) for _, target_box in gt_rows), default=0.0) < 0.5:
                        wrong_track_counts[(video, track_id)] += 1

            for gt_id, matches in matched_by_gt.items():
                previous = None
                for _, track_id in sorted(matches):
                    if track_id is not None and previous is not None:
                        fragmentation_links += 1
                        fragmentation_switches += int(track_id != previous)
                    if track_id is not None:
                        previous = track_id

            total_gt += gt_records
            category = query_categories(sentence)
            precision = source_tp / max(source_selected, 1)
            recall = source_tp / max(gt_records, 1)
            rank_info = rank_summary(expr_ranks)
            record = {
                "video": video,
                "expression_id": expression_id,
                "sentence": sentence,
                "categories": category,
                "gt_records": gt_records,
                "candidate_frames": len([frame for frame in frames_with_gt if tracker.get(frame)]),
                "no_candidate_frames": no_candidate_frames,
                "no_selected_frames": no_selected_frames,
                "candidate_recall_iou_0.5": all_hits_50 / max(gt_records, 1),
                "candidate_recall_iou_0.75": all_hits_75 / max(gt_records, 1),
                "source_selected_recall_iou_0.5": selected_hits_50 / max(gt_records, 1),
                "source_selected_precision_iou_0.5": precision,
                "source_selected_rows": source_selected,
                "source_true_positive_rows": source_tp,
                "source_false_positive_rows": source_fp,
                "high_confidence_false_positive_rows_margin_ge_4": high_conf_false,
                "score_quantiles": quantiles(expr_scores),
                "positive_score_quantiles": quantiles([score for score, label in zip(expr_scores, expr_labels) if label]),
                "negative_score_quantiles": quantiles([score for score, label in zip(expr_scores, expr_labels) if not label]),
                "ranking": rank_info,
            }
            coverage_records.append(record)
            ranking_records.append({
                "video": video,
                "expression_id": expression_id,
                "sentence": sentence,
                "categories": category,
                "candidate_count": len(expr_scores),
                "positive_candidate_count": sum(expr_labels),
                "auroc_iou_0.5": auroc(expr_scores, expr_labels),
                "average_precision_iou_0.5": average_precision(expr_scores, expr_labels),
                "ece_10_bins": ece(expr_scores, expr_labels),
                "ranking": rank_info,
                "high_confidence_false_positive_rows_margin_ge_4": high_conf_false,
            })
            semantic_records.append({
                "video": video,
                "expression_id": expression_id,
                "sentence": sentence,
                "categories": category,
                "source_selected_precision": precision,
                "source_selected_recall": recall,
                "top1_positive_rank_rate": rank_info["top1"],
                "high_confidence_false_positive_rows": high_conf_false,
            })
            for category_name in category:
                category_values[category_name].append({"precision": precision, "recall": recall, "top1": rank_info["top1"]})

    def aggregate(records: list[dict[str, float]]) -> dict[str, float | int]:
        return {"expressions": len(records), **{key: float(mean(item[key] for item in records)) for key in records[0]} } if records else {"expressions": 0}

    threshold_summary = {}
    for threshold, counter in threshold_counts.items():
        selected = counter["selected"]
        tp = counter["tp"]
        threshold_summary[threshold] = {
            **counter,
            "precision": tp / max(selected, 1),
            "recall": tp / max(tp + counter["fn"], 1),
            "false_positive_to_true_positive": counter["fp"] / max(tp, 1),
        }

    similarity = []
    for video, expression_map in top_sets.items():
        expression_ids = sorted(expression_map)
        for left_index, left in enumerate(expression_ids):
            for right in expression_ids[left_index + 1:]:
                common = sorted(set(expression_map[left]) & set(expression_map[right]))
                values = []
                for frame in common:
                    a = expression_map[left][frame]
                    b = expression_map[right][frame]
                    values.append(len(a & b) / max(len(a | b), 1))
                if values:
                    similarity.append(float(mean(values)))

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_coverage.json").write_text(json.dumps({
        "format": "flexhook-v2-dance-d1-v1",
        "gt_used_only_in_offline_diagnostic": True,
        "scope": {"videos": len(videos), "expressions": len(coverage_records), "gt_records": total_gt, "candidate_rows_repeated_per_expression": total_candidate_rows},
        "candidate_frames_with_any_tracker_row": total_candidate_frames,
        "source_selected_rows": total_source_selected,
        "tracker_id_fragmentation": {"matched_links": fragmentation_links, "switches": fragmentation_switches, "switch_rate": fragmentation_switches / max(fragmentation_links, 1)},
        "per_expression": coverage_records,
    }, indent=2) + "\n", encoding="utf-8")
    (output / "score_distribution.json").write_text(json.dumps({
        "format": "flexhook-v2-dance-d2-score-distribution-v1",
        "scope": {"candidate_rows": len(score_values), "positive_iou_0.5": sum(score_labels), "negative_iou_0.5": len(score_labels) - sum(score_labels)},
        "all": quantiles(score_values),
        "positive_iou_0.5": quantiles(positive_scores),
        "negative_iou_0.5": quantiles(negative_scores),
        "source_threshold": 0.0,
        "source_threshold_positive_fraction": sum(value > 0 for value in score_values) / max(len(score_values), 1),
        "calibration_error_10_bins": ece(score_values, score_labels),
    }, indent=2) + "\n", encoding="utf-8")
    (output / "ranking_metrics.json").write_text(json.dumps({
        "format": "flexhook-v2-dance-d2-ranking-v1",
        "gt_used_only_in_offline_diagnostic": True,
        "global": {"auroc_iou_0.5": auroc(score_values, score_labels), "average_precision_iou_0.5": average_precision(score_values, score_labels), "ece_10_bins": ece(score_values, score_labels)},
        "per_expression": ranking_records,
        "threshold_sweep_development_only": threshold_summary,
    }, indent=2) + "\n", encoding="utf-8")
    (output / "semantic_error_breakdown.json").write_text(json.dumps({
        "format": "flexhook-v2-dance-d3-semantic-breakdown-v1",
        "classification": "auditable keyword multi-label categories; unknown retained",
        "category_aggregate": {key: aggregate(values) for key, values in sorted(category_values.items())},
        "per_expression": semantic_records,
        "top_repeated_wrong_track_ids": [{"video": video, "track_id": track, "wrong_selected_rows": count} for (video, track), count in wrong_track_counts.most_common(50)],
        "top5_cross_expression_jaccard_mean": float(mean(similarity)) if similarity else None,
        "top5_cross_expression_pair_count": len(similarity),
        "semantic_relation_labels": "No target semantic graph is assumed; non-identical query overlap is reported descriptively, not labeled equivalent/subset.",
    }, indent=2) + "\n", encoding="utf-8")
    (output / "oracle_bounds_development_only.json").write_text(json.dumps({
        "format": "flexhook-v2-dance-oracle-bounds-development-only-v1",
        "warning": "These GT-based threshold and candidate-label numbers are offline diagnostics. They were not fed into any TTA runtime or parameter choice.",
        "candidate_recall_iou_0.5": sum(item["candidate_recall_iou_0.5"] * item["gt_records"] for item in coverage_records) / max(total_gt, 1),
        "candidate_recall_iou_0.75": sum(item["candidate_recall_iou_0.75"] * item["gt_records"] for item in coverage_records) / max(total_gt, 1),
        "source_selected_recall_iou_0.5": sum(item["source_selected_recall_iou_0.5"] * item["gt_records"] for item in coverage_records) / max(total_gt, 1),
        "threshold_sweep": threshold_summary,
        "interpretation": "The gap between all-candidate and source-selected recall estimates how much loss is introduced by the referring score gate, while the all-candidate recall is the fixed MOT ceiling.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "videos": len(videos), "expressions": len(coverage_records), "gt_records": total_gt, "candidate_rows": len(score_values)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
