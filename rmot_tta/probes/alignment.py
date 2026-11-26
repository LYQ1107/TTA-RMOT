"""Prediction-box alignment between the main stream and frozen probes."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from rmot_tta.types import AlignedCandidate, FrameSnapshot, ProbeResult


def box_iou_xyxy(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return left.new_zeros((left.shape[0], right.shape[0]))
    lt = torch.maximum(left[:, None, :2], right[None, :, :2])
    rb = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area_left = (left[:, 2] - left[:, 0]).clamp_min(0) * (left[:, 3] - left[:, 1]).clamp_min(0)
    area_right = (right[:, 2] - right[:, 0]).clamp_min(0) * (right[:, 3] - right[:, 1]).clamp_min(0)
    return inter / (area_left[:, None] + area_right[None, :] - inter).clamp_min(1e-8)


def _query_candidates(result: ProbeResult) -> tuple[torch.Tensor, list]:
    boxes = torch.tensor([candidate.box_xyxy for candidate in result.candidates], dtype=torch.float32)
    return boxes, list(result.candidates)


def align_probe_results(main: FrameSnapshot, probe_results: Iterable[ProbeResult],
                        min_iou: float = 0.6, ambiguity_iou_margin: float = 0.05) -> list[AlignedCandidate]:
    """Match each probe at most once to the main prediction boxes.

    The matching is based only on predicted boxes.  A close second candidate
    is marked ambiguous and becomes UNKNOWN rather than silently choosing an
    identity in a crowded frame.
    """
    results = {result.query_type: result for result in probe_results}
    main_indices = [int(index) for index in torch.where(main.valid_mask)[0].tolist()]
    # Matching is a small CPU-side bookkeeping operation.  Keeping this path
    # on CPU also makes probe alignment independent of the source model device.
    main_boxes = main.boxes_xyxy[main_indices].detach().float().cpu()
    per_query: dict[str, dict[int, tuple[int, float, str | None]]] = {}
    for query_type, result in results.items():
        probe_boxes, candidates = _query_candidates(result)
        if not len(main_indices) or not len(candidates):
            per_query[query_type] = {}
            continue
        ious = box_iou_xyxy(main_boxes, probe_boxes).cpu().numpy()
        row, col = linear_sum_assignment(1.0 - ious)
        mapping: dict[int, tuple[int, float, str | None]] = {}
        for r, c in zip(row.tolist(), col.tolist()):
            score = float(ious[r, c])
            if score < min_iou:
                continue
            alternatives = np.delete(ious[r], c)
            second = float(alternatives.max()) if alternatives.size else 0.0
            ambiguity = None
            if second >= min_iou and score - second < ambiguity_iou_margin:
                ambiguity = "best_vs_second_iou_margin"
            mapping[main_indices[r]] = (c, score, ambiguity)
        per_query[query_type] = mapping

    output: list[AlignedCandidate] = []
    for main_index in main_indices:
        probe_indices: dict[str, int | None] = {}
        iou_by_query: dict[str, float] = {}
        ambiguity_by_query: dict[str, str | None] = {}
        probabilities: dict[str, float | None] = {"q0": None, "q_eq": None, "q_cf": None}
        quality_values: list[float] = []
        status = "OK"
        for query_type, result in results.items():
            item = per_query.get(query_type, {}).get(main_index)
            if item is None:
                probe_indices[query_type] = None
                iou_by_query[query_type] = 0.0
                ambiguity_by_query[query_type] = "not_matched"
                status = "UNKNOWN"
                continue
            probe_index, iou, ambiguity = item
            probe_indices[query_type] = int(probe_index)
            iou_by_query[query_type] = float(iou)
            ambiguity_by_query[query_type] = ambiguity
            quality_values.append(float(iou))
            if ambiguity is not None:
                status = "UNKNOWN"
            candidate = results[query_type].candidates[probe_index]
            if query_type in probabilities:
                probabilities[query_type] = float(candidate.probability)
        if any(value is not None for value in (probabilities["q0"], probabilities["q_eq"], probabilities["q_cf"])):
            quality = float(sum(quality_values) / max(1, len(quality_values)))
        else:
            quality = 0.0
        output.append(AlignedCandidate(
            main_index=main_index,
            main_uid=main.candidate_uid[main_index],
            probe_indices=probe_indices,
            iou_by_query=iou_by_query,
            ambiguity_by_query=ambiguity_by_query,
            r0=probabilities["q0"],
            req=probabilities["q_eq"],
            rcf=probabilities["q_cf"],
            alignment_quality=quality,
            status=status,
        ))
    return output
