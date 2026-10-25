"""DEVELOPMENT-ONLY candidate-rank recoverability statistics.

Requires post-hoc GT-oracle match flags. This file MUST NOT be imported into
TTA runtimes. No value calculated here is a permitted pseudo label or test-time
selection threshold. This determines whether a positive-free rank hypothesis
has enough *source* ordering signal to warrant a costly four-GPU experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RankedCandidate:
    track_id: int
    margin: float
    matches_gt: bool


def rank_recoverability(frames: Iterable[Sequence[RankedCandidate]], k_values=(1, 3, 5)) -> dict:
    hits = {k: 0.0 for k in k_values}
    random_expectation = {k: 0.0 for k in k_values}
    valid_target_frames = 0
    no_candidate = 0
    unmatched_gt = 0
    observed_rows = 0
    for frame in frames:
        ordered = sorted(frame, key=lambda p: (-p.margin, p.track_id))
        observed_rows += len(ordered)
        if not ordered:
            no_candidate += 1
            continue
        num_pos = sum(bool(p.matches_gt) for p in ordered)
        if num_pos == 0:
            unmatched_gt += 1
            continue
        valid_target_frames += 1
        n = len(ordered)
        for k in k_values:
            kk = min(k, n)
            # Whether ANY positive candidate appears in top-k. The exact
            # null probability is hypergeometric, not k/n for multi-positive.
            hits[k] += float(any(p.matches_gt for p in ordered[:kk]))
            absent = 1.0
            for j in range(kk):
                absent *= (n - num_pos - j) / (n - j) if n - num_pos - j > 0 else 0.0
            random_expectation[k] += 1.0 - absent
    return {
        "development_only_gt_used": True,
        "frames_with_retrievable_gt": valid_target_frames,
        "frames_without_candidates": no_candidate,
        "frames_without_correct_candidate": unmatched_gt,
        "rows_observed": observed_rows,
        "conditional_r_at_k": {str(k): hits[k] / max(valid_target_frames, 1) for k in k_values},
        "random_r_at_k_matched_candidate_counts": {
            str(k): random_expectation[k] / max(valid_target_frames, 1) for k in k_values
        },
    }
