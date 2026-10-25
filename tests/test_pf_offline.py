"""Small assignment checks; no real GT or experiment is opened by these tests."""
import numpy as np
from scripts.analyze_pf_full_offline import matched_gt, overlaps, auroc


def test_assignment_counts_duplicates_as_false_positives():
    gt, candidates = matched_gt(np.array([[.9], [.8]]), [0, 1])
    assert gt == {0} and candidates == {0}


def test_assignment_maximizes_valid_match_cardinality_first():
    gt, candidates = matched_gt(np.array([[.99, .51], [.51, .49]]), [0, 1])
    assert len(gt) == len(candidates) == 2


def test_rank_metric_and_empty_boxes():
    assert auroc([0., 1., 2.], [0, 1, 1]) == 1.
    assert overlaps([(0., 0., 1., 1.)], []).shape == (1, 0)
