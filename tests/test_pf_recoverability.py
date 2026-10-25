import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flex_tta.diagnostics.recoverability import RankedCandidate, rank_recoverability


def test_recovery_above_random_baseline():
    frames=[[
        RankedCandidate(1,-1,True),RankedCandidate(2,-2,False),
        RankedCandidate(3,-3,False),RankedCandidate(4,-4,False),
    ] for _ in range(4)]
    r=rank_recoverability(frames)
    assert r['frames_with_retrievable_gt']==4
    assert r['conditional_r_at_k']['1']==1
    assert abs(r['random_r_at_k_matched_candidate_counts']['1']-0.25)<1e-9


def test_no_positives_does_not_force_recovery():
    r=rank_recoverability([[RankedCandidate(1,1.0,False)],[]])
    assert r['frames_with_retrievable_gt']==0
    assert r['frames_without_candidates']==1
    assert r['frames_without_correct_candidate']==1
    assert r['conditional_r_at_k']['1']==0


def test_multi_positive_random_probability():
    frame=[RankedCandidate(i, float(5-i), i in (1,2)) for i in (1,2,3,4)]
    r=rank_recoverability([frame],k_values=(1,2))
    assert r['conditional_r_at_k']['2']==1
    assert abs(r['random_r_at_k_matched_candidate_counts']['1']-0.5)<1e-9
    assert abs(r['random_r_at_k_matched_candidate_counts']['2']-(5/6))<1e-9
