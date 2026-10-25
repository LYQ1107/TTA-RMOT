"""Necessary scientific checks for post-hoc counts, score controls and protocol."""
import numpy as np
from scripts.diag_pf_protocol import normalized_label_xywh
from scripts.diag_pf_countmatched import topk_ids
from scripts.diag_pf_query_information import average_precision,binary_summary,decompose,frame_metrics
from scripts.diag_pf_frozen_features import sampled_frames,cross_video_nn


def test_official_labels_are_top_left_not_center_xywh():
    assert normalized_label_xywh((.1,.2,.3,.4),100,50)==(10.,10.,30.,20.)


def test_countmatching_uses_requested_count_even_when_frozen_is_all_negative():
    ids=[2,1,3];scores={2:-1.,1:-1.,3:-3.}
    assert topk_ids(ids,scores,0)==set()
    assert topk_ids(ids,scores,1)=={1}
    assert topk_ids(ids,scores,2)=={1,2}


def test_average_precision_handles_score_ties_as_threshold_groups():
    assert average_precision([1.,1.,1.,1.],[1,0,1,0])==.5
    assert average_precision([4.,3.,2.],[1,1,0])==1.


def test_single_class_presence_does_not_invent_auroc():
    r=binary_summary([1.,2.],[0,0])
    assert r["AUROC"] is None and r["AUPRC"] is None
    assert binary_summary([1.,2.],[1,1])["AUPRC"]==1.


def test_visual_query_interaction_variance_is_an_exact_decomposition():
    s=np.array([[1.,2.,5.],[2.,-1.,4.],[6.,3.,2.]])
    v,q,i,ss=decompose(s)
    assert np.allclose(s,v+q-s.mean()+i)
    assert np.isclose(ss["total_ss"],sum(ss[k] for k in ("candidate_visual","query_marginal","interaction")))
    assert v.shape==(3,1) and q.shape==(1,3)


def test_text_swap_does_not_call_overlapping_real_query_a_negative():
    s=np.array([[3.,3.,0.],[1.,1.,1.],[0.,0.,3.]])
    y=np.array([[1,1,0],[0,0,0],[0,0,1]],dtype=bool)
    rows,_=frame_metrics(s,y)
    r=rows[0]
    assert r["all_other_top1"]==.5
    assert r["disjoint_top1"]==r["hard_disjoint_top1"]==0.
    assert r["mixed_source_AUROC"]==1.


def test_all_target_frames_are_not_rank_discrimination_evidence():
    rows,_=frame_metrics(np.array([[1.,2.],[3.,4.]]),np.ones((2,2),dtype=bool))
    assert rows[0]["mixed_frames"]==0 and rows[0]["random_top1"]==1.
    assert "mixed_source_AUROC" not in rows[0]


def test_feature_frames_are_fixed_index_quantiles_without_gt_inputs():
    assert sampled_frames(list(range(1,102)))==[11,51,91]


def test_feature_retrieval_excludes_same_video_even_with_identical_features():
    f=np.array([[1.,0.],[1.,0.],[0.,1.],[.1,1.]])
    y=np.array([[1,0],[1,0],[0,1],[0,1]],dtype=bool)
    hit,random,retrievable=cross_video_nn(f,y,np.array(["a","a","b","b"]))
    assert np.all(hit==0) and np.all(random==0) and not retrievable.any()
