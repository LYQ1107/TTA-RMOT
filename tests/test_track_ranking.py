"""Only checks needed for trusted CPU diagnostic quantities."""
from fractions import Fraction
import numpy as np
from scipy.special import expit,logit
from scripts.track_ranking_common import (
    aggregate,auc,binary_f1,bisect_result,flip_labels,frame_segments,prefix_ceiling,
)


def test_segments_are_frame_windows_not_observation_chunks():
    frames=np.asarray([10,11,30,31,50,54])
    assert [a.tolist() for a in frame_segments(frames,20)]==[[0,1],[2,3,4,5]]
    assert [a.tolist() for a in frame_segments([10,11,30,31,50,59],20)]==[[0,1],[2,3],[4,5]]
    assert [a.tolist() for a in frame_segments([10,11,70,71,72],20)]==[[0,1,2,3,4]]
    assert frame_segments([3,4,5],40)[0].tolist()==[0,1,2]


def test_four_aggregates_and_stable_logit_mean():
    x=np.asarray([-3.,-2.,0.,1.,4.]);r=aggregate(x)
    assert r['MEAN']==0 and r['MEDIAN']==0 and r['TOP25']==2.5
    assert np.isclose(r['LOGIT_MEAN'],logit(expit(x).mean()))
    assert np.isfinite(aggregate([-1000.,-999.])['LOGIT_MEAN'])
    assert np.isfinite(aggregate([999.,1000.])['LOGIT_MEAN'])


def test_auroc_undefined_classes_and_score_ties():
    assert auc([2,2],[False,True])==.5
    assert auc([1,2],[False,True])==1
    assert auc([1,2],[True,False])==0
    assert auc([1,2],[False,False]) is None
    assert auc([],[]) is None


def test_prefix_ceiling_all_boundaries_against_exact_fractions():
    import itertools
    for n in range(1,9):
        for y in itertools.product((False,True),repeat=n):
            p=sum(y)
            values=[Fraction(2*sum(y[:k]),k+p) if k+p else Fraction(1) for k in range(n+1)]
            f,k=prefix_ceiling(np.arange(n,0,-1),y,np.arange(n))
            assert f==float(max(values)) and k==values.index(max(values))
    assert prefix_ceiling([1.,1.],[False,True],[5,2])==(1.,1)


def test_bisection_is_same_as_search_feasibility_and_dense_perfect():
    for n in range(1,33):
        for k in range(n+1):
            y=np.arange(n)<k
            r=bisect_result(np.arange(n,0,-1),y,y,np.arange(n))
            assert r['selected_count']==k and r['F1']==1
            assert r['questions']<=int(np.ceil(np.log2(n+1)))
            assert binary_f1(np.flatnonzero(y),y)==1


def test_noise_answers_are_reused_per_physical_query_not_model_order():
    y=np.asarray([True,False,False,True])
    assert np.array_equal(flip_labels(y,'Dance','v','q',1,0),np.zeros(4,dtype=bool))
    assert np.array_equal(flip_labels(y,'Dance','v','q',0,1),np.ones(4,dtype=bool))
    assert np.array_equal(flip_labels(y,'Dance','v','q',.2,.3),flip_labels(y,'Dance','v','q',.2,.3))


def test_matching_precedes_short_track_filter():
    from scripts.analyze_pf_full_offline import overlaps,matched_gt
    boxes=[[0.,0.,10.,10.],[0.,0.,10.,10.]]
    _,matched=matched_gt(overlaps(boxes,boxes[:1]),[0,1])
    labels=np.asarray([i in matched for i in range(2)])
    assert labels.tolist()==[True,False]
    # Removing the first (short) track does not reassign its match to the second.
    assert labels[np.asarray([False,True])].tolist()==[False]


def test_requested_source_inventory_has_nine_complete_not_nine_total_banks():
    import json
    from scripts.track_ranking_common import ROOT,TARGETS,MODELS
    rows=json.loads((ROOT/'results/multidomain_generality/margin_paths.json').read_text())['files']
    complete=[r for r in rows if r['coverage']=='complete']
    assert len(complete)==9
    assert {(r['target'],r['model']) for r in complete}=={(t,m) for t in TARGETS for m in MODELS}
    assert any(r['coverage']=='historical' for r in rows)


def test_bootstrap_weights_keep_all_queries_of_a_video_together():
    import pandas as pd
    from scripts.summarize_track_ranking import paired_samples,decisions
    data=pd.DataFrame(dict(video=['a','b','b','b'],difference=[.2,.5,.6,.7]))
    weights=np.asarray([[1,1],[2,0],[0,2]])
    actual=paired_samples(data,'difference',['a','b'],weights)
    assert np.allclose(actual,[.5,.2,.6])
    assert decisions(.7,.75,.8)[:2]==('轨迹级排序可靠','前缀约束可接受')
    assert decisions(.6,.7,.7)[:2]==('轨迹级更好但仍不够可靠','前缀约束代价过大')
    assert decisions(.7,.71,.9)[0]=='轨迹级没有明显改善'
