import numpy as np
from flex_uda.active.belief import BeliefState
from flex_uda.active.map_calibration import source_map_fast_step


def test_unqueried_confidence_cannot_train_map():
    state=BeliefState([[1.,-1.]],[1.],[.5],slope=8.)
    before=state.prior.copy();result=source_map_fast_step(state,[[0.]],source_intercepts=state.a.copy())
    assert result['identity']['unqueried_confident_excluded']==2
    assert result['status'].startswith('SKIP') and np.array_equal(before,state.prior)


def test_single_class_presence_and_identity_skip_without_saturation():
    state=BeliefState([[1.,-1.]],[1.],[.9])
    for _ in range(4):state.observe(0,1,False,'CROP',(.8,.8))
    state.absent[0]=True
    result=source_map_fast_step(state,[[20.,21.]],source_intercepts=state.a.copy())
    assert result['status'].startswith('SKIP')
    assert result['identity']['negative']==1 and result['identity']['positive']==0
    assert result['temperature']==1. and result['bias']==0. and result['slope']==1.


def test_both_classes_update_with_source_prior_and_without_reapplying_evidence():
    state=BeliefState([[1.,-1.],[1.,-1.]],[1.,1.],[.8,.2])
    anchor=state.a.copy();state.a[1]=-2.3
    for _ in range(4):
        state.observe(0,0,True,'CROP',(.8,.8));state.observe(0,1,False,'CROP',(.8,.8))
    state.confirmed_present[0]=True;state.absent[1]=True
    evidence=state.evidence.copy()
    result=source_map_fast_step(state,[[3.,4.],[2.,3.]],source_intercepts=anchor)
    assert result['identity']['status']==result['presence']['status']=='MAP_UPDATED'
    for item in (result['identity'],result['presence']):
        assert item['objective_after']<=item['objective_before']+1e-8
    assert result['slope']>0 and result['temperature']>0
    assert np.isfinite([result['slope'],result['temperature'],result['bias']]).all()
    assert state.a[1]==-2.3 and np.array_equal(evidence,state.evidence)


def test_prior_dominated_label_opposing_teacher_is_not_supported():
    state=BeliefState([[1.,-1.]],[1.],[.5],slope=10.)
    state.observe(0,0,False,'CROP',(.8,.8));state.observe(0,1,True,'CROP',(.8,.8))
    result=source_map_fast_step(state,[[0.]],source_intercepts=state.a.copy())
    assert result['identity']['queried_confident_without_agreeing_evidence']==2
    assert result['identity']['positive']==result['identity']['negative']==0
    assert result['status'].startswith('SKIP')


def test_separable_supported_labels_have_finite_regularized_solution():
    state=BeliefState([np.r_[np.ones(50),-np.ones(50)]],[50.],[.5])
    for track in range(100):
        for _ in range(4):state.observe(0,track,track<50,'CROP',(.8,.8))
    result=source_map_fast_step(state,[[0.]],source_intercepts=state.a.copy())
    assert result['identity']['status']=='MAP_UPDATED'
    assert result['identity']['positive']==result['identity']['negative']==50
    assert 1.<result['slope']<10.
    assert result['identity']['objective_after']<result['identity']['objective_before']


def test_conflicting_presence_is_unknown_not_an_extra_positive():
    state=BeliefState([[1.,-1.],[1.,-1.]],[1.,1.],[.8,.2])
    state.confirmed_present[0]=state.absent[0]=True;state.absent[1]=True
    result=source_map_fast_step(state,[[10.],[10.]],source_intercepts=state.a.copy())
    assert result['presence']['conflicting_queries']==1 and result['presence']['positive']==0
    assert result['presence']['status'].startswith('SKIP')
