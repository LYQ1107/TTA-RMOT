"""Verify the conditional proxy assumptions and paid-observation contract."""
from itertools import combinations
import numpy as np
from flex_uda.active.observation_design import ConditionalObjective,plan


def test_fixed_proxy_is_monotone_submodular_on_exhaustive_small_ground_set():
    kernel=np.array([[1,.4,.1],[.4,1,.3],[.1,.3,1.]])
    actions=[dict(track_index=0,frame=1,contrast_squared=.02),dict(track_index=0,frame=2,contrast_squared=.08),
        dict(track_index=1,frame=0,contrast_squared=0),dict(track_index=2,frame=0,contrast_squared=0)]
    objective=ConditionalObjective(kernel,np.ones(3),kernel[:,0],actions,2)
    sets=[frozenset(c) for n in range(5) for c in combinations(range(4),n)]
    for a in sets:
        for b in sets:
            if not a<=b:continue
            for e in set(range(4))-b:
                da=objective.value(a|{e})-objective.value(a);db=objective.value(b|{e})-objective.value(b)
                assert da>=-1e-12 and db>=-1e-12 and da>=db-1e-12
    # A weaker contrast sharing an already stronger paid anchor adds no
    # independent information or coverage at this fixed state.
    assert abs(objective.value([1,0])-objective.value([1]))<1e-12


def test_planner_pays_for_anchor_and_uses_unique_actual_frames():
    x=np.eye(4);samples=[[0,1,2] for _ in range(4)];positions=[{0:.1,1:.5,2:.9} for _ in range(4)]
    actions,receipt=plan(x,np.ones(4),np.array([.8,.6,.3,.1]),samples,[1]*4,positions,6,1)
    paid={(a['track_index'],a['frame']) for a in actions if a['kind']=='PAID_ANCHOR'}
    assert len(actions)==6 and receipt['state_contrasts']==2
    for a in actions:
        if a['kind']=='STATE_CONTRAST':assert (a['track_index'],a['paid_anchor_frame']) in paid


def test_no_relation_signal_spends_available_budget_on_object_coverage():
    samples=[[0,1,2] for _ in range(5)];positions=[{0:.1,1:.5,2:.9} for _ in range(5)]
    actions,receipt=plan(np.eye(5),np.ones(5),np.linspace(.9,.1,5),samples,[1]*5,positions,4,0)
    assert receipt['distinct_objects']==4 and receipt['state_contrasts']==0
    assert len({(a['track_index'],a['frame']) for a in actions})==4
