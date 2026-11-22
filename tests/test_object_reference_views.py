"""A multi-view counterexample and a correctly ordered reference identity."""
import copy
import numpy as np
from scipy.special import expit
from flex_uda.active.object_reference import transport_targets
from flex_uda.active.object_reference_views import reference_frame_supervision,group_canonical_frames
from flex_uda.active.relational_features import frame_supervision


def example():
    records = []
    for frame,x in enumerate((.15,.85)):
        for view,intercept in (('CROP',2.),('FRAME_BOX',-1.)):
            records.append(dict(query_id='q',track_id=11,frame_id=frame,effective_view=view,
                input=dict(query='cars on the right',frame=frame,view=view,bbox=[100*x-5,10,100*x+5,30]),
                yes_probability=float(expit(intercept+2.*(x-.5)))))
    return records,[dict(expression_id='q',query_text='cars on the right')]


def test_view_first_removes_state_but_mean_first_does_not_with_view_intercepts():
    records,queries = example()
    frames = reference_frame_supervision(records,queries,[11],100,50,np.array([1]),2.)
    expected = .5*(expit(2.)+expit(-1.))
    np.testing.assert_allclose(frames['reference_frame_probability'],expected,atol=1e-12)
    cells,raw,coords,_ = frame_supervision(records,queries,[11],100,50)
    assert np.array_equal(cells,frames['cells'])
    np.testing.assert_allclose(raw,frames['raw_frame_probability'],atol=1e-12)
    mean_first = transport_targets(raw,coords,np.ones(len(raw),int),2.)
    assert np.ptp(mean_first) > .05
    np.testing.assert_allclose(group_canonical_frames(frames)['reference_target'],expected,atol=1e-12)


def test_unqualified_queries_keep_the_original_mean_then_clip_convention():
    records,queries = example()
    records[0]['yes_probability'] = .999
    records[1]['yes_probability'] = .6
    frames = reference_frame_supervision(records,queries,[11],100,50,np.array([0]),4.)
    _,raw,_,_ = frame_supervision(records,queries,[11],100,50)
    assert np.array_equal(frames['reference_frame_probability'],np.clip(raw,.02,.98))
    assert np.array_equal(frames['control_frame_probability'],np.clip(raw,.02,.98))
    duplicated = copy.deepcopy(records)+copy.deepcopy(records[:1])
    repeated = reference_frame_supervision(duplicated,queries,[11],100,50,np.array([0]),4.)
    assert np.array_equal(repeated['reference_frame_probability'],frames['reference_frame_probability'])
