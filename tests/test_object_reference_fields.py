"""Check same-fallback and temporal controls in the scientific comparison."""
import numpy as np
from scipy.special import expit,logit
from scripts.a100_object_reference import reference_fields


def example(beta):
    ds = np.array([[.1,.2],[.3,.4],[.6,.7]])
    learned = np.array([[.2,.4],[.5,.6],[.7,.3]])
    control = np.array([[.3,.5],[.5,.6],[.6,.4]])
    direct = np.arange(12).reshape(3,4)/15.
    residual = direct/2
    coords = np.array([[.1,.4,.1,.2],[.3,.5,.1,.3],
                       [.8,.4,.1,.2],[.7,.5,.1,.3]])
    columns = np.array([0,1,0,1])
    directions = np.array([1,0,-1])
    obs = np.array([[0,11],[0,12],[1,11],[1,12]])
    tracks = np.array([11,12])
    values = reference_fields(ds,direct,residual,learned,control,coords,columns,
        directions,beta,obs,tracks,'synthetic-video')
    return values,ds,learned,direct,coords,columns,obs


def test_reference_reconstruction_and_same_fallback_ordinary_hybrid():
    values,ds,learned,direct,coords,columns,obs = example(2.)
    expected = ds[:,columns].copy()
    expected[0] = expit(logit(learned[0,columns])+2.*(coords[:,0]-.5))
    expected[2] = expit(logit(learned[2,columns])+2.*(.5-coords[:,0]))
    np.testing.assert_allclose(values['REFERENCE_FULL'],expected,atol=1e-12)
    assert np.array_equal(values['FRAME_DIRECT_HYBRID'][[0,2]],direct[[0,2]])
    assert np.array_equal(values['FRAME_DIRECT_HYBRID'][1],ds[1,columns])
    for name,value in values.items():
        if name not in ('FRAME_DIRECT','FRAME_RESIDUAL'):
            assert np.array_equal(value[1],ds[1,columns])
    for track in (11,12):
        ids = np.flatnonzero(obs[:,1] == track)
        np.testing.assert_allclose(values['REFERENCE_TIME_AVERAGE'][:,ids],
            np.broadcast_to(expected[:,ids].mean(1)[:,None],(3,len(ids))),atol=1e-12)
        np.testing.assert_array_equal(np.sort(values['REFERENCE_SHUFFLED'][:,ids],axis=1),
                                     np.sort(expected[:,ids],axis=1))


def test_zero_slope_does_not_activate_reference_predictions_or_the_hybrid():
    values,ds,_,_,_,columns,_ = example(0.)
    for name,value in values.items():
        if name not in ('FRAME_DIRECT','FRAME_RESIDUAL'):
            assert np.array_equal(value,ds[:,columns])
