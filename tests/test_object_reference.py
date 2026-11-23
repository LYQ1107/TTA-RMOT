"""Check representation identities that motivate the candidate refinement."""
import numpy as np
from scipy.special import expit, logit
from flex_uda.active.object_reference import (
    transport_targets, grouped_targets, reconstruct, oriented_position)


def test_exact_local_model_has_a_constant_reference_and_reconstructs_state():
    coords = np.array([[.1,.5,.1,.2],[.6,.5,.1,.2],[.9,.5,.1,.2]])
    for direction in (1,-1):
        d = np.full(3,direction)
        position = oriented_position(coords,d)
        beta, reference_logit = 1.7,.25
        actual = expit(reference_logit+beta*(position-.5))
        target = transport_targets(actual,coords,d,beta)
        np.testing.assert_allclose(target,expit(reference_logit),atol=1e-12)
        np.testing.assert_allclose(reconstruct(target,position,beta),actual,atol=1e-12)


def test_grouping_preserves_the_weighted_loss_for_arbitrary_head_logits():
    cells = np.array([[0,0,1],[0,0,3],[0,1,2],[1,0,2],[1,0,5]])
    raw = np.array([.12,.76,.4,.2,.91])
    coords = np.array([[.1,.5,.2,.2],[.8,.4,.1,.2],[.4,.6,.1,.1],
                       [.2,.3,.3,.2],[.9,.1,.1,.1]])
    result = grouped_targets(cells,raw,coords,np.array([1,-1]),2.)
    for logits in (np.array([.3,-2.,1.1]),np.array([-1.,3.,.7])):
        frame_logits = logits[result['frame_to_group']]
        frame_loss = np.logaddexp(0.,frame_logits)-result['reference_frame_target']*frame_logits
        grouped_loss = np.logaddexp(0.,logits)-result['reference_target']*logits
        assert abs(np.dot(frame_loss,result['raw_frame_weight'])-
                   np.dot(grouped_loss,result['sample_weight'])) < 1e-12


def test_zero_slope_or_ineligible_description_keeps_the_target_and_field():
    coords = np.array([[.1,.5,.1,.2],[.9,.5,.1,.2]])
    raw = np.array([.2,.8])
    assert np.array_equal(transport_targets(raw,coords,np.ones(2,int),0.),raw)
    assert np.array_equal(transport_targets(raw,coords,np.zeros(2,int),4.),raw)
    assert np.array_equal(reconstruct(raw,coords[:,0],0.),raw)
    assert np.array_equal(reconstruct(raw,coords[:,0],4.,np.zeros(2,bool)),raw)


def test_reference_error_and_slope_error_bound_probability_error():
    reference = np.array([-.4,.2,.8])
    position = np.array([.1,.5,.9])
    reference_error = np.array([.02,-.1,.2])
    beta, slope_error = 2.,.3
    truth = expit(reference+beta*(position-.5))
    estimated = reconstruct(expit(reference+reference_error),position,beta+slope_error)
    logit_error = reference_error+slope_error*(position-.5)
    np.testing.assert_allclose(logit(estimated)-logit(truth),logit_error,atol=1e-12)
    assert np.all(np.abs(estimated-truth) <= .25*np.abs(logit_error)+1e-12)
