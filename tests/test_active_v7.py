import numpy as np
from scipy.special import expit,logit
from flex_uda.active.v7 import BeliefStateV7,OnlinePresenceEB,balanced_weights,weak_rank_logits,questions_v7,fast_step_v7
from flex_uda.active.v6 import ReliabilityV6


def state(nq=1, **kwargs):
    return BeliefStateV7(np.tile(np.arange(6),(nq,1)),np.tile([1.,.8,.6,.4,.2,0.],(nq,1)),
        np.linspace(.1,.9,nq),track_lengths=[4]*6,**kwargs)


def test_identity_answers_never_change_presence_or_eb_pi_sample():
    s=state();p=s.presence_probability.copy()
    s.observe(0,0,True,'CROP',(.8,.8))
    assert np.array_equal(p,s.presence_probability) and not s.presence_answers
    assert s.pseudo_labels()[0,0]==1 and (s.pseudo_labels()[0,1:]==-1).all()


def test_presence_likelihood_uses_mixture_and_per_answer_cap():
    s=state();s.recompute_presence({'CROP':(.999,.999),'CLIP':(.8,.8),'FRAME_BOX':(.8,.8)})
    s.observe(0,0,True,'CROP',(.999,.999),presence_question=True)
    raw=np.log(.5*.999+.5*.001)-np.log(.001)
    assert np.isclose(s.presence_logLR_unclipped_sum[0],raw)
    assert s.presence_logLR_clipped_sum[0]==2 and s.presence_logLR_clip_count[0]==1
    assert s.presence_probability[0]==expit(2.) and s.events[-1]['delta']==2.


def test_pi_uses_full_likelihood_once_per_pair_not_presence_clipped_value():
    s=state();values={'CROP':(.999,.999),'CLIP':(.8,.8),'FRAME_BOX':(.8,.8)}
    s.recompute_presence(values)
    for _ in range(2):s.observe(0,0,True,'CROP',values['CROP'],presence_question=True)
    assert s.presence_probability[0]>.9
    eb=OnlinePresenceEB();r=eb.finish_video(s,values)
    expected=.999**2/(.999**2+.001**2)
    assert len(r['pi_pair_soft_counts'])==1
    assert np.isclose(r['pi_pair_soft_counts'][0]['probability'],expected)
    assert eb.rho==2/3 and np.isclose(eb.pi,(1+expected)/3)
    assert r['pi_pair_soft_counts'][0]['probability']>expit(4)


def test_no_absolute_presence_hard_gate_and_absence_is_stopped():
    s=state();s.refresh_presence([1e-12]);assert not s.absent[0]
    for t in range(5):s.observe(0,t,False,'CROP',(.8,.8),presence_question=True)
    assert s.absent[0] and not questions_v7(s,['red person'],[4]*6,ReliabilityV6().values,2)
    assert s.presence_probability[0]<.1


def test_positive_recheck_is_prioritized_and_keeps_effective_view():
    s=state();s.observe(0,3,True,'CLIP',(.8,.8))
    assert .5<=s.posterior[0,3]<.85
    possible=questions_v7(s,['red person'],[4]*6,ReliabilityV6().values,1)
    assert possible[0].kind=='POSITIVE_RECHECK' and possible[0].track==3 and possible[0].view=='CLIP'
    s.observe(0,3,True,'CLIP',(.8,.8))
    assert not any(q.kind=='POSITIVE_RECHECK' and q.track==3 for q in questions_v7(s,['red person'],[4]*6,ReliabilityV6().values,2))


def test_combined_prior_clipped_and_frequency_weights_exact():
    s=state();s.replace_prior(intercept=[10.]);assert s.prior.max()==2
    assert balanced_weights(1,100,exclusive=False)['binary_positive_weight']==5
    assert balanced_weights(0,100,exclusive=False)['binary_positive_weight']==1
    assert balanced_weights(0,100,exclusive=True)['none_class_weight']==.2
    assert balanced_weights(30,100,exclusive=True)['none_class_weight']==.3
    assert balanced_weights(30,0,exclusive=True)['none_class_weight']==1


def test_rank_ties_and_online_state_are_not_dataset_gt_proportions():
    assert np.array_equal(weak_rank_logits([1,1,1]),[0,0,0])
    assert np.array_equal(weak_rank_logits([0,1,2]),[-1,0,1])
    assert weak_rank_logits([1])[0]==0
    eb=OnlinePresenceEB();assert eb.rho==eb.pi==.5


def test_fast_does_not_use_presence_decisions_without_stratified_teacher_support():
    s=state(2,rho_hat=.999)
    before=s.evidence.copy();r=fast_step_v7(s,np.zeros((2,3)))
    assert r['presence']['positive']==0 and r['presence']['negative']==0
    assert r['presence']['status']=='SKIP_INSUFFICIENT_CONFIRMED_CLASSES'
    assert np.array_equal(before,s.evidence)


def test_replay_recheck_uses_original_eligibility_but_keeps_changed_posterior():
    import pytest
    from scripts.active_v7_validate import validate_recheck
    original=dict(kind='POSITIVE_RECHECK',posterior_probability_before=.7,requested_view='CLIP',frame_id=12)
    replay=dict(original,posterior_probability_before=.95)
    history=[dict(yes_probability=.9,effective_view='CLIP',frame_id=2)]
    validate_recheck(replay,history,original)
    with pytest.raises(AssertionError):validate_recheck(replay,history)


def test_six_direction_gate_is_not_rescued_by_five_control_wins():
    from scripts.active_v7_report import verdict_for
    values=[38.849551,25.313445,65.345262,45.656912,30.331540,30.204325]
    controls=[24.107948,24.413313,58.600791,46.098939,27.588195,24.107948]
    rows=[dict(pair='C'+str(i+1),MINUS_SLOW_v7=v,YT_PRIOR=c,UNIFORM_K=c) for i,(v,c) in enumerate(zip(values,controls))]
    result=verdict_for('MINUS_SLOW_v7',rows,False)
    assert result['directions_above_both_controls']==5
    assert result['result']=='FAIL' and not result['Dance_both_within_two_of_v5']
    assert verdict_for('FULL_v7',rows,False)['result']=='NOT_EVALUATED'
