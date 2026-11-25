"""CPU equivalence and single-factor checks before the conditional stage4."""
import numpy as np
from flex_uda.active.v8 import BeliefStateV8, questions_v8, PresenceEM
from flex_uda.active.v8_variants import setup, variant_questions, FixedPresenceEM, V7PresenceEstimator
from flex_uda.active.v7 import OnlinePresenceEB
from flex_uda.active.acquisition import VIEWS
from flex_uda.active.baselines import video_rng

REL={v:(.8,.8) for v in VIEWS}


def state(kind=BeliefStateV8,**kwargs):
    f=np.tile(np.linspace(.95,.05,9),(3,1))
    return kind(f,f,np.array([.2,.5,.8]),track_lengths=[8]*9,**kwargs)


def test_no_ablation_is_the_original_state_estimator_and_question_order():
    ds,em,kind=setup(None);a=state();b=state(kind)
    assert type(em) is PresenceEM
    assert np.array_equal(a.posterior,b.posterior)
    assert variant_questions(b,['walking person']*3,[8]*9,ds.values,0,
        ablation=None,rng=video_rng('v'))==questions_v8(a,['walking person']*3,[8]*9,REL,0)


def test_minus_em_collects_answers_but_never_changes_half_parameters():
    s=state();s.observe(0,0,True,'CROP',REL['CROP'])
    em=FixedPresenceEM();em.add_video('v',s);r=em.fit(REL)
    assert r['rho_hat']==.5 and r['pi_strata']==[.5]*3 and r['records']==1
    assert r['convergence_not_applicable'] and r['iterations']==0


def test_v7_estimator_values_exactly_follow_the_old_formula_on_same_state():
    s=state()
    for t in (0,1,2):s.observe(0,t,True,'CROP',REL['CROP'],presence_question=True)
    s.presence_probability[0]=.99;s.confirmed_present[0]=True
    old=OnlinePresenceEB();s.pi_start=old.pi;old.finish_video(s,REL)
    new=V7PresenceEstimator();new.add_video('v',s);r=new.fit(REL)
    assert r['rho_hat']==old.rho and r['pi_strata']==[old.pi]*3


def test_minus_glance_does_not_remove_other_question_types():
    s=state();base=questions_v8(s,['walking person']*3,[8]*9,REL,0)
    got=variant_questions(s,['walking person']*3,[8]*9,REL,0,
        ablation='MINUS_GLANCE',rng=video_rng('v'))
    assert got==[q for q in base if q.kind!='GLANCE']


def test_minus_ds_is_fixed_after_answers_and_fit():
    ds,_,_=setup('MINUS_DS')
    ds.add(('v','q',1),'CROP',True);ds.fit()
    assert ds.values==REL and ds.receipt()['identity_annotator'] is False


def test_minus_presence_never_rejects_or_generates_presence_questions():
    _,_,kind=setup('MINUS_PRES');s=state(kind,rho_hat=.05)
    for t in range(9):s.observe(0,t,False,'CROP',REL['CROP'],presence_question=False)
    assert not s.absent.any() and not s.low_tentative.any() and not s.output_absence().any()
    assert not s.confirmed_present.any()  # no artificial positive T/beta supervision
    got=variant_questions(s,['walking person']*3,[8]*9,REL,0,
        ablation='MINUS_PRES',rng=video_rng('v'))
    assert got and all(q.kind=='IDENTITY' for q in got)


def test_minus_route_changes_only_first_two_round_views():
    s=state()
    for r in range(4):
        original=questions_v8(s,['walking person']*3,[8]*9,REL,r)
        got=variant_questions(s,['walking person']*3,[8]*9,REL,r,
            ablation='MINUS_ROUTE',rng=video_rng('v'))
        assert [(x.query,x.track,x.kind,x.information_gain) for x in got]==[
            (x.query,x.track,x.kind,x.information_gain) for x in original]
        assert all(x.view=='CROP' for x in got) if r<2 else got==original


def test_random_acquisition_seed0_preserves_types_and_membership():
    s=state();args=(s,['walking person']*3,[8]*9,REL,0)
    a=variant_questions(*args,ablation='RANDOM_ACQ',rng=video_rng('v'))
    b=variant_questions(*args,ablation='RANDOM_ACQ',rng=video_rng('v'))
    base=questions_v8(*args)
    assert a==b and a!=base
    assert [x.kind for x in a]==[x.kind for x in base]
    assert set(a)==set(base)
