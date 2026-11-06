"""Numerical likelihood checks only; no teacher, GPU or target GT."""
import numpy as np
import subprocess
import sys
from scipy.special import logsumexp
from flex_uda.active.v8 import PresenceEM
from flex_uda.active.v9_offline import RandomEffectsPresence,RandomEffectsBelief,decision_llr,GH_LOGW,GH_X

REL={'CROP':(.8,.8),'CLIP':(.7,.9),'FRAME_BOX':(.85,.65)}


def example():
    em=RandomEffectsPresence()
    em.records=[dict(video='v',query=0,tracks=[(0,[('CROP',True),('CLIP',False)]),(1,[('CROP',False)])]),
                dict(video='v',query=1,tracks=[(2,[('FRAME_BOX',True)])])]
    return em


def test_tau_zero_decision_is_v8_per_track_clipped_sum():
    em=PresenceEM();em.pi=np.array([.8,.3,.05])
    tracks=example().records[0]['tracks']
    got,_=decision_llr(tracks,em.pi,0.,REL)
    assert np.isclose(got,sum(em.track_terms(k,a,REL)[0] for k,a in tracks),atol=1e-13)
    assert decision_llr([],em.pi,1.,REL)==(0.,0)


def test_exact_objective_analytic_gradient_matches_finite_difference():
    em=example();data=em.arrays(REL);theta=np.array([.6,.7,.3,.2,1.])
    value,grad=em.objective(theta,data);numeric=[]
    for j in range(5):
        a=theta.copy();b=theta.copy();a[j]-=1e-6;b[j]+=1e-6
        numeric.append((em.objective(b,data)[0]-em.objective(a,data)[0])/2e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(grad,numeric,rtol=2e-5,atol=1e-7)


def test_B1_fit_is_unclipped_even_when_decision_is_capped():
    rel={'CROP':(.999,.999)};tracks=[(0,[('CROP',True)]*3)]
    mu=np.array([.8,.3,.05])
    raw,_=decision_llr(tracks,mu,1.,rel,capped=False)
    clipped,_=decision_llr(tracks,mu,1.,rel,capped=True)
    assert raw>clipped and clipped<=2.+1e-12
    em=RandomEffectsPresence();em.records=[dict(video='v',query=0,tracks=tracks)]
    theta=np.r_[.6,mu,1.]
    lp=3*np.log(.999);ln=3*np.log(.001)
    exact=np.logaddexp(np.log(.6)+raw+ln,np.log(.4)+ln)
    prior=np.log(.6)+np.log(.4)+np.log(mu).sum()+np.log1p(-mu).sum()-1/8
    assert np.isclose(em.objective(theta,em.arrays(rel))[0],-exact-prior)


def test_random_effects_coverage_and_B2_output_remain_distinct():
    f=np.linspace(.95,.05,9)[None]
    s=RandomEffectsBelief(f,f,[.5],track_lengths=[8]*9,rho_hat=.05,pi_strata=[.9]*3,tau=1.)
    assert s.low_tentative[0] and not s.absent[0] and s.output_absence()[0]
    for t in (0,3):
        s.observe(0,t,False,'CROP',REL['CROP'])
        assert not s.absent[0]
    s.observe(0,6,False,'CROP',REL['CROP'])
    assert s.absent[0] and not s.low_tentative[0]


def test_estimator_rejects_the_new_derived_gt_artifacts():
    code='''from scripts.active_v9_inputs import estimation_guard,PUBLIC
attempts=estimation_guard()
try: (PUBLIC/'presence_truth.csv').open()
except PermissionError: pass
else: raise AssertionError('derived GT must not enter an estimator')
assert len(attempts)==1
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
