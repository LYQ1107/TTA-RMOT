"""Exact missing-opinion marginalization and MAP update contract, no task GT."""
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.semantic_teacher_patterns import SemanticPatterns,PATTERNS
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_semantic_pattern_teacher_primary import OUT,PUBLIC,PLAN,STEM,registration,input_path


def explicit(observer,probability,prior=None):
    probabilities=[]
    likelihoods=[]
    for i,row in enumerate(probability):
        weight=observer.prevalence if prior is None else np.clip(prior[i],.05,.95)
        mass=[]
        for latent in (0,1):
            total=0.
            for k,pattern in enumerate(itertools.product((0,1),repeat=3)):
                if all(not np.isfinite(p) or p==.5 or int(p>.5)==bit for p,bit in zip(row,pattern)):
                    total+=observer.table[latent,k]
            mass.append(total*(weight if latent else 1-weight))
        probabilities.append(mass[1]/sum(mass))
        likelihoods.append(np.log(sum(mass)))
    return np.asarray(probabilities),float(sum(likelihoods))


def main():
    plan=registration()
    guard=generation_guard()
    base=SemanticPatterns()
    assert np.allclose(base.base.sum(1),1.) and (base.base>0).all()
    for y in (0,1):
        for view in range(3):
            correct=10*base.base[y,PATTERNS[:,view]==y].sum()
            incorrect=10*base.base[y,PATTERNS[:,view]!=y].sum()
            assert abs(correct-8)<1e-12 and abs(incorrect-2)<1e-12
    p=np.asarray(list(itertools.product((.1,.9,np.nan,.5),repeat=3)))
    # A dependent table ensures that this is not just the product-model test.
    table=base.table.copy()
    table[0]+=np.array([.05,-.025,-.025,0,-.002,0,0,.002])
    table[1]+=np.array([-.002,0,0,0,0,0,-.002,.004])
    assert (table>0).all()
    observer=SemanticPatterns(table,prevalence=.7)
    maximum=0.
    for prior in (None,np.linspace(0,1,len(p))):
        q,ll=explicit(observer,p,prior)
        maximum=max(maximum,float(np.abs(q-observer.posterior(p,prior)).max()),abs(ll-observer.log_likelihood(p,prior)))
        previous=observer.objective(p,prior)
        for _ in range(10):
            row=observer.update(p,prior)
            assert row['after']>=previous-1e-8
            previous=row['after']
        np.testing.assert_allclose(SemanticPatterns.from_state(observer.state()).posterior(p,prior),observer.posterior(p,prior),atol=1e-14)
    assert maximum<1e-12
    neutral=SemanticPatterns()
    np.testing.assert_allclose(neutral.posterior(np.full((1,3),np.nan)),[.5])
    assert neutral.posterior(np.ones((1,3)))[0]>.5
    assert neutral.posterior(np.zeros((1,3)))[0]<.5
    cases=[]
    for case in plan['cases']:
        path=input_path(case['pair'])
        assert file_sha(path)==case['data_sha256']
        with np.load(path) as data:
            probability=data['probability']
        initial_path=OUT/case['pair']/'initial_observer.json'
        if initial_path.exists():
            initial=json.loads(initial_path.read_text())
            assert initial['registration_sha256']==file_sha(PLAN) and initial['data_sha256']==case['data_sha256']
            fitted=SemanticPatterns.from_state(initial['state'])
        else:
            fitted=SemanticPatterns()
            history=fitted.fit(probability)
            initial=dict(state=fitted.state(),history=history,data_sha256=case['data_sha256'],
                         registration_sha256=file_sha(PLAN),GT_open_attempts=0)
            atomic_json(initial_path,initial)
        q,ll=explicit(fitted,probability)
        np.testing.assert_allclose(q,fitted.posterior(probability),atol=1e-12)
        assert abs(ll-fitted.log_likelihood(probability))<1e-8
        independent_penalty=10*np.sum(fitted.base*np.log(fitted.table))+np.log(fitted.prevalence)+np.log1p(-fitted.prevalence)
        assert abs(ll+independent_penalty-fitted.objective(probability))<1e-8
        cases.append(dict(pair=case['pair'],events=len(probability),teacher_only_steps=len(initial['history']),
                          selected_prevalence=fitted.prevalence,mean_posterior=float(q.mean()),
                          observer_sha256=file_sha(initial_path),GT_open_attempts=0))
    assert not guard
    report=dict(status='PASS_EXACT_PATTERN_MARGINAL_MAP_AND_REAL_INITIALIZATION',
                synthetic_opinion_patterns=64,maximum_explicit_marginal_error=maximum,
                every_pattern_positive_under_both_classes=True,per_view_prior_correct_incorrect=[8,2],
                cases=cases,GT_open_attempts=0,no_neural_fit_new_teacher_call_or_HOTA=True,
                semantic_identifiability_or_quality_not_proved=True)
    atomic_json(OUT/'contract.json',report)
    atomic_json(PUBLIC/(STEM+'_contract.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
