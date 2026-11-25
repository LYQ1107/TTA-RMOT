"""Exact adaptive-with-replacement cancellation and neural-gradient check.

No random draws, dataset labels, teacher inference, neural fitting or HOTA.
The old PURE/LURE contract is unchanged and not rerun.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.residual_risk import ResidualRisk,coverage_distribution,numpy_risk


def main():
    import torch
    from flex_uda.active.residual_risk import torch_risk,bounded_logits
    query=np.asarray([0,0,1,1,1]);teacher=np.asarray([0.,1.,.2,.8,.4])
    x=np.asarray([[1.,-.4],[1.,1.1],[1.,-1.3],[1.,.7],[1.,.2]])
    theta=np.asarray([.3,-.8]);raw=x@theta;z=8*np.tanh(raw/8)
    initial=np.asarray([.85,.1,.9,.15,.75]);paths=[]
    schedule=[0,1,0,1]

    def walk(at,history,records,mass):
        if at==len(schedule):
            risk=ResidualRisk(query)
            for r in records:risk.record(**r)
            c=risk.coefficients();w=risk.population_weights()
            paths.append((mass,c,numpy_risk(z,c,w)));return
        q=schedule[at];group=np.flatnonzero(query==q)
        # Forecast and policy depend only on already revealed synthetic
        # outcomes. They are fixed before the next teacher value is revealed.
        past=float(np.mean([v for _,v in history])) if history else .5
        proxy=.7*initial[group]+.3*past
        priority=np.exp(.5*x[group,1]+past*x[group,1]**2)
        proposal=coverage_distribution(priority)
        for local,index in enumerate(group):
            record=dict(query=q,proxy_before=proxy.copy(),selected=int(index),
                        teacher_probability=float(teacher[index]),proposal_probability=float(proposal[local]))
            walk(at+1,history+[(int(index),float(teacher[index]))],records+[record],mass*proposal[local])

    walk(0,[],[],1.)
    mass=np.asarray([p for p,c,r in paths]);coeff=np.stack([c for p,c,r in paths])
    weights=ResidualRisk(query).population_weights()
    expected_coefficient=mass@coeff
    risk_expected=float(sum(p*r for p,c,r in paths));risk_true=numpy_risk(z,teacher,weights)
    assert abs(mass.sum()-1)<1e-12
    assert np.max(np.abs(expected_coefficient-teacher))<1e-12
    assert abs(risk_expected-risk_true)<1e-12
    # An autograd check of the expected bounded-neural-logit objective, not
    # a claim that a data-dependent optimizer inherits fixed-model unbiasedness.
    xt=torch.as_tensor(x,dtype=torch.float64);wt=torch.as_tensor(weights,dtype=torch.float64)
    th=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    expected_loss=sum(float(p)*torch_risk(bounded_logits(xt@th),torch.as_tensor(c),wt) for p,c,r in paths)
    grad=torch.autograd.grad(expected_loss,th)[0]
    truth_loss=torch_risk(bounded_logits(xt@th),torch.as_tensor(teacher),wt)
    truth_grad=torch.autograd.grad(truth_loss,th)[0]
    grad_error=float((grad-truth_grad).abs().max());assert grad_error<1e-12
    clipped=mass@np.clip(coeff,0,1);clipping_bias=float(np.max(np.abs(clipped-teacher)))
    assert clipping_bias>.01
    proxy_risk=numpy_risk(z,initial,weights)
    report=dict(status='PASS_EXACT_ADAPTIVE_RESIDUAL_RISK_AND_GRADIENT_IDENTITY',
        synthetic_teacher_action_units=5,adaptive_complete_paths=len(paths),
        with_replacement_draws_per_query=2,fixed_query_schedule=schedule,
        path_probability_sum=float(mass.sum()),maximum_coefficient_expectation_error=float(np.max(np.abs(expected_coefficient-teacher))),
        fixed_model_teacher_risk=risk_true,expected_residual_risk=risk_expected,
        expected_gradient_maximum_error=grad_error,proxy_only_risk=proxy_risk,
        probability_clipping_breaks_identity_maximum_bias=clipping_bias,
        signed_coefficients_not_teacher_probabilities=True,
        no_unbiased_claim_for_the_fitted_model_semantics_HOTA_or_outside_pool=True,
        support_and_prelabel_prediction_required=True,
        no_estimated_or_reconstructed_old_propensities=True,
        no_random_draws_neural_training_new_teacher_data_GT_or_HOTA=True,
        classical_augmented_importance_sampling_not_claimed_original=True,
        primary_references=[
          dict(url='https://arxiv.org/html/2607.08347',read='Fixed-model setup and control-variate/LURE formulation; our with-replacement illustration is not its full theorem.'),
          dict(url='https://proceedings.neurips.cc/paper_files/paper/2021/file/09e7655fc1dc8fa7c9d6c4478313d5e6-Paper.pdf',read='Adaptive-fitting assumptions, introduction and estimator context; no import of its asymptotic CI guarantees.'),
          dict(url='https://proceedings.neurips.cc/paper_files/paper/2022/file/9b9cfd5428153ccfbd4ba34b7e007305-Paper-Conference.pdf',read='Abstract/setup: active surrogate is model evaluation, not an automatic guarantee for training.')],
        source_sha256={n:file_sha(ROOT/n) for n in ('scripts/a100_residual_risk_contract.py','flex_uda/active/residual_risk.py')},goal_status='ACTIVE')
    atomic_json(ROOT/'docs/a100_six_direction_20261122/residual_risk_contract_20261125.json',report)
    atomic_json(ROOT/'a100_runs/visual_classical_20261121/supervisor/residual_risk_contract_20261125.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
