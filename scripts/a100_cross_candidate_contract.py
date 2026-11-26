"""Numerical contracts for the prospective cross-object learner, no GT."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.cross_candidate_learning import CrossCandidateHead, OrdinaryCrossCandidateKD, teacher_risk
from scripts.a100_six_direction import PUBLIC


def main():
    guard = generation_guard(); torch.manual_seed(0); torch.set_num_threads(2)
    n, k, d = 7, 3, 11
    x = dict(anchors_A=torch.randn(n,k,d,dtype=torch.float64),
             anchors_B=torch.randn(n,k,d,dtype=torch.float64),
             first=torch.randn(n,d,dtype=torch.float64), second=torch.randn(n,d,dtype=torch.float64),
             present_A=torch.ones(n,k,dtype=torch.bool), present_B=torch.ones(n,k,dtype=torch.bool))
    x['present_A'][0,-1] = False; x['present_B'][1,-1] = False
    swapped = dict(anchors_A=x['anchors_B'], anchors_B=x['anchors_A'],
                   present_A=x['present_B'], present_B=x['present_A'], first=x['second'], second=x['first'])
    errors = {}; gradient = {}
    for name, model in (('conditional',CrossCandidateHead(d)), ('ordinary',OrdinaryCrossCandidateKD(d))):
        model = model.double()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p)*.1)
        y = model(**x)
        errors[name+'_swap'] = float((y-model(**swapped)[:,[1,0,2,3]]).abs().max().detach())
        errors[name+'_normalization'] = float((y.exp().sum(-1)-1).abs().max().detach())
        assert max(errors.values()) < 1e-12
        if name != 'conditional':
            continue
        full = copy.deepcopy(model); full.conditional_estimation = False
        errors['conditional_full_forward'] = float((y-full(**x)).abs().max().detach())
        for label, learner in (('conditional',model), ('full',full)):
            target = torch.zeros(n,4,dtype=torch.float64); target[:,-1] = 1
            learner.zero_grad(); teacher_risk(learner(**x),target).backward()
            gradient[label+'_No_only_utility'] = sum(float(p.grad.abs().sum()) for name,p in learner.named_parameters()
                if name.startswith(('utility_encoder.','identity.','state.')) and p.grad is not None)
        assert gradient['conditional_No_only_utility'] == 0 and gradient['full_No_only_utility'] > 1e-8
        model.zero_grad(); target = torch.zeros(n,4,dtype=torch.float64);target[:,0] = 1
        teacher_risk(model(**x),target).backward()
        gradient['exactly_one_identity'] = sum(float(p.grad.abs().sum()) for p in model.identity.parameters())
        assert gradient['exactly_one_identity'] > 1e-8
        om, os = torch.randn(d,dtype=torch.float64),torch.rand(d,dtype=torch.float64)+.5
        nm, ns = torch.randn(d,dtype=torch.float64),torch.rand(d,dtype=torch.float64)+.5
        raw = {key:torch.randn_like(v) if v.dtype == torch.float64 else v for key,v in x.items()}
        old = {key:(v-om)/os if v.dtype == torch.float64 else v for key,v in raw.items()}
        new = {key:(v-nm)/ns if v.dtype == torch.float64 else v for key,v in raw.items()}
        before=model(**old).detach(); model.renormalize(om,os,nm,ns)
        errors['prefix_coordinate_change'] = float((before-model(**new)).abs().max().detach())
        assert errors['prefix_coordinate_change'] < 1e-12
    # Common intercept cancels, candidate identity remains.
    u = torch.tensor([1.7,-.8],dtype=torch.float64)
    reference = torch.softmax(u,0)
    errors['common_intercept_cancellation'] = max(float((torch.softmax(u+b,0)-reference).abs().max()) for b in (-30.,0.,30.))
    assert errors['common_intercept_cancellation'] < 1e-12 and float(reference[0]) > .9
    assert not guard
    report = dict(status='PASS_CROSS_CANDIDATE_LEARNING_NUMERICAL_CONTRACTS', errors=errors, gradients=gradient,
        only_seed0=True, synthetic_only=True, no_teacher_or_GPU_or_GT_or_quality_runs=True,
        candidate_identity_receives_conditional_discrimination_gradients=True,
        No_only_count_supervision_cannot_update_conditional_utility=True,
        common_intercept_cancellation_is_classical_not_new_theorem=True,
        no_guarantee_of_cross_domain_HOTA_or_teacher_accuracy=True,
        sources={p:file_sha(ROOT/p) for p in ('flex_uda/active/cross_candidate_learning.py','scripts/a100_cross_candidate_contract.py')})
    atomic_json(PUBLIC/'cross_candidate_learning_contract.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
