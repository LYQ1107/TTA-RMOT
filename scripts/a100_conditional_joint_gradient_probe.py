"""Check the proposed estimator's mechanism, including its negative control."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.conditional_joint_observation import ConditionalJointObservationHead
from flex_uda.active.joint_observation import count_conditional_KL,SWAP


def maximum_gradient(parameters):
    return max((float(p.grad.abs().max()) for p in parameters if p.grad is not None),default=0.)


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    anchors=torch.sin(torch.arange(3*3*7,dtype=torch.float64).reshape(3,3,7)/11)
    present=torch.tensor([[True,True,False],[True,True,True],[False,True,True]])
    first,second=anchors[:,0]+.17,anchors[:,2]-.23
    target=torch.tensor([[.1,.2,.5,.2],[.45,.45,.05,.05],[0.,0.,.5,.5]],dtype=torch.float64)
    model=ConditionalJointObservationHead(7,width=5).double()
    with torch.no_grad():
        for n,p in enumerate(model.parameters()):p.copy_(torch.sin(torch.arange(p.numel(),dtype=p.dtype).reshape(p.shape)+n)*.13)
    joint=copy.deepcopy(model);joint.conditional_estimation=False
    value_error=float((model(anchors,present,first,second)-joint(anchors,present,first,second)).abs().max())
    count,contrast=count_conditional_KL(target,model(anchors,present,first,second))
    count.sum().backward();blocked=maximum_gradient((*model.state_encoder.parameters(),*model.local.parameters()))
    count_signal=maximum_gradient((*model.count_encoder.parameters(),*model.object.parameters(),*model.interaction.parameters()))
    assert blocked < 1e-12 and count_signal > 1e-6
    joint_count,_=count_conditional_KL(target,joint(anchors,present,first,second));joint_count.sum().backward()
    unblocked=maximum_gradient((*joint.state_encoder.parameters(),*joint.local.parameters()))
    assert unblocked > 1e-6
    model.zero_grad(set_to_none=True)
    _,contrast=count_conditional_KL(target,model(anchors,present,first,second));contrast.sum().backward()
    contrast_nuisance=maximum_gradient((*model.count_encoder.parameters(),*model.object.parameters(),*model.interaction.parameters()))
    contrast_signal=maximum_gradient((*model.state_encoder.parameters(),*model.local.parameters()))
    assert contrast_nuisance < 1e-12 and contrast_signal > 1e-6
    symmetry=float((model(anchors,present,first,second)-model(anchors,present,second,first)[...,SWAP]).abs().max())
    no_reference=copy.deepcopy(model);no_reference.object_reference=False
    local_before=no_reference.local_scores(anchors,present,first,second)
    local_after=no_reference.local_scores(anchors+.19,present,first,second)
    local_scope_error=max(float((a-b).abs().max()) for a,b in zip(local_before,local_after))
    reference_before=model.local_scores(anchors,present,first,second)
    reference_after=model.local_scores(anchors+.19,present,first,second)
    reference_signal=max(float((a-b).abs().max()) for a,b in zip(reference_before,reference_after))
    assert local_scope_error == 0 and reference_signal > 1e-8
    before=model(anchors,present,first,second).detach()
    om=torch.linspace(-.2,.2,7,dtype=torch.float64);os=torch.linspace(.7,1.3,7,dtype=torch.float64)
    nm=torch.linspace(.1,-.1,7,dtype=torch.float64);ns=torch.linspace(1.1,.9,7,dtype=torch.float64)
    shifted=copy.deepcopy(model);shifted.renormalize(om,os,nm,ns)
    transform=lambda x:(x*os+om-nm)/ns
    after=shifted(transform(anchors),present,transform(first),transform(second))
    normalization_error=float((before-after).abs().max())
    assert max(value_error,symmetry,normalization_error) < 1e-12
    report=dict(status='PASS_CONDITIONAL_JOINT_COUNT_STATE_GRADIENT_ISOLATION',
        source_sha256={n:file_sha(ROOT/n) for n in ['scripts/a100_conditional_joint_gradient_probe.py','flex_uda/active/conditional_joint_observation.py']},
        identical_parameter_forward_error=value_error,conditional_count_loss_state_gradient=blocked,
        full_likelihood_negative_control_state_gradient=unblocked,count_branch_learning_signal=count_signal,
        conditional_relative_loss_count_parameter_gradient=contrast_nuisance,relative_branch_learning_signal=contrast_signal,
        swap_equivariance_error=symmetry,two_encoder_normalization_function_error=normalization_error,
        identical_architecture_parameter_count_for_estimator_ablation=True,
        identical_parameter_count_for_reference_scope_ablation=True,
        no_reference_state_context_sensitivity=local_scope_error,reference_state_context_sensitivity=reference_signal,
        not_universal_robustness_HOTA_or_better_efficiency_guarantee=True,
        conditional_sample_weights_can_still_change_empirical_optimum=True,
        no_GT_GPU_teacher_RMOT_fits_or_HOTA=True,seeds=[0],candidate_not_accepted=True,goal_status='ACTIVE')
    atomic_json(ROOT/'docs/a100_six_direction_20261122/conditional_joint_gradient_math.json',report)
    print(json.dumps(report))


if __name__ == '__main__':main()
