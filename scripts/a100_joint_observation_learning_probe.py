"""Nontrivial algebra/architecture checks for an unaccepted candidate learner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.joint_observation import (JointObservationHead, SWAP,
    pair_log_probabilities, marginal_logits, count_conditional_KL, categorical_KL)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    a = torch.tensor([.3, -.7, .8, -1.], dtype=torch.float64)
    b = torch.tensor([-.2, .9, .1, .5], dtype=torch.float64)
    coupling = torch.tensor([1., -1.4, 0., -.3], dtype=torch.float64)
    log_p = pair_log_probabilities(a, b, coupling)
    p = log_p.exp()
    recovered = torch.stack((log_p[:, 0]-log_p[:, 3], log_p[:, 1]-log_p[:, 3],
        log_p[:, 2]+log_p[:, 3]-log_p[:, 0]-log_p[:, 1]), -1)
    potential_error = float((recovered-torch.stack((a,b,coupling),-1)).abs().max())
    marginal_error = float((marginal_logits(log_p).sigmoid()-torch.stack((p[:, 0]+p[:, 2], p[:, 1]+p[:, 2]), -1)).abs().max())
    target = torch.tensor([[.1,.2,.5,.2], [.45,.45,.05,.05], [0.,0.,.5,.5], [0.,0.,0.,1.]], dtype=torch.float64)
    count, contrast = count_conditional_KL(target, log_p)
    chain_error = float((categorical_KL(target,log_p)-count-contrast).abs().max())
    shift = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
    j = torch.tensor(-.2, dtype=torch.float64, requires_grad=True)
    _, conditional = count_conditional_KL(target, pair_log_probabilities(a+shift,b+shift,torch.ones_like(a)*j))
    gradient = torch.autograd.grad(conditional.sum(), (shift,j))
    cancellation_error = max(float(g.abs()) for g in gradient)
    factorized = pair_log_probabilities(a,b,torch.zeros_like(a))
    marginal_a = torch.stack((target[:,0]+target[:,2],target[:,1]+target[:,3]),-1)
    marginal_b = torch.stack((target[:,1]+target[:,2],target[:,0]+target[:,3]),-1)
    independent_target = torch.stack((marginal_a[:,0]*marginal_b[:,1],
        marginal_a[:,1]*marginal_b[:,0], marginal_a[:,0]*marginal_b[:,0],
        marginal_a[:,1]*marginal_b[:,1]),-1)
    # Keep 0*log0 terms zero, including a truly absent target.
    independent_log = independent_target.clamp_min(torch.finfo(target.dtype).tiny).log()
    information = categorical_KL(target, independent_log)
    # Bernoulli log probabilities are log-sigmoid(z), log-sigmoid(-z).
    marginal_kl = categorical_KL(marginal_a, torch.stack((F.logsigmoid(a),F.logsigmoid(-a)),-1))
    marginal_kl += categorical_KL(marginal_b, torch.stack((F.logsigmoid(b),F.logsigmoid(-b)),-1))
    marginal_identity_error = float((categorical_KL(target,factorized)-information-marginal_kl).abs().max())
    assert max(potential_error,marginal_error,chain_error,cancellation_error,marginal_identity_error) < 1e-12
    checks = []
    anchors = torch.sin(torch.arange(3*3*7,dtype=torch.float64).reshape(3,3,7)/11)
    present = torch.tensor([[True,True,False],[True,True,True],[False,True,True]])
    first, second = anchors[:,0]+.17, anchors[:,2]-.23
    for configuration in (dict(),dict(shared=False),dict(coupled=False),dict(generic=True)):
        model = JointObservationHead(7,width=5,**configuration).double()
        # Nonzero deterministic weights ensure invariance checks are not a
        # vacuous consequence of uniform-output initialization.
        with torch.no_grad():
            for n, parameter in enumerate(model.parameters()):
                parameter.copy_(torch.sin(torch.arange(parameter.numel(),dtype=torch.float64).reshape(parameter.shape)+n)*.13)
        original = model(anchors,present,first,second)
        swapped = model(anchors,present,second,first)[...,SWAP]
        permutation = [2,0,1]
        permuted = model(anchors[:,permutation],present[:,permutation],first,second)
        missing = anchors.clone();missing[~present] = 1000.
        absent = model(missing,present,first,second)
        error = max(float((original-swapped).abs().max()),float((original-permuted).abs().max()),float((original-absent).abs().max()))
        assert error < 1e-12 and torch.isfinite(original).all()
        categorical_KL(target[:3],original).sum().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        checks.append(dict(configuration=configuration,nonzero_weight_symmetry_error=error,finite_gradient=True))
    covariance = p[:,2]-(p[:,0]+p[:,2])*(p[:,1]+p[:,2])
    assert covariance[0] > 0 and covariance[1] < 0
    report = dict(status='PASS_CANDIDATE_JOINT_LEARNING_ALGEBRA_AND_NONZERO_ARCHITECTURE',
        source_sha256={name:file_sha(ROOT/name) for name in ['scripts/a100_joint_observation_learning_probe.py','flex_uda/active/joint_observation.py']},
        potential_recovery_error=potential_error,marginal_error=marginal_error,
        count_conditional_KL_chain_error=chain_error,shared_shift_and_coupling_cancellation_gradient=cancellation_error,
        factorized_joint_KD_equals_marginal_KD_plus_target_information_error=marginal_identity_error,
        permits_positive_and_negative_association=True,architecture_checks=checks,neural_initialization_seed=0,
        no_empirical_RMOT_training_or_GPU_teacher_GT_HOTA=True,not_a_new_classical_theorem=True,
        potentials_not_ground_truth_identity_or_existence=True,candidate_not_accepted=True,goal_status='ACTIVE')
    atomic_json(ROOT/'docs/a100_six_direction_20261122/joint_observation_learning_math.json',report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
