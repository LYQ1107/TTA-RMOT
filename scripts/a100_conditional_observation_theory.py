"""Synthetic checks of a conditional learning objective, not RMOT efficacy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.conditional_observation import (
    BagContextHead, joint_log_probability, joint_task_logit, weighted_bernoulli_loss, relative_choice_kl)


def main():
    torch.set_num_threads(1); torch.manual_seed(0)
    # A frame No is compatible both with an empty and a nonempty bag.
    no_witnesses = [dict(Y_t=0, A=0), dict(Y_t=0, A=1)]
    observed = np.asarray([.2, .6]); witnesses = []
    for support in (.6, .8, 1.):
        local = observed/support
        assert (local <= 1).all() and np.allclose(support*local, observed)
        witnesses.append(dict(bag_support=support, conditional_local=local.tolist(),
            relative_choice=(local/local.sum()).tolist()))
    weights = torch.tensor([1.5, 2., 3.], dtype=torch.float64)
    target = torch.tensor([.15, .6, .8], dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.long)
    conditional = torch.tensor([-1.2, .8, 1.7], dtype=torch.float64, requires_grad=True)
    bag = torch.tensor(-.4, dtype=torch.float64, requires_grad=True)
    logp = joint_log_probability(bag.expand(3), conditional)
    relative = relative_choice_kl(logp, target, groups, weights)
    bag_grad, local_grad = torch.autograd.grad(relative, (bag, conditional), retain_graph=True)
    assert abs(float(bag_grad)) < 1e-12 and float(local_grad.abs().sum()) > 0
    shifted = joint_log_probability((bag+2.).expand(3), conditional)
    assert abs(float(relative-relative_choice_kl(shifted, target, groups, weights))) < 1e-12
    joint = weighted_bernoulli_loss(joint_task_logit(bag.expand(3), conditional), target, weights)
    assert torch.isfinite(joint)
    zero = relative_choice_kl(torch.log(target), target, groups, weights)
    assert abs(float(zero)) < 1e-12
    singleton = relative_choice_kl(logp, target, torch.arange(3), weights)
    assert abs(float(singleton)) < 1e-12
    observations = torch.randn(2, 3, 7, dtype=torch.float64)
    present = torch.tensor([[True, True, False], [True, True, True]])
    current = torch.randn(2, 7, dtype=torch.float64)
    model_checks = []
    for kind in ('conditional', 'direct_context', 'local_two_factor'):
        model = BagContextHead(7, width=4, kind=kind).double()
        initial = model(observations, present, current)
        assert torch.allclose(initial.sigmoid(), torch.full((2,), .5, dtype=torch.float64))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter)*.01)
        baseline = model(observations, present, current)
        permutation = [2, 0, 1]
        permuted = model(observations[:, permutation], present[:, permutation], current)
        padded = observations.clone(); padded[0, 2] = 1e6
        padding_check = model(padded, present, current)
        error = max(float((baseline-permuted).abs().max()), float((baseline-padding_check).abs().max()))
        assert error < 1e-12
        model_checks.append(dict(kind=kind, initial_task_probability=.5,
            permutation_and_missing_slot_max_error=error))
    extreme = torch.tensor(200., dtype=torch.float64, requires_grad=True)
    extreme_loss = weighted_bernoulli_loss(joint_task_logit(extreme, extreme).reshape(1),
        torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64))
    extreme_gradient = torch.autograd.grad(extreme_loss, extreme)[0]
    assert torch.isfinite(extreme_loss) and float(extreme_gradient) > .99
    report = dict(status='PASS_SYNTHETIC_CONDITIONAL_OBSERVATION_OBJECTIVE',
        instance_No_compatible_bag_states=no_witnesses,
        partial_identification_witnesses=witnesses,
        chain_scope='Y_t implies bag A; P(Y_t|I)=P(A|I)P(Y_t|A,I). A is existence of a match in the provided candidate bag, not certified physical identity or a pure clothing attribute.',
        relative_choice='Uniformly sample an observation index (or fixed positive design weights), condition on a positive match: index probabilities are proportional to w_t*q_t. A common positive support factor cancels.',
        relative_loss=float(relative), joint_bernoulli_loss=float(joint),
        direct_bag_logit_gradient_from_relative_loss=float(bag_grad),
        conditional_local_gradient_L1=float(local_grad.abs().sum()),
        encoder_checks=model_checks,
        confidently_wrong_No_finite_gradient_check=dict(logits=200., loss=float(extreme_loss), gradient=float(extreme_gradient)),
        gradient_boundary='Cancellation holds for the direct common bag-logit path with conditional logits held fixed; a conditional head depending on shared context still backpropagates through that context.',
        learning_boundary='Joint frame No is not an atom-wise or bag-negative label, but joint optimization can still choose a bad latent decomposition. The relative term is redundant at exact per-instance fit; benefits under finite data/noisy teacher need actual ablation.',
        teacher_boundary='Raw Qwen probabilities are not proven calibrated or multiplicatively biased. Matching actual views avoids one confound, not all. CLIP temporal scope remains an observation-model assumption, not known frame truth.',
        no_target_GT_or_teacher_answers_or_predictions_loaded=True,
        new_GPU_or_real_data_fits=0, no_RMOT_gain_or_identified_latent_claim=True, goal_status='ACTIVE')
    atomic_json(ROOT/'docs/a100_six_direction_20261122/conditional_observation_theory_audit.json', report)
    print(json.dumps(dict(status=report['status'], bag_gradient=float(bag_grad),
        local_gradient_L1=float(local_grad.abs().sum()), models_checked=3, no_target_data=True)))


if __name__ == '__main__':
    main()
