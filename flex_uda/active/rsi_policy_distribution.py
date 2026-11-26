"""Finite observation-policy updates measured in distribution space.

This is an independent prospective experiment. The running six-setting
protocol still uses its pinned SGD observer. A local KL constraint on
recorded decision states is not a guarantee of RMOT improvement.
"""
import math
import torch
from flex_uda.active.rsi_curriculum import (
    group_of, exploration_log_probabilities, select_round, update_challenger)


def recorded_selection(actor, features, actions, quotas, reservation, served, stage, generator):
    chosen, logp = select_round(actor, features, actions, quotas, reservation, served, stage, generator)
    order = sorted(actions); index = {u:i for i,u in enumerate(order)}
    values = torch.as_tensor(features, dtype=next(actor.parameters()).dtype)
    scores = actor(values); used = set(served)
    seen = {group_of(actions[u]) for u in used}; decisions = []
    for uid in chosen:
        qi = actions[uid]['local_query']
        available = [u for u in order if actions[u]['local_query'] == qi and u not in used
                     and group_of(actions[u]) not in reservation['groups']]
        fresh = [u for u in available if group_of(actions[u]) not in seen]
        eligible = fresh or available
        ids = torch.tensor([index[u] for u in eligible], dtype=torch.long)
        at = eligible.index(uid)
        old = exploration_log_probabilities(scores[ids]).detach()
        decisions.append((ids, at, old))
        used.add(uid); seen.add(group_of(actions[uid]))
    actor._distribution_trace = (values, decisions)
    if decisions:
        exact = torch.stack([old[at] for _, at, old in decisions]).mean()
        assert torch.allclose(exact, logp.detach(), rtol=0, atol=1e-12)
    return chosen, logp


def flat_gradient(value, parameters, create_graph=False):
    parts = torch.autograd.grad(value, parameters, create_graph=create_graph, retain_graph=True)
    return torch.cat([p.reshape(-1) for p in parts])


def install(parameters, vector):
    start = 0
    with torch.no_grad():
        for p in parameters:
            n = p.numel(); p.copy_(vector[start:start+n].reshape_as(p)); start += n
    assert start == len(vector)


def trust_update(actor, optimizer, log_probability, before, after, baseline,
                 *, frozen=False, reward_override=None, KL_limit=.01, damping=.01):
    reward = float(reward_override) if reward_override is not None else math.tanh(before-after)
    assert math.isfinite(reward) and -1 <= reward <= 1
    advantage = reward-baseline
    parameters = list(actor.parameters())
    original = torch.cat([p.detach().flatten() for p in parameters]).clone()
    values, decisions = actor._distribution_trace

    def objective():
        scores = actor(values); gains = []; divergences = []
        for ids, at, old in decisions:
            new = exploration_log_probabilities(scores[ids])
            gains.append(advantage * torch.exp(new[at]-old[at]))
            divergences.append((old.exp()*(old-new)).sum())
        zero = scores.sum()*0
        return (torch.stack(gains).mean(), torch.stack(divergences).mean()) if decisions else (zero,zero)

    accepted = False; step_fraction = 0.; old_gain = float(advantage)
    gradient_norm = 0.; cg_steps = 0
    if not frozen and before is not None and decisions and advantage != 0:
        gain, KL = objective()
        gradient = flat_gradient(gain, parameters).detach()
        KL_gradient = flat_gradient(KL, parameters, create_graph=True)
        gradient_norm = float(gradient.norm())

        def fisher(vector):
            return flat_gradient((KL_gradient*vector).sum(),parameters).detach()+damping*vector

        direction = torch.zeros_like(gradient); residual = gradient.clone(); conjugate = residual.clone()
        squared = residual.double().dot(residual.double())
        for cg_steps in range(1,11):
            if squared <= 1e-20: break
            value = fisher(conjugate)
            curvature = conjugate.double().dot(value.double())
            if curvature <= 0: break
            alpha = float(squared/curvature)
            direction += alpha*conjugate; residual -= alpha*value
            new_squared = residual.double().dot(residual.double())
            conjugate = residual + float(new_squared/squared)*conjugate; squared = new_squared
        curvature = float(direction.double().dot(fisher(direction).double()))
        if curvature > 0:
            full_step = direction*math.sqrt(2*KL_limit/curvature)
            for attempt in range(10):
                fraction = .5**attempt
                install(parameters, original+fraction*full_step)
                with torch.no_grad(): new_gain, new_KL = objective()
                if bool(torch.isfinite(new_gain) & torch.isfinite(new_KL)) and float(new_KL)<=KL_limit+1e-10 and float(new_gain)>old_gain:
                    accepted = True; step_fraction = fraction; break
        if not accepted: install(parameters, original)
    with torch.no_grad(): final_gain, final_KL = objective()
    change = torch.cat([p.detach().flatten() for p in parameters])-original
    assert float(final_KL)<=KL_limit+1e-8
    return dict(reward=reward, baseline_before=baseline,
        baseline_after=.9*baseline+.1*reward if before is not None else baseline,
        teacher_probe_CE_before=before, teacher_probe_CE_after=after,
        actor_parameter_L2_change=float(change.double().norm()), frozen_actor=frozen,
        fresh_teacher_probe=before is not None, reward_is_teacher_risk_proxy_not_RMOT_truth=True,
        optimizer='natural_gradient_recorded_state_KL_backtracking',
        mean_recorded_state_KL=float(final_KL), KL_limit=KL_limit,
        surrogate_before=old_gain, surrogate_after=float(final_gain), accepted=accepted,
        step_fraction=step_fraction, gradient_norm=gradient_norm, CG_steps=cg_steps,
        recorded_decision_states=len(decisions), trajectory_KL_or_task_return_guarantee=False)


def gated_update(update, credit, *args, **kwargs):
    """Remove score-function noise where this protocol's reward is constant 0.

    Teacher information depends on the independent fixed paid probe, not the
    selected training actions. Below the risk resolution the reward function
    is identically zero. Its expected score-function gradient is zero; skip
    the finite-sample baseline term while preserving the baseline recursion.
    """
    frozen = kwargs.get('frozen',False)
    information = credit.get('information')
    skip = information is None or information < .005
    kwargs['frozen'] = frozen or skip
    report = update(*args,**kwargs)
    report.update(frozen_actor=frozen, skipped_zero_information_context=skip,
                  probe_information=information,
                  zero_reward_context_independent_of_training_action=True)
    return report


SGD_UPDATE = update_challenger
