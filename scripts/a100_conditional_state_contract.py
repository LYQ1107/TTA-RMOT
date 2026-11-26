"""Independent enumeration/gradient contract for a new latent memory learner.

Synthetic arithmetic only: no RMOT accuracy, target GT, images, teacher calls
or GPU. This is a prerequisite for, never a substitute for, the full primary.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.conditional_state_learning import (
    ScopeEvent, sequence_layout, forward_backward, scoped_bound,
    event_log_likelihood, complete_state_objective, accepted_fraction,
    channel_expected_counts, observable_changes, StateObserver)

SUP = ROOT/'a100_runs/visual_classical_20261121/supervisor'
DEST = SUP/'conditional_state_contract_20261127'
OWN = DEST/'registration.json'
SOURCES = ('scripts/a100_conditional_state_contract.py',
           'flex_uda/active/conditional_state_learning.py')
PARENT = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126/registration.json'


def prepare():
    if OWN.exists():
        raise FileExistsError(OWN)
    original = json.loads(PARENT.read_text())
    for name, digest in original['source_sha256'].items():
        assert file_sha(ROOT/name) == digest, name
    atomic_json(OWN, dict(status='REGISTERED_BEFORE_CONTRACT_EXECUTION',
        source_sha256={name: file_sha(ROOT/name) for name in SOURCES},
        original42_source_sha256=original['source_sha256'],
        no_target_images_teacher_GPU_GT_or_neural_quality=True,
        tests='Independent path enumeration; exact decision likelihood; scoped ELBO; Fisher identity; sufficient statistics; protocol and rejection boundaries; objective backtracking; finite neural writer update.',
        not_primary_implementation_or_quality_completion=True))
    print(json.dumps(dict(status='REGISTERED',sources=len(SOURCES))), flush=True)


def enumerate_paths(prior, keep, obs, events, reliability, protocol, rho):
    """Exhaust all binary states; no forward/backward or bound helper calls."""
    nq, n = prior.shape
    previous = np.full(n, -1, dtype=int)
    for track in sorted(set(obs[:, 1])):
        rows = sorted(np.flatnonzero(obs[:, 1] == track), key=lambda i: obs[i, 0])
        for left, right in zip(rows, rows[1:]):
            previous[right] = left
    log_evidence, probabilities = [], []
    for q in range(nq):
        total, mass = 0., np.zeros(n)
        for raw in itertools.product((0, 1), repeat=n):
            z = np.array(raw)
            probability = 1.
            for i in range(n):
                b = prior[q, i] if z[i] else 1-prior[q, i]
                probability *= b if previous[i] < 0 else (
                    keep[q, i]*int(z[previous[i]] == z[i])+(1-keep[q, i])*b)
            for event in events:
                if event.query != q:
                    continue
                sensitivity, specificity = reliability[event.view]
                yes = [sensitivity if z[node] else 1-specificity for node in event.nodes]
                probability *= np.mean(yes) if event.answer else 1-np.mean(yes)
            total += probability
            mass += probability*z
        if protocol == 'CHECKLIST':
            all_zero = 1.
            for event in events:
                if event.query == q:
                    specificity = reliability[event.view, 1]
                    all_zero *= 1-specificity if event.answer else specificity
            mass *= rho
            total = rho*total+(1-rho)*all_zero
        log_evidence.append(np.log(total))
        probabilities.append(mass/total)
    return np.array(log_evidence), np.array(probabilities)


def run():
    started = time.perf_counter()
    reg = json.loads(OWN.read_text())
    for group in ('source_sha256', 'original42_source_sha256'):
        for name, digest in reg[group].items():
            assert file_sha(ROOT/name) == digest, name
    if (DEST/'result.json').exists():
        raise FileExistsError(DEST/'result.json')
    rng = np.random.default_rng(0)
    obs = np.array([[2, 7], [1, 3], [1, 7], [3, 3]])
    p = rng.uniform(.1, .9, (2, 4));r = rng.uniform(.05, .95, (2, 4))
    rel = np.array([[.83, .78], [.74, .91], [.89, .72]])
    singleton = (ScopeEvent(0, (0,), 1, 0, 'a'), ScopeEvent(0, (3,), 0, 2, 'b'),
                 ScopeEvent(1, (2,), 0, 0, 'c'), ScopeEvent(1, (1,), 1, 2, 'd'))
    clips = (ScopeEvent(0, (2, 0), 1, 1, 'clip-a'),
             ScopeEvent(0, (1, 3), 0, 1, 'clip-b'), singleton[2], singleton[3])
    checks, errors = [], []

    def close(name, actual, expected, tolerance=3e-10):
        error = float(np.max(np.abs(np.asarray(actual)-expected)))
        assert error <= tolerance, (name, error)
        checks.append(name);errors.append(error)

    for protocol in ('PRESUPPOSED', 'CHECKLIST'):
        exact = scoped_bound(p, r, obs, singleton, rel, protocol=protocol,
                             existence_prior=.37, iterations=3)
        ll, post = enumerate_paths(p, r, obs, singleton, rel, protocol, .37)
        close(protocol+'_exact_log_evidence', exact['bound'], ll)
        close(protocol+'_exact_state_posterior', exact['posterior'], post)
        close(protocol+'_posterior_mass', exact['reset_positive']+exact['reset_negative']
              +exact['copy'], np.broadcast_to(exact['existence'][:, None], p.shape))
        counts = channel_expected_counts(singleton, exact)
        close(protocol+'_one_count_per_event', counts.sum(), len(singleton))

        approx = scoped_bound(p, r, obs, clips, rel, protocol=protocol,
                              existence_prior=.37, iterations=30)
        joint_ll, _ = enumerate_paths(p, r, obs, clips, rel, protocol, .37)
        assert np.all(approx['bound'] <= joint_ll+1e-12)
        assert np.min(np.diff(approx['coordinate_bound_trace'], axis=0)) >= -2e-10
        checks.extend([protocol+'_clip_bound_below_exact_joint', protocol+'_coordinate_nondecrease'])
        close(protocol+'_clip_responsibility_mass', [x.sum() for x in approx['responsibilities']], 1.)
        close(protocol+'_one_clip_count_not_frame_count', channel_expected_counts(clips, approx).sum(), len(clips))
        if protocol == 'CHECKLIST':
            derivative = exact

    epsilon = 1e-5
    z = np.log(p/(1-p));w = np.log(r/(1-r))
    for query, node in itertools.product(range(2), range(4)):
        high, low = z.copy(), z.copy();high[query, node] += epsilon;low[query, node] -= epsilon
        values = [scoped_bound(expit(x), r, obs, singleton, rel, protocol='CHECKLIST',
                              existence_prior=.37, iterations=1)['bound'].sum() for x in (high, low)]
        numerical = (values[0]-values[1])/(2*epsilon)
        analytic = (derivative['reset_positive']-(derivative['reset_positive']
                    +derivative['reset_negative'])*p)[query, node]
        close('solver_Fisher_identity_'+str((query, node)), numerical, analytic, 2e-9)
        high, low = w.copy(), w.copy();high[query, node] += epsilon;low[query, node] -= epsilon
        values = [scoped_bound(p, expit(x), obs, singleton, rel, protocol='CHECKLIST',
                              existence_prior=.37, iterations=1)['bound'].sum() for x in (high, low)]
        numerical = (values[0]-values[1])/(2*epsilon)
        first = derivative['layout']['previous'][node] < 0
        analytic = 0. if first else (derivative['copy']-(derivative['copy']+
                    derivative['reset_positive']+derivative['reset_negative'])*r)[query, node]
        close('observer_Fisher_identity_'+str((query, node)), numerical, analytic, 2e-9)

    empty = scoped_bound(p, r, obs, (), rel, protocol='CHECKLIST', existence_prior=.37)
    close('no_evidence_existence_stays_prior', empty['existence'], .37)
    close('no_evidence_log_marginal_zero', empty['bound'], 0.)
    independent = scoped_bound(p, np.zeros_like(r), obs, singleton, rel)
    # At unobserved nodes a zero-copy state is independent of all other events.
    close('zero_copy_unobserved_state_prior', independent['posterior'][0, 1:3], p[0, 1:3])
    rejected = 0
    for bad in [(singleton[0], singleton[0]), (ScopeEvent(0, (0, 1), 1, 1, 'cross-ID'),),
                (ScopeEvent(0, (0, 2), 1, 1, 'backwards'),)]:
        try:
            scoped_bound(p, r, obs, bad, rel)
        except ValueError:
            rejected += 1
    assert rejected == 3;checks.append('duplicate_cross_ID_and_nonchronological_events_rejected')
    fraction, trials = accepted_fraction(1., lambda a: 1.2 if a >= .5 else .9)
    assert fraction == .25 and len(trials) == 3
    fraction, trials = accepted_fraction(1., lambda a: float('nan'))
    assert fraction == 0 and len(trials) == 6
    checks.append('backtracking_accepts_only_finite_nonincreasing_objective')

    import torch
    torch.manual_seed(0);torch.set_num_threads(1)
    observer = StateObserver()
    change, base, _ = observable_changes(obs, rng.normal(size=(4, 5)),
        rng.normal(size=(2, 4, 5)), np.array([[.1,.1,.2,.3]]*4), np.array([False, True]))
    x = torch.tensor(change.reshape(-1, 8));b = torch.tensor(base.reshape(-1))
    before = observer(x, b).detach().clone()
    target = torch.tensor(derivative['copy'].reshape(-1), dtype=torch.float32)
    has_previous = np.tile(sequence_layout(obs)['previous'] >= 0, 2)
    mask = torch.tensor(has_previous)
    optimizer = torch.optim.Adam(observer.parameters(), lr=.001)
    output = observer(x, b)
    loss = nn_bce(output[mask], target[mask]) + (observer.reliability()-.75).square().mean()
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in observer.parameters() if t.grad is not None)
    assert any(float(t.grad.abs().sum()) > 0 for t in observer.parameters() if t.grad is not None)
    optimizer.step()
    after = observer(x, b).detach()
    assert (after-before).abs().max() > 0
    checks.append('neural_observer_has_finite_nonzero_update_CPU_only')
    for group in ('source_sha256', 'original42_source_sha256'):
        for name, digest in reg[group].items():assert file_sha(ROOT/name) == digest, name
    receipt = dict(status='PASS_INDEPENDENT_ENUMERATION_SCOPED_BOUND_AND_EM_GRADIENTS',
        checks=checks, numerical_cells=len(errors), maximum_numeric_error=max(errors),
        original42sources_unchanged=True, source_sha256=reg['source_sha256'],
        current_target_GT_opens=0, real_RMOT_images=0, teacher_calls=0, GPU_calls=0,
        target_neural_models_trained=0, no_HOTA_or_RSI_quality_claim=True,
        scoped_observation_assumption='A clip observes one uniformly latent member of its actual time scope; this is a modeling assumption, not verified Qwen semantics.',
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(DEST/'result.json', receipt)
    print(json.dumps(dict(status=receipt['status'], checks=len(checks),
                         maximum_numeric_error=max(errors))), flush=True)


def nn_bce(prediction, target):
    return torch_bce(prediction, target)


def torch_bce(prediction, target):
    import torch.nn.functional as F
    return F.binary_cross_entropy(prediction, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
