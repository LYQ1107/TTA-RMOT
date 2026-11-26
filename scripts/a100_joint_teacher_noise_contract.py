"""Independent Gaussian marginals, rollback and exact-input risk contracts."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from scipy.special import expit, logsumexp
    from scipy.stats import multivariate_normal
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_teacher_noise import CorrelatedTeacher, make_solver, fit_solver, accept_solver, risk_floor, entropy
    from scripts.a100_joint_teacher_noise_pilot import OUT, PUBLIC, registration

    guard = generation_guard()
    plan = registration()
    torch.set_num_threads(2)
    mean = np.array([[-1., -2., -.5], [1., 1.5, .5]])
    covariance = np.array([[2., .8, .2], [.8, 1., .3], [.2, .3, .7]])
    observer = CorrelatedTeacher(mean, np.stack((covariance, covariance*1.2)))
    z = np.array([[-2., -1., -2.], [2., 1., 0.], [np.nan, 1., 2.], [1., np.nan, np.nan]])
    probability = expit(z)
    actual = observer.log_density(probability)
    expected = np.empty_like(actual)
    for i, row in enumerate(z):
        axes = np.flatnonzero(np.isfinite(row))
        for k in range(2):
            expected[i, k] = multivariate_normal.logpdf(row[axes], mean[k, axes], observer.covariance[k][np.ix_(axes, axes)])
    density_error = float(np.abs(expected-actual).max())
    assert density_error < 1e-10
    diagonal = CorrelatedTeacher(mean, np.asarray([np.diag(np.diag(v)) for v in observer.covariance]))
    assert np.abs(diagonal.log_density(probability)-actual).max() > .01
    prior = np.array([.1, .9, .8, .3])
    expected += np.column_stack((np.log1p(-prior), np.log(prior)))
    posterior = np.exp(expected[:, 1]-logsumexp(expected, axis=1))
    assert np.allclose(posterior, observer.posterior(probability, prior), atol=1e-12)
    # A deliberately bad proposed solver must be rolled back without reading GT.
    model = make_solver('cpu')
    for p in model.parameters():
        p.data.zero_()
    initial = copy.deepcopy(model.state_dict())
    model.net[-1].bias.data.fill_(-10.)
    report = accept_solver(model, initial, observer, np.full((4, 3), .9), np.zeros((4, 5508), 'float32'), 'cpu')
    assert report['accepted_fraction'] == 0 and report['after'] == report['before']
    assert all(torch.equal(model.state_dict()[k], v) for k, v in initial.items())
    duplicate = np.zeros((2, 5508), 'float32')
    floor = risk_floor(duplicate, np.array([.1, .9]), np.array([1., 3.]))
    assert abs(floor-float(entropy(.7))) < 1e-12
    random = np.random.default_rng(0)
    x = random.normal(size=(8, 5508)).astype('float32')
    solver = make_solver('cpu')
    before = copy.deepcopy(solver.state_dict())
    fit_solver(solver, x, np.linspace(.1, .9, 8), 'cpu', steps=2)
    assert any(not torch.equal(before[k], v) for k, v in solver.state_dict().items())
    cases = []
    for case in plan['cases']:
        folder = OUT/case['pair']
        with np.load(folder/'teacher_inputs.npz') as saved:
            p = saved['probability']
        state = CorrelatedTeacher.initialize(p)
        history = state.fit(p, 20)
        assert all(r['after'] >= r['before']-1e-8 for r in history)
        assert np.linalg.eigvalsh(state.covariance).min() >= .01-1e-10
        assert state.mean[1].mean() > state.mean[0].mean()
        initial = dict(state=state.state(), history=history, data_sha256=case['data_sha256'])
        path = folder/'initial_observer.json'
        if path.exists():
            assert json.loads(path.read_text()) == initial
        else:
            atomic_json(path, initial)
        cases.append(dict(pair=case['pair'], events=len(p), initial_loglikelihood=history[0]['before'],
                          final_loglikelihood=history[-1]['after'],
                          maximum_observer_change=max(r['maximum_parameter_change'] for r in history),
                          minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(state.covariance).min()),
                          actual_initial_observer_sha256=file_sha(path)))
    assert not guard
    result = dict(status='PASS_JOINT_NOISE_GAUSSIAN_MARGINAL_ROLLBACK_RISK_AND_REAL_INITIAL_OBSERVERS',
                  scipy_independent_density_maximum_error=density_error,
                  covariance_changes_likelihood=True, missing_views_marginalized_not_imputed_as_votes=True,
                  wrong_solver_update_restored=True, weighted_exact_input_floor_correct=True,
                  tiny_CPU_neural_fit_only=True, initial_observers=cases,
                  manual_GT_reads=0, new_teacher_GPU_calls=0, new_RMOT_predictions=0,
                  not_task_quality_or_RSIcontribution_evidence=True, source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'contract.json', result)
    (PUBLIC/'joint_teacher_noise_primary_contract_20261125.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
