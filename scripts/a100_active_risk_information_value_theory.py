"""Data-free mathematical checks; no model, teacher, prediction, or GT data."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data3/liuyeqiang/FlexHook-TTA')
SUP = ROOT / 'a100_runs/visual_classical_20261121/supervisor'
BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'


def main():
    plan = json.loads((BASE / 'registration.json').read_text())
    paths = ['flex_uda/active/fullscope_visual_runtime.py',
             'flex_uda/active/evidence_visual_learning.py']
    hashes = {}
    for name in paths:
        value = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        expected = [v for k, v in plan['source_sha256'].items() if k.endswith(name)]
        assert expected == [value]
        hashes[name] = value
    namespace = {'np': np}
    tree = ast.parse((ROOT / paths[0]).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('predicted_student_risk', 'proposal')]
    assert len(functions) == 2
    exec(compile(ast.Module(body=functions, type_ignores=[]), paths[0], 'exec'), namespace)

    weights, y, p = np.array([2/3, 1/3]), np.array([.2, .8]), .37
    mean = float(weights @ y)
    variance = float(weights @ (y - mean)**2)
    z = np.log(p / (1 - p))
    raw = 8 * z / (8 - abs(z))
    jac = 64 / (8 + abs(raw))**2 * np.array([.7, -1.3, 1.])
    gradients = (p - y[:, None]) * jac
    lhs = float(weights @ (gradients**2).sum(1))
    rhs = float(namespace['predicted_student_risk'](mean, variance, p) * (jac @ jac))
    assert abs(lhs - rhs) < 1e-12

    epistemic, observation = np.array([.005, .05]), np.array([.2, .002])
    predictive = epistemic + observation
    risk = namespace['predicted_student_risk'](np.array([.5, .5]), predictive,
                                               np.array([.5, .5]))
    after = 1 / (1 / epistemic + 1 / observation)
    gain = epistemic - after
    direct = epistemic**2 / predictive
    assert np.allclose(gain, direct, rtol=0, atol=1e-15)
    information = .5 * np.log1p(epistemic / observation)
    assert np.argmax(risk) == 0 and np.argmax(gain) == np.argmax(information) == 1

    covariance = np.eye(2)
    a = np.array([1., 0.])
    noise = .1
    posterior = covariance - np.outer(covariance @ a, covariance @ a) / (noise + a @ covariance @ a)
    same_direction, new_direction = np.array([1., 0.]), np.array([0., 1.])
    variance_next = np.array([v @ posterior @ v for v in [same_direction, new_direction]])
    direction_information = .5 * np.log1p(variance_next / noise)
    # Three different candidate IDs, but A and B share a feature direction.
    allocation = namespace['proposal'](np.ones(3), np.array([True, False, False]),
                                       np.arange(3))
    assert allocation[1] == allocation[2] == .5
    assert direction_information[0] < direction_information[1]

    result = dict(
        status='PASS_DATA_FREE_GRADIENT_RISK_AND_INFORMATION_VALUE_SCOPE_CHECK',
        current_source_sha256=hashes,
        gradient_identity=dict(assumptions='Fixed x and its logit Jacobian; a hypothetical bounded soft-response distribution with calibrated mean and variance; unweighted BCE.',
            expected_gradient_norm_squared=lhs, conditional_risk_times_jacobian_norm_squared=rhs,
            absolute_error=abs(lhs-rhs), current_R_prequential_residual_is_not_certified_conditional_variance=True),
        scalar_Gaussian_counterexample=dict(
            assumptions='Independent scalar Gaussian latent parameters, known independent Gaussian observation noise, posterior means equal student predictions. This is not a model of Qwen response probabilities.',
            epistemic_variance=epistemic.tolist(), observation_noise_variance=observation.tolist(),
            predictive_squared_error=risk.tolist(), posterior_variance_reduction=gain.tolist(),
            mutual_information_nats=information.tolist(), risk_prefers='A', learning_information_prefers='B'),
        frozen_batch_direction_counterexample=dict(
            assumptions='Fixed pre-batch equal priorities and norm-one features; distinct candidate IDs; Gaussian linear model with identity covariance and noise variance .1.',
            features=[[1., 0.], [1., 0.], [0., 1.]], first_observation='A',
            current_proposal_probability_B=float(allocation[1]), current_proposal_probability_C=float(allocation[2]),
            additional_information_B_nats=float(direction_information[0]),
            additional_information_C_nats=float(direction_information[1]),
            not_actual_RSI_gradient_geometry_or_measured_HOTA=True),
        primary_sources=[
            dict(title='Bayesian Active Learning for Classification and Preference Learning',
                 url='https://arxiv.org/html/1112.5745v1', sections='2 equations1-2;4 maximum entropy discussion',
                 scope='Information about model parameters distinguishes model uncertainty from conditional observation uncertainty; requires a specified posterior and likelihood.'),
            dict(title='Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds',
                 url='https://arxiv.org/html/1906.03671v2', sections='3 Algorithm1 and Proposition1',
                 scope='Uses predicted-label final-layer gradient vectors and k-means++ batch diversity; this is prior art and not an RMOT/noisy-Qwen guarantee.')],
        boundaries=[
            'The gradient identity is a conditional mathematical identity, not proof that R estimates the required moments.',
            'Magnitude of expected local training gradients is not expected held-out risk reduction, semantic correctness, or HOTA gain.',
            'Actual R/S changes between rounds; only priorities within a selected batch are frozen. The direction example is hypothetical, not a diagnosis of learned real features.',
            'Gaussian posterior information requires known probabilistic assumptions absent from the current deterministic Qwen/R/S setup.',
            'Classic information gain and gradient diversity are not new inventions. Do not rename BALD/BADGE or the existing object-uniform control as our contribution.',
            'No algorithm is changed or selected here. Wait for full186/180 reproduction and registered postseal evidence before a new primary experiment.'],
        current_study_GT_predictions_or_neural_parameters_read=0,
        new_GPU_teacher_or_training_operations=0, new_sampling_or_training_seeds=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        goal_achieved=False)
    out = SUP / 'active_risk_information_value_theory_20261127.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), status=result['status'],
                          gradient_identity=result['gradient_identity'],
                          scalar_Gaussian_counterexample=result['scalar_Gaussian_counterexample'],
                          batch_direction=result['frozen_batch_direction_counterexample'],
                          no_current_quality_conclusion=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
