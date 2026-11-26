"""Independent post-seal targets, ordinary CPU risks and text-count audit."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def text_counts(path):
    return Counter(int(float(line.split(',')[0])) for line in Path(path).read_text().splitlines() if line.strip())


def independent_ds(probability):
    import numpy as np
    from scipy.special import expit
    observed = np.isfinite(probability)
    yes = np.where(observed, probability > .5, 0.).astype(float)
    no = observed-yes
    sensitivity, specificity, prevalence = np.full(3, .8), np.full(3, .8), .5
    for iteration in range(20):
        positive = np.log(prevalence)+yes@np.log(sensitivity)+no@np.log1p(-sensitivity)
        negative = np.log1p(-prevalence)+yes@np.log1p(-specificity)+no@np.log(specificity)
        posterior = expit(positive-negative)
        sensitivity = (posterior@yes+8)/(posterior@observed+10)
        specificity = ((1-posterior)@no+8)/((1-posterior)@observed+10)
        prevalence = (posterior.sum()+1)/(len(posterior)+2)
    evidence = yes@np.log(sensitivity/(1-specificity))+no@np.log((1-sensitivity)/specificity)
    return np.clip(expit(evidence/observed.sum(1)), .02, .98)


def independent_gaussian(probability, state):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    p = np.clip(probability, .001, .999)
    z = np.log(p)-np.log1p(-p)
    known = np.isfinite(p)
    mean, covariance = np.asarray(state['mean']), np.asarray(state['covariance'])
    density = np.empty((len(p), 2))
    for pattern in np.unique(known, axis=0):
        rows = np.flatnonzero(np.all(known == pattern, axis=1))
        axes = np.flatnonzero(pattern)
        for component in range(2):
            density[rows, component] = multivariate_normal.logpdf(z[np.ix_(rows, axes)],
                mean[component, axes], covariance[component][np.ix_(axes, axes)])
    return np.exp(density[:, 1]-logsumexp(density, axis=1))


def independent_cpu_risk(model, features, target, weight):
    import numpy as np
    import torch
    z = []
    with torch.no_grad():
        for begin in range(0, len(target), 256):
            z.extend(model(torch.as_tensor(features[begin:begin+256])).double().numpy())
    z = np.asarray(z)
    risk = float(np.sum(weight*(np.logaddexp(0, z)-target*z))/weight.sum())
    groups = defaultdict(list)
    for i, row in enumerate(np.ascontiguousarray(features, dtype='float32')):
        groups[row.tobytes()].append(i)
    numerator = 0.
    for indices in groups.values():
        total = weight[indices].sum()
        p = np.clip(np.sum(weight[indices]*target[indices])/total, 1e-12, 1-1e-12)
        numerator += total*(-p*np.log(p)-(1-p)*np.log1p(-p))
    floor = float(numerator/weight.sum())
    return dict(risk=risk, floor=floor, excess_KL=risk-floor)


def main():
    import numpy as np
    import torch
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_teacher_noise import make_solver, CorrelatedTeacher, predict
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_joint_teacher_noise_pilot import OUT, PUBLIC, METHODS, ORDINARY, PRIMARY, ordinary_targets, registration
    from scripts.a100_factor_rsi_primary_quality import subset_metadata

    torch.set_num_threads(2)
    guard = generation_guard()
    plan = registration()
    barrier = json.loads((OUT/'all75_sealed_before_GT.json').read_text())
    assert len(barrier['tasks']) == 75 and barrier['GT_open_attempts'] == 0
    assert barrier['registration_sha256'] == file_sha(OUT/'registration.json')
    seals = {}
    for task in barrier['tasks']:
        key = task['pair'], task['method']
        assert key not in seals
        seals[key] = verify_predictions(task['folder'])
    initial_shas = {c['pair']: c['actual_initial_observer_sha256']
                    for c in json.loads((OUT/'contract.json').read_text())['initial_observers']}
    heads = self_checks = projected_files = feedback_rounds = 0
    reports = []
    for case in plan['cases']:
        pair = case['pair']
        folder = OUT/pair
        done = json.loads((folder/'main_completion.json').read_text())
        assert done['GT_open_attempts'] == 0 and len(done['tasks']) == 25
        assert file_sha(folder/'initial_observer.json') == initial_shas[pair]
        initial = json.loads((folder/'initial_observer.json').read_text())
        assert initial['data_sha256'] == case['data_sha256']
        with np.load(folder/'teacher_inputs.npz') as arrays:
            features, probability = arrays['features'], arrays['probability']
        known = np.isfinite(probability)
        mean, targets, ds_report = ordinary_targets(probability)
        assert np.allclose(mean, np.nansum(probability, axis=1)/known.sum(1), atol=1e-15)
        assert np.allclose(targets['DS_FUSED_SOFT_KD'][1], independent_ds(probability), atol=1e-12)
        gaussian = independent_gaussian(probability, initial['state'])
        actual = CorrelatedTeacher.from_state(initial['state']).posterior(probability)
        assert np.allclose(gaussian, actual, atol=1e-10)
        targets['GAUSSIAN_FUSED_SOFT_KD'] = (np.ones(len(features), bool), actual, np.ones(len(features)))
        ordinary = {}
        for name in ORDINARY:
            path = folder/'ordinary'/(name+'.pt')
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['name'] == name and state['seed'] == 0 and state['data_sha256'] == case['data_sha256']
            mask, y, weight = targets[name]
            assert hashlib.sha256(y.tobytes()+weight.tobytes()).hexdigest() == state['target_sha256']
            report = state['fit']
            assert 6000 <= report['steps'] <= 16000 and report['adequate_requested'] and report['adequate_passed']
            assert report['excess_KL'] <= .005 and not report['target_manual_GT_used']
            model = make_solver('cpu')
            model.load_state_dict(state['model'])
            model.eval()
            cpu = independent_cpu_risk(model, features[mask], y, weight)
            error = max(abs(cpu[k]-report[k]) for k in cpu)
            assert error < 1e-5 and cpu['excess_KL'] <= .005+1e-6, (pair, name, cpu, report)
            ordinary[name] = dict(checkpoint_sha256=file_sha(path), steps=report['steps'],
                                  independently_computed_CPU_risk=cpu, maximum_report_error=error)
            heads += 1
        final = torch.load(folder/'joint_primary/final.pt', map_location='cpu', weights_only=False)
        bootstrap = torch.load(folder/'joint_primary/bootstrap.pt', map_location='cpu', weights_only=False)
        assert final['seed'] == 0 and final['no_GT_selection'] and final['data_sha256'] == case['data_sha256']
        assert len(final['rounds']) == 4 and bootstrap['fit']['steps'] == 500
        previous_after = None
        for stage, row in enumerate(final['rounds']):
            r, s = row['observer_update'], row['solver_acceptance']
            assert r['after'] >= r['before']-1.01e-9 and s['after'] >= s['before']-1.01e-7
            assert abs(r['after']-s['before']) < 1e-7
            if previous_after is not None:
                assert abs(previous_after-r['before']) < 1e-7
            previous_after = s['after']
            assert row['solver_fit_proposal']['steps'] == (4000 if stage == 3 else 500)
            feedback_rounds += 1
        covariance = np.asarray(final['observer']['covariance'])
        assert np.linalg.eigvalsh(covariance).min() >= .01-1e-9
        change_r = max(float(np.abs(np.asarray(final['observer'][key])-np.asarray(initial['state'][key])).max())
                       for key in ('mean', 'covariance'))
        change_s = max(float((value-bootstrap['model'][name]).abs().max()) for name, value in final['model'].items())
        current = make_solver('cpu')
        current.load_state_dict(final['model'])
        current.eval()
        cpu_likelihood = CorrelatedTeacher.from_state(final['observer']).likelihood(probability, predict(current, features, 'cpu'))
        # Report CPU/GPU likelihood drift; proper task quality is evaluated by
        # official predictions, not by treating this scalar as semantic truth.
        likelihood_error = abs(cpu_likelihood-previous_after)
        metadata = subset_metadata(case['target'], case['evaluated_video'])
        obs = np.asarray(metadata['videos'][case['evaluated_video']]['observations'])
        query_count = len(metadata['episodes'])
        for name in METHODS:
            root = folder/'predictions'/name
            score, bits = np.load(root/'scores.npy'), np.load(root/'mask.npy')
            assert score.shape == bits.shape == (query_count, len(obs))
            assert np.isfinite(score).all() and np.array_equal(bits, score > 0)
            for frame in np.unique(obs[:, 0]):
                indices = np.flatnonzero(obs[:, 0] == frame)
                for qi in range(query_count):
                    k = int(bits[qi, indices].sum())
                    chosen = indices[np.lexsort((indices, -score[qi, indices]))][:k]
                    assert set(chosen) == set(indices[bits[qi, indices]])
            self_checks += 1
        for (p, name), seal in seals.items():
            if p != pair or 'count_method' not in seal:
                continue
            by = {(r['video'], r['expression']): r for r in seals[p, seal['count_method']]['episodes']}
            assert len(by) == len(seal['episodes']) == query_count
            for row in seal['episodes']:
                assert text_counts(row['prediction_path']) == text_counts(by[row['video'], row['expression']]['prediction_path'])
                projected_files += 1
        reports.append(dict(pair=pair, ordinary=ordinary, joint_observer_parameter_change=change_r,
                            joint_solver_parameter_change=change_s, final_CPU_likelihood=cpu_likelihood,
                            final_CPU_GPU_likelihood_absolute_difference=likelihood_error,
                            likelihood_increase_not_ground_truth_or_HOTA=True))
    assert heads == 18 and feedback_rounds == 12 and self_checks == 27 and not guard
    result = dict(status='PASS_ALL75_JOINT_NOISE_SEALS_18_ORDINARY_CPU_RISKS_TARGETS_AND_COUNTS',
                  records=75, adequate_ordinary_heads_recomputed=heads, feedback_rounds_checked=feedback_rounds,
                  exact_CountSelf_aliases=self_checks, projected_query_files=projected_files,
                  cases=reports, manual_GT_reads=0, new_fits_teacher_calls_or_RMOT_predictions=0,
                  latent_posterior_or_likelihood_not_semantic_accuracy_claim=True,
                  source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json', result)
    (PUBLIC/'joint_teacher_noise_primary_sealed_audit_20261125.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
