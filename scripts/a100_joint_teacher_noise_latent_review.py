"""Inspect learned observation components without treating teacher votes as GT."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from scipy.special import expit
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_teacher_noise import CorrelatedTeacher, make_solver, predict
    from scripts.a100_joint_teacher_noise_pilot import OUT, PUBLIC, registration

    torch.set_num_threads(2)
    guard = generation_guard()
    plan = registration()
    barrier = json.loads((OUT/'all75_sealed_before_GT.json').read_text())
    assert len(barrier['tasks']) == 75 and barrier['GT_open_attempts'] == 0
    cases = {}
    for case in plan['cases']:
        folder = OUT/case['pair']
        with np.load(folder/'teacher_inputs.npz') as saved:
            probability, features = saved['probability'], saved['features']
        state = torch.load(folder/'joint_primary/final.pt', map_location='cpu', weights_only=False)
        initial = json.loads((folder/'initial_observer.json').read_text())
        observer = CorrelatedTeacher.from_state(state['observer'])
        model = make_solver('cpu')
        model.load_state_dict(state['model'])
        prior = predict(model, features, 'cpu')
        latent = observer.posterior(probability, prior)
        teacher_only = CorrelatedTeacher.from_state(initial['state']).posterior(probability)
        mean = np.nanmean(probability, axis=1)
        known = np.isfinite(probability)
        scopes = {'all_observed_events': np.ones(len(mean), bool),
                  'complete_three_view_events': known.all(1),
                  'complete_strong_yes': known.all(1) & (np.nanmin(probability, axis=1) >= .9),
                  'complete_strong_no': known.all(1) & (np.nanmax(probability, axis=1) <= .1)}
        summary = {}
        for name, mask in scopes.items():
            summary[name] = dict(events=int(mask.sum()),
                teacher_mean=float(mean[mask].mean()) if mask.any() else None,
                teacher_only_latent_mean=float(teacher_only[mask].mean()) if mask.any() else None,
                joint_latent_mean=float(latent[mask].mean()) if mask.any() else None,
                solver_mean=float(prior[mask].mean()) if mask.any() else None,
                latent_differs_from_mean_teacher_threshold=int(((latent >= .5) != (mean >= .5))[mask].sum()),
                solver_differs_from_mean_teacher_threshold=int(((prior >= .5) != (mean >= .5))[mask].sum()))
        covariance = observer.covariance
        sd = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        correlation = covariance/(sd[:, :, None]*sd[:, None, :])
        result = dict(initial_logit_centres=initial['state']['mean'], final_logit_centres=observer.mean.tolist(),
                      final_component_teacher_probability_medians=expit(observer.mean).tolist(),
                      fitted_latent_component_correlations=correlation.tolist(),
                      component1_centre_mean_is_positive=bool(observer.mean[1].mean() > 0),
                      component0_centre_mean_is_negative=bool(observer.mean[0].mean() < 0),
                      scopes=summary, final_model_sha256=file_sha(folder/'joint_primary/final.pt'))
        cases[case['pair']] = result
        print(json.dumps(dict(pair=case['pair'], result=result)), flush=True)
    assert not guard
    receipt = dict(status='COMPLETE_JOINT_NOISE_TEACHER_ONLY_LATENT_COMPONENT_REVIEW', cases=cases,
                   manual_GT_reads=0, new_fits_teacher_calls_or_MOT_prediction_files=0,
                   agreement_with_teacher_not_accuracy_and_disagreement_not_correction=True,
                   fitted_component_correlation_not_measured_true_error_correlation=True,
                   class_mean_orientation_not_proof_of_semantic_identification=True,
                   source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'teacher_only_latent_review.json', receipt)
    (PUBLIC/'joint_teacher_noise_primary_latent_review_20261125.json').write_text(json.dumps(receipt, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
