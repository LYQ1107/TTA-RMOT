"""Whole primary screen: correlated teacher observer R <-> neural solver S.

Same target-only auxiliary events and complete inputs for all competitors.
No target labels before all75 raw/count outputs have been sealed.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_six_direction import OUT as SIX

OUT = SIX/'joint_teacher_noise_primary_20261125'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PLAN = OUT/'registration.json'
PRIMARY = 'JOINT_CORRELATED_TEACHER_RSI'
ORDINARY = ('MEAN_SOFT_KD', 'MAJORITY_HARD_KD', 'CONFIDENCE_SOFT_KD',
            'CROP_SOFT_KD', 'DS_FUSED_SOFT_KD', 'GAUSSIAN_FUSED_SOFT_KD')
COMPARATORS = (*ORDINARY, 'MEAN_SOFT_POOLED', 'MAJORITY_HARD_POOLED')
METHODS = (PRIMARY, *COMPARATORS)
SOURCES = ('flex_uda/active/joint_teacher_noise.py', 'scripts/a100_joint_teacher_noise_pilot.py',
           'scripts/a100_joint_teacher_noise_contract.py', 'scripts/a100_joint_teacher_noise_statistics.py',
           'scripts/a100_joint_teacher_noise_prepare.py', 'flex_uda/active/factor_rsi_learning.py',
           'flex_uda/active/joint_observation_inputs.py', 'flex_uda/active/observation_learning_inputs.py',
           'flex_uda/active/native_binary_inputs.py', 'scripts/a100_observation_local_cache.py',
           'flex_uda/active/view_evidence.py', 'scripts/a100_factor_rsi_primary_quality.py',
           'flex_uda/a100/evaluation_reuse.py', 'flex_uda/presence/predictions.py',
           'flex_uda/a100/mechanism.py', 'flex_uda/active/deterministic_observation_runtime.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    data = json.loads((OUT/'data_preparation.json').read_text())
    cases = []
    for case in data['cases']:
        case = dict(case)
        folder = OUT/case['pair']
        assert file_sha(folder/'teacher_inputs.npz') == case['data_sha256']
        assert file_sha(folder/'row_keys.json') == case['row_keys_sha256']
        assert case['GT_open_attempts'] == 0 and case['input_dimension'] == 5508
        cases.append(case)
    plan = dict(status='PREREGISTERED_SAME_EVENT_JOINT_CORRELATED_TEACHER_MAIN',
        primary=PRIMARY, comparators=COMPARATORS, ordinary_neural_heads=ORDINARY, cases=cases,
        source_sha256={name: file_sha(ROOT/name) for name in SOURCES},
        data_preparation_sha256=file_sha(OUT/'data_preparation.json'),
        hypothesis='Same-event correlated teacher observations and image-conditioned latent membership can jointly improve referencing beyond direct and fuse-then-distill teacher guidance.',
        solver='5508->128 SiLU->128 SiLU->1; identical ordinary capacity/init, full ordered anchors/current/coarse, no lost state coordinates.',
        observer='Two3D Gaussian components in clipped teacher-logit space; full covariance, exact missing-view marginalization and conditional moments.',
        observer_initialization='Per-view20/80percentiles, initial means bounded below/above-1/+1; variances>=1;20teacher-onlyEM updates, neutral prior.5.',
        covariance_eigenvalue_floor=.01, teacher_logit_clip=[.001, .999], solver_prior_clip=[.05, .95],
        latent_class_orientation='Component1 mean teacher logit exceeds component0; weak teacher-Yes convention, no targetGT orientation.',
        feedback='BootstrapS500 mean-KD steps; four R<->S generalized-EM rounds with S500/500/500/4000 steps; actual observed-teacher likelihood checks for both updates.',
        optimizer='Adam lr.001, batches512, no random minibatch shuffle, neural seed0.',
        ordinary_minimum_steps=6000, ordinary_maximum_steps=16000, ordinary_proper_excess_KL_limit=.005,
        event_preprocessing='Same-event/same-effective-view repeats averaged once for EVERY method; ordinary examples uniformly weighted by event except stated confidence or CROP-subset recipes.',
        hard_baseline='Majority of observed effective-view binary(p>=.5) votes, tied majority positive.',
        confidence_baseline='Per-view weight1+3abs(2p-1), weighted view target and mean-view-confidence event weight.',
        fusion_baselines='Existing prior-regularized DS on SAME-event matrix, and exact shared teacher-only correlated observer initialization followed by adequate KD.',
        pooled_controls='Mean of predicted probabilities over each provided TrackID, computed with stable log-sigmoid sums; no track editing.',
        all_target_auxiliary_teacher_events_and_features_given_to_all_methods=True,
        auxiliary_scope='100 complete3view events from SAME physical target collection, including other videos; explicitly collection-transductive, not video-online.',
        source_teacher_replies=[c['logical_source_responses'] for c in cases], new_teacher_GPU_calls=0,
        source_features_prepaid=True, actual_online_acquisition_or_fresh_cold_cost_not_tested=True,
        no_module_ablation_matrix=True, neural_seeds=[0], raw_records=27, dual_count_records=48,
        official_statistical_records=75, paired_effects=72, CountSelf_exact_aliases=27,
        all75_outputs_sealed_before_current_study_GT=True,
        success_screen='Unchanged strict whole-main screen: raw and BOTH CountMatched effects positive versus ALL8 controls in EVERY one-video case.',
        no_manual_GT_for_training_or_checkpoint_selection=True, all_current_data_DEV=True,
        likelihood_improvement_not_semantic_accuracy_or_HOTA_guarantee=True,
        no_RSIfresh_acquisition_or_three_innovation_acceptance_claim=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/'joint_teacher_noise_primary_registration_20261125.json', plan)
    print(json.dumps(dict(status=plan['status'], cases=[c['pair'] for c in cases], pins=len(SOURCES))), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    assert file_sha(OUT/'data_preparation.json') == plan['data_preparation_sha256']
    for name, expected in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    for case in plan['cases']:
        assert file_sha(OUT/case['pair']/'teacher_inputs.npz') == case['data_sha256']
        assert file_sha(OUT/case['pair']/'row_keys.json') == case['row_keys_sha256']
    return plan


def ordinary_targets(probability):
    import numpy as np
    from flex_uda.active.view_evidence import targets
    known = np.isfinite(probability)
    mean = np.nanmean(probability, axis=1)
    majority = ((np.where(known, probability >= .5, 0).sum(1)/known.sum(1)) >= .5).astype(float)
    confidence = np.where(known, 1+3*np.abs(2*np.nan_to_num(probability)-1), 0.)
    weighted = (np.nan_to_num(probability)*confidence).sum(1)/confidence.sum(1)
    confidence_weight = confidence.sum(1)/known.sum(1)
    _, _, ds, ds_report = targets(probability)
    crop = known[:, 0]
    return mean, dict(MEAN_SOFT_KD=(np.ones(len(mean), bool), mean, np.ones(len(mean))),
        MAJORITY_HARD_KD=(np.ones(len(mean), bool), majority, np.ones(len(mean))),
        CONFIDENCE_SOFT_KD=(np.ones(len(mean), bool), weighted, confidence_weight),
        CROP_SOFT_KD=(crop, probability[crop, 0], np.ones(int(crop.sum()))),
        DS_FUSED_SOFT_KD=(np.ones(len(mean), bool), ds, np.ones(len(mean)))), ds_report


def solve_primary(features, probability, initial, folder, device):
    import copy
    import numpy as np
    import torch
    from flex_uda.active.joint_teacher_noise import make_solver, CorrelatedTeacher, predict, fit_solver, accept_solver
    root = folder/'joint_primary'
    root.mkdir(parents=True, exist_ok=True)
    model = make_solver(device)
    observer = CorrelatedTeacher.from_state(initial)
    final = root/'final.pt'
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        assert state['data_sha256'] == file_sha(folder/'teacher_inputs.npz')
        model.load_state_dict(state['model'])
        return model, state
    rounds, completed = [], -1
    saved = root/'closed.pt'
    if saved.exists():
        state = torch.load(saved, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        observer = CorrelatedTeacher.from_state(state['observer'])
        rounds, completed = state['rounds'], state['completed']
        bootstrap = state['bootstrap']
    else:
        bootstrap_path = root/'bootstrap.pt'
        if bootstrap_path.exists():
            state = torch.load(bootstrap_path, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'])
            bootstrap = state['fit']
        else:
            bootstrap = fit_solver(model, features, np.nanmean(probability, axis=1), device, steps=500)
            torch.save(dict(model=model.state_dict(), fit=bootstrap), bootstrap_path)
    for stage in range(completed+1, 4):
        prior = predict(model, features, device)
        before_posterior = observer.posterior(probability, prior)
        observer_update = observer.update(probability, prior)
        posterior = observer.posterior(probability, prior)
        previous = copy.deepcopy(model.state_dict())
        fit = fit_solver(model, features, posterior, device, steps=4000 if stage == 3 else 500)
        accepted = accept_solver(model, previous, observer, probability, features, device)
        row = dict(stage=stage, observer_update=observer_update, solver_fit_proposal=fit,
                   solver_acceptance=accepted,
                   observer_mean_posterior_change=float(np.abs(posterior-before_posterior).mean()),
                   posterior_is_a_latent_model_estimate_not_truth=True)
        rounds.append(row)
        atomic_json(root/('round'+str(stage)+'.json'), row)
        state = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     observer=observer.state(), rounds=rounds, completed=stage, bootstrap=bootstrap,
                     data_sha256=file_sha(folder/'teacher_inputs.npz'), seed=0,
                     no_GT_selection=True, no_new_teacher_calls=True)
        temp = saved.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(saved)
        print(json.dumps(dict(status='JOINT_NOISE_CLOSED_FEEDBACK_ROUND', case=folder.name, stage=stage,
                              observer_change=observer_update['maximum_parameter_change'],
                              solver_change=accepted['maximum_parameter_change'])), flush=True)
    temp = final.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(final)
    return model, state


def pack(arrays):
    import numpy as np
    return np.column_stack((arrays['bags'].reshape(len(arrays['current']), -1), arrays['present'],
                            arrays['current'], arrays['coarse'])).astype('float32')


def worker(pair_name):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from flex_uda.a100.mechanism import generation_guard, PAIRS
    from flex_uda.active.joint_teacher_noise import CorrelatedTeacher, make_solver, fit_solver
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_factor_rsi_primary_quality import subset_metadata, projection

    plan = registration()
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features()
    case = next(c for c in plan['cases'] if c['pair'] == pair_name)
    folder = OUT/pair_name
    if (folder/'main_completion.json').exists():
        return
    start = time.perf_counter()
    with np.load(folder/'teacher_inputs.npz') as saved:
        features, probability = saved['features'], saved['probability']
    initial_path = folder/'initial_observer.json'
    if initial_path.exists():
        initial = json.loads(initial_path.read_text())
        assert initial['data_sha256'] == case['data_sha256']
    else:
        observer = CorrelatedTeacher.initialize(probability)
        history = observer.fit(probability, 20)
        initial = dict(state=observer.state(), history=history, data_sha256=case['data_sha256'])
        atomic_json(initial_path, initial)
    primary, primary_state = solve_primary(features, probability, initial['state'], folder, 'cuda:0')
    _, target, ds_report = ordinary_targets(probability)
    fused = CorrelatedTeacher.from_state(initial['state']).posterior(probability)
    target['GAUSSIAN_FUSED_SOFT_KD'] = (np.ones(len(features), bool), fused, np.ones(len(features)))
    fitted, fits = {PRIMARY: primary}, {}
    normal = folder/'ordinary'
    normal.mkdir(parents=True, exist_ok=True)
    for name in ORDINARY:
        model = make_solver('cuda:0')
        path = normal/(name+'.pt')
        mask, y, weight = target[name]
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['data_sha256'] == case['data_sha256'] and state['fit']['adequate_passed']
            model.load_state_dict(state['model'])
            fit = state['fit']
        else:
            try:
                fit = fit_solver(model, features[mask], y, 'cuda:0', steps=6000, weight=weight, adequate=True)
            except Exception:
                torch.save(dict(model=model.state_dict(), name=name, data_sha256=case['data_sha256']), normal/(name+'_unfinished.pt'))
                raise
            torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            name=name, fit=fit, data_sha256=case['data_sha256'], seed=0,
                            target_sha256=__import__('hashlib').sha256(y.tobytes()+weight.tobytes()).hexdigest()), path)
        fitted[name], fits[name] = model, fit
        print(json.dumps(dict(status='ADEQUATE_JOINT_DATA_ORDINARY', case=pair_name, method=name,
                              steps=fit['steps'], proper_excess_KL=fit['excess_KL'])), flush=True)
    video = JointObservationVideo(pair_name, case['evaluated_video'])
    inputs = Inputs(video)
    keys = json.loads((folder/'row_keys.json').read_text())['keys']
    query_index = {r['expression_id']: i for i, r in enumerate(video.queries)}
    track_index = {int(t): i for i, t in enumerate(video.tracks)}
    points = [i for i, row in enumerate(keys) if row[0] == video.video]
    records = [dict(local_query=query_index[keys[i][1]], local_track=track_index[int(keys[i][2])],
                    frame_id=int(keys[i][3]), track_id=int(keys[i][2])) for i in points]
    assert np.array_equal(pack(inputs.arrays(records)), features[points]), 'Predict with exactly the same feature normalization'
    scores = {name: np.empty((len(video.queries), len(video.obs)), dtype='float32') for name in fitted}
    with torch.no_grad():
        for qi in range(len(video.queries)):
            for begin in range(0, len(video.obs), 512):
                rows = np.arange(begin, min(begin+512, len(video.obs)))
                records = [dict(local_query=qi, local_track=int(video.columns[row]), frame_id=int(video.obs[row, 0]),
                                track_id=int(video.obs[row, 1])) for row in rows]
                x = torch.as_tensor(pack(inputs.arrays(records)), device='cuda:0')
                for name, model in fitted.items():
                    scores[name][qi, rows] = model(x).cpu().numpy()
    for original, pooled in (('MEAN_SOFT_KD', 'MEAN_SOFT_POOLED'), ('MAJORITY_HARD_KD', 'MAJORITY_HARD_POOLED')):
        z = scores[original].astype('float64')
        result = np.empty_like(z)
        for ti in range(len(video.tracks)):
            rows = np.flatnonzero(video.columns == ti)
            value = logsumexp(-np.logaddexp(0, -z[:, rows]), axis=1)-logsumexp(-np.logaddexp(0, z[:, rows]), axis=1)
            result[:, rows] = value[:, None]
        scores[pooled] = result.astype('float32')
    bits = {name: values > 0 for name, values in scores.items()}
    metadata = subset_metadata(case['target'], case['evaluated_video'])
    tasks = []
    for name in METHODS:
        root = folder/'predictions'/name
        root.mkdir(parents=True, exist_ok=True)
        assert np.isfinite(scores[name]).all()
        np.save(root/'scores.npy', scores[name])
        np.save(root/'mask.npy', bits[name])
        assert np.array_equal(projection(scores[name], bits[name], video.obs), bits[name])
        seal_mask(root, metadata, bits[name].flatten(), target=case['target'], model=PAIRS[pair_name][0],
                  method=name, GT_open_attempts=len(guard), extra=dict(seed=0, auxiliary_target_events=100,
                  matched_data_sha256=case['data_sha256'], no_new_teacher_calls=True, full_raw_logits_retained=True))
        tasks.append(dict(pair=pair_name, method=name, folder=str(root)))
    for rank, count in [(PRIMARY, name) for name in COMPARATORS]+[(name, PRIMARY) for name in COMPARATORS]:
        name = rank+'__AT_'+count+'_COUNT'
        root = folder/'predictions'/name
        selected = projection(scores[rank], bits[count], video.obs)
        seal_mask(root, metadata, selected.flatten(), target=case['target'], model=PAIRS[pair_name][0],
                  method=name, GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=count,
                  counts_equal_per_query_frame=True, counts_from_GT=False))
        tasks.append(dict(pair=pair_name, method=name, folder=str(root)))
    assert len(tasks) == 25 and not guard
    atomic_json(folder/'main_completion.json', dict(status='SEALED_JOINT_NOISE_ALL25_CASE_OUTPUTS_NO_GT',
                tasks=tasks, GT_open_attempts=0, ordinary_fits=fits, DS_fusion=ds_report,
                joint_rounds=primary_state['rounds'], execution=execution,
                wall_seconds=time.perf_counter()-start, no_new_teacher_calls=True, not_full_cold_deployment=True))


def evaluate_one(task):
    from flex_uda.a100.evaluation_reuse import evaluate
    return task, evaluate(task['folder'], task['pair'])


def main():
    plan = registration()
    supervisor = OUT/'supervisor'
    supervisor.mkdir(parents=True, exist_ok=True)
    pending = [(gpu, case) for gpu, case in zip((0, 1, 3), plan['cases']) if not (OUT/case['pair']/'main_completion.json').exists()]
    for gpu, case in pending:
        pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        assert not pids and used < 512, (gpu, pids, used)
    jobs = []
    for gpu, case in pending:
        log = supervisor/(case['pair']+'.log')
        stream = log.open('a', buffering=1)
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', case['pair']], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        jobs.append((proc, stream, log, case))
        atomic_json(supervisor/(case['pair']+'_launch.json'), dict(pid=proc.pid, GPU=gpu, pair=case['pair']))
        print(json.dumps(dict(status='STARTED_JOINT_CORRELATED_TEACHER_MAIN', GPU=gpu, pair=case['pair'], pid=proc.pid)), flush=True)
    failed = []
    for proc, stream, log, case in jobs:
        code = proc.wait()
        stream.close()
        if code:
            failed.append(dict(pair=case['pair'], exit_code=code, tail=log.read_text()[-5000:]))
    if failed:
        raise RuntimeError(failed)
    from flex_uda.presence.predictions import verify_predictions
    tasks = []
    for case in plan['cases']:
        done = json.loads((OUT/case['pair']/'main_completion.json').read_text())
        assert done['GT_open_attempts'] == 0
        tasks.extend(done['tasks'])
    assert len(tasks) == 75
    for task in tasks:
        verify_predictions(task['folder'])
    atomic_json(OUT/'all75_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0, registration_sha256=file_sha(PLAN)))
    print('ALL75_JOINT_NOISE_OUTPUTS_SEALED_ISOLATED_OFFICIAL_EVALUATION', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(evaluate_one, tasks))
    results = {c['pair']: {} for c in plan['cases']}
    for task, value in completed:
        results[task['pair']][task['method']] = value
    comparisons = []
    for pair, values in results.items():
        primary = values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            reference = values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair, comparator=other, primary_HOTA=primary, ordinary_HOTA=reference,
                raw_delta_HOTA=primary-reference,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-reference,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible = all(min(row['raw_delta_HOTA'], row['primary_at_ordinary_count_delta_HOTA'],
                       row['ordinary_at_primary_count_delta_HOTA']) > 0 for row in comparisons)
    atomic_json(OUT/'quality_results.json', dict(status='COMPLETE_JOINT_CORRELATED_TEACHER_PRIMARY', results=results,
        comparisons=comparisons, candidate_eligible_for_larger_screen=eligible,
        all_data_DEV=True, no_six_setting_or_three_innovation_or_fresh_cost_claim=True, goal_status='ACTIVE'))
    for flag in ('--collect', '--reproduce'):
        subprocess.run([sys.executable, '-u', 'scripts/a100_joint_teacher_noise_statistics.py', flag], cwd=ROOT, check=True)
    print(json.dumps(dict(status='COMPLETE_JOINT_CORRELATED_TEACHER_MAIN_REPRODUCED', comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker)
    else:
        main()
