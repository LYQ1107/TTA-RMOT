"""New complete primary: anchored visual challenges with robust peer teaching.

Previous cohort is terminal and rejected. Six sufficiently trained ordinary
references are reused exactly. New raw/dual-count outputs seal before GT.
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
from scripts import a100_reciprocal_memory_pilot as previous

OUT = previous.OUT.parent/'anchored_reciprocal_primary_20261125'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PLAN = OUT/'registration.json'
PRIMARY = 'ANCHOR_CHALLENGED_RECIPROCAL_RSI'
COMPARATORS = previous.COMPARATORS
METHODS = (PRIMARY, *COMPARATORS)
SOURCES = ('scripts/a100_anchored_reciprocal_pilot.py',
           'flex_uda/active/anchored_reciprocal.py',
           'scripts/a100_anchored_reciprocal_contract.py',
           'scripts/a100_anchored_reciprocal_statistics.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    parent = previous.registration()
    old_result = json.loads((previous.OUT/'quality_results.json').read_text())
    assert not old_result['candidate_eligible_for_larger_screen']
    audit = json.loads((previous.OUT/'independent_sealed_prediction_audit.json').read_text())
    assert audit['records'] == 57 and audit['final_student_models_independently_recomputed'] == 18
    cases = []
    for case in parent['cases']:
        case = dict(case)
        case['paid_answers'] = str(previous.OUT/case['pair']/'paid_answers.json')
        assert file_sha(case['paid_answers']) == case['paid_answers_sha256']
        case['ordinary_score_sha256'] = {
            n: file_sha(previous.OUT/case['pair']/'predictions'/n/'scores.npy') for n in COMPARATORS}
        cases.append(case)
    plan = dict(status='PREREGISTERED_NEW_ANCHORED_ROBUST_RECIPROCAL_MAIN',
                primary=PRIMARY, comparators=COMPARATORS, cases=cases,
                source_sha256={n: file_sha(ROOT/n) for n in SOURCES},
                parent_registration_sha256=file_sha(previous.PLAN),
                parent_all57_sealed_sha256=file_sha(previous.OUT/'all57_sealed_before_GT.json'),
                parent_adequacy_audit_sha256=file_sha(previous.OUT/'independent_sealed_prediction_audit.json'),
                hypothesis='Direct unlabeled visual challenges at paid teacher anchors; robust reciprocal weights permit noisy-teacher disagreement while preserving every query-role supervision mass.',
                candidate_contributions=['Visual anchor coverage in the learning curriculum',
                                         'Bounded positive peer reweighting with unchanged query-role supervision mass',
                                         'Reciprocal frame-memory students, evolving challenges and heldout-teacher expert feedback'],
                human_vision='Reinspect the actual observed object when evidence conflicts; preserve identity without suppressing visible state innovations.',
                classical_ideas='Edge covers on paths and bounded-influence M-estimation; conditional query-role gradient-mass bounds only, never a HOTA guarantee.',
                reciprocal_model_architectures_inputs_and_initial_seed_identical_to_parent=True,
                ordinary_full_teacher_fit_gates_unchanged_and_independently_passed=True,
                new_primary_teacher_KL_interpolation_gate=False,
                new_primary_objective_change='Intentional robust learning from noisy teacher guidance; report actual teacher fidelity, do not call mismatches task corrections without GT evaluation.',
                primary_steps=[500, 500, 500, 4000], teacher_folds_and_probes_identical_to_parent=True,
                final_checkpoint_fixed_by_schedule_not_GT_or_teacher_validation_selection=True,
                challenge='Minimum topological edge cover of nonisolated paid image groups plus1024 current reciprocal-disagreement/exploration edges.',
                orphan_image_groups='Reported explicitly; no temporal edge fabricated for an isolated observation.',
                peer_weight='0.2+0.8/(1+peer_teacher_KL/log2), stop gradient, rescale within query-role to preserve original row mass.',
                teacher_labels_not_guaranteed_clean=True, model_agreement_not_truth=True,
                neural_seeds=[0], new_teacher_GPU_calls=0, fixed_paid_QA_replay=True,
                raw_records=21, dual_CountMatched_records=36, official_statistical_records=57,
                exact_parent_raw_references=18, CountSelf_exact_aliases=21,
                same_candidates_TrackIDs_visual_information_queries_and_paid_answers=True,
                all57_prediction_seals_before_this_study_GT=True,
                success_screen='Unchanged strict screen: raw and BOTH CountMatched effects positive versus ALL six ordinary references in EVERY case.',
                no_primary_ablation_matrix=True, no_fresh_online_acquisition_or_full_cold_cost_claim=True,
                all_data_DEV=True, no_target_manual_GT_in_learning=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/'anchored_reciprocal_primary_registration_20261125.json', plan)
    print(json.dumps(dict(status=plan['status'], cases=[c['pair'] for c in cases], source_pins=len(SOURCES))), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    previous.registration()
    assert file_sha(previous.PLAN) == plan['parent_registration_sha256']
    assert file_sha(previous.OUT/'all57_sealed_before_GT.json') == plan['parent_all57_sealed_sha256']
    assert file_sha(previous.OUT/'independent_sealed_prediction_audit.json') == plan['parent_adequacy_audit_sha256']
    for name, expected in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    return plan


def solve(inputs, paid, heldout, folder, device):
    import numpy as np
    import torch
    from flex_uda.active.reciprocal_memory import models, probabilities, update_experts
    from flex_uda.active.anchored_reciprocal import challenge, train

    root = folder/'fits/PRIMARY'
    root.mkdir(parents=True, exist_ok=True)
    pair = models(device)
    weights = np.full((2, 3), .5)
    generator = np.random.default_rng(0)
    rounds = []
    completed = -1
    final = root/'final.pt'
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        assert state['paid_answers_sha256'] == file_sha(folder/'paid_answers.json')
        for model, values in zip(pair, state['models']):
            model.load_state_dict(values)
            model.eval()
        return pair, np.asarray(state['weights']), state
    saved = root/'closed.pt'
    if saved.exists():
        state = torch.load(saved, map_location='cpu', weights_only=False)
        for model, values in zip(pair, state['models']):
            model.load_state_dict(values)
        weights = np.asarray(state['weights'])
        generator.bit_generator.state = state['generator_state']
        rounds, completed = state['rounds'], state['completed']

    def states():
        return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in pair]

    def selected(stage, training):
        path = root/('selection'+str(stage)+'.pt')
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=False)
            selection = old['selection']
            generator.bit_generator.state = old['generator_state']
            return inputs.edge_data(np.asarray(selection['selected_edge_ids'])), selection
        edges, selection = challenge(pair, inputs, training, weights, generator, device)
        torch.save(dict(selection=selection, generator_state=generator.bit_generator.state), path)
        return edges, selection

    for stage in range(completed+1, 3):
        forbidden = {i for fold in heldout[stage:] for i in fold}
        ids = [i for i in range(len(paid)) if i not in forbidden]
        training = [paid[i] for i in ids]
        probe = [paid[i] for i in heldout[stage]]
        edges, selection = selected(stage, training)
        fit = train(pair, inputs, training, device=device, steps=500, edges=edges, weights=weights)
        predictions = probabilities(pair, inputs.arrays(probe), device)
        weights, critic = update_experts(weights, predictions, probe)
        row = dict(stage=stage, fit=fit, critic=critic, curriculum=selection,
                   training_indices=ids, withheld_indices=heldout[stage],
                   current_and_future_probe_groups_excluded=True)
        rounds.append(row)
        atomic_json(root/('round'+str(stage)+'.json'), row)
        checkpoint = dict(models=states(), weights=weights, rounds=rounds, completed=stage,
                          generator_state=generator.bit_generator.state)
        temp = saved.with_suffix('.tmp')
        torch.save(checkpoint, temp)
        temp.replace(saved)
        print(json.dumps(dict(status='CLOSED_ANCHORED_RECIPROCAL_ROUND', case=folder.name,
                              stage=stage, support=selection['covered_paid_image_groups'],
                              groups=selection['unique_paid_image_groups'], teacher_KL=fit['maximum_excess_KL'])), flush=True)
    edges, selection = selected('final', paid)
    fit = train(pair, inputs, paid, device=device, steps=4000, edges=edges, weights=weights)
    state = dict(models=states(), weights=weights, rounds=rounds, final_fit=fit,
                 final_curriculum=selection, paid_answers_sha256=file_sha(folder/'paid_answers.json'),
                 mean=inputs.mean, std=inputs.std, seed=0, no_GT_selection=True)
    temp = final.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(final)
    return pair, weights, state


def worker(pair_name):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import generation_guard, PAIRS
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.reciprocal_memory import MemoryInputs, tensor_data
    from scripts.a100_factor_rsi_primary_quality import subset_metadata, projection
    from flex_uda.presence.predictions import seal_mask

    plan = registration()
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features()
    case = next(c for c in plan['cases'] if c['pair'] == pair_name)
    folder = OUT/pair_name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completion.json').exists():
        return
    path = Path(case['paid_answers'])
    assert file_sha(path) == case['paid_answers_sha256']
    copied = folder/'paid_answers.json'
    if copied.exists():
        assert file_sha(copied) == file_sha(path)
    else:
        copied.write_bytes(path.read_bytes())
    paid = json.loads(copied.read_text())['paid']
    start = time.perf_counter()
    video = JointObservationVideo(pair_name, case['video'])
    inputs = MemoryInputs(video)
    models, weights, state = solve(inputs, paid, case['heldout_rows'], folder, 'cuda:0')
    scores = np.empty((len(video.queries), len(video.obs)), dtype='float32')
    with torch.no_grad():
        for qi in range(len(video.queries)):
            for begin in range(0, len(video.obs), 512):
                ids = np.arange(begin, min(begin+512, len(video.obs)))
                data = tensor_data(inputs.at(np.full(len(ids), qi), ids), 'cuda:0')
                p = sum(float(weights[i, 0])*model(**data)[:, 0].sigmoid() for i, model in enumerate(models))
                p = p.cpu().numpy().clip(1e-7, 1-1e-7)
                scores[qi, ids] = np.log(p)-np.log1p(-p)
    values = {PRIMARY: scores}
    for name in COMPARATORS:
        path = previous.OUT/pair_name/'predictions'/name/'scores.npy'
        assert file_sha(path) == case['ordinary_score_sha256'][name]
        values[name] = np.load(path)
    masks = {n: s > 0 for n, s in values.items()}
    metadata = subset_metadata(case['target'], case['video'])
    tasks = []
    for name in METHODS:
        root = folder/'predictions'/name
        root.mkdir(parents=True, exist_ok=True)
        assert np.isfinite(values[name]).all()
        np.save(root/'scores.npy', values[name])
        np.save(root/'mask.npy', masks[name])
        assert np.array_equal(projection(values[name], masks[name], video.obs), masks[name])
        seal_mask(root, metadata, masks[name].flatten(), target=case['target'], model=PAIRS[pair_name][0],
                  method=name, GT_open_attempts=len(guard), extra=dict(new_teacher_calls=0,
                  exact_parent_reference=name in COMPARATORS, source_features_prepaid=True))
        tasks.append(dict(pair=pair_name, method=name, folder=str(root)))
    for rank, count in [(PRIMARY, n) for n in COMPARATORS]+[(n, PRIMARY) for n in COMPARATORS]:
        name = rank+'__AT_'+count+'_COUNT'
        root = folder/'predictions'/name
        bits = projection(values[rank], masks[count], video.obs)
        seal_mask(root, metadata, bits.flatten(), target=case['target'], model=PAIRS[pair_name][0],
                  method=name, GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=count,
                  counts_equal_per_query_frame=True, counts_from_GT=False))
        tasks.append(dict(pair=pair_name, method=name, folder=str(root)))
    assert len(tasks) == 19 and not guard
    atomic_json(folder/'completion.json', dict(status='SEALED_ALL19_ANCHORED_MAIN_OUTPUTS_NO_GT',
                tasks=tasks, GT_open_attempts=0, final_fit=state['final_fit'],
                final_curriculum=state['final_curriculum'], execution=execution,
                wall_seconds=time.perf_counter()-start, parent_ordinary_refs=6,
                new_fits=1, new_teacher_calls=0, CountSelf_exact=True, not_full_cold_deployment=True))


def main():
    plan = registration()
    jobs = []
    supervisor = OUT/'supervisor'
    supervisor.mkdir(parents=True, exist_ok=True)
    for gpu, case in zip((0, 1, 3), plan['cases']):
        if (OUT/case['pair']/'completion.json').exists():
            continue
        pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        assert not pids and used < 512, (gpu, pids, used)
        log = supervisor/(case['pair']+'.log')
        stream = log.open('a', buffering=1)
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', case['pair']], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
        jobs.append((proc, stream, log, case))
        atomic_json(supervisor/(case['pair']+'_launch.json'), dict(pid=proc.pid, GPU=gpu, pair=case['pair']))
        print(json.dumps(dict(status='STARTED_NEW_ANCHORED_ROBUST_MAIN', GPU=gpu, pair=case['pair'], pid=proc.pid)), flush=True)
    failures = []
    for proc, stream, log, case in jobs:
        code = proc.wait()
        stream.close()
        if code:
            failures.append(dict(pair=case['pair'], exit_code=code, tail=log.read_text()[-5000:]))
    if failures:
        raise RuntimeError(failures)
    from flex_uda.presence.predictions import verify_predictions
    tasks = []
    for case in plan['cases']:
        done = json.loads((OUT/case['pair']/'completion.json').read_text())
        assert done['GT_open_attempts'] == 0
        tasks.extend(done['tasks'])
    assert len(tasks) == 57
    for task in tasks:
        verify_predictions(task['folder'])
    atomic_json(OUT/'all57_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0,
                                                      registration_sha256=file_sha(PLAN)))
    print('ALL57_ANCHORED_PRIMARY_OUTPUTS_SEALED_ISOLATED_OFFICIAL_EVALUATION', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(previous.evaluate_one, tasks))
    results = {c['pair']: {} for c in plan['cases']}
    for task, value in completed:
        results[task['pair']][task['method']] = value
    comparisons = []
    for pair, values in results.items():
        p = values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            o = values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair, comparator=other, primary_HOTA=p, ordinary_HOTA=o,
                raw_delta_HOTA=p-o,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-o,
                ordinary_at_primary_count_delta_HOTA=p-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible = all(min(row['raw_delta_HOTA'], row['primary_at_ordinary_count_delta_HOTA'],
                       row['ordinary_at_primary_count_delta_HOTA']) > 0 for row in comparisons)
    atomic_json(OUT/'quality_results.json', dict(status='COMPLETE_ANCHORED_RECIPROCAL_MAIN_SCREEN',
                results=results, comparisons=comparisons, candidate_eligible_for_larger_screen=eligible,
                all_data_DEV=True, no_three_contribution_or_six_setting_claim=True, goal_status='ACTIVE'))
    for flag in ('--collect', '--reproduce'):
        subprocess.run([sys.executable, '-u', 'scripts/a100_anchored_reciprocal_statistics.py', flag], cwd=ROOT, check=True)
    print(json.dumps(dict(status='COMPLETE_ANCHORED_RECIPROCAL_MAIN_REPRODUCED', comparisons=comparisons)), flush=True)


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
