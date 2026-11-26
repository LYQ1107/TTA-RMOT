"""Small online factor-evidence RSI main-versus-ordinary pilot, seed0.

Three first videos cover appearance, spatial/motion and sports queries.
All old native partial outputs stay untouched and unscored. No new module
ablation matrix is scheduled. Only complete new-cohort outputs permit GT.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from scripts.a100_six_direction import OUT as SIX
from scripts.a100_observation_design_prepare import OUT as PREP

OUT = SIX/'factor_rsi_primary_pilot_20261125'
PLAN = OUT/'registration.json'
PRIMARY = 'FACTOR_RSI_NEURAL'
COMPARATORS = ('ORDINARY_SOFT_MULTITASK_KD', 'ORDINARY_HARD_MULTITASK_KD',
               'ORDINARY_SOFT_WHOLE_KD', 'ORDINARY_HARD_WHOLE_KD')
SOURCES = ('scripts/a100_factor_rsi_primary_pilot.py', 'flex_uda/active/factor_evidence.py',
    'flex_uda/active/factor_rsi_learning.py', 'flex_uda/active/rsi_curriculum.py',
    'flex_uda/active/rsi_policy_distribution.py', 'flex_uda/active/native_binary_inputs.py',
    'flex_uda/active/joint_observation_inputs.py', 'flex_uda/active/observation_learning_inputs.py',
    'flex_uda/active/teacher_client.py', 'flex_uda/teacher/yestrack_backend.py',
    'flex_uda/teacher/decompose_inputs.py', 'flex_uda/teacher/teacher_cache.py',
    'flex_uda/a100/runtime.py', 'flex_uda/active/deterministic_observation_runtime.py',
    'scripts/a100_factor_rsi_contract.py', 'scripts/a100_factor_evidence_contract.py',
    'configs/teacher/base_qwen2b.yaml', 'configs/teacher/prompts_v1.yaml')


def prepare():
    guard = generation_guard(); cases = []
    for pair in ('C1', 'C2', 'C4'):
        md = metadata(PAIRS[pair][1]); video = sorted(md['videos'])[0]
        path = PREP/pair/video/'UNIFORM_requests.json'
        requests = json.loads(path.read_text())['requests']
        quotas = dict(Counter(int(r['local_query']) for r in requests))
        assert min(quotas.values()) >= 1 and sum(n >= 3 for n in quotas.values()) >= 3
        cases.append(dict(pair=pair, video=video, target=PAIRS[pair][1], budget=len(requests),
            query_count=len(quotas), quota=quotas, old_input_manifest=str(path), old_input_sha256=file_sha(path)))
    plan = dict(status='PREREGISTERED_FACTOR_RSI_MAIN_VS_ADEQUATE_ORDINARY', cases=cases,
        primary=PRIMARY, comparators=COMPARATORS, neural_seeds=[0],
        source_sha256={p:file_sha(ROOT/p) for p in SOURCES},
        hypotheses=['Separately paid appearance/state evidence identifies teacher-opinion factors that a whole-expression No cannot identify.',
                    'Current-appearance retrieval avoids unconditional trust in an impure Track ID; a separate state representation preserves changing relations.',
                    'A learned observer chooses real factor questions using solver state; new withheld external teacher pairs measure solver progress and update the observer.'],
        input_scope='Official fixed candidates; referring adaptation only; historical PCD/RGB prepaid and shared.',
        observer_features=8, observer_rounds=3, observer_update='Original natural-gradient recorded-state KL<=.01; no hard fresh-track priority',
        probe_budget='Three pairs of WHOLE questions, same query/frame/view and different tracks, paid inside original per-query quotas; current probe labels withheld until after this solver update.',
        reward='Teacher JS-weighted progress in conditional relative discrimination; common solver-logit shifts cannot improve it. Proxy, not correctness or HOTA.',
        learner_round_steps=500, learner_final_steps=4000, learner_width=64,
        ordinary_initial_width=128, ordinary_max_width=256, ordinary_min_final_steps=4000,
        ordinary_max_final_steps=16000, ordinary_per_role_proper_excess_KL_limit=.005,
        all_competitors_get_identical_visual_inputs_and_all_paid_answers=True,
        ordinary_WHOLE_variants_voluntarily_use_whole_labels_only_but_MT_variants_use_every_role=True,
        all_raw_and_both_query_frame_CountMatched_outputs_seal_before_GT=True,
        no_GT_for_acquisition_rewards_fitting_hyperparameters_or_checkpoint_selection=True,
        scope='Three first-video DEV cases; primary screening only, no six-setting or independent-generalization claim.',
        no_module_ablation_matrix=True, old_native_incomplete_and_not_scored=True,
        full_cold_cost_not_measured_by_this_prepaid_feature_pilot=True, goal_status='ACTIVE')
    assert not guard; atomic_json(PLAN, plan)
    print(json.dumps(dict(status=plan['status'], cases=cases)), flush=True)


def verify_plan():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    return plan


def run_case(case, client):
    import numpy as np
    import torch
    from flex_uda.active.factor_evidence import FactorStudent
    from flex_uda.active.factor_rsi_learning import (Observer, Inputs, make_pool, probes_and_seeds,
        choose, uid, fit, tensor_data, discrimination_credit, trust_update)
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from scripts.a100_observation_design_collect import TEACHER_SHA
    pair = case['pair']; folder = OUT/pair; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'primary_fit_complete.json').exists(): return
    torch.manual_seed(0); generator = np.random.default_rng(0)
    guard = generation_guard(); use_verified_local_features()
    v = JointObservationVideo(pair, case['video']); inputs = Inputs(v)
    original = json.loads(Path(case['old_input_manifest']).read_text())['requests']
    assert file_sha(Path(case['old_input_manifest'])) == case['old_input_sha256']
    quotas = {int(k):n for k, n in case['quota'].items()}; pool = make_pool(v, quotas)
    probes, seeds, remaining, reserved = probes_and_seeds(v, original, quotas, pool)
    atomic_json(folder/'questions_and_probe_scope_before_answers.json', dict(
        pool_UIDs=sorted(pool), seeds=seeds, probes=probes, quota=quotas,
        first_video_only=True, input_manifest_sha256=case['old_input_sha256'], GT_open_attempts=len(guard)))
    model = FactorStudent().cuda(); actor = Observer(); paid = []; prefixes = []; rounds = []
    checkpoint = folder/'paid_answers_checkpoint.json'
    recovered = json.loads(checkpoint.read_text())['paid'] if checkpoint.exists() else []
    paid_map = {uid(r):r for r in recovered}
    def ask(requests, role, stage):
        missing = [r for r in requests if uid(r) not in paid_map]
        for begin in range(0, len(missing), 4):
            batch = missing[begin:begin+4]; responses = client.ask_many(batch)
            for r, answer in zip(batch, responses):
                assert answer['frozen_weight_sha256'] == TEACHER_SHA
                assert all(answer[k] == r[k] for k in ('target','video_id','query_id','track_id','frame_id','factor_mode'))
                assert uid(r) not in paid_map
                paid_map[uid(r)] = dict(answer, local_query=int(r['local_query']), local_track=int(r['local_track']),
                    observer_role=role, observer_stage=stage)
            atomic_json(checkpoint, dict(paid=list(paid_map.values()), stats=client.stats, GT_open_attempts=len(guard)))
            print(json.dumps(dict(status='NEW_FACTOR_RSI_REAL_QWEN', pair=pair, stage=stage,
                paid=len(paid_map), total=case['budget'])), flush=True)
        return [paid_map[uid(r)] for r in requests]
    saved = folder/'closed_round.pt'; completed = -1
    if saved.exists():
        previous = torch.load(saved, map_location='cpu', weights_only=False)
        model.load_state_dict(previous['model']); actor.load_state_dict(previous['actor'])
        paid, prefixes, rounds, completed = [previous[k] for k in ('paid','prefixes','rounds','completed')]
        generator.bit_generator.state = previous['generator_state']
    else:
        paid.extend(ask(seeds, 'seed', -1)); prefixes.append(len(paid))
        report = fit(model, inputs, paid, steps=500, device='cuda:0')
        atomic_json(folder/'seed_fit.json', report)
    for stage in range(completed+1, 3):
        stage_quota = {q:n//3+int(stage < n%3) for q, n in remaining.items()}
        selection_path = folder/('selection_'+str(stage)+'.pt')
        if selection_path.exists():
            selection = torch.load(selection_path, map_location='cpu', weights_only=False)
            selected, trace = selection['selected'], selection['trace']
            actor._distribution_trace = selection['distribution_trace']
            generator.bit_generator.state = selection['generator_state']
        else:
            selected, trace = choose(actor, model, inputs, pool, stage_quota, paid, reserved, generator)
            torch.save(dict(selected=selected, trace=trace, distribution_trace=actor._distribution_trace,
                generator_state=generator.bit_generator.state), selection_path)
        with torch.no_grad(): before = model(**tensor_data(inputs.arrays(probes[stage]), 'cuda:0'))[:, 0].cpu().numpy()
        # New probe responses are requested only AFTER this solver fit.
        new = ask(selected, 'train', stage); paid.extend(new); prefixes.append(len(paid))
        report = fit(model, inputs, paid, steps=500, device='cuda:0')
        with torch.no_grad(): after = model(**tensor_data(inputs.arrays(probes[stage]), 'cuda:0'))[:, 0].cpu().numpy()
        answers = ask(probes[stage], 'probe', stage)
        assert not any(uid(a) in {uid(r) for r in paid} for a in answers)
        credit = discrimination_credit(before, after, answers)
        update = trust_update(actor, None, None, credit['before'], credit['after'], 0.,
            reward_override=credit['reward'], frozen=credit['teacher_JS'] <= 0)
        paid.extend(answers)
        row = dict(stage=stage, fit=report, credit=credit, observer=update,
            selected_requests=len(selected), one_step_conditional_choice_changes=sum(
                r['uid'] != r['one_step_frozen_choice_uid'] for r in trace), selection_trace=trace,
            current_probe_labels_excluded_from_this_fit=True)
        rounds.append(row); atomic_json(folder/('round_'+str(stage)+'.json'), row)
        temp = saved.with_suffix('.tmp'); torch.save(dict(model=model.cpu().state_dict(), actor=actor.state_dict(),
            paid=paid, prefixes=prefixes, rounds=rounds, completed=stage,
            generator_state=generator.bit_generator.state), temp); temp.replace(saved); model.cuda()
        print(json.dumps(dict(status='NEW_FACTOR_RSI_CLOSED_ROUND', pair=pair, stage=stage,
            reward=credit['reward'], actor_change=update['actor_parameter_L2_change'], paid=len(paid))), flush=True)
    assert len(paid) == len(paid_map) == case['budget']
    assert Counter(int(r['local_query']) for r in paid) == quotas and not guard
    prefixes.append(len(paid)); final = fit(model, inputs, paid, steps=4000, device='cuda:0')
    atomic_json(folder/'paid_answers.json', dict(paid=paid, budget=case['budget'], fit_prefix_lengths=prefixes,
        modes=dict(Counter(r['factor_mode'] for r in paid)), manual_GT_open_attempts=0,
        all_answers_available_to_ordinary_comparators=True))
    torch.save(dict(model=model.cpu().state_dict(), mean=inputs.mean, std=inputs.std, width=64,
        source_case=case, final_fit=final, rounds=rounds, fit_prefix_lengths=prefixes), folder/'primary.pt')
    atomic_json(folder/'primary_fit_complete.json', dict(status='PASS_NEW_FACTOR_RSI_PRIMARY_FIT_NO_GT',
        final=final, rounds=len(rounds), paid=len(paid), actual_bidirectional_feedback=True,
        positive_RMOT_or_RSI_contribution_unproven=True, target_GT_read=False))
    del model, v, inputs; torch.cuda.empty_cache()


def worker(pairs):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.a100.runtime import teacher
    from flex_uda.active.factor_evidence import FactorTeacher
    from scripts.active_run import video_spec
    from scripts.a100_observation_design_collect import TEACHER_SHA
    plan = verify_plan(); guard = generation_guard(); torch.set_num_threads(2)
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.45)
    folder = OUT/('teacher_gpu'+os.environ['CUDA_VISIBLE_DEVICES']); folder.mkdir(parents=True, exist_ok=True)
    cases = [next(c for c in plan['cases'] if c['pair'] == p) for p in pairs]
    specs = [video_spec(c['target'], c['video'], metadata(c['target'])['videos'][c['video']]) for c in cases]
    begin = time.perf_counter(); reservation = torch.empty(128<<20, dtype=torch.float32, device='cuda:0')
    atomic_json(folder/'progress.json', dict(status='LOADING_NEW_FACTOR_RSI_FROZEN_TEACHER', pid=os.getpid(), pairs=pairs))
    client = teacher(specs, folder/'raw.sqlite'); client.__class__ = FactorTeacher
    assert client.backend.weight_sha == TEACHER_SHA
    del reservation; torch.cuda.empty_cache()
    for case in cases: run_case(case, client)
    receipt = client.close(); assert not guard
    atomic_json(folder/'completion.json', dict(status='PASS_NEW_FACTOR_RSI_EVIDENCE_AND_PRIMARY_FITS',
        pairs=pairs, teacher=receipt, execution=execution, wall_seconds=time.perf_counter()-begin,
        actual_GPU=os.environ['CUDA_VISIBLE_DEVICES'], source_features_prepaid=True,
        not_single_method_cold_cost=True, target_GT_read=False))
    print(json.dumps(dict(status='PASS_NEW_FACTOR_RSI_EVIDENCE_AND_PRIMARY_FITS', pairs=pairs,
        new_calls=receipt['new_calls'], cache_hits=receipt['cache_hits'])), flush=True)


def main():
    verify_plan(); jobs = []; folder = OUT/'supervisor'; folder.mkdir(parents=True, exist_ok=True)
    needed = [(gpu,pairs) for gpu,pairs in ((0, ('C1','C2')), (3, ('C4',)))
              if not (OUT/('teacher_gpu'+str(gpu))/'completion.json').exists()]
    # Preflight both assignments before starting either, preserving unrelated owners.
    for gpu, pairs in needed:
        owners = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip()); assert not owners and used < 512, (gpu, used, owners)
    for gpu, pairs in needed:
        log = folder/('gpu'+str(gpu)+'.log'); handle = log.open('a', buffering=1)
        proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',*pairs], cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        jobs.append((gpu,proc,handle,log))
        atomic_json(folder/('gpu'+str(gpu)+'_launch.json'), dict(pid=proc.pid, GPU=gpu, pairs=pairs, source_sha256=file_sha(Path(__file__))))
        print(json.dumps(dict(status='STARTED_NEW_FACTOR_RSI_PRIMARY_WORKER',GPU=gpu,pid=proc.pid,pairs=pairs)),flush=True)
    failures = []
    for gpu,proc,handle,log in jobs:
        code = proc.wait(); handle.close()
        row = dict(GPU=gpu,pid=proc.pid,exit_code=code,log=str(log))
        atomic_json(folder/('gpu'+str(gpu)+'_terminal.json'),row)
        if code: failures.append(dict(row,tail=log.read_text()[-4000:]))
    if failures: raise RuntimeError(failures)
    atomic_json(OUT/'primary_collection_complete.json',dict(status='PASS_ALL_NEW_FACTOR_RSI_PRIMARY_CASES',
        target_GT_read=False,adequate_ordinary_and_quality_still_required=True))
    print('PASS_ALL_NEW_FACTOR_RSI_PRIMARY_CASES',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',nargs='+');a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
    else:main()
