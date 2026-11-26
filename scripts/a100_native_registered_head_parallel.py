"""Prepare one existing registered native model on a spare GPU.

Uses the original replay_solver, exact original own-policy prefixes and warm
checkpoint. No teacher/actor/prediction/evaluation or new experimental arm.
The original worker can load the prepared head through its existing cache.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for thread_option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[thread_option] = '2'
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as native

MAIN_NAMES = ('FULL_CONDITIONAL_REFERENCE_LEARNED', 'FULL_CONDITIONAL_REFERENCE_FROZEN')


def spare_gpu_preflight(gpu):
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    owners = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert used < 512 and not owners, ('GPU no longer spare; preserve its owner', gpu, used, owners)
    return dict(physical_GPU=gpu, used_MiB=used, no_resident_compute_owner=True)


def main(pair, gpu, arm):
    ARM = arm
    plan = native.registration()
    assert ARM in native.RECIPES and ARM not in MAIN_NAMES
    policy = native.RECIPES[ARM]['policy']
    worker_order = [*MAIN_NAMES, *[n for n in native.RECIPES if n not in MAIN_NAMES]]
    before_arm = worker_order.index(ARM)
    assert gpu in (0, 1, 2, 3) and pair in native.PAIRS
    root = native.OUT/pair/'seed0'
    assert not (root/'completion.json').exists()
    assert not (native.OUT/'all_predictions_sealed_before_GT.json').exists()
    preflight = spare_gpu_preflight(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.native_binary_inputs import inventory, NativeVisual
    from flex_uda.active.native_binary_curriculum import replay_solver

    guard = generation_guard()
    torch.set_num_threads(2)
    use_verified_local_features()
    begin = time.perf_counter()
    stream = root/'stream.pt'
    stream_sha = file_sha(stream)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    md = metadata(native.PAIRS[pair][1])
    videos = sorted(md['videos'])
    index = saved['next_video']
    assert 0 < index < len(videos), 'Requires an existing warm prefix and unfinished current video'
    video = videos[index]
    local = root/'videos'/video
    destination = local/'heads'/(ARM+'.pt')
    assert not destination.exists(), 'Original registered head already complete; never overwrite'
    current_closed = local/'policies'/policy/'closed.pt'
    assert current_closed.is_file(), 'Wait for this policy to close; do not duplicate teacher/actor work'
    current = torch.load(current_closed, map_location='cpu', weights_only=False)['checkpoint']
    assert current['recipe'] == native.RECIPES['FULL_CONDITIONAL_REFERENCE_'+policy]
    previous = saved['previous'][ARM]
    assert previous['recipe'] == native.RECIPES[ARM]
    progress = json.loads((root/'progress.json').read_text())
    assert progress['video'] == video and progress['completed_arms'] < before_arm
    inputs = {str(stream):stream_sha, str(current_closed):file_sha(current_closed)}
    total = np.zeros(1101); square = np.zeros(1101); number = 0; past = []
    for vi, name in enumerate(videos[:index+1]):
        v = JointObservationVideo(pair, name)
        actions, _ = inventory(pair, name)
        visual = NativeVisual(v, actions)
        total += visual.norm_sum; square += visual.norm_square; number += visual.norm_rows
        policy_folder = root/'videos'/name/'policies'/policy
        paid_path = policy_folder/'paid.json'
        inputs[str(paid_path)] = file_sha(paid_path)
        paid = json.loads(paid_path.read_text())['paid']
        if vi < index:
            past.append(visual.record(paid))
        else:
            sequence_path = policy_folder/'sequence.json'
            inputs[str(sequence_path)] = file_sha(sequence_path)
            sequence = json.loads(sequence_path.read_text())['sequence']
    mean = total/number
    std = np.sqrt(np.maximum(square/number-mean**2, 0)).clip(min=.01)
    assert np.array_equal(mean, current['mean']) and np.array_equal(std, current['std'])
    assert file_sha(stream) == stream_sha and not guard
    preflight_after_inputs = spare_gpu_preflight(gpu)
    staging = local/'parallel_registered_heads'/ARM
    staging.mkdir(parents=True, exist_ok=True)
    prepared = staging/'prepared.pt'
    plan_path = staging/'plan.json'
    receipt_path = staging/'receipt.json'
    binding = dict(pair=pair, video=video, arm=ARM, policy=policy,
                   original_worker_arm_number=before_arm+1, neural_seed=0, physical_GPU=gpu,
                   original_registration_sha256=file_sha(native.PLAN), original_sources=plan['source_sha256'],
                   exact_input_file_sha256=inputs, normalizer_matches_original_closed_checkpoint=True,
                   previous_arm_recipe_identical=True, prefix_video_count=index+1,
                   normalization_rows=number, original_replay_solver_unmodified=True,
                   entry_sha256=file_sha(Path(__file__)), original_head_destination=str(destination))
    if plan_path.exists():
        assert json.loads(plan_path.read_text())['binding'] == binding
    else:
        atomic_json(plan_path, dict(binding=binding, preflight=preflight,
            preflight_after_inputs=preflight_after_inputs, existing_registered_arm_only=True,
            new_teacher_actor_question_path_predictions_GT_or_neural_seeds=0, goal_status='ACTIVE'))
    print(json.dumps(dict(status='FITTING_EXISTING_REGISTERED_MODEL_IN_PARALLEL',
        pid=os.getpid(), pair=pair, video=video, arm=ARM, physical_GPU=gpu,
        plan=str(plan_path), new_experimental_conditions=0)), flush=True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.15)
    fit_begin = time.perf_counter()
    if prepared.exists():
        prepared_data = torch.load(prepared, map_location='cpu', weights_only=False)
        assert prepared_data['parallel_binding'] == binding
        report = prepared_data['report']
        fit_reused = True
    else:
        model, early, checkpoint, report = replay_solver(visual, paid, sequence,
            native.RECIPES[ARM], mean, std, past=past, previous=previous,
            device='cuda:0', scheduled=native.scheduled(native.RECIPES[ARM]))
        torch.cuda.synchronize()
        assert len(report['fits']) == (4 if native.scheduled(native.RECIPES[ARM]) else 1) and not guard
        if native.RECIPES[ARM]['family'] != 'structured':
            assert all(f['teacher_fit_gate_passed'] and f['excess_risk'] <= .005 for f in report['fits'])
        report['registered_parallel_preparation'] = dict(pair=pair, video=video, arm=ARM,
            physical_GPU=gpu, pid=os.getpid(), exact_binding=binding,
            fitting_wall_seconds=time.perf_counter()-fit_begin,
            diagnostic_ablation_work_not_primary_method_cold_deployment=True)
        prepared_data = dict(checkpoint=checkpoint,
            early={k:a.detach().cpu().clone() for k,a in early.state_dict().items()},
            report=report, parallel_binding=binding)
        temporary = prepared.with_suffix('.tmp')
        torch.save(prepared_data, temporary); temporary.replace(prepared)
        fit_reused = False
    native.registration()
    assert not guard
    inputs_still_identical = all(Path(p).is_file() and file_sha(Path(p)) == sha for p,sha in inputs.items())
    progress = json.loads((root/'progress.json').read_text())
    ahead = progress['video'] == video and progress['completed_arms'] < before_arm
    installed = False
    if inputs_still_identical and ahead and not (root/'completion.json').exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(prepared, destination)
            installed = True
        except FileExistsError:
            pass  # The original worker won; preserve its artifact.
    receipt = dict(status='PASS_EXISTING_REGISTERED_ARM_PREPARED_PARENT_REUSE_PENDING'
                        if installed else 'COMPLETED_REGISTERED_ARM_PRIVATE_CACHE_NOT_INSTALLED',
        pair=pair, video=video, arm=ARM, pid=os.getpid(), physical_GPU=gpu,
        installed_before_original_arm=installed, actual_parent_reuse_verified=False,
        parent_reuse_requires_original_video_receipt_marker=True,
        source_inputs_still_identical=inputs_still_identical, original_progress_at_install=progress,
        prepared_sha256=file_sha(prepared), original_destination=str(destination),
        prepared=str(prepared), prepared_fit_reused=fit_reused, binding=binding, execution=execution,
        fit_records=[dict(excess_risk=r['excess_risk'], teacher_fit_gate_passed=r['teacher_fit_gate_passed'],
                          actual_final_Adam_steps=r['actual_final_Adam_steps']) for r in report['fits']],
        wall_seconds=time.perf_counter()-begin, GT_open_attempts=len(guard),
        new_teacher_actor_questions_predictions_evaluations_or_neural_seeds=0,
        registered_original_ablation_fitting_not_a_new_quality_condition=True,
        parallel_diagnostic_cost_not_single_primary_method_deployment_cost=True, goal_status='ACTIVE')
    atomic_json(receipt_path, receipt)
    print(json.dumps(dict(status=receipt['status'], pid=os.getpid(), pair=pair, video=video,
        arm=ARM, installed=installed, parent_reuse_verified=False,
        receipt=str(receipt_path), wall_seconds=receipt['wall_seconds'], GT_open_attempts=len(guard))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', choices=tuple(native.PAIRS), required=True)
    parser.add_argument('--gpu', type=int, choices=(0,1,2,3), required=True)
    parser.add_argument('--arm', choices=tuple(n for n in native.RECIPES if n not in MAIN_NAMES), required=True)
    args = parser.parse_args()
    main(args.pair, args.gpu, args.arm)
