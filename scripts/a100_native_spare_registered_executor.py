"""Execute existing unstarted C4 replay heads on one spare GPU.

This is a numerical-work executor, not a process/GPU monitor or a new study.
It waits for the original worker's closed inputs, reserves an existing arm,
then invokes the unchanged registered-head helper.  The original worker always
remains responsible for predictions and any head not prepared in time.
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
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as native

ENTRY = ROOT / 'scripts/a100_native_registered_head_parallel.py'
ENTRY_SHA = '5e8084ef546190363ab5814f14745e04e5d6759f62415e07cc10e8da6b79f6b0'
PLAN_SHA = '9a19d1643c9c6a51ad92f723f558edcbbc496cafcb05e1e2a2f7161186ad28ec'
# Conservative execution windows, unrelated to losses, labels or model choice.
# The helper independently verifies the original exact deadline and all inputs.
WINDOWS = (
    ('FULL_LIKELIHOOD_LOCAL_LEARNED', 1),
    ('NO_IDENTITY_CONDITIONAL_REFERENCE_LEARNED', 13),
    ('FULL_LIKELIHOOD_LOCAL_FROZEN', 8),
    ('FULL_LIKELIHOOD_REFERENCE_FROZEN', 6),
    ('JOINT_KD_FROZEN', 12),
    ('POINT_KD_FROZEN', 16),
    ('COARSE_UNIT_SOFT_FROZEN', 16),
    ('COARSE_UNIT_HARD_FROZEN', 18),
    ('COARSE_CONF_SOFT_FROZEN', 19),
    ('COARSE_CONF_HARD_FROZEN', 19),
)


def main(gpu):
    assert gpu in (0, 3)
    assert file_sha(ENTRY) == ENTRY_SHA and file_sha(native.PLAN) == PLAN_SHA
    native.registration()
    order = ['FULL_CONDITIONAL_REFERENCE_LEARNED', 'FULL_CONDITIONAL_REFERENCE_FROZEN']
    order += [name for name in native.RECIPES if name not in order]
    assert all(arm in native.RECIPES and maximum < order.index(arm)
               for arm, maximum in WINDOWS)
    root = native.OUT / 'C4/seed0'
    status_path = native.OUT / 'supervisor' / ('spare_registered_gpu' + str(gpu) + '_' + str(os.getpid()) + '.json')
    state = dict(status='RUNNING_EXISTING_C4_REGISTERED_WORK_EXECUTOR', pid=os.getpid(),
        physical_GPU=gpu, pair='C4', entry=str(ENTRY), entry_sha256=ENTRY_SHA,
        executor_sha256=file_sha(Path(__file__)), original_controller_session=14883,
        completed=[], new_quality_conditions=0, new_neural_seeds=0,
        no_GT_or_task_metric_selection=True,
        ablation_execution_not_single_method_cold_deployment_cost=True,
        goal_status='ACTIVE')
    atomic_json(status_path, state)
    print(json.dumps({k: state[k] for k in ('status', 'pid', 'physical_GPU', 'pair',
        'new_quality_conditions')}), flush=True)
    while not (root / 'completion.json').is_file():
        if (native.OUT / 'all_predictions_sealed_before_GT.json').is_file():
            break
        progress = json.loads((root / 'progress.json').read_text())
        video = progress['video']
        local = root / 'videos' / video
        chosen = None
        for arm, maximum in WINDOWS:
            if progress['completed_arms'] > maximum:
                continue
            if (local / 'heads' / (arm + '.pt')).is_file():
                continue
            policy = native.RECIPES[arm]['policy']
            if not (local / 'policies' / policy / 'closed.pt').is_file():
                continue
            locks = local / 'spare_execution_locks'
            locks.mkdir(parents=True, exist_ok=True)
            lock = locks / (arm + '.json')
            try:
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, 'w') as f:
                json.dump(dict(executor_pid=os.getpid(), physical_GPU=gpu,
                    pair='C4', video=video, arm=arm,
                    original_progress_when_reserved=progress), f)
            chosen = arm
            break
        if chosen is None:
            time.sleep(1)
            continue
        if (root / 'completion.json').is_file():
            break
        if (local / 'heads' / (chosen + '.pt')).is_file():
            continue
        assert file_sha(ENTRY) == ENTRY_SHA
        native.registration()
        command = [sys.executable, '-u', str(ENTRY), '--pair', 'C4',
                   '--gpu', str(gpu), '--arm', chosen]
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        state.update(current_video=video, current_arm=chosen, current_child_pid=child.pid)
        atomic_json(status_path, state)
        print(json.dumps(dict(status='EXECUTING_EXISTING_REGISTERED_C4_ARM',
            pid=os.getpid(), child_pid=child.pid, physical_GPU=gpu,
            pair='C4', video=video, arm=chosen)), flush=True)
        records = []
        for line in child.stdout:
            print(line, end='', flush=True)
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        code = child.wait()
        if code:
            state.update(status='STOPPED_SPARE_EXECUTOR_KEEP_ORIGINAL_WORKER',
                actual_child_exit_code=code)
            atomic_json(status_path, state)
            raise SystemExit(code)
        result = next((r for r in reversed(records) if 'receipt' in r), None)
        assert result is not None, 'Missing actual registered-helper completion receipt'
        state['completed'].append(dict(selected_video=video, arm=chosen,
            child_pid=child.pid, actual_result=result,
            actual_parent_reuse_verified=False))
        state.update(current_video=None, current_arm=None, current_child_pid=None)
        atomic_json(status_path, state)
    state.update(status='PASS_EXISTING_C4_REGISTERED_WORK_EXECUTOR_FINISHED',
        current_video=None, current_arm=None, current_child_pid=None)
    atomic_json(status_path, state)
    print(json.dumps(dict(status=state['status'], pid=os.getpid(), physical_GPU=gpu,
        completed_registered_heads=len(state['completed']), receipt=str(status_path),
        new_quality_conditions=0, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(0, 3), required=True)
    main(parser.parse_args().gpu)
