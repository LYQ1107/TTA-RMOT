"""Execute existing NoIdentity fits using the verified shared-device wrapper.

One CPU distributor per physical GPU 1/2 starts a fresh wrapper process for
each eligible original unit. It waits for original closed inputs, never polls
device utilization, and stops on ownership refusal or numerical failure.
The original worker retains all prediction and whole-study evaluation work.
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

ENTRY = ROOT / 'scripts/a100_native_registered_shared_head.py'
ENTRY_SHA = 'a6f99fb9789c5ef46d04113413301c2db1d22e739707a9c91b1bba680c0715be'
HELPER_SHA = '5e8084ef546190363ab5814f14745e04e5d6759f62415e07cc10e8da6b79f6b0'
PLAN_SHA = '9a19d1643c9c6a51ad92f723f558edcbbc496cafcb05e1e2a2f7161186ad28ec'
ARM = 'NO_IDENTITY_CONDITIONAL_REFERENCE_LEARNED'
PROOF = native.PUBLIC / 'native_project_shared_head_parent_reuse_20261125.json'


def verified_bootstrap():
    proof = json.loads(PROOF.read_text())
    assert proof['status'] == 'PASS_PROJECT_SHARED_HEAD_ACTUAL_ORIGINAL_PARENT_REUSE'
    assert proof['actual_parent_reuse_verified'] and proof['GT_open_attempts'] == 0
    assert proof['entry_sha256'] == ENTRY_SHA and proof['original_helper_sha256'] == HELPER_SHA
    assert proof['original_registration_sha256'] == PLAN_SHA
    assert file_sha(ENTRY) == ENTRY_SHA and file_sha(native.PLAN) == PLAN_SHA
    assert file_sha(ROOT / 'scripts/a100_native_registered_head_parallel.py') == HELPER_SHA
    native.registration()


def eligible_tokens(attempted):
    if (native.OUT / 'all_predictions_sealed_before_GT.json').exists():
        return []
    result = []
    for pair in ('C3', 'C4'):
        root = native.OUT / pair / 'seed0'
        if (root / 'completion.json').exists():
            continue
        progress = json.loads((root / 'progress.json').read_text())
        token = (pair, progress['video'])
        if token in attempted or progress['completed_arms'] > 8:
            continue
        local = root / 'videos' / progress['video']
        if not (local / 'policies/LEARNED/closed.pt').exists():
            continue
        if any(path.exists() for path in (
            local / 'heads' / (ARM + '.pt'),
            local / 'parallel_registered_heads' / ARM / 'plan.json',
            local / 'spare_execution_locks' / (ARM + '.json'))):
            continue
        result.append(token)
    return result


def claim_slot(gpu):
    lock = native.OUT / 'supervisor' / ('shared_executor_gpu' + str(gpu) + '.lock')
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(pid=os.getpid(), physical_GPU=gpu), f)
    return lock


def main(gpu):
    assert gpu in (1, 2)
    verified_bootstrap()
    runtime_sha = file_sha(Path(__file__))
    lock = claim_slot(gpu)
    folder = native.OUT / 'supervisor' / ('shared_executor_gpu' + str(gpu) + '_' + str(os.getpid()))
    folder.mkdir(parents=True, exist_ok=False)
    state_path = folder / 'state.json'
    state = dict(status='WAITING_ORIGINAL_CLOSED_INPUTS_FOR_REGISTERED_NO_IDENTITY_FITS',
        pid=os.getpid(), physical_GPU=gpu, executor_sha256=runtime_sha,
        wrapper_sha256=ENTRY_SHA, helper_sha256=HELPER_SHA,
        original_registration_sha256=PLAN_SHA, actual_bootstrap_proof_sha256=file_sha(PROOF),
        original_controller_session=14883, completed=[], attempted_tokens=[],
        current_child_pid=None, new_quality_conditions=0,
        new_teacher_questions_predictions_GT_or_neural_seeds=0,
        original_workers_preserved=True, goal_status='ACTIVE',
        costs_are_ablation_execution_not_primary_cold_deployment=True)
    atomic_json(state_path, state)
    print(json.dumps(dict(status=state['status'], pid=os.getpid(), physical_GPU=gpu,
        state_file=str(state_path), new_quality_conditions=0)), flush=True)
    attempted = set()
    try:
        while True:
            controller = json.loads(native.STATE.read_text())
            if controller.get('failure') is not None:
                state['status'] = 'SHARED_EXECUTOR_NATIVE_FAILURE_REPORTED_KEEP_ORIGINAL_WORKERS'
                break
            job = controller['active'].get(str(gpu))
            if job is None or (native.OUT / job['pair'] / 'seed0/completion.json').exists():
                state['status'] = 'SHARED_EXECUTOR_PRIMARY_SLOT_FINISHED_KEEP_REMAINING_WORKERS'
                break
            tokens = eligible_tokens(attempted)
            if not tokens:
                time.sleep(1)
                continue
            assert file_sha(Path(__file__)) == runtime_sha
            verified_bootstrap()
            log = folder / (str(len(state['completed'])) + '.log')
            command = [sys.executable, '-u', str(ENTRY), '--gpu', str(gpu)]
            with log.open('x', buffering=1) as handle:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                state.update(status='EXECUTING_EXISTING_SHARED_REGISTERED_HEAD',
                    current_child_pid=child.pid, candidate_tokens=tokens, current_log=str(log))
                atomic_json(state_path, state)
                print(json.dumps(dict(status=state['status'], pid=os.getpid(),
                    child_pid=child.pid, physical_GPU=gpu, candidate_tokens=tokens)), flush=True)
                records = []
                for line in child.stdout:
                    handle.write(line)
                    print(line, end='', flush=True)
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                code = child.wait()
            row = dict(child_pid=child.pid, actual_exit_code=code, log=str(log),
                candidate_tokens=tokens, actual_parent_reuse_verified=False)
            state.update(current_child_pid=None,
                attempted_tokens=[list(t) for t in sorted(attempted)])
            if code:
                row['error_tail'] = log.read_text()[-3000:]
                state['completed'].append(row)
                state['status'] = 'STOPPED_SHARED_EXECUTOR_AFTER_CHILD_FAILURE_KEEP_ORIGINAL_WORKERS'
                atomic_json(state_path, state)
                raise SystemExit(code)
            terminal = next((r for r in reversed(records) if r.get('status') in (
                'PASS_EXISTING_SHARED_REGISTERED_HEAD_PREPARED_PARENT_REUSE_PENDING',
                'NO_ELIGIBLE_EXISTING_UNIT_KEEP_ORIGINAL_WORKERS',
                'SHARING_PREFLIGHT_REFUSED_KEEP_ORIGINAL_WORKERS')), None)
            assert terminal is not None, 'Missing actual wrapper terminal record'
            row.update(status=terminal['status'], actual_result=terminal)
            if terminal['status'].startswith('PASS_'):
                result = json.loads(Path(terminal['receipt']).read_text())
                assert result['pid'] == child.pid and result['entry_sha256'] == ENTRY_SHA
                assert result['helper_result']['GT_open_attempts'] == 0
                assert len(result['preflights']) == 2
                attempted.add((result['pair'], result['video']))
                row.update(pair=result['pair'], video=result['video'], arm=result['arm'],
                    installed=result['helper_result']['installed_before_original_arm'],
                    receipt=terminal['receipt'],
                    prepared_sha256=result['helper_result']['prepared_sha256'])
            else:
                attempted.update(tuple(t) for t in tokens)
            state['completed'].append(row)
            state.update(status='WAITING_ORIGINAL_CLOSED_INPUTS_FOR_REGISTERED_NO_IDENTITY_FITS',
                attempted_tokens=[list(t) for t in sorted(attempted)])
            atomic_json(state_path, state)
            print(json.dumps(dict(event='EXISTING_SHARED_REGISTERED_UNIT_RETURNED',
                executor_pid=os.getpid(), physical_GPU=gpu, **row)), flush=True)
            if terminal['status'] == 'SHARING_PREFLIGHT_REFUSED_KEEP_ORIGINAL_WORKERS':
                state['status'] = 'SHARED_EXECUTOR_PREFLIGHT_REFUSED_KEEP_ORIGINAL_WORKERS'
                break
        atomic_json(state_path, state)
        print(json.dumps(dict(status=state['status'], pid=os.getpid(), physical_GPU=gpu,
            completed_units=len(state['completed']), receipt=str(state_path),
            goal_status='ACTIVE')), flush=True)
    finally:
        if lock.exists() and json.loads(lock.read_text())['pid'] == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(1, 2), required=True)
    main(parser.parse_args().gpu)
