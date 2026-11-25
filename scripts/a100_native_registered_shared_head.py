"""Prepare one existing head with explicit project-only GPU sharing.

Only the immutable helper's device preflight is replaced in this new process.
All model, input, optimization, seed and exclusive-install code stays original.
The original helper's plan records both this runtime digest and the verified
resident worker. These are ablation execution costs, not deployment timings.
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
from scripts import a100_native_registered_head_parallel as original

HELPER_SHA = '5e8084ef546190363ab5814f14745e04e5d6759f62415e07cc10e8da6b79f6b0'
ARM = 'NO_IDENTITY_CONDITIONAL_REFERENCE_LEARNED'


class SharingRefused(RuntimeError):
    pass


def validate_snapshot(gpu, used, total, owners, worker, argv):
    if gpu not in (1, 2) or worker is None or worker['pair'] not in ('C3', 'C4'):
        raise SharingRefused('No current registered primary worker on this device')
    if set(owners) != {worker['pid']} or len(owners) != 1:
        raise SharingRefused('Resident compute owners are not exactly the registered primary worker')
    expected_entry = ROOT / 'scripts/a100_native_binary_six_setting.py'
    if (len(argv) != 5 or argv[1] != '-u' or argv[3:] != ['--pair', worker['pair']]
            or Path(argv[0]).resolve() != Path(sys.executable).resolve()
            or (ROOT / argv[2]).resolve() != expected_entry):
        raise SharingRefused('Resident process identity does not match the original native worker')
    if total < 16384 or used < 0 or used > 4096 or used + .15 * total > .5 * total:
        raise SharingRefused('Insufficient conservative device memory allowance')
    return dict(physical_GPU=gpu, used_MiB=used, total_MiB=total,
        resident_registered_worker_pid=worker['pid'], resident_pair=worker['pair'],
        original_per_process_memory_fraction=.15, unknown_compute_owners=0,
        execution_preflight_amendment='PROJECT_ONLY_SHARED_DEVICE',
        no_model_input_fit_gate_or_seed_change=True)


def main(gpu):
    plan = native.registration()
    helper = ROOT / 'scripts/a100_native_registered_head_parallel.py'
    assert file_sha(helper) == HELPER_SHA
    runtime_sha = file_sha(Path(__file__))
    candidates = []
    for pair in ('C3', 'C4'):
        root = native.OUT / pair / 'seed0'
        if (root / 'completion.json').exists():
            continue
        progress = json.loads((root / 'progress.json').read_text())
        local = root / 'videos' / progress['video']
        if progress['completed_arms'] > 8:
            continue
        if not (local / 'policies/LEARNED/closed.pt').exists():
            continue
        if (local / 'heads' / (ARM + '.pt')).exists():
            continue
        if (local / 'parallel_registered_heads' / ARM / 'plan.json').exists():
            continue
        order = [*original.MAIN_NAMES, *[a for a in native.RECIPES if a not in original.MAIN_NAMES]]
        earlier = [a for a in order[progress['completed_arms'] + 1:order.index(ARM)]
            if not (local / 'heads' / (a + '.pt')).exists()
            and not (local / 'spare_execution_locks' / (a + '.json')).exists()]
        if sum(native.scheduled(native.RECIPES[a]) for a in earlier) < 3:
            continue
        candidates.append((-(progress['total_videos'] - progress['completed_videos']), pair, progress, local))
    chosen = None
    for _, pair, progress, local in sorted(candidates):
        lock = local / 'spare_execution_locks' / (ARM + '.json')
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, 'w') as f:
            json.dump(dict(executor_pid=os.getpid(), physical_GPU=gpu, pair=pair,
                video=progress['video'], arm=ARM, runtime_sha256=runtime_sha,
                original_progress_when_reserved=progress), f)
        chosen = pair, progress, local, lock
        break
    if chosen is None:
        print(json.dumps(dict(status='NO_ELIGIBLE_EXISTING_UNIT_KEEP_ORIGINAL_WORKERS',
            new_models=0, new_quality_conditions=0)), flush=True)
        return
    pair, progress, local, lock = chosen
    receipt_path = native.OUT / 'supervisor' / ('shared_registered_head_' + str(os.getpid()) + '.json')
    record = dict(status='PREPARING_EXISTING_HEAD_WITH_EXPLICIT_PROJECT_GPU_SHARING',
        pid=os.getpid(), physical_GPU=gpu, pair=pair, video=progress['video'], arm=ARM,
        entry_sha256=runtime_sha, original_helper_sha256=HELPER_SHA,
        original_registration_sha256=file_sha(native.PLAN), original_sources=plan['source_sha256'],
        original_progress_when_reserved=progress, preflights=[], new_quality_conditions=0,
        new_teacher_questions_predictions_GT_or_neural_seeds=0, goal_status='ACTIVE',
        ablation_execution_not_primary_cold_deployment=True, started_unix=time.time())
    atomic_json(receipt_path, record)

    def shared_preflight(device):
        assert device == gpu and file_sha(Path(__file__)) == runtime_sha
        state = json.loads(native.STATE.read_text())
        worker = state['active'].get(str(device))
        memory = subprocess.check_output(['nvidia-smi', '-i', str(device),
            '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
        used, total = map(int, memory.strip().split(','))
        raw = subprocess.check_output(['nvidia-smi', '-i', str(device),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        owners = [int(p) for p in raw.split()]
        argv = []
        if worker is not None and owners == [worker['pid']]:
            argv = [p.decode() for p in (Path('/proc') / str(worker['pid']) / 'cmdline').read_bytes().split(b'\0')[:-1]]
        snapshot = validate_snapshot(device, used, total, owners, worker, argv)
        snapshot.update(runtime_entry_sha256=runtime_sha, numerical_helper_sha256=HELPER_SHA)
        record['preflights'].append(snapshot)
        atomic_json(receipt_path, record)
        return snapshot

    original.spare_gpu_preflight = shared_preflight
    print(json.dumps(dict(status=record['status'], pid=os.getpid(), pair=pair,
        video=progress['video'], physical_GPU=gpu, arm=ARM, receipt=str(receipt_path))), flush=True)
    try:
        original.main(pair, gpu, ARM)
    except SharingRefused as exc:
        record.update(status='SHARING_PREFLIGHT_REFUSED_KEEP_ORIGINAL_WORKERS', reason=str(exc),
            actual_fits=0)
        assert not (local / 'parallel_registered_heads' / ARM / 'plan.json').exists()
        assert json.loads(lock.read_text())['executor_pid'] == os.getpid()
        lock.unlink()
        atomic_json(receipt_path, record)
        print(json.dumps(record), flush=True)
        return
    helper_receipt = local / 'parallel_registered_heads' / ARM / 'receipt.json'
    result = json.loads(helper_receipt.read_text())
    assert result['pid'] == os.getpid() and result['video'] == progress['video']
    assert result['GT_open_attempts'] == 0 and len(record['preflights']) == 2
    record.update(status='PASS_EXISTING_SHARED_REGISTERED_HEAD_PREPARED_PARENT_REUSE_PENDING',
        helper_receipt=str(helper_receipt), helper_result=result,
        actual_parent_reuse_verified=False)
    atomic_json(receipt_path, record)
    print(json.dumps(dict(status=record['status'], pid=os.getpid(), pair=pair,
        video=progress['video'], physical_GPU=gpu, arm=ARM,
        installed=result['installed_before_original_arm'], receipt=str(receipt_path))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(1, 2), required=True)
    main(parser.parse_args().gpu)
