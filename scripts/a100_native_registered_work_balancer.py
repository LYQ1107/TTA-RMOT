"""Balance only already registered C3/C4 replay fits across spare GPUs 0/3.

The original workers still select questions, generate every prediction, and
evaluate the full study.  This executor calls the immutable existing helper;
it changes execution placement, never a recipe, prefix, fit gate or seed.
An older executor is retired only while idle, with its identity and absence
of children checked again after stopping that executor alone. Running fits
are allowed to finish. No experimental worker or helper receives a signal.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as native

ENTRY = ROOT / 'scripts/a100_native_registered_head_parallel.py'
ENTRY_SHA = '5e8084ef546190363ab5814f14745e04e5d6759f62415e07cc10e8da6b79f6b0'
OLD_ENTRY = ROOT / 'scripts/a100_native_spare_registered_executor.py'
OLD_SHA = '10d00f88bd479b8151dfa40288ffe29a2947d2e6b40256488f9c1cb1d6a11170'
PLAN_SHA = '9a19d1643c9c6a51ad92f723f558edcbbc496cafcb05e1e2a2f7161186ad28ec'
MAIN = ('FULL_CONDITIONAL_REFERENCE_LEARNED', 'FULL_CONDITIONAL_REFERENCE_FROZEN')
ORDER = [*MAIN, *[arm for arm in native.RECIPES if arm not in MAIN]]


def candidates():
    """Use unfinished work and execution deadlines, never scores or losses."""
    rows = []
    if (native.OUT / 'all_predictions_sealed_before_GT.json').exists():
        return rows
    for pair in ('C3', 'C4'):
        root = native.OUT / pair / 'seed0'
        if (root / 'completion.json').exists():
            continue
        progress = json.loads((root / 'progress.json').read_text())
        local = root / 'videos' / progress['video']
        for arm in ORDER:
            if arm in MAIN:
                continue
            # Give scheduled four-prefix fits more room than single fits.
            lead = 3 if native.scheduled(native.RECIPES[arm]) else 2
            if ORDER.index(arm) - progress['completed_arms'] < lead:
                continue
            if (local / 'heads' / (arm + '.pt')).exists():
                continue
            policy = native.RECIPES[arm]['policy']
            if not (local / 'policies' / policy / 'closed.pt').exists():
                continue
            lock = local / 'spare_execution_locks' / (arm + '.json')
            if lock.exists():
                continue
            # Cached or externally reserved heads can be skipped immediately
            # by the original worker; they are not available execution lead.
            earlier = [name for name in ORDER[progress['completed_arms']:ORDER.index(arm)]
                if not (local / 'heads' / (name + '.pt')).exists()
                and not (local / 'spare_execution_locks' / (name + '.json')).exists()]
            required = 2 if native.scheduled(native.RECIPES[arm]) else 1
            if sum(native.scheduled(native.RECIPES[name]) for name in earlier) < required:
                continue
            if len(earlier) < 2:
                continue
            rows.append(dict(pair=pair, video=progress['video'], arm=arm,
                progress=progress, lock=str(lock),
                remaining=progress['total_videos'] - progress['completed_videos']))
    return sorted(rows, key=lambda r: (-r['remaining'], ORDER.index(r['arm']), r['pair']))


def reserve(row, gpu):
    lock = Path(row['lock'])
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(executor_pid=os.getpid(), physical_GPU=gpu,
            pair=row['pair'], video=row['video'], arm=row['arm'],
            original_progress_when_reserved=row['progress']), f)
    return True


def children(pid):
    return (Path('/proc') / str(pid) / 'task' / str(pid) / 'children').read_text().split()


def retire_if_idle(pid, gpu):
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return dict(status='OLD_EXECUTOR_ALREADY_ABSENT', pid=pid, physical_GPU=gpu)
    expected = proc.joinpath('cmdline').read_bytes().split(b'\0')[:-1]
    argv = [s.decode() for s in expected]
    assert len(argv) == 5 and argv[1] == '-u' and argv[3:] == ['--gpu', str(gpu)]
    assert Path(argv[0]).resolve() == Path(sys.executable).resolve()
    assert (ROOT / argv[2]).resolve() == OLD_ENTRY
    path = native.OUT / 'supervisor' / ('spare_registered_gpu' + str(gpu) + '_' + str(pid) + '.json')
    state = json.loads(path.read_text())
    assert state['pid'] == pid and state['physical_GPU'] == gpu
    if state.get('current_child_pid') is not None or children(pid):
        return None
    stopped = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            status = proc.joinpath('status').read_text()
            if any(line.startswith('State:') and '\tT' in line for line in status.splitlines()):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Owned idle executor did not stop; keep the original executor')
        state = json.loads(path.read_text())
        if state.get('current_child_pid') is not None or children(pid):
            return None
        assert proc.joinpath('cmdline').read_bytes().split(b'\0')[:-1] == expected
        # SIGTERM targets this idle distributor only, then SIGCONT delivers it.
        os.kill(pid, signal.SIGTERM)
        return dict(status='INTENTIONALLY_RETIRED_IDLE_EXECUTOR_ONLY', pid=pid,
            physical_GPU=gpu, no_children_verified_while_stopped=True,
            expected_cmdline=argv, source_sha256=file_sha(OLD_ENTRY),
            prior_state=state, original_workers_and_all_model_fits_signalled=False)
    finally:
        if stopped and proc.exists():
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def main(old_executors, dry_run=False):
    assert file_sha(ENTRY) == ENTRY_SHA and file_sha(native.PLAN) == PLAN_SHA
    assert file_sha(OLD_ENTRY) == OLD_SHA
    native.registration()
    if dry_run:
        print(json.dumps(dict(status='DRY_RUN_EXISTING_REGISTERED_WORK_ONLY',
            candidates=candidates(), retired_executors=0, spawned_fits=0,
            source_pins_unchanged=True)), flush=True)
        return
    assert set(old_executors) == {0, 3}
    assert all(pid is not None and pid > 1 for pid in old_executors.values())
    lock = native.OUT / 'supervisor/registered_work_balancer.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(pid=os.getpid(), old_executors=old_executors), f)
    folder = native.OUT / 'supervisor' / ('registered_balancer_' + str(os.getpid()))
    folder.mkdir(parents=True, exist_ok=False)
    state_path = folder / 'state.json'
    source_sha = file_sha(Path(__file__))
    state = dict(status='RUNNING_ORIGINAL_REGISTERED_FITS_NO_NEW_STUDY',
        pid=os.getpid(), source_sha256=source_sha, helper_sha256=ENTRY_SHA,
        original_registration_sha256=PLAN_SHA, original_controller_session=14883,
        available_GPUs=[], retired=[], completed=[], active={},
        new_quality_conditions=0, new_teacher_or_predictions_or_GT_or_seeds=0,
        original_workers_preserved=True, goal_status='ACTIVE',
        costs_are_research_ablation_execution_not_primary_cold_deployment=True)
    active = {}
    available = set()
    failure = None
    atomic_json(state_path, state)
    while True:
        changed = False
        for gpu, job in list(active.items()):
            code = job['process'].poll()
            if code is None:
                continue
            job['handle'].close()
            row = dict(job['row'], gpu=gpu, child_pid=job['process'].pid,
                actual_exit_code=code, log=str(job['log']), actual_parent_reuse_verified=False)
            if code:
                failure = dict(row, error_tail=job['log'].read_text()[-3000:])
            else:
                receipt = native.OUT / row['pair'] / 'seed0/videos' / row['video'] / 'parallel_registered_heads' / row['arm'] / 'receipt.json'
                data = json.loads(receipt.read_text())
                assert data['pid'] == row['child_pid'] and data['GT_open_attempts'] == 0
                assert data['arm'] == row['arm'] and data['video'] == row['video']
                row.update(receipt=str(receipt), installed=data['installed_before_original_arm'],
                    prepared_sha256=data['prepared_sha256'], wall_seconds=data['wall_seconds'])
            state['completed'].append(row)
            del active[gpu]
            changed = True
            print(json.dumps(dict(status='EXISTING_REGISTERED_FIT_RETURNED', **row)), flush=True)
        unfinished = [p for p in ('C3', 'C4') if not (native.OUT / p / 'seed0/completion.json').exists()]
        if not failure and unfinished:
            for gpu, old_pid in old_executors.items():
                if gpu in available:
                    continue
                try:
                    retired = retire_if_idle(old_pid, gpu)
                except (FileNotFoundError, ProcessLookupError):
                    if (Path('/proc') / str(old_pid)).exists():
                        raise
                    retired = dict(status='OLD_EXECUTOR_ALREADY_ABSENT',
                        pid=old_pid, physical_GPU=gpu)
                if retired is not None:
                    state['retired'].append(retired)
                    available.add(gpu)
                    changed = True
                    print(json.dumps(dict(status=retired['status'], pid=old_pid,
                        physical_GPU=gpu, no_numerical_worker_or_fit_interrupted=True)), flush=True)
            for gpu in sorted(available - active.keys()):
                assert file_sha(Path(__file__)) == source_sha and file_sha(ENTRY) == ENTRY_SHA
                native.registration()
                row = next((r for r in candidates() if reserve(r, gpu)), None)
                if row is None:
                    continue
                log = folder / (str(len(state['completed'])) + '_' + str(gpu) + '_' + row['pair'] + '_' + row['video'] + '_' + row['arm'] + '.log')
                handle = log.open('x', buffering=1)
                command = [sys.executable, '-u', str(ENTRY), '--pair', row['pair'],
                    '--gpu', str(gpu), '--arm', row['arm']]
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = dict(process=child, handle=handle, log=log, row=row)
                changed = True
                print(json.dumps(dict(status='FITTING_EXISTING_REGISTERED_ARM',
                    gpu=gpu, pid=child.pid, pair=row['pair'], video=row['video'],
                    arm=row['arm'], original_progress=row['progress'])), flush=True)
        state.update(available_GPUs=sorted(available), failure=failure,
            active={str(g):dict(j['row'], child_pid=j['process'].pid, log=str(j['log']))
                for g, j in active.items()})
        if changed:
            atomic_json(state_path, state)
        if (failure or not unfinished) and not active:
            break
        time.sleep(1)
    state.update(status='STOPPED_KEEP_ORIGINAL_WORKERS_AFTER_HELPER_FAILURE' if failure
        else 'PASS_EXISTING_REGISTERED_WORK_BALANCER_FINISHED', failure=failure)
    atomic_json(state_path, state)
    print(json.dumps(dict(status=state['status'], completed=len(state['completed']),
        receipt=str(state_path), goal_status='ACTIVE')), flush=True)
    if failure:
        raise RuntimeError(json.dumps(failure))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-gpu0-pid', type=int)
    parser.add_argument('--old-gpu3-pid', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    main({0: args.old_gpu0_pid, 3: args.old_gpu3_pid}, args.dry_run)
