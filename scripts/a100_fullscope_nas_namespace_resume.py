"""Resume the original study after verifying NAS SSH/SFTP namespace differences.

Only the transport preflight and its provenance change. The original numerical
worker, selected questions, answers, optimizer states, and evaluation stay fixed.
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
from scripts import a100_fullscope_nas_resume as legacy

OUT = legacy.OUT / 'namespace_preflight_20261126'
OWN = OUT / 'registration.json'
SSH_ROOT = '/home/Valadmin/A800'


def prepare():
    legacy.validate()
    assert not OWN.exists()
    atomic_json(OWN, dict(
        status='REGISTERED_SSH_EXEC_NAMESPACE_PREFLIGHT_CORRECTION_ONLY',
        source_sha256=file_sha(__file__),
        original_transport_source_sha256=file_sha(legacy.__file__),
        original_transport_registration_sha256=file_sha(legacy.OWN),
        SSH_exec_root=SSH_ROOT, SFTP_mount_root='/home/A800',
        reason='Authenticated SSH exec uses /home/Valadmin/A800; the same authenticated SFTP endpoint exposes /home/A800. All three formerly failing mounted images decode and exactly match fresh SSH-side SHA256. No remount is needed.',
        preflight='Require the existing exact read-only FUSE mount, authenticated fresh SSH hashes of all three failed images, and bounded mounted-image decoding with matching content hashes.',
        numerical_worker=str(Path(legacy.__file__).relative_to(ROOT))+' --worker',
        original42sources_and_numerical_recovery_unchanged=True,
        questions_answers_checkpoints_optimizer_RNG_budgets_gate_seed_unchanged=True,
        GPU_workers=[0, 1, 2, 3], preserve_GPU2_service_PID=legacy.GPU2_SERVICE_PID,
        archive_and_all_failed_attempts_retained=True,
        no_new_scientific_arm_or_module_ablation=True,
        no_GT_before_original4588_and_all_teacher_attempt_seals=True,
        extra_reload_preflight_retry_cost_not_free=True, goal_status='ACTIVE'))
    print(json.dumps(dict(status='PREPARED_NAMESPACE_ONLY_RESUME', registration=str(OWN))), flush=True)


def validate():
    numerical, plan = legacy.validate()
    own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__)
    assert own['original_transport_source_sha256'] == file_sha(legacy.__file__)
    assert own['original_transport_registration_sha256'] == file_sha(legacy.OWN)
    return numerical, plan


def preflight(attempt):
    started = time.perf_counter()
    record = dict(GPU_workers_started=0, target_GT_opened=False, SSH_exec_root=SSH_ROOT,
                  SFTP_mount_root='/home/A800', namespace_registration_sha256=file_sha(OWN), image_results=[])
    try:
        mounts = json.loads(subprocess.check_output(
            ['findmnt', '-T', str(legacy.MOUNT), '-J', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'], text=True))
        mount = mounts['filesystems'][0]
        record['mount'] = mount
        assert mount['target'] == str(legacy.MOUNT)
        assert mount['source'] == 'Valadmin@127.0.0.1:/home/A800'
        assert mount['fstype'] == 'fuse.sshfs' and 'ro' in mount['options'].split(',')
        options = ['-p', '24557', '-i', str(ROOT.parent/'.ssh/id_ed25519_nas_readwrite'),
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'HostKeyAlias=nas-readwrite', '-o', 'UserKnownHostsFile='+str(ROOT.parent/'.ssh/known_hosts_nas_readwrite'),
            '-o', 'ConnectTimeout=8']
        remote = [SSH_ROOT+'/'+str(path.relative_to(legacy.MOUNT)) for path in legacy.PROBES]
        # These are fixed registered paths without shell metacharacters.
        connection = subprocess.run(['ssh', *options, 'Valadmin@127.0.0.1', 'sha256sum '+' '.join(remote)],
                                    capture_output=True, text=True, timeout=20)
        record['ssh'] = dict(returncode=connection.returncode, stderr=connection.stderr[-2000:],
                             fresh_image_hashes=connection.stdout)
        assert connection.returncode == 0, record['ssh']
        hashes = {}
        for line in connection.stdout.splitlines():
            digest, path = line.split(None, 1)
            assert len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)
            hashes[path.strip()] = digest
        assert set(hashes) == set(remote)
        code = ('import hashlib,io,json,sys; from pathlib import Path; from PIL import Image; '
                'p=Path(sys.argv[1]); b=p.read_bytes(); im=Image.open(io.BytesIO(b)); im.load(); '
                'print(json.dumps(dict(path=str(p),bytes=len(b),size=im.size,sha256=hashlib.sha256(b).hexdigest())))')
        def decode(path):
            try:
                result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, timeout=20)
                return dict(path=str(path), returncode=result.returncode,
                    decoded=json.loads(result.stdout) if result.returncode == 0 else None, stderr=result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                return dict(path=str(path), returncode=124, stderr='NAS mounted-image decode timed out')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            record['image_results'] = list(pool.map(decode, legacy.PROBES))
        for row, remote_path in zip(record['image_results'], remote):
            assert row['returncode'] == 0, row
            assert row['decoded']['sha256'] == hashes[remote_path], row
        record['status'] = 'PASS_AUTHENTICATED_FRESH_REMOTE_HASHES_MATCH_READONLY_MOUNTED_PIXELS'
    except Exception as error:
        record.update(status='FAILED_NO_GPU_WORKERS_STARTED', error=repr(error))
        raise
    finally:
        record['wall_seconds'] = time.perf_counter()-started
        atomic_json(attempt/'NAS_preflight.json', record)


def seal_and_evaluate(numerical, plan):
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_fullscope_visual_six import METHODS, PRIMARY, COMPARATORS, evaluate_and_collect
    expected = set(METHODS)
    expected.update(PRIMARY+'__AT_'+name+'_COUNT' for name in COMPARATORS)
    expected.update(name+'__AT_'+PRIMARY+'_COUNT' for name in COMPARATORS)
    assert len(expected) == 31
    tasks = []
    for case in plan['cases']:
        path = legacy.PARENT/'units'/case['pair']/case['evaluated_video']/'generation_complete.json'
        done = json.loads(path.read_text())
        assert done['GT_open_attempts'] == 0 and done['registration_sha256'] == file_sha(legacy.PARENT/'registration.json')
        assert len(done['tasks']) == 31 and {task['method'] for task in done['tasks']} == expected
        assert all(task['pair'] == case['pair'] and task['video'] == case['evaluated_video'] for task in done['tasks'])
        tasks.extend(done['tasks'])
    assert len(tasks) == 4588 and len({task['folder'] for task in tasks}) == 4588
    for task in tasks:
        verify_predictions(task['folder'])
    sessions = []
    for path in sorted((legacy.PARENT/'supervisor').glob('GPU*_teacher_session_*.json')):
        teacher = json.loads(path.read_text())['teacher']
        if teacher is not None:
            assert teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == legacy.TEACHER_SHA
            sessions.append(dict(path=str(path), sha256=file_sha(path)))
    assert len(sessions) >= 4
    for gpu in (0, 1, 2, 3):
        teacher = json.loads((legacy.PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())['teacher']
        assert teacher is not None and teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == legacy.TEACHER_SHA
    atomic_json(legacy.PARENT/'all4588_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0,
        teacher_final_weight_seals=4, all_attempt_teacher_sessions=sessions,
        registration_sha256=file_sha(legacy.PARENT/'registration.json'),
        explicit_numerical_recovery_registration_sha256=file_sha(numerical.OWN),
        explicit_NAS_transport_resume_registration_sha256=file_sha(legacy.OWN),
        explicit_NAS_namespace_preflight_registration_sha256=file_sha(OWN)))
    print('ALL4588_AND_ALL_ATTEMPT_TEACHER_SEALS_BEFORE_ISOLATED_GT_AFTER_NAMESPACE_RESUME', flush=True)
    evaluate_and_collect(plan, tasks)


def main(preflight_only=False):
    numerical, plan = validate()
    attempt = OUT/('attempt_'+str(time.time_ns())+'_'+str(os.getpid()))
    attempt.mkdir(parents=True)
    started = time.perf_counter()
    preflight(attempt)
    if preflight_only:
        print(json.dumps(dict(status='NAS_NAMESPACE_PREFLIGHT_PASSED_NO_GPU_WORKERS', attempt=str(attempt))), flush=True)
        return
    owners = {}
    for gpu in (0, 1, 2, 3):
        result = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name',
            '--format=csv,noheader,nounits'], text=True).strip()
        rows = [row.split(',', 1) for row in result.splitlines() if row.strip()]
        allowed = {legacy.GPU2_SERVICE_PID} if gpu == 2 else set()
        assert all(int(row[0]) in allowed for row in rows), (gpu, result)
        owners[gpu] = result
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for gpu in (0, 1, 2, 3):
            proc = subprocess.Popen([sys.executable, '-u', str(Path(legacy.__file__)), '--worker', str(gpu), '--attempt', str(attempt)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            future = pool.submit(numerical.drain, proc, attempt/f'GPU{gpu}.log')
            jobs.append((gpu, proc, future))
            atomic_json(attempt/f'GPU{gpu}_launch.json', dict(GPU=gpu, pid=proc.pid, previous_owners_preserved=owners[gpu],
                NAS_resume_registration_sha256=file_sha(legacy.OWN), namespace_registration_sha256=file_sha(OWN)))
            print(json.dumps(dict(status='STARTED_UNCHANGED_CHECKPOINT_RESUME_AFTER_VERIFIED_NAS_RECOVERY',
                GPU=gpu, pid=proc.pid, attempt=str(attempt))), flush=True)
        failures = []
        for gpu, proc, future in jobs:
            code = proc.wait()
            future.result()
            if code:
                failures.append([gpu, code])
    atomic_json(attempt/'workers_final.json', dict(failures=failures, wall_seconds=time.perf_counter()-started,
        original_source_registration_sha256=file_sha(legacy.PARENT/'registration.json'), goal_status='ACTIVE'))
    assert not failures, failures
    validate()
    seal_and_evaluate(numerical, plan)
    atomic_json(attempt/'completion.json', dict(status='COMPLETE_ORIGINAL_SIX_STUDY_AFTER_NAS_NAMESPACE_RESUME',
        namespace_registration_sha256=file_sha(OWN), extra_attempt_cost_not_free=True,
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        main(args.preflight)
