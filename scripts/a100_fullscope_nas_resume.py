"""Resume the unchanged full-six study after the measured NAS I/O outage.

This changes transport and scheduling only. Existing numerical recovery,
selected questions, answers, checkpoints, budgets, and gates are reused.
No target GT is opened until the original whole-study barrier succeeds.
"""
import argparse
import concurrent.futures
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

PARENT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OUT = PARENT / 'nas_io_resume_20261126'
OWN = OUT / 'registration.json'
INCIDENT = ROOT / 'a100_runs/visual_classical_20261121/supervisor/fullscope_NAS_transport_failure_20261126.json'
MOUNT = Path('/data3/liuyeqiang/nas/A800')
TEACHER_SHA = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
GPU2_SERVICE_PID = 243079
PROBES = (
    MOUNT / 'data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance/DanceTrack/training/image_02/dancetrack0026/000127.png',
    MOUNT / 'data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val/v_dw7LOz17Omg_c053/img1/000328.jpg',
    MOUNT / 'data2/user/LocateMOT_FullDataV2_spill/datasets/SportsMOT/val/v_dw7LOz17Omg_c053/img1/000333.jpg',
)


def original():
    import scripts.a100_fullscope_fit_recovery as numerical
    plan, _ = numerical.validate()
    return numerical, plan


def prepare():
    numerical, plan = original()
    assert not OWN.exists()
    incident = json.loads(INCIDENT.read_text())
    assert incident['GT_open_attempts'] == 0
    assert len(incident['actual_sealed_unit_receipts']) == 36
    archive = OUT / 'original_failure_state'
    preserved = []
    for row in incident['preserved_partial_files']:
        source = Path(row['path'])
        source = source if source.is_absolute() else ROOT / source
        assert file_sha(source) == row['sha256']
        destination = archive / source.resolve().relative_to(PARENT.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
            destination.chmod(0o444)
        assert file_sha(destination) == row['sha256']
        preserved.append(dict(source=str(source), archive=str(destination), sha256=row['sha256'], bytes=row['bytes']))
    for row in incident['actual_sealed_unit_receipts']:
        path = Path(row['receipt'])
        assert file_sha(path if path.is_absolute() else ROOT/path) == row['sha256']
    atomic_json(OWN, dict(status='REGISTERED_TRANSPORT_ONLY_RESUME_OF_UNFINISHED_FULL_SIX_STUDY',
        source_sha256=file_sha(__file__), parent_registration_sha256=file_sha(PARENT/'registration.json'),
        numerical_recovery_registration_sha256=file_sha(numerical.OWN),
        numerical_recovery_source_sha256=file_sha(numerical.__file__), incident_sha256=file_sha(INCIDENT),
        original42sources=plan['source_sha256'], original_failure_state_archive=preserved,
        sealed_units_before_resume=36, original_planned_units=148, original_outputs=4588,
        failed_actual_sessions=[23947, 78495], actual_exit_codes=[1, 1],
        reason='NAS EIO on real Dance/Sports images; direct SSH banner exchange timed out. Not an algorithmic result.',
        GPU_workers=[0, 1, 2, 3], preserved_GPU2_service_PID=GPU2_SERVICE_PID,
        numerical_worker='Existing a100_fullscope_fit_recovery.worker on each GPU; only its output-metadata directory is redirected per attempt.',
        transport='Before loading any GPU model, require readonly mount, authenticated SSH, and complete decoding of all three failed images. Retry only transient OSError from the identical PairedViews.frame call, up to four attempts with 2/5/10-second waits; never substitute or omit an image.',
        checkpoint='Original final/closed/bootstrap/optimizer/RNG and before-answer selections, plus per-policy journals. Already sealed units skipped; cached or received answers never grant extra unique budget.',
        parameters_architecture_objectives_teacher_budgets_and_seed_unchanged=True, original_gate=.005, seed=0,
        preserve_failed_attempt_teacher_seals_and_cost=True, additional_cold_load_IO_retry_and_resume_cost_not_free=True,
        original42numerical_sources_unchanged=True, no_GT_before_whole4588_and_all_teacher_attempt_seals=True,
        no_new_scientific_arm_or_module_ablation=True, goal_status='ACTIVE'))
    print(json.dumps(dict(status='PREPARED_NAS_TRANSPORT_RESUME', archived_files=len(preserved), sealed_units=36)), flush=True)


def validate():
    numerical, plan = original()
    own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__)
    assert own['parent_registration_sha256'] == file_sha(PARENT/'registration.json')
    assert own['numerical_recovery_registration_sha256'] == file_sha(numerical.OWN)
    assert own['numerical_recovery_source_sha256'] == file_sha(numerical.__file__)
    assert own['incident_sha256'] == file_sha(INCIDENT)
    return numerical, plan


def preflight_nas(attempt):
    started = time.perf_counter()
    options = ['-p', '24557', '-i', str(ROOT.parent/'.ssh/id_ed25519_nas_readwrite'),
        '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'HostKeyAlias=nas-readwrite', '-o', 'UserKnownHostsFile='+str(ROOT.parent/'.ssh/known_hosts_nas_readwrite'),
        '-o', 'ConnectTimeout=8']
    try:
        connection = subprocess.run(['ssh', *options, 'Valadmin@127.0.0.1', 'test -r /home/A800'],
            capture_output=True, text=True, timeout=15)
        ssh = dict(returncode=connection.returncode, stderr=connection.stderr[-2000:])
    except subprocess.TimeoutExpired:
        ssh = dict(returncode=124, stderr='Authenticated NAS SSH preflight timed out')
    record = dict(ssh=ssh, GPU_workers_started=0, image_results=[], target_GT_opened=False)
    if ssh['returncode']:
        record['wall_seconds'] = time.perf_counter()-started
        atomic_json(attempt/'NAS_preflight.json', record)
        raise RuntimeError('NAS transport unavailable; no GPU worker or teacher model was started')
    mount_options = subprocess.check_output(['findmnt', '-T', str(MOUNT), '-n', '-o', 'OPTIONS'], text=True).strip()
    assert 'ro' in mount_options.split(','), mount_options
    probe_code = ('import hashlib,io,json,sys; from pathlib import Path; from PIL import Image; '
        'p=Path(sys.argv[1]); b=p.read_bytes(); im=Image.open(io.BytesIO(b)); im.load(); '
        'print(json.dumps(dict(path=str(p),bytes=len(b),size=im.size,sha256=hashlib.sha256(b).hexdigest())))')
    def probe(path):
        try:
            result = subprocess.run([sys.executable, '-c', probe_code, str(path)],
                capture_output=True, text=True, timeout=20)
            return dict(path=str(path), returncode=result.returncode, decoded=json.loads(result.stdout) if result.returncode == 0 else None,
                stderr=result.stderr[-2000:])
        except subprocess.TimeoutExpired:
            return dict(path=str(path), returncode=124, stderr='NAS image decode timed out')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        record['image_results'] = list(pool.map(probe, PROBES))
    record.update(readonly_mount_options=mount_options, wall_seconds=time.perf_counter()-started)
    atomic_json(attempt/'NAS_preflight.json', record)
    assert all(row['returncode'] == 0 for row in record['image_results']), record


def install_transport_retry():
    from flex_uda.teacher.decompose_inputs import PairedViews
    frame = PairedViews.frame
    transient = {errno.EIO, errno.ETIMEDOUT, errno.ECONNRESET, errno.ENOTCONN, errno.ENETUNREACH, errno.EHOSTUNREACH}
    def retried(self, *args, **kwargs):
        for attempt in range(4):
            try:
                return frame(self, *args, **kwargs)
            except OSError as error:
                if error.errno not in transient or attempt == 3:
                    raise
                delay = (2, 5, 10)[attempt]
                print(json.dumps(dict(status='RETRY_IDENTICAL_NAS_FRAME_AFTER_TRANSIENT_IO',
                    errno=error.errno, attempt=attempt+1, wait_seconds=delay, path=error.filename)), flush=True)
                time.sleep(delay)
    PairedViews.frame = retried


def worker(gpu, attempt):
    numerical, _ = validate()
    assert attempt.resolve().is_relative_to(OUT.resolve())
    install_transport_retry()
    numerical.OUT = attempt  # Metadata only; numerical.OWN and numerical.PARENT stay fixed.
    numerical.worker(gpu)


def seal_and_evaluate(numerical, plan):
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_fullscope_visual_six import METHODS, PRIMARY, COMPARATORS
    expected = set(METHODS)
    expected.update(PRIMARY+'__AT_'+name+'_COUNT' for name in COMPARATORS)
    expected.update(name+'__AT_'+PRIMARY+'_COUNT' for name in COMPARATORS)
    assert len(expected) == 31
    tasks = []
    for case in plan['cases']:
        path = PARENT/'units'/case['pair']/case['evaluated_video']/'generation_complete.json'
        done = json.loads(path.read_text())
        assert done['GT_open_attempts'] == 0 and done['registration_sha256'] == file_sha(PARENT/'registration.json')
        assert len(done['tasks']) == 31
        assert {task['method'] for task in done['tasks']} == expected
        assert all(task['pair'] == case['pair'] and task['video'] == case['evaluated_video'] for task in done['tasks'])
        tasks.extend(done['tasks'])
    assert len(tasks) == 4588
    assert len({task['folder'] for task in tasks}) == 4588
    for task in tasks:
        verify_predictions(task['folder'])
    sessions = []
    for path in sorted((PARENT/'supervisor').glob('GPU*_teacher_session_*.json')):
        teacher = json.loads(path.read_text())['teacher']
        if teacher is not None:
            assert teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == TEACHER_SHA
            sessions.append(dict(path=str(path), sha256=file_sha(path)))
    assert len(sessions) >= 4
    for gpu in (0, 1, 2, 3):
        teacher = json.loads((PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())['teacher']
        assert teacher is not None and teacher['backend_unmodified'] and teacher['teacher_weight_sha256'] == TEACHER_SHA
    atomic_json(PARENT/'all4588_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0,
        teacher_final_weight_seals=4, all_attempt_teacher_sessions=sessions,
        registration_sha256=file_sha(PARENT/'registration.json'),
        explicit_numerical_recovery_registration_sha256=file_sha(numerical.OWN),
        explicit_NAS_transport_resume_registration_sha256=file_sha(OWN)))
    print('ALL4588_AND_ALL_ATTEMPT_TEACHER_SEALS_BEFORE_ISOLATED_GT_AFTER_NAS_RESUME', flush=True)
    from scripts.a100_fullscope_visual_six import evaluate_and_collect
    evaluate_and_collect(plan, tasks)


def main(preflight_only=False):
    numerical, plan = validate()
    attempt = OUT/('attempt_'+str(time.time_ns())+'_'+str(os.getpid()))
    attempt.mkdir(parents=True)
    started = time.perf_counter()
    preflight_nas(attempt)
    if preflight_only:
        print(json.dumps(dict(status='NAS_PREFLIGHT_PASSED_NO_GPU_WORKERS', attempt=str(attempt))), flush=True)
        return
    owners = {}
    for gpu in (0, 1, 2, 3):
        result = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name',
            '--format=csv,noheader,nounits'], text=True).strip()
        rows = [row.split(',', 1) for row in result.splitlines() if row.strip()]
        allowed = {GPU2_SERVICE_PID} if gpu == 2 else set()
        assert all(int(row[0]) in allowed for row in rows), (gpu, result)
        owners[gpu] = result
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for gpu in (0, 1, 2, 3):
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', str(gpu), '--attempt', str(attempt)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            future = pool.submit(numerical.drain, proc, attempt/f'GPU{gpu}.log')
            jobs.append((gpu, proc, future))
            atomic_json(attempt/f'GPU{gpu}_launch.json', dict(GPU=gpu, pid=proc.pid, previous_owners_preserved=owners[gpu],
                NAS_resume_registration_sha256=file_sha(OWN)))
            print(json.dumps(dict(status='STARTED_UNCHANGED_FULLSCOPE_CHECKPOINT_RESUME_AFTER_NAS_RECOVERY',
                GPU=gpu, pid=proc.pid, attempt=str(attempt))), flush=True)
        failures = []
        for gpu, proc, future in jobs:
            code = proc.wait()
            future.result()
            if code:
                failures.append([gpu, code])
    atomic_json(attempt/'workers_final.json', dict(failures=failures, wall_seconds=time.perf_counter()-started,
        original_source_registration_sha256=file_sha(PARENT/'registration.json'), goal_status='ACTIVE'))
    assert not failures, failures
    seal_and_evaluate(numerical, plan)
    atomic_json(attempt/'completion.json', dict(status='COMPLETE_ORIGINAL_SIX_STUDY_AFTER_NAS_TRANSPORT_RESUME',
        extra_attempt_cost_not_free=True, wall_seconds=time.perf_counter()-started, goal_status='ACTIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=int, choices=(0, 1, 2, 3))
    parser.add_argument('--attempt', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        assert args.attempt is not None
        worker(args.worker, args.attempt)
    else:
        main(args.preflight)
