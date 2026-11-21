"""Run the fixed sparse-evidence/observation study on physical GPUs 0..3.

This controller does not evaluate GT. Every new condition must finish sealing
before a separate post-seal analysis starts. Failed conditions are preserved.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_budget_prefix import BUDGET, prepare_evidence
from scripts.a100_visual_classical_observation_prepare import OBSERVATION

SUPERVISOR = OUT / 'supervisor'
STATE = SUPERVISOR / 'sparse_observation_state.json'


def conditions():
    jobs = []
    def budget(pair, rho, seed):
        jobs.append(dict(kind='BUDGET', pair=pair, fraction=rho, seed=seed,
            folder=str(BUDGET / f'rho{rho:g}_seed{seed}' / pair),
            args=['scripts/a100_visual_classical_budget_prefix.py', '--pair', pair,
                  '--fraction', str(rho), '--seed', str(seed)]))
    def observation(pair, seed):
        jobs.append(dict(kind='OBSERVATION', pair=pair, seed=seed,
            folder=str(OBSERVATION / f'seed{seed}' / pair),
            args=['scripts/a100_visual_classical_observation_prefix.py', '--pair', pair, '--seed', str(seed)]))
    for pair in ('C4', 'C1'):
        budget(pair, .1, 0)
    for pair in ('C4', 'C1'):
        observation(pair, 0)
    for rho in (.05, .2):
        for pair in ('C4', 'C1'):
            budget(pair, rho, 0)
    for seed in (17, 42):
        for pair in ('C4', 'C1'):
            budget(pair, .1, seed)
            observation(pair, seed)
    return jobs


def main():
    audit = json.loads((OUT / 'analysis/benchmark_scope_audit.json').read_text())
    assert audit['status'] == 'PASS_BASELINE_SCOPE_AND_UNCACHED_EVALUATION_AUDIT'
    prereg = ROOT / 'docs/a100_visual_classical_20261121/preregistration_sparse_observation_v1.json'
    assert prereg.is_file()
    begin = time.perf_counter()
    source_files = ['flex_uda/active/observation_memory.py',
        'scripts/a100_visual_classical_observation_prepare.py',
        'scripts/a100_visual_classical_observation_prefix.py',
        'scripts/a100_visual_classical_budget_prefix.py', __file__]
    code = {p: file_sha(ROOT / p) for p in source_files}
    pending = conditions()
    active, completed, failure = {}, [], None
    def state(status):
        atomic_json(STATE, dict(status=status, pid=os.getpid(), pending=pending,
            active={str(g): record for g, (_, _, record) in active.items()},
            completed=completed, total_conditions=16, failure=failure,
            preregistration_sha256=file_sha(prereg), source_sha256=code,
            wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE'))
    state('PREPARING_FIXED_SPARSE_EVIDENCE')
    for pair in ('C4', 'C1'):
        for rho in (.05, .1, .2):
            prepare_evidence(pair, rho)
    prep_jobs = json.loads((SUPERVISOR / 'observation_prepare_launch.json').read_text())['jobs']
    prep_by_pair = {j['pair']: j for j in prep_jobs}
    while pending or active:
        changed = False
        for gpu, (process, handle, record) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            handle.close()
            record = dict(record, exit_code=rc)
            completion = Path(record['folder']) / 'completion.json'
            if rc or not completion.is_file():
                failure = record
            else:
                receipt = json.loads(completion.read_text())
                assert receipt['GT_open_attempts'] == 0
                record['sealed_methods'] = len(receipt['methods'])
            completed.append(record)
            del active[gpu]
            changed = True
            print(json.dumps(dict(event='SPARSE_OBSERVATION_JOB_DONE', **record)), flush=True)
        if not failure:
            for gpu in range(4):
                if gpu in active:
                    continue
                chosen = None
                for index, record in enumerate(pending):
                    completion = Path(record['folder']) / 'completion.json'
                    if completion.is_file():
                        receipt = json.loads(completion.read_text())
                        assert receipt['GT_open_attempts'] == 0
                        completed.append(dict(record, reused_sealed_condition=True, exit_code=0,
                            sealed_methods=len(receipt['methods'])))
                        chosen = index
                        break
                    if record['kind'] == 'OBSERVATION':
                        ready = OBSERVATION / 'inputs' / record['pair'] / 'completion.json'
                        if not ready.is_file():
                            job = prep_by_pair[record['pair']]
                            proc = Path('/proc') / str(job['pid']) / 'stat'
                            dead = not proc.is_file() or proc.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
                            if dead:
                                failure = dict(kind='OBSERVATION_PREPARATION', **job)
                                break
                            continue
                    chosen = index
                    log = SUPERVISOR / ('sparse_' + record['kind'].lower() + '_' + record['pair'] +
                        (f"_rho{record['fraction']:g}" if 'fraction' in record else '') + f"_seed{record['seed']}.log")
                    handle = log.open('a', buffering=1)
                    process = subprocess.Popen([sys.executable]+record['args'], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                    record = dict(record, GPU=gpu, pid=process.pid, log=str(log))
                    active[gpu] = process, handle, record
                    print(json.dumps(dict(event='SPARSE_OBSERVATION_JOB_START', **record)), flush=True)
                    break
                if chosen is not None:
                    del pending[chosen]
                    changed = True
                if failure:
                    break
        if changed or failure:
            state('STOPPING_AFTER_FAILURE_PRESERVE_OTHER_RUNNING_JOBS' if failure else 'RUNNING_FIXED_SPARSE_OBSERVATION')
        if failure and not active:
            state('FAILED_SPARSE_OBSERVATION_REQUIRES_ROOT_CAUSE')
            raise RuntimeError('Preserved failed unit: '+json.dumps(failure))
        if pending or active:
            time.sleep(3)
    assert len(completed) == 16 and all(r['exit_code'] == 0 for r in completed)
    state('ALL_SPARSE_OBSERVATION_PREDICTIONS_SEALED_NO_NEW_GT_YET')
    print(json.dumps(dict(event='SPARSE_OBSERVATION_GENERATION_COMPLETE', conditions=16,
        methods=sum(r['sealed_methods'] for r in completed), goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if STATE.is_file():
            record = json.loads(STATE.read_text())
        else:
            record = dict(pid=os.getpid(), goal_status='ACTIVE')
        record.update(status='FAILED_SPARSE_OBSERVATION_REQUIRES_ROOT_CAUSE', error=repr(exc))
        atomic_json(STATE, record)
        raise
