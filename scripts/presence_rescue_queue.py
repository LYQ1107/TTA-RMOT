"""Recover lost export slots without stopping any healthy original worker.

Only the two original physical GPUs are eligible. Nonblocking bank locks and
inherited lock descriptors avoid duplicate GPU work; completed video seals
stay immutable. All resource checks and waits are internal to this command.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--feature-admission', type=Path, required=True); a = p.parse_args()
    from scripts.select_gpu import query
    registered = json.loads(a.feature_admission.read_text())['selected']
    if len(registered) > 2: raise ValueError('At most two registered physical GPUs')
    out = RUN / 'rescue_launchers' / str(time.time_ns()); out.mkdir(parents=True)
    own = [(t, m) for t, m in IN_DOMAIN.items()]
    jobs = own + [j for j in reversed([(t, m) for t in IN_DOMAIN for m in ('V2-best', 'Dance-best', 'LaMOT-best')]) if j not in own]
    def root(job): return RUN / 'features' / job[0] / job[1]
    def complete(job): return (root(job) / 'export_receipt.json').exists()
    atomic_json(out / 'plan.json', dict(registered_GPUs=registered, jobs=jobs,
        only_incomplete_banks=True, immutable_completed_videos=True, foreign_processes_stopped=0,
        original_healthy_workers_not_restarted=True))
    def slot(gpu):
        records = []
        while True:
            pending = [j for j in jobs if not complete(j)]
            if not pending: return records
            current = next(g for g in query() if g['uuid'] == gpu['uuid'])
            if current['free_mib'] < 20480:
                time.sleep(20); continue
            chosen = lock = None
            for job in pending:
                root(job).mkdir(parents=True, exist_ok=True)
                candidate = (root(job) / '.export.lock').open('a')
                try: fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close(); continue
                if complete(job): candidate.close(); continue
                chosen, lock = job, candidate; break
            if chosen is None:
                time.sleep(20); continue
            target, model = chosen
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['physical_gpu']),
                PRESENCE_EXPORT_LOCK_FD=str(lock.fileno()), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            log = out / (target+'_'+model+'.log'); start = time.perf_counter()
            atomic_json(out / (target+'_'+model+'_admission.json'), dict(current=current,
                maximum_physical_GPUs=2, registered_GPUs=registered, allocator_cap_mib=16384))
            try:
                with log.open('a') as stream:
                    child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/presence_export_target.py'), '--target', target, '--model', model],
                        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, pass_fds=(lock.fileno(),))
                    print(json.dumps(dict(event='RESCUE_EXPORT_STARTED', target=target, model=model,
                        physical_gpu=gpu['physical_gpu'], pid=child.pid, log=str(log))), flush=True)
                    for line in child.stdout:
                        stream.write(line); stream.flush()
                        if line.startswith('{'): print(line, end='', flush=True)
                    code = child.wait()
            finally: lock.close()
            row = dict(target=target, model=model, exit_code=code, wall_seconds=time.perf_counter()-start,
                       physical_gpu=gpu['physical_gpu'], log=str(log))
            records.append(row); atomic_json(out / (target+'_'+model+'_completion.json'), row)
            if code: raise RuntimeError('Export recovery failed; completed shards preserved: '+str(log))
    with ThreadPoolExecutor(max_workers=len(registered)) as pool:
        records = [r for rr in pool.map(slot, registered) for r in rr]
    atomic_json(out / 'completion.json', dict(status='ALL_NINE_TARGET_BANKS_COMPLETE', workers=records))


if __name__ == '__main__': main()
