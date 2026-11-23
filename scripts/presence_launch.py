"""Two admitted physical GPUs at most; one blocking launcher, video recovery."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['target-smoke', 'target-full', 'source-smoke', 'source-full', 'A-export'], required=True)
    p.add_argument('--max-gpus', type=int, choices=[1, 2], default=2)
    a = p.parse_args()
    from scripts.launch_uda_workers import audit_gpus
    from flex_uda.presence.runtime import RUN, IN_DOMAIN
    if a.phase.startswith('source'):
        jobs = [(target, model, 'source') for target, model in IN_DOMAIN.items()]
    elif a.phase == 'target-smoke':
        jobs = [(target, model, 'target') for target, model in IN_DOMAIN.items()]
    else:
        jobs = [(target, model, 'target') for target in IN_DOMAIN for model in ('V2-best', 'Dance-best', 'LaMOT-best')]
        if a.phase == 'A-export':
            # Source heads can train as soon as their own-domain feature pair
            # completes, without waiting for the cross-domain feature banks.
            jobs = [(t, m, 'source') for t, m in IN_DOMAIN.items()] + jobs
    marker = 'smoke_receipt.json' if a.phase.endswith('smoke') else 'export_receipt.json'
    jobs = [(t, m, k) for t, m, k in jobs
            if not (RUN / 'features' / ('source_train' if k == 'source' else '') / t / m / marker).exists()]
    if not jobs:
        print(json.dumps(dict(status='ALREADY_COMPLETE_NO_RERUN', phase=a.phase)), flush=True)
        return
    selected, all_gpus, apps = audit_gpus(min(a.max_gpus, len(jobs)), minimum_free_mib=20480)
    out = RUN / 'launchers' / str(time.time_ns()); out.mkdir(parents=True)
    atomic_json(out / 'admission.json', dict(selected=selected, all_gpus=all_gpus,
        processes=apps, phase=a.phase, max_GPUs=2, sharing_authorized=True, foreign_processes_stopped=0))
    queue = Queue()
    for job in jobs: queue.put(job)
    def slot(gpu):
        result = []
        while True:
            try: target, model, kind = queue.get_nowait()
            except Empty: return result
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['physical_gpu']),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1', PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            command = [sys.executable, str(ROOT / ('scripts/presence_export_source.py' if kind == 'source' else 'scripts/presence_export_target.py')), '--target', target]
            if kind == 'target': command.extend(['--model', model])
            if a.phase.endswith('smoke'): command.append('--smoke')
            log = out / (kind + '_' + target + '_' + model + '.txt'); start = time.perf_counter()
            with log.open('x') as handle:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print(json.dumps(dict(event='STARTED', target=target, model=model, gpu=gpu['physical_gpu'], pid=child.pid, log=str(log))), flush=True)
                for line in child.stdout:
                    handle.write(line); handle.flush()
                    if line.startswith('{'): print(line, end='', flush=True)
                code = child.wait()
            result.append(dict(target=target, model=model, returncode=code, seconds=time.perf_counter()-start, log=str(log)))
            if code:
                raise RuntimeError(f'{target}/{model} failed; first root cause in {log}; preserve completed shards')
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        receipts = [r for result in pool.map(slot, selected) for r in result]
    atomic_json(out / 'launcher_receipt.json', dict(status='COMPLETE', phase=a.phase, workers=receipts))


if __name__ == '__main__': main()
