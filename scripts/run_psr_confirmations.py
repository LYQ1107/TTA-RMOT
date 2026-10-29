#!/usr/bin/env python3
"""Registered seeds 0/42, unchanged teacher caches and budgets; max four GPUs.

One blocking scheduler owns the physical lanes. Each child reuses the existing
single-GPU trainer and full425 inference driver, preserving completed work.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_uda_workers import audit_gpus
from flex_tta.runtime.pf_receipts import atomic_json


def run(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    _,gpus,processes=audit_gpus(1,minimum_free_mib=20480)
    chosen=sorted([g for g in gpus if g['free_mib']>=20480],
                  key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))[:4]
    if not chosen:raise RuntimeError('no memory-fit GPU; no other process is stopped')
    attempt=root/'launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',dict(chosen=chosen,all_gpus=gpus,processes=processes,
        sharing_authorized=True,maximum_physical_GPUs=4,other_own_GPU_jobs=0,
        immutable_sampling_seed=17,student_seeds=[0,42],optimizer_updates_per_student=8973,
        checkpoint_selection='source validation CE only',target_GT_used=False))
    queue=[(seed,method) for seed in (0,42) for method in ('NAIVE_PAIRED','PSR')]
    lock=threading.Lock();start=time.perf_counter()
    def lane(gpu):
        results=[]
        while True:
            with lock:
                if not queue:break
                seed,method=queue.pop(0)
            command=[sys.executable,str(ROOT/'scripts/run_psr_matrix.py'),'--root',str(root/f'seed{seed}'),
                '--methods',method,'--workers','1','--seed',str(seed),'--gpu-ids',str(gpu['physical_gpu'])]
            record=dict(seed=seed,method=method,physical_gpu=gpu['physical_gpu'],command=command)
            begin=time.perf_counter()
            with (attempt/f'seed{seed}_{method}.txt').open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,bufsize=1)
                record['pid']=process.pid
                print(json.dumps(dict(event='PSR_CONFIRMATION_STARTED',**record)),flush=True)
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith('{'):
                        print(json.dumps(dict(event='PSR_CONFIRMATION_CHILD',seed=seed,method=method,
                            message=json.loads(line))),flush=True)
                code=process.wait()
            record.update(exit_code=code,wall_seconds=time.perf_counter()-begin)
            atomic_json(attempt/f'seed{seed}_{method}.json',record);results.append(record)
            print(json.dumps(dict(event='PSR_CONFIRMATION_FINISHED',**record)),flush=True)
            if code:break
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        for future in as_completed([pool.submit(lane,g) for g in chosen]):results.extend(future.result())
    complete=not queue and all(r['exit_code']==0 for r in results)
    atomic_json(attempt/'launcher_status.json',dict(status='COMPLETE' if complete else 'FAILED_UNITS_RETAINED',
        results=results,pending=queue,wall_seconds=time.perf_counter()-start,
        physical_GPUs=[g['physical_gpu'] for g in chosen],target_GT_opened=False))
    if not complete:raise RuntimeError('resume unfinished confirmation units only; completed outputs are preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/runs/psr_confirm_20261030_v1')
    run(p.parse_args().root)
