"""Repair only the two missing-registry workers; preserve healthy GPU jobs."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT,source

STATE=OUT/'supervisor/lamot_recovery_state.json'


def one(job):
    pair,gpu=job;log=OUT/'supervisor'/('recovery_'+pair+'.log');start=time.perf_counter()
    with log.open('a',buffering=1) as h:
        child=subprocess.Popen([sys.executable,'-u','scripts/a100_six_direction.py','--pair',pair,'--collect'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        code=child.wait()
    if code:raise RuntimeError(pair+' '+log.read_text()[-7000:])
    completion=json.loads((source(pair)/'completion.json').read_text());assert completion['GT_open_attempts']==0
    return dict(pair=pair,physical_GPU=gpu,actual_retry_process_wall_seconds=time.perf_counter()-start,log=str(log))


def main():
    jobs=[('C5',2),('C6',3)];pending=list(jobs);done=[]
    atomic_json(STATE,dict(status='WAITING_RELEASED_GPU2_3_FOR_LAMOT_ONLY',pid=os.getpid(),completed=0,goal_status='ACTIVE'))
    # Preflight is internal to the job allocator. No existing process is killed.
    while pending:
        rows=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.strip().splitlines()}
        if all(memory[g]<512 for _,g in pending):break
        time.sleep(3)
    atomic_json(OUT/'supervisor/lamot_recovery_allocation.json',dict(used_MiB=memory,only_start_below_MiB=512))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,j) for j in jobs]
        atomic_json(STATE,dict(status='RUNNING_LAMOT_REGISTRY_RECOVERY_NO_GT',pid=os.getpid(),completed=0,goal_status='ACTIVE'))
        for f in concurrent.futures.as_completed(futures):
            done.append(f.result());atomic_json(STATE,dict(status='RUNNING_LAMOT_REGISTRY_RECOVERY_NO_GT',pid=os.getpid(),completed=done,goal_status='ACTIVE'))
    atomic_json(STATE,dict(status='PASS_LAMOT_REGISTRY_RECOVERY_EVIDENCE_SEALED',pid=os.getpid(),completed=done,goal_status='ACTIVE'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_LAMOT_RECOVERY_REQUIRES_ROOT_CAUSE',pid=os.getpid(),error=repr(error),goal_status='ACTIVE'));raise
