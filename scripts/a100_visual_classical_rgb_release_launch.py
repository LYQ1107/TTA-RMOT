"""Reuse our released GPUs1/2 for the two remaining preregistered RGB seeds."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_visual_classical_prepare import OUT

SUP=OUT/'supervisor';STATE=SUP/'strict_rgb_release_state.json'
SUCCESS='PASS_RGB_RELEASE_HELPER_RESEARCH_GOAL_ACTIVE'


def main():
    begin=time.perf_counter();done=[]
    amend=ROOT/'docs/a100_visual_classical_20261121/strict_rgb_execution_resource_amendment_v2.json'
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),completed_jobs=done,
            resource_amendment_sha256=file_sha(amend),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    while True:
        sources=[json.loads((SUP/(name+'_state.json')).read_text()) for name in ('final_single_clip_mean','final_single_clip_mean_cohort')]
        if any(r['status'].startswith('FAILED_') for r in sources):raise RuntimeError('An independent source job failed; do not occupy its GPU.')
        if all(r['status']=='PASS_FRESH_BUDGET_PAIR_RESEARCH_GOAL_ACTIVE' for r in sources):break
        state('WAITING_FOR_OUR_INDEPENDENT_SOURCE_GPUS_TO_RELEASE');time.sleep(15)
    state('RUNNING_EXISTING_C4_RGB_SEEDS_ON_RELEASED_GPUS')
    def one(gpu,seed):
        log=SUP/f'strict_rgb_release_C4_seed{seed}.log'
        with log.open('a') as handle:
            process=subprocess.run([sys.executable,'-u','scripts/a100_visual_classical_rgb_parity.py','--pair','C4','--seed',str(seed)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        if process.returncode:raise RuntimeError(str(log)+'\n'+log.read_text()[-6000:])
        return dict(pair='C4',seed=seed,requested_GPU=gpu,log=str(log),exit_code=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,g,s) for g,s in ((1,17),(2,42))]
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result());state('RUNNING_EXISTING_C4_RGB_SEEDS_ON_RELEASED_GPUS')
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_RGB_RELEASE_HELPER_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
