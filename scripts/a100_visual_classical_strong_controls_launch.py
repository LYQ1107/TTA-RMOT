"""Persistent companion supervisor for strengthened KD quantity/stats controls."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def job(item):
    name,args=item
    with (OUT/'supervisor'/('recovery_'+name+'.log')).open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    assert code==0,(name,code)


def main():
    begin=time.perf_counter();groups=['strong_'+p+'_seed0' for p in ('cover','uniform')]
    state=OUT/'supervisor/strong_controls_state.json'
    atomic_json(state,dict(status='RUNNING_COUNT_CONTROLS',pid=os.getpid()))
    jobs=[('strong_counts_'+p+'_'+pair,['scripts/a100_visual_classical_strong_countmatched.py','--pair',pair,'--group',g])
        for p,g in zip(('cover','uniform'),groups) for pair in ('C4','C1')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,jobs))
    atomic_json(state,dict(status='EVALUATING_COUNT_CONTROLS',pid=os.getpid()))
    for group in groups:
        job((group+'_counts_evaluate',['scripts/a100_visual_classical_evaluate_group.py','--group',group+'_counts',
            '--primary','COHORT__AT_COHORT_COUNT','--workers','4']))
    job(('strong_analysis',['scripts/a100_visual_classical_strong_analyze.py']))
    atomic_json(state,dict(status='PASS_STRONG_COUNT_CONTROLS_AND_ANALYSIS',wall_seconds=time.perf_counter()-begin,pid=os.getpid()))


if __name__=='__main__':main()
