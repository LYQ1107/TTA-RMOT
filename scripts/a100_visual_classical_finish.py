"""Generate both domains, then count controls, then isolated official metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
OUT=ROOT/'a100_runs/visual_classical_20261121'


def phase(script,*,gpu_jobs):
    folder=OUT/'supervisor';children=[]
    for pair,gpu in [('C4',2),('C1',3)]:
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1')
        if gpu_jobs:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        log=(folder/(script+'__'+pair+'.log')).open('a')
        cmd=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts'/script),'--pair',pair]
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((pair,gpu,child,log))
    receipts=[]
    for pair,gpu,child,log in children:
        code=child.wait();log.close();receipts.append(dict(pair=pair,gpu=gpu if gpu_jobs else None,exit_code=code))
    atomic_json(folder/(script+'.json'),dict(jobs=receipts))
    assert all(r['exit_code']==0 for r in receipts),receipts


def main():
    begin=time.perf_counter()
    for pair in ('C4','C1'):assert (OUT/'evidence'/pair/'completion.json').is_file()
    phase('a100_visual_classical_generate.py',gpu_jobs=True)
    phase('a100_visual_classical_frame_kd.py',gpu_jobs=True)
    phase('a100_visual_classical_countmatched.py',gpu_jobs=False)
    # Ground truth is first opened below, after ALL methods in both domains
    # are sealed. Evaluation cannot affect any first-round method parameter.
    phase('a100_visual_classical_evaluate.py',gpu_jobs=False)
    atomic_json(OUT/'supervisor/finish_completion.json',dict(status='PASS_ALL_FIRST_ROUND_UNITS',
        wall_seconds=time.perf_counter()-begin))
    print(json.dumps(dict(event='FIRST_ROUND_COMPLETE',wall_seconds=time.perf_counter()-begin)),flush=True)


if __name__=='__main__':main()
