#!/usr/bin/env python3
"""Four independent read-only evaluations, only after complete coverage training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.launch_uda_workers import audit_gpus

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--parity-only',action='store_true');a=p.parse_args()
    root=a.output_root.resolve();training=a.training_root.resolve();root.mkdir(parents=True,exist_ok=True)
    for method in a.methods:
        receipt=json.loads((training/method/'training_receipt.json').read_text())
        if receipt['status']!='COMPLETE_FIXED_COVERAGE_DIAGNOSTIC' or receipt['steps']!=8973:
            raise ValueError('do not evaluate incomplete/relabelled coverage runs')
    selected,all_gpus,apps=audit_gpus(len(a.methods));workers=[];start=time.perf_counter()
    mode='parity' if a.parity_only else 'evaluation'
    atomic_json(root/f'{mode}_gpu_admission.json',dict(selected=selected,all_gpus=all_gpus,existing_processes=apps,no_foreign_process_killed=True))
    for method,gpu in zip(a.methods,selected):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],UDA_PHYSICAL_GPU=str(gpu['physical_gpu']),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        log=(root/f'{method}_{mode}_stdout.txt').open('a')
        command=[sys.executable,str(ROOT/'scripts/evaluate_uda_full.py'),'--checkpoint',str(training/method/'checkpoint_best_source_selected.pt'),
            '--method',method,'--output-root',str(root)]
        if a.parity_only:command.append('--parity-only')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        workers.append((proc,log,dict(method=method,pid=proc.pid,**gpu)))
    print(json.dumps(dict(event='evaluation_started',workers=[w[2] for w in workers],parity_only=a.parity_only)),flush=True)
    results=[]
    for proc,log,worker in workers:
        code=proc.wait();log.close();row=dict(worker,exit_code=code);results.append(row)
        print(json.dumps(dict(event='evaluation_finished',**row)),flush=True)
    atomic_json(root/f'{mode}_launcher_status.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'FAILED',
        workers=results,wall_seconds=time.perf_counter()-start))
    raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
