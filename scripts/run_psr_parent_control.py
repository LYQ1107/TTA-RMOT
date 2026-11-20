#!/usr/bin/env python3
"""Queue the positive-weight control behind one registered confirmation lane."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_uda_workers import audit_gpus
from flex_tta.runtime.pf_receipts import atomic_json


def run(confirm,root):
    confirm,root=Path(confirm).resolve(),Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    attempts=sorted((confirm/'launchers').glob('*/gpu_admission.json'))
    if not attempts:raise ValueError('start the registered confirmation scheduler first')
    attempt=attempts[-1].parent;admission=json.loads(attempts[-1].read_text())
    print(json.dumps(dict(event='PARENT_WEIGHT_WAIT_ONE_CONFIRMATION_LANE_RELEASE',GPU_calls=0)),flush=True)
    while True:
        finished=list(attempt.glob('seed*_*.json'))
        if finished:break
        time.sleep(20)
    completed={json.loads(p.read_text())['physical_gpu'] for p in finished}
    active={r['physical_gpu'] for r in admission['chosen']}-completed
    _,gpus,processes=audit_gpus(1,minimum_free_mib=20480)
    eligible=sorted([g for g in gpus if g['physical_gpu'] not in active and g['free_mib']>=20480],
                    key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))
    if not eligible:raise RuntimeError('no memory-fit released slot; existing confirmation workers untouched')
    gpu=eligible[0]
    atomic_json(root/'gpu_admission.json',dict(selected=gpu,all_gpus=gpus,processes=processes,
        other_confirmation_physical_GPUs=sorted(active),maximum_simultaneous_own_GPUs=4,sharing_authorized=True))
    env=os.environ.copy();env.update(PSR_PHYSICAL_GPU=str(gpu['physical_gpu']),CUDA_VISIBLE_DEVICES=gpu['uuid'],
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
    method='PSR_PARENT_WEIGHT_CONTROL';records=[]
    commands=[(scope,[sys.executable,str(ROOT/'scripts/train_psr.py'),'--method',method,'--scope',scope,
        '--val-authorized','--output',str(root/scope/method)]) for scope in ('interface','smoke')]
    commands.append(('full_training_and_inference',[sys.executable,str(ROOT/'scripts/run_psr_matrix.py'),
        '--root',str(root),'--methods',method,'--workers','1','--seed','17',
        '--gpu-ids',str(gpu['physical_gpu']),'--other-active-gpus',*map(str,sorted(active))]))
    for phase,command in commands:
        receipt=root/phase/method/'training_receipt.json'
        if phase!='full_training_and_inference' and receipt.exists():
            expected={'interface':'INTERFACE_PASS_NO_OPTIMIZER_STEP','smoke':'STUDENT_SMOKE_PASS'}[phase]
            if json.loads(receipt.read_text())['status']==expected:
                records.append(dict(phase=phase,status='PRESERVED',GPU_calls=0));continue
        begin=time.perf_counter()
        with (root/(phase+'_stdout.txt')).open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            print(json.dumps(dict(event='PARENT_WEIGHT_PHASE_STARTED',phase=phase,physical_gpu=gpu['physical_gpu'],pid=child.pid)),flush=True)
            for line in child.stdout:
                log.write(line);log.flush()
                if line.startswith('{"event":') or line.startswith('{"step":'):print(line.rstrip(),flush=True)
            code=child.wait()
        record=dict(phase=phase,exit_code=code,wall_seconds=time.perf_counter()-begin)
        atomic_json(root/(phase+'_completion.json'),record);records.append(record)
        print(json.dumps(dict(event='PARENT_WEIGHT_PHASE_FINISHED',**record)),flush=True)
        if code:raise RuntimeError('fix/resume only failed parent-control phase; all original predictions preserved')
    atomic_json(root/'completion.json',dict(status='COMPLETE_PARENT_WEIGHT_CONTROL',phases=records,
        target_GT_opened=False,independent_new_weak_labels=0,scientific_gain='PENDING_INDEPENDENT_OFFLINE_EVALUATION'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation-root',type=Path,default=ROOT/'results/runs/psr_confirm_20261030_v1')
    p.add_argument('--root',type=Path,default=ROOT/'results/runs/psr_parent_weight_20261030_v1')
    a=p.parse_args();run(a.confirmation_root,a.root)
