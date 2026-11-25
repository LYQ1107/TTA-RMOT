#!/usr/bin/env python3
"""One blocking parent for real pre/post-change compatibility, no polling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json

RUN=ROOT/'results/runs/diag_fit_20261030_v1'


def run():
    folder=RUN/'validation';folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    commands=[('real_zero_alpha',['diag_fit_compatibility.py','--mode','parity','--output',folder/'real_zero_alpha']),
        ('legacy_K1',['diag_fit_compatibility.py','--mode','legacy','--legacy-root',RUN/'legacy_source','--output',folder/'legacy_K1']),
        ('current_K1',['train_psr.py','--method','NAIVE_PAIRED','--scope','smoke','--val-authorized','--config',RUN/'legacy_source/psr.yaml','--output',folder/'current_K1'])]
    receipts=[]
    for name,args in commands:
        completed=folder/(name+'_process.json')
        if completed.exists() and json.loads(completed.read_text())['exit_code']==0:continue
        begin=time.perf_counter()
        with (folder/(name+'_stdout.txt')).open('a') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/args[0]),*map(str,args[1:])],cwd=ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            print(json.dumps(dict(event='D_FIT_VALIDATION_STARTED',name=name,pid=child.pid)),flush=True)
            for line in child.stdout:
                log.write(line);log.flush()
                if line.startswith('{"event":') or line.startswith('{"step":'):print(line.rstrip(),flush=True)
            code=child.wait()
        record=dict(name=name,exit_code=code,seconds=time.perf_counter()-begin)
        atomic_json(completed,record);receipts.append(record)
        print(json.dumps(dict(event='D_FIT_VALIDATION_EXIT',**record)),flush=True)
        if code:raise RuntimeError('inspect the retained first compatibility failure: '+name)
    records={name:[json.loads(s) for s in (folder/name/'metrics_train.jsonl').read_text().splitlines()][:5] for name in ('legacy_K1','current_K1')}
    errors={key:max(abs(a[key]-b[key]) for a,b in zip(records['legacy_K1'],records['current_K1']))
        for key in ('source_loss','target_KD_loss')}
    if any(v>1e-4 for v in errors.values()):raise RuntimeError('actual K1 first5 loss regression: '+str(errors))
    if len(records['legacy_K1'])!=5 or len(records['current_K1'])!=5:raise RuntimeError('missing actual five steps')
    for name in records:
        receipt=json.loads((folder/name/'training_receipt.json').read_text())
        if receipt['target_GT_open_attempts'] or not receipt['frozen_parameters_unchanged']:raise RuntimeError('default training isolation changed')
    atomic_json(folder/'completion.json',dict(status='PASS_REAL_GPU_PARITY_AND_ACTUAL_OLD_NEW_FIRST5_STEPS',
        max_loss_errors=errors,atol=1e-4,target_GT_open_attempts=0,legacy_source_commit='965e71fe5055a4299ae6f46f89dbe4e993cfad11',
        legacy_trainer_source_original_common_default_adapter_and_loss_separately_bitwise_tested=True))
    print(json.dumps(dict(event='D_FIT_COMPATIBILITY_COMPLETE',errors=errors)),flush=True)


if __name__=='__main__':run()
