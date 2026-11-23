#!/usr/bin/env python3
"""Memory-admitted queued student jobs, at most four physical GPUs, resumable."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_uda_workers import audit_gpus
from scripts.train_psr import METHODS
from flex_tta.runtime.pf_receipts import atomic_json


def launch(root,methods,scope='full',workers=4,resume=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);pending=[];preserved=[]
    expected={'interface':'INTERFACE_PASS_NO_OPTIMIZER_STEP','smoke':'STUDENT_SMOKE_PASS','full':'COMPLETE_FIXED_COVERAGE_PSR_STUDY'}[scope]
    for method in methods:
        receipt=root/method/'training_receipt.json'
        if receipt.exists() and json.loads(receipt.read_text())['status']==expected:
            preserved.append(method);continue
        pending.append(method)
    if not pending:print(json.dumps(dict(status='PRESERVED_COMPLETE_STUDENTS',methods=preserved)));return
    _,all_gpus,processes=audit_gpus(1,minimum_free_mib=20480)
    chosen=sorted([g for g in all_gpus if g['free_mib']>=20480],key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))[:min(workers,4,len(pending))]
    if not chosen:raise RuntimeError('no memory-fit GPU; existing completed units preserved')
    attempt=root/'launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',dict(chosen=chosen,all_gpus=all_gpus,processes=processes,sharing_authorized=True,
        allocator_GiB=16,headroom_GiB=4,maximum_physical_GPUs=4))
    lock=threading.Lock();queue=list(pending);started=time.perf_counter()
    def worker(gpu):
        results=[]
        while True:
            with lock:
                if not queue:break
                method=queue.pop(0)
            folder=root/method;env=os.environ.copy();env.update(PSR_PHYSICAL_GPU=str(gpu['physical_gpu']))
            command=[sys.executable,str(ROOT/'scripts/train_psr.py'),'--method',method,'--output',str(folder),'--scope',scope,'--val-authorized']
            if (folder/'checkpoint_latest.pt').exists() or resume:command.append('--resume')
            start=time.perf_counter();record=dict(method=method,physical_gpu=gpu['physical_gpu'],uuid=gpu['uuid'],scope=scope)
            with (root/(method+'_stdout.txt')).open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);record['pid']=process.pid
                print(json.dumps(dict(event='PSR_STUDENT_STARTED',**record)),flush=True)
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith('{"step":'):
                        item=json.loads(line)
                        if item['step']%1000==0 or scope!='full':print(json.dumps(dict(event='PSR_STUDENT_PROGRESS',method=method,**item)),flush=True)
                code=process.wait()
            record.update(exit_code=code,elapsed_seconds=time.perf_counter()-start);results.append(record)
            print(json.dumps(dict(event='PSR_STUDENT_FINISHED',**record)),flush=True)
            if code:break
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in chosen]
        # One blocking parent wait. Worker output is streamed; no agent polling.
        for future in as_completed(futures):results.extend(future.result())
    status='COMPLETE' if not queue and all(r['exit_code']==0 for r in results) else 'FAILED_UNITS_RETAINED'
    result=dict(status=status,scope=scope,workers=results,preserved=preserved,pending=queue,wall_seconds=time.perf_counter()-started)
    atomic_json(attempt/'launcher_status.json',result)
    if status!='COMPLETE':raise RuntimeError('inspect failed method logs; resume only failed/unfinished units')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--methods',nargs='+',choices=METHODS,default=['NAIVE_PAIRED','PSR'])
    p.add_argument('--scope',choices=('interface','smoke','full'),default='full');p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args();launch(a.output,a.methods,a.scope,a.workers,a.resume)
