#!/usr/bin/env python3
"""Independent frozen teachers; shared exact-input cache and one parent wait."""
import argparse
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


def launch(pseudo_root,output,smoke=False):
    pseudo_root,output=Path(pseudo_root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    policies=['PSR'] if smoke else ['NAIVE','PSR','UNIFORM','CONFIDENCE']
    pending=[]
    for policy in policies:
        done=output/policy/'receipt.json'
        if done.exists():
            saved=json.loads(done.read_text())
            if saved['status']==('FUNCTIONALITY_SMOKE_ONLY' if smoke else 'SEALED_FIXED_PROBE'):continue
        pending.append(policy)
    if not pending:print(json.dumps(dict(status='PRESERVED_ALL_COMPLETED_TEACHERS')));return
    gpus,all_gpus,processes=audit_gpus(len(pending),minimum_free_mib=12288)
    attempt=output/'launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',dict(chosen=gpus,all_gpus=all_gpus,processes=processes,sharing_authorized=True,
        maximum_task_physical_GPUs=4,teacher_allocator_GiB=8,free_headroom_GiB=4))
    children=[];started=time.perf_counter()
    for policy,gpu in zip(pending,gpus):
        folder=output/policy;log=(output/(policy+'_stdout.txt')).open('a')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu['physical_gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        command=['/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python',str(ROOT/'scripts/run_teacher_probe.py'),
            '--config',str(ROOT/'configs/teacher/base_qwen2b.yaml'),'--manifest',str(pseudo_root/policy/'manifest.jsonl'),
            '--staging-root',str(pseudo_root/'inputs'),'--method','YES_FULL','--output-root',str(folder),
            '--cache','/data2/user/TTA-RMOT-assets/psr/cache/raw.sqlite','--memory-budget-gib','8','--batch-size','8']
        if smoke:command+=['--limit','4']
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        record=dict(policy=policy,pid=process.pid,physical_gpu=gpu['physical_gpu'],uuid=gpu['uuid'],start_epoch=time.time())
        def relay(proc=process,handle=log,row=record):
            for line in proc.stdout:
                handle.write(line);handle.flush()
                if line.startswith('{"method":'):print(json.dumps(dict(event='teacher_progress',policy=row['policy'],message=json.loads(line))),flush=True)
            row['end_epoch']=time.time()
        thread=threading.Thread(target=relay);thread.start();children.append((process,log,thread,record))
    print(json.dumps(dict(event='PSR_TEACHER_WORKERS_STARTED',workers=[r[3] for r in children],smoke=smoke)),flush=True)
    results=[]
    for process,log,thread,record in children:
        code=process.wait();thread.join();log.close();record['exit_code']=code;results.append(record)
        print(json.dumps(dict(event='PSR_TEACHER_WORKER_FINISHED',**record)),flush=True)
    result=dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'FAILED_UNITS_PRESERVED',workers=results,wall_seconds=time.perf_counter()-started,smoke=smoke)
    atomic_json(attempt/'launcher_status.json',result)
    if result['status']!='COMPLETE':raise RuntimeError('repair only failed teacher units; keep caches/completed outputs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pseudo-root',type=Path,default=Path('/data2/user/TTA-RMOT-assets/psr/val_v1'));p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();launch(a.pseudo_root,a.output,a.smoke)
