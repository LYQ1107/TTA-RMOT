#!/usr/bin/env python3
"""Four independent one-GPU methods; one blocking wait, no foreign kills."""
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json

CONFIGS=('source_rehearsal','mean_teacher','weak_mil','combined')
METHODS=('SOURCE_REHEARSAL','MEAN_TEACHER_UDA','WEAK_MIL_UDA','COMBINED_UDA')


def audit_gpus(required=4):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    counts=Counter(line.split(',')[0].strip() for line in apps.splitlines() if line.strip())
    rows=[]
    for fields in csv.reader(io.StringIO(text)):
        index,uuid,name,free,total=[x.strip() for x in fields]
        rows.append(dict(physical_gpu=int(index),uuid=uuid,name=name,free_mib=int(free),total_mib=int(total),processes=counts[uuid]))
    explicit=[x.strip() for x in os.environ.get('UDA_GPU_IDS','').split(',') if x.strip()]
    if explicit:
        selected=[next(g for g in rows if x in (str(g['physical_gpu']),g['uuid'])) for x in explicit]
    else:selected=sorted([g for g in rows if g['free_mib']>=32768],key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))[:required]
    if len(selected)!=required or len({g['uuid'] for g in selected})!=required:raise RuntimeError(f'{required} distinct memory-fit physical GPUs required')
    if any(g['free_mib']<32768 for g in selected):raise RuntimeError('selected GPU lacks24GiB budget plus8GiB admission headroom')
    return selected,rows,apps


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scope',choices=('plan','smoke','full'),default='plan')
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--launcher-record-dir',type=Path,help='separate admission/wait receipts when repairing a finished worker alongside healthy peers')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    a=p.parse_args();root=a.output_root.resolve();record_root=(a.launcher_record_dir or root).resolve()
    selected,all_gpus,apps=audit_gpus(len(a.methods))
    configurations=dict(zip(METHODS,CONFIGS))
    plan=[dict(method=m,config=configurations[m],**g) for m,g in zip(a.methods,selected)]
    print(json.dumps(dict(scope=a.scope,workers=plan),indent=2),flush=True)
    if a.scope=='plan':raise SystemExit(0)
    if os.environ.get('UDA_VAL_DIAGNOSTIC_AUTHORIZED')!='YES':raise PermissionError('explicit Val diagnostic authorization required')
    for method in METHODS:
        if (root/method).exists() and not a.resume:raise FileExistsError('preserve existing run: '+str(root/method))
    root.mkdir(parents=True,exist_ok=True)
    record_root.mkdir(parents=True,exist_ok=True)
    if a.resume:
        history=record_root/'launcher_attempts'/str(time.time_ns());history.mkdir(parents=True)
        for name in ('gpu_admission.json','launcher_status.json'):
            if (record_root/name).exists():shutil.copy2(record_root/name,history/name)
    atomic_json(record_root/'gpu_admission.json',dict(selected_workers=plan,all_gpus=all_gpus,existing_processes=apps,
        sharing_authorized=True,no_foreign_process_killed=True,per_worker_allocator_budget_mib=24576,
        admission_headroom_mib=8192,resource_snapshot_is_not_reservation=True))
    children=[];started=time.perf_counter()
    atomic_json(record_root/'launcher_status.json',dict(status='RUNNING',workers=plan,scope=a.scope))
    for worker in plan:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=worker['uuid'],UDA_PHYSICAL_GPU=str(worker['physical_gpu']),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        path=root/(worker['method']+'_stdout.txt');handle=path.open('a' if a.resume else 'w')
        cmd=[sys.executable,str(ROOT/'scripts/train_flexhook_uda.py'),'--config',str(ROOT/'configs/uda'/(worker['config']+'.yaml')),
            '--output',str(root/worker['method']),'--scope',a.scope]
        if a.resume:cmd.append('--resume')
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        children.append((process,handle,worker));worker['pid']=process.pid
    results=[]
    # Parent owns the ONLY completion wait; do not poll via agent ps/tail.
    for process,handle,worker in children:
        code=process.wait();handle.close();result=dict(worker,exit_code=code)
        results.append(result);print(json.dumps(dict(event='worker_finished',**result)),flush=True)
    status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'FAILED_WORKERS_RETAINED_FOR_TARGETED_RESUME'
    atomic_json(record_root/'launcher_status.json',dict(status=status,workers=results,wall_seconds=time.perf_counter()-started,scope=a.scope))
    raise SystemExit(0 if status=='COMPLETE' else 1)
