"""v8P Stage A, admitted only after the original v8 GPU queue releases slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue,Empty
import subprocess
import sys
import time
from threading import Event
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase');p.add_argument('--pilot-videos',type=int,default=0);p.add_argument('--jobs-json',type=Path,required=True);a=p.parse_args()
    execution,_=authorize_active();plan=json.loads(a.jobs_json.read_text())
    assert execution.manifest['version']=='v8P'
    assert file_sha(plan['prereg_path'])==plan['prereg_sha256']
    assert file_sha(execution.public/'BC_cost_estimate.json')==plan['cost_estimate_sha256']
    assert all(j in execution.manifest['allowed_registered_jobs'] for j in plan['jobs'])
    dependency=Path(execution.manifest['prior_v8_durable_completion'])
    if not dependency.exists():
        print(json.dumps(dict(event='V8P_WAITING_ORIGINAL_V8_SLOTS',completion=str(dependency),maximum_total_project_GPUs=2)),flush=True)
    while not dependency.exists():time.sleep(5)
    previous=json.loads(dependency.read_text())
    atomic_json(execution.public/'original_v8_queue_dependency.json',dict(path=str(dependency),sha256=file_sha(dependency),receipt=previous))
    print(json.dumps(dict(event='ORIGINAL_V8_GPU_SLOTS_RELEASED',receipt=previous)),flush=True)
    def folder(job):return execution.active/job['pair']/job['method']/('rho_'+str(job['rho']))
    jobs=[j for j in plan['jobs'] if not (folder(j)/'predictions_seal.json').exists() and not (a.pilot_videos and (folder(j)/'pilot_receipt.json').exists())]
    if not jobs:
        print(json.dumps(dict(event='ALREADY_COMPLETE_NO_RERUN')),flush=True);return
    jobs.sort(key=lambda j:({'MINUS_SLOW':0,'FULL':1,'SLOW_REPLAY':2}.get(j['method'],3),j['pair']))
    from scripts.select_gpu import query
    all_gpus=query()
    slow_jobs=any(j['method'] in ('FULL','SLOW_REPLAY') for j in jobs)
    minimum=43008 if slow_jobs else 24576
    selected=sorted([g for g in all_gpus if g['free_mib']>=minimum],
        key=lambda g:(g['compute_processes'],-g['free_mib'],g['index']))[:min(2,len(jobs))]
    if not selected:raise RuntimeError('No shared GPU passes registered memory admission; do not evict foreign work')
    root=execution.launchers/('workers_'+str(time.time_ns()));root.mkdir(parents=True)
    atomic_json(root/'admission.json',dict(selected=selected,all_gpus=all_gpus,jobs=jobs,
        required_mib=minimum,maximum_physical_GPUs=2,sharing_authorized=True,foreign_processes_stopped=0))
    queue=Queue()
    failed=Event()
    for job in jobs:queue.put(job)
    def slot(gpu):
        receipts=[]
        while True:
            if failed.is_set():return receipts
            try:job=queue.get_nowait()
            except Empty:return receipts
            if job['method']=='SLOW_REPLAY':
                source=execution.active/job['pair']/'MINUS_SLOW'/('rho_'+str(job['rho']))/'predictions_seal.json'
                if not source.exists():raise RuntimeError('Replay dependency must complete first; never invent answers')
            needed=43008 if job['method'] in ('FULL','SLOW_REPLAY') else 24576
            while next(g for g in query() if g['uuid']==gpu['uuid'])['free_mib']<needed:time.sleep(20)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PRESENCE_PHYSICAL_GPU=str(gpu['index']),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            if job['method'].startswith('YESTRACK'):
                command=[sys.executable,str(ROOT/'scripts/active_baseline_run.py'),'--pair',job['pair'],'--method',job['method'],'--rho',str(job['rho'])]
            else:
                command=[sys.executable,str(ROOT/'scripts/active_v8p_run.py'),'--pair',job['pair'],'--variant',job['method'],'--rho',str(job['rho'])]
            if a.pilot_videos:
                assert job['method']=='MINUS_SLOW' and job['pair'] in ('C1','C2') and a.pilot_videos==2
                command+=['--pilot-videos',str(a.pilot_videos)]
            log=root/(job['pair']+'_'+job['method']+'_rho_'+str(job['rho'])+'.log');start=time.perf_counter()
            with log.open('x') as handle:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                print(json.dumps(dict(event='V8P_JOB_STARTED',job=job,physical_gpu=gpu['index'],pid=child.pid,log=str(log))),flush=True)
                for line in child.stdout:
                    handle.write(line);handle.flush()
                    if line.startswith('{'):print(line,end='',flush=True)
                code=child.wait()
            receipt=dict(job,exit_code=code,physical_gpu=gpu['index'],seconds=time.perf_counter()-start,log=str(log))
            atomic_json(log.with_name(log.stem+'_completion.json'),receipt);receipts.append(receipt)
            if code:
                failed.set()
                raise RuntimeError('Retain completed streams; first v8 error '+str(log))
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:receipts=[r for group in pool.map(slot,selected) for r in group]
    atomic_json(root/'completion.json',dict(status='COMPLETE',jobs=receipts))


if __name__=='__main__':main()
