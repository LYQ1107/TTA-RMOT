#!/usr/bin/env python3
"""One blocking parent per preparation stage, <=4 shared memory-fit GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl,write_jsonl,official_splits
from flex_uda.data.train2val import ASSETS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.launch_uda_workers import audit_gpus

RUN=ROOT/'results/runs/train2val_20261031'
TEACHER_PY='/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python'


def child(command,gpu,log,tag):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['physical_gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    begin=time.perf_counter();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as handle:
        p=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps(dict(event='T2V_PREP_STARTED',tag=tag,pid=p.pid,physical_gpu=gpu['physical_gpu'])),flush=True)
        for line in p.stdout:
            handle.write(line);handle.flush()
            if line.startswith('{"event":') or line.startswith('{"method":') or 'Traceback' in line or 'Error:' in line:
                print(tag+' '+line.rstrip(),flush=True)
        code=p.wait()
    result=dict(tag=tag,physical_gpu=gpu['physical_gpu'],pid=p.pid,exit_code=code,worker_seconds=time.perf_counter()-begin,log=str(log))
    print(json.dumps(dict(event='T2V_PREP_EXIT',**result)),flush=True)
    return result


def main(phase,workers):
    started=time.perf_counter();attempt=RUN/'prep_launchers'/phase/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=True)
    # Admission can wait internally; the agent does not issue GPU/process polls.
    while True:
        try:
            chosen,all_gpus,processes=audit_gpus(workers,minimum_free_mib=12288);break
        except RuntimeError as e:
            print(json.dumps(dict(event='T2V_PREP_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    atomic_json(attempt/'admission.json',dict(selected=chosen,all_gpus=all_gpus,processes=processes,
        sharing_authorized=True,maximum_task_GPUs=4,allocator_GiB=8,headroom_GiB=4))
    jobs=[]
    if phase=='scores':
        videos=sorted(official_splits()['dance']['train'])
        for i,gpu in enumerate(chosen):
            command=[sys.executable,ROOT/'scripts/train2val_sparse_scores.py','--videos',*videos[i::len(chosen)],'--output',ASSETS/'frozen_sparse']
            jobs.append((command,gpu,attempt/f'scores_{i}.log',f'scores_{i}'))
    else:
        manifest=ASSETS/'pseudo/NAIVE/manifest.jsonl';rows=read_jsonl(manifest);stage=ASSETS/'pseudo/inputs'
        if phase=='teacher-smoke':
            chunks=[rows];chosen=chosen[:1]
        else:
            from scripts.launch_teacher_probe_sharded import whole_video_shards
            chunks=whole_video_shards(rows,len(chosen))
        manifests=[];phase_root=RUN/('teacher_smoke' if phase=='teacher-smoke' else 'teacher_full')
        for i,(chunk,gpu) in enumerate(zip(chunks,chosen)):
            path=phase_root/'shards'/f'shard{i:02d}'/'manifest.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if read_jsonl(path)!=chunk:raise ValueError('preserve shard assignment')
            else:write_jsonl(path,chunk)
            manifests.append(path);folder=path.parent/'YES_FULL'
            done=folder/'receipt.json'
            if done.exists():
                old=json.loads(done.read_text())
                if old['predictions_sha256']!=file_sha(folder/'teacher_outputs.jsonl'):raise ValueError('changed teacher shard')
                continue
            command=[TEACHER_PY,ROOT/'scripts/run_teacher_probe.py','--config',ROOT/'configs/teacher/base_qwen2b.yaml',
                '--manifest',path,'--staging-root',stage,'--method','YES_FULL','--output-root',folder,
                '--cache',ASSETS/'cache/raw.sqlite','--memory-budget-gib','8','--batch-size','8']
            if phase=='teacher-smoke':command+=['--limit','4']
            jobs.append((command,gpu,attempt/f'teacher_{i}.log',f'{phase}_{i}'))
        atomic_json(phase_root/'shard_plan.json',dict(videos=[[r['video_id'] for r in c[::20]] for c in chunks],
            manifests=[dict(path=str(p),sha256=file_sha(p)) for p in manifests],whole_video_shards=True))
    results=[]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        for f in as_completed([pool.submit(child,*args) for args in jobs]):results.append(f.result())
    failed=any(r['exit_code'] for r in results)
    result=dict(status='FAILED_UNITS_RETAINED' if failed else 'COMPLETE',phase=phase,workers=results,
        actual_wall_seconds=time.perf_counter()-started,maximum_physical_GPUs=4)
    atomic_json(attempt/'completion.json',result)
    if failed:raise RuntimeError('inspect first actionable failed unit; completed units/caches retained')
    if phase=='teacher':
        from scripts.launch_teacher_probe_sharded import merge_method
        merged=merge_method('YES_FULL',manifest,phase_root,manifests)
        atomic_json(ROOT/'results/train2val/teacher_receipt.json',merged)
    atomic_json(RUN/(phase+'_completion.json'),result)
    print(json.dumps(dict(event='T2V_PREP_COMPLETE',**result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('scores','teacher-smoke','teacher'));p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('at most four physical GPUs')
    main(a.phase,1 if a.phase=='teacher-smoke' else a.workers)
