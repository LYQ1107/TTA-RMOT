#!/usr/bin/env python3
"""Four fixed physical-GPU lanes: each job trains then independently evaluates.

Healthy completed training/inference is reused. No GT is read anywhere in this
driver. Official evaluation is a later, independent CPU command after all seals.
"""
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
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

DEFAULT=['NAIVE_PAIRED','PSR','UNIFORM_PAIRED','CONFIDENCE_PAIRED','PSR_DISCOVERY','PSR_GEOMETRY','NAIVE_TEMPORAL','MEAN_TEACHER_MATCHED']


def run(root,methods,workers=4,other_active_gpus=(),seed=None,gpu_ids=None):
    root=Path(root).resolve();training=root/'students';evaluation=root/'evaluation/merged'
    evaluation.mkdir(parents=True,exist_ok=True);training.mkdir(exist_ok=True)
    pending=[];preserved=[]
    for method in methods:
        training_receipt=training/method/'training_receipt.json'
        if seed is not None and training_receipt.exists() and json.loads(training_receipt.read_text())['protocol']['seed']!=seed:
            raise ValueError('use a separate run root for each registered seed; preserve existing results')
        receipt=evaluation/method/'run_metadata.json';checkpoint=training/method/'checkpoint_best_source_selected.pt'
        if receipt.exists() and checkpoint.exists():
            old=json.loads(receipt.read_text())
            if old['status']=='SEALED' and old['valid_episodes']==425 and old['checkpoint_sha256']==file_sha(checkpoint):
                for row in old['episodes']:
                    if file_sha(Path(row['prediction_path']))!=row['prediction_sha256'] or file_sha(Path(row['margins_path']))!=row['margins_sha256']:raise ValueError('sealed output changed')
                preserved.append(method);continue
        pending.append(method)
    if not pending:print(json.dumps(dict(status='PRESERVED_ALL_COMPLETE_MATRIX',methods=preserved)));return
    _,gpus,processes=audit_gpus(1,minimum_free_mib=20480)
    other_active_gpus=set(map(int,other_active_gpus))
    allowed=None if gpu_ids is None else set(map(int,gpu_ids))
    chosen=sorted([g for g in gpus if g['free_mib']>=20480 and g['physical_gpu'] not in other_active_gpus
        and (allowed is None or g['physical_gpu'] in allowed)],
        key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))[:min(4-len(other_active_gpus),workers,len(pending))]
    if not chosen:raise RuntimeError('no memory-fit GPU slots after preserving healthy peer lanes')
    attempt=root/'matrix_launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',dict(chosen=chosen,all_gpus=gpus,processes=processes,sharing_authorized=True,
        maximum_physical_GPUs=4,other_healthy_task_GPUs=sorted(other_active_gpus),
        each_lane_serializes_its_own_training_and_inference=True,allocator_GiB=16,headroom_GiB=4))
    queue=list(pending);lock=threading.Lock();start=time.perf_counter()
    def command_run(command,env,log_path,record):
        phase=record['phase'];begin=time.perf_counter();video_count=0
        with log_path.open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            record.update(pid=proc.pid,start_epoch=time.time())
            print(json.dumps(dict(event='PSR_MATRIX_PHASE_STARTED',**record)),flush=True)
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith('{"step":'):
                    step=json.loads(line)
                    if step['step']%2000==0:print(json.dumps(dict(event='PSR_MATRIX_TRAIN_PROGRESS',method=record['method'],**step)),flush=True)
                elif line.startswith('{"event": "video_sealed"'):
                    video_count+=1
                    if video_count%5==0:print(json.dumps(dict(event='PSR_MATRIX_INFERENCE_PROGRESS',method=record['method'],videos=video_count,of=25)),flush=True)
            code=proc.wait()
        record.update(exit_code=code,elapsed_seconds=time.perf_counter()-begin,end_epoch=time.time())
        print(json.dumps(dict(event='PSR_MATRIX_PHASE_FINISHED',**record)),flush=True)
        return code
    def lane(gpu):
        results=[]
        while True:
            with lock:
                if not queue:break
                method=queue.pop(0)
            records=[];folder=training/method;train_receipt=folder/'training_receipt.json'
            ready=train_receipt.exists() and json.loads(train_receipt.read_text())['status']=='COMPLETE_FIXED_COVERAGE_PSR_STUDY'
            env=os.environ.copy();env.update(PSR_PHYSICAL_GPU=str(gpu['physical_gpu']),CUDA_VISIBLE_DEVICES=gpu['uuid'],UDA_PHYSICAL_GPU=str(gpu['physical_gpu']),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
            if not ready:
                command=[sys.executable,str(ROOT/'scripts/train_psr.py'),'--method',method,'--scope','full','--val-authorized','--output',str(folder)]
                if seed is not None:command+=['--seed',str(seed)]
                if (folder/'checkpoint_latest.pt').exists():command.append('--resume')
                record=dict(method=method,phase='training',physical_gpu=gpu['physical_gpu'],uuid=gpu['uuid'])
                code=command_run(command,env,training/(method+'_stdout.txt'),record);records.append(record)
                if code:
                    results.append(dict(method=method,status='FAILED_TRAINING_RETAINED',phases=records));break
            checkpoint=folder/'checkpoint_best_source_selected.pt'
            train=json.loads(train_receipt.read_text())
            if train['steps']!=8973 or train['target_GT_open_attempts'] or not train['frozen_parameters_unchanged']:raise ValueError('invalid actual training receipt: '+method)
            method_out=evaluation/method;method_out.mkdir(exist_ok=True)
            clock=method_out/'first_inference_launch_epoch.json'
            if not clock.exists():atomic_json(clock,dict(epoch=time.time(),checkpoint_sha256=file_sha(checkpoint)))
            command=[sys.executable,str(ROOT/'scripts/evaluate_mllm_distill.py'),'--checkpoint',str(checkpoint),'--output-root',str(evaluation),'--method',method]
            record=dict(method=method,phase='inference',physical_gpu=gpu['physical_gpu'],uuid=gpu['uuid'])
            code=command_run(command,env,evaluation/(method+'_stdout.txt'),record);records.append(record)
            atomic_json(method_out/'inference_attempts'/f"{time.time_ns()}.json",record)
            if code:
                results.append(dict(method=method,status='FAILED_INFERENCE_RETAINED',phases=records));break
            path=method_out/'run_metadata.json';metadata=json.loads(path.read_text())
            if metadata['valid_episodes']!=425 or metadata['test_time_updates'] or metadata['target_GT_open_attempts']:raise ValueError('incomplete/invalid425 inference')
            history=[json.loads(p.read_text()) for p in (method_out/'inference_attempts').glob('*.json')]
            metadata.update(worker_seconds=sum(r['elapsed_seconds'] for r in history),actual_scheduler_wall_seconds=time.time()-json.loads(clock.read_text())['epoch'],
                all_attempt_worker_costs_included=True,execution_mode='independent single_GPU full425 checkpoint, four method lanes',
                shared_backbone_across_methods=False,inference_worker_attempts=history)
            atomic_json(path,metadata);results.append(dict(method=method,status='COMPLETE',phases=records))
        return results
    results=[]
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        # One blocking parent wait; no agent-level process checks or polling.
        for future in as_completed([pool.submit(lane,gpu) for gpu in chosen]):results.extend(future.result())
    complete=not queue and all(r['status']=='COMPLETE' for r in results)
    result=dict(status='COMPLETE' if complete else 'FAILED_UNITS_RETAINED',methods=methods,results=results,preserved=preserved,pending=queue,
        actual_matrix_wall_seconds=time.perf_counter()-start,registered_physical_GPUs=[g['physical_gpu'] for g in chosen],
        target_GT_opened=False,scientific_metrics='NOT_COMPUTED_BEFORE_ALL_PREDICTIONS_SEALED')
    atomic_json(attempt/'launcher_status.json',result)
    if not complete:raise RuntimeError('retain all completed methods and inspect failed phase; resume unfinished units only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'results/runs/psr_val_20261029_v1');p.add_argument('--methods',nargs='+',choices=METHODS,default=DEFAULT);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--other-active-gpus',nargs='*',type=int,default=[],help='preserved healthy peer lanes count toward the four-card total')
    p.add_argument('--seed',type=int,help='registered confirmation seed, with a new independent run root')
    p.add_argument('--gpu-ids',nargs='+',type=int,help='physical lane allowlist assigned by an outer fixed-GPU scheduler')
    a=p.parse_args();run(a.root,a.methods,a.workers,a.other_active_gpus,a.seed,a.gpu_ids)
