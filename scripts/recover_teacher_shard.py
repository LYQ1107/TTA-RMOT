#!/usr/bin/env python3
"""Recover only a failed P1 shard, preserving other healthy GPU workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_teacher_probe import resources,PYTHON
from scripts.audit_teacher_leakage import RUN,validate_receipt
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--exclude-gpus',nargs='+',type=int,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not 0<=a.shard<4 or len(set(a.exclude_gpus))!=3:raise ValueError('one recovery slot beside three healthy task GPUs')
    phase=RUN/'full/T0';folder=phase/'shards'/f'shard{a.shard:02d}';manifest=folder/'manifest.jsonl'
    config=ROOT/'configs/teacher/base_qwen2b.yaml';rules=ROOT/'results/teacher_probe/source_calibration_rules_T0.json'
    stage=Path('/data2/user/TTA-RMOT-assets/teacher-p1/staging_full');count=sum(1 for _ in manifest.open())
    methods=['YES_ENSEMBLE_B_PLUS','YESTRACK_TCP_TRP']
    print(json.dumps(dict(shard=a.shard,methods=methods,exclude_gpus=a.exclude_gpus,execute=a.execute,
        GPU_budget_gib=8,measured_smoke_peak_GiB=5.15,numeric_model_configuration_unchanged=True)),flush=True)
    if not a.execute:return
    attempt=phase/'launcher_attempts'/('repair_'+str(time.time_ns()));attempt.mkdir(parents=True)
    records=[];started=time.perf_counter()
    status=dict(status='RUNNING_FAILED_SHARD_RECOVERY',phase='full',condition='T0',workers=records,
        actual_wall_seconds=0.,failed_shard=a.shard,healthy_GPUs_not_stopped=a.exclude_gpus,
        allocation_budget_gib=8,physical_free_margin_gib=4,teacher_model_dtype_batch_prompts_unchanged=True)
    def save():
        status['actual_wall_seconds']=time.perf_counter()-started
        atomic_json(attempt/'launcher_status.json',status)
    save()
    for method in methods:
        out=folder/method;receipt=out/'receipt.json'
        if receipt.exists():
            old=json.loads(receipt.read_text());validate_receipt(old,method,file_sha(manifest),count)
            records.append(dict(method=method,shard=a.shard,status='PRESERVED_SEALED_RESULT',receipt=str(receipt)));save();continue
        while True:
            try:admission=resources(1,exclude=a.exclude_gpus,memory_budget_gib=8);break
            except RuntimeError as error:
                status['resource_wait']=str(error);save();time.sleep(15)
        gpu=admission['selected'][0];atomic_json(attempt/(method+'_gpu_admission.json'),admission)
        command=[PYTHON,str(ROOT/'scripts/run_teacher_probe.py'),'--config',str(config),'--manifest',str(manifest),
            '--staging-root',str(stage),'--method',method,'--output-root',str(out),
            '--reliability-config',str(rules),'--memory-budget-gib','8']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        log_path=phase/f'repair_shard{a.shard:02d}_{method}_stdout.txt'
        begin=time.perf_counter();epoch=time.time()
        with log_path.open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            item=dict(method=method,shard=a.shard,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=child.pid,
                start_epoch_seconds=epoch,samples=count,log=str(log_path),receipt=str(receipt))
            records.append(item);save()
            print(json.dumps(dict(started=method,shard=a.shard,physical_gpu=gpu['index'],pid=child.pid)),flush=True)
            for line in child.stdout:
                log.write(line);log.flush()
                if '"completed":' in line or 'Traceback' in line or 'Error:' in line:print(line.rstrip(),flush=True)
            code=child.wait()
        item.update(exit_code=code,elapsed_seconds=time.perf_counter()-begin);save()
        print(json.dumps(dict(finished=method,shard=a.shard,exit_code=code,elapsed_seconds=item['elapsed_seconds'])),flush=True)
        if code:
            status['status']='FAILED_RECOVERY_ATTEMPT_RETAINED';save();raise SystemExit(code)
    status['status']='COMPLETE_FAILED_SHARD_RECOVERY';save();print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
