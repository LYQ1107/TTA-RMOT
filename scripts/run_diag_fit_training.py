#!/usr/bin/env python3
"""Three registered D-FIT arms, training only. No implicit Val inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.launch_uda_workers import audit_gpus

RUN=ROOT/'results/runs/diag_fit_20261030_v1'
CONFIGS={'DIAG_FIT_OPT':'diag_fit_opt','DIAG_FIT_OPT_POS':'diag_fit_opt_pos','DIAG_FIT_CAP':'diag_fit_cap'}


def child(command,env,log_path,event):
    begin=time.perf_counter();log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a') as log:
        process=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print(json.dumps(dict(event='D_FIT_STARTED',pid=process.pid,**event)),flush=True)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith('{"event":'):print(line.rstrip(),flush=True)
            elif line.startswith('{"step":'):
                record=json.loads(line)
                if record['step']%1000==0 or record['step']==200:print(json.dumps(dict(event='D_FIT_PROGRESS',**event,**record)),flush=True)
        code=process.wait()
    record=dict(exit_code=code,seconds=time.perf_counter()-begin,**event)
    print(json.dumps(dict(event='D_FIT_EXIT',**record)),flush=True);return record


def lane(scope,method,gpu):
    logical=RUN/('students' if scope=='full' else 'smoke')/method
    if (logical/'training_receipt.json').exists():
        receipt=json.loads((logical/'training_receipt.json').read_text())
        config=json.loads((logical/'config.json').read_text())
        if receipt['target_GT_open_attempts'] or not receipt['frozen_parameters_unchanged']:raise RuntimeError('retained isolation failure')
        return dict(status='COMPLETE',method=method,directory=str(logical.resolve()),attempts=[],
            fallback_lr=config['optimizer']['lr']==.0001,trainable_parameters=receipt['trainable_parameters'],
            steps=receipt['steps'],target_GT_open_attempts=0,fit_subset_sha256=receipt['fit_subset_sha256'],preserved=True)
    base=yaml.safe_load((ROOT/'configs/uda'/(CONFIGS[method]+'.yaml')).read_text())
    previous=RUN/'smoke'/method/'training_receipt.json'
    smoke_record=RUN/'smoke'/method/'config.json'
    if scope=='full' and not previous.exists():
        status=json.loads((RUN/'smoke'/f'{method}_status.json').read_text())
        return dict(status='UNTRAINABLE_IN_SMOKE',method=method,smoke_status=status)
    fallback_used=scope=='full' and yaml.safe_load(smoke_record.read_text())['optimizer']['lr']==.0001
    if fallback_used:base['optimizer']['lr']=.0001
    records=[]
    for retry in range(2 if not fallback_used else 1):
        config=json.loads(json.dumps(base))
        if retry:config['optimizer']['lr']=.0001
        folder=RUN/'attempts'/scope/method/('fallback_lr1e4' if retry or fallback_used else 'initial')
        retained_attempts=[json.loads(p.read_text()) for p in (folder/'attempt_runtimes').glob('*.json')]
        if (folder/'training_receipt.json').exists():
            receipt=json.loads((folder/'training_receipt.json').read_text())
            if receipt['status']==('STUDENT_SMOKE_PASS' if scope=='smoke' else 'COMPLETE_FIXED_COVERAGE_PSR_STUDY'):
                record=dict(status='REUSED_COMPLETE',exit_code=0);config_path=folder/'launch_config.yaml'
            else:raise RuntimeError('unknown retained training receipt')
        elif retained_attempts:
            prior=retained_attempts[-1]
            if prior['exception_type']!='FloatingPointError':raise RuntimeError('archive/fix the retained code failure before a fresh run: '+str(folder))
            record=dict(exit_code=1,seconds=prior['elapsed_seconds'],preserved_failed_attempt=True)
        else:
            config_path=RUN/'launch_configs'/scope/(method+('_fallback' if retry or fallback_used else '')+'.yaml')
            atomic_json(config_path,config)  # JSON is valid YAML; no implicit config search.
            env=dict(os.environ,PSR_PHYSICAL_GPU=str(gpu['physical_gpu']),CUDA_VISIBLE_DEVICES=gpu['uuid'],
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
            record=child([sys.executable,ROOT/'scripts/train_psr.py','--method',method,'--scope',scope,
                '--seed','17','--val-authorized','--config',config_path,'--output',folder],env,
                RUN/'logs'/scope/(method+f'_attempt{retry}.txt'),dict(scope=scope,method=method,retry=retry,physical_gpu=gpu['physical_gpu'],lr=config['optimizer']['lr']))
        records.append(record)
        if record['exit_code']==0:
            receipt=json.loads((folder/'training_receipt.json').read_text())
            if receipt['target_GT_open_attempts'] or not receipt['frozen_parameters_unchanged']:raise RuntimeError('hard D-FIT data/parameter isolation failure')
            if method=='DIAG_FIT_CAP' and not receipt['head_changed']:raise RuntimeError('CAP head did not actually change')
            logical=RUN/('students' if scope=='full' else 'smoke')/method;logical.parent.mkdir(parents=True,exist_ok=True)
            if not logical.exists():logical.symlink_to(folder,target_is_directory=True)
            elif logical.resolve()!=folder.resolve():raise RuntimeError('preserve an existing different arm')
            return dict(status='COMPLETE',method=method,directory=str(folder),attempts=records,
                fallback_lr=config['optimizer']['lr']==.0001,trainable_parameters=receipt['trainable_parameters'],
                steps=receipt['steps'],target_GT_open_attempts=0,fit_subset_sha256=receipt['fit_subset_sha256'])
        attempts=[json.loads(p.read_text()) for p in (folder/'attempt_runtimes').glob('*.json')]
        if not attempts:raise RuntimeError('code/admission failure, not a numerical fallback; inspect '+str(folder))
        error=attempts[-1]['exception_type']
        if error!='FloatingPointError':raise RuntimeError('code/protocol failure: '+error+'; retain and restart affected arm after fix')
        if retry or fallback_used:
            return dict(status='UNTRAINABLE_AFTER_REGISTERED_FALLBACK',method=method,attempts=records,target_GT_open_attempts=0)
        print(json.dumps(dict(event='D_FIT_REGISTERED_NUMERICAL_FALLBACK',method=method,scope=scope,lr=.0001,head_lr_unchanged=True)),flush=True)


def run(scope):
    validation=json.loads((RUN/'validation/completion.json').read_text())
    if not validation['status'].startswith('PASS_'):raise ValueError('real old/new compatibility must pass first')
    if scope=='full' and not (RUN/'baselines/completion.json').exists():raise ValueError('finish fixed baseline fit before full training')
    attempt=RUN/'launchers'/scope/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=True)
    # One parent command owns this resource wait and all child waits.
    while True:
        try:
            chosen,gpus,processes=audit_gpus(3,minimum_free_mib=20480);break
        except RuntimeError as error:
            print(json.dumps(dict(event='D_FIT_WAIT_MEMORY_FIT_GPUS',reason=str(error))),flush=True);time.sleep(20)
    atomic_json(attempt/'admission.json',dict(chosen=chosen,all_gpus=gpus,processes=processes,sharing_authorized=True,maximum_own_GPUs=3))
    records={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(lane,scope,m,g):m for m,g in zip(CONFIGS,chosen)}
        for future in as_completed(jobs):
            result=future.result();records[jobs[future]]=result
            dest=RUN/('smoke' if scope=='smoke' else 'students');dest.mkdir(parents=True,exist_ok=True)
            atomic_json(dest/(jobs[future]+'_status.json'),result)
    hashes={r['fit_subset_sha256'] for r in records.values() if r['status']=='COMPLETE'}
    if len(hashes)>1:raise RuntimeError('three arms used different fit subsets')
    atomic_json(RUN/(scope+'_completion.json'),dict(status='COMPLETE_ALL_REGISTERED_ARMS_REPORTED',arms=records,
        fit_subset_sha256=next(iter(hashes),None),Val_inference_started=False,target_GT_open_attempts=0))
    print(json.dumps(dict(event='D_FIT_TRAINING_PHASE_COMPLETE',scope=scope,arms=records)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('smoke','full'),required=True);run(p.parse_args().scope)
