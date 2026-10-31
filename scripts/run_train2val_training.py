#!/usr/bin/env python3
"""Train-only four fixed arms; never launches Val inference or reads Val GT."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
from flex_uda.data.train2val import ASSETS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.launch_uda_workers import audit_gpus
from scripts.run_diag_fit_training import child

RUN=ROOT/'results/runs/train2val_20261031'
PUBLIC=ROOT/'results/train2val'
ARMS={'T2V_D1_s17':('DIAG_FIT_OPT',17),'T2V_D1_s23':('DIAG_FIT_OPT',23),
      'T2V_D1_s42':('DIAG_FIT_OPT',42),'T2V_D3_s17':('DIAG_FIT_CAP',17)}


def configuration(method,seed):
    filename='diag_fit_opt.yaml' if method=='DIAG_FIT_OPT' else 'diag_fit_cap.yaml'
    config=yaml.safe_load((ROOT/'configs/uda'/filename).read_text())
    config['protocol'].update(name='FORMAL_TRAIN_TO_VAL_PSR_STUDY',target_split='train',
        evaluation_split='OFFICIAL_VAL25_VIDEO_DISJOINT_8_QUERY_STEMS_SHARED',seed=seed)
    config['data'].update(pseudo_directory=str(ASSETS/'pseudo/NAIVE'),frozen_reference_root=str(ASSETS/'frozen_sparse'))
    config['input_policy']='TRAIN_ONLY_NO_VAL'
    return config


def lane(arm,gpu,scope):
    method,seed=ARMS[arm];logical=RUN/('smoke' if scope=='smoke' else 'students')/arm
    if (logical/'training_receipt.json').exists():
        r=json.loads((logical/'training_receipt.json').read_text())
        if r['target_GT_open_attempts'] or r['forbidden_input_open_attempts']:raise RuntimeError('retained isolation failure')
        return dict(status='COMPLETE',arm=arm,path=str(logical.resolve()),preserved=True)
    base=configuration(method,seed);smoke=RUN/'smoke'/('T2V_D1_s17' if method=='DIAG_FIT_OPT' else 'T2V_D3_s17')
    if scope=='full':
        ready=json.loads((smoke/'training_receipt.json').read_text())
        if ready['status']!='STUDENT_SMOKE_PASS' or ready['target_GT_open_attempts'] or ready['forbidden_input_open_attempts']:
            raise RuntimeError('require real Train-only smoke before full training')
        if json.loads((smoke/'config.json').read_text())['optimizer']['lr']==.0001:base['optimizer']['lr']=.0001
    records=[]
    for retry in range(1 if base['optimizer']['lr']==.0001 else 2):
        config=copy.deepcopy(base)
        if retry:config['optimizer']['lr']=.0001
        folder=RUN/'attempts'/scope/arm/('fallback_lr1e4' if config['optimizer']['lr']==.0001 else 'initial')
        config_path=RUN/'launch_configs'/scope/(arm+('_fallback' if retry else '')+'.json')
        atomic_json(config_path,config)
        if (folder/'training_receipt.json').exists():
            record=dict(exit_code=0,preserved=True)
        else:
            env=dict(os.environ,PSR_PHYSICAL_GPU=str(gpu['physical_gpu']),CUDA_VISIBLE_DEVICES=gpu['uuid'],
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1')
            command=[sys.executable,ROOT/'scripts/train_psr.py','--method',method,'--scope',scope,
                '--seed',str(seed),'--config',config_path,'--output',folder]
            if (folder/'checkpoint_latest.pt').exists():command.append('--resume')
            record=child(command,env,RUN/'logs'/scope/(arm+f'_attempt{retry}.txt'),
                dict(scope=scope,arm=arm,method=method,seed=seed,retry=retry,physical_gpu=gpu['physical_gpu'],lr=config['optimizer']['lr']))
        records.append(record)
        if record['exit_code']==0:
            r=json.loads((folder/'training_receipt.json').read_text())
            if r['target_GT_open_attempts'] or r['forbidden_input_open_attempts'] or not r['frozen_parameters_unchanged']:
                raise RuntimeError('hard Train→Val isolation/frozen-parameter violation')
            if scope=='full' and r['steps']!=8973:raise RuntimeError('fixed update budget changed')
            if method=='DIAG_FIT_CAP' and not r['head_changed']:raise RuntimeError('D3 head did not update')
            logical.parent.mkdir(parents=True,exist_ok=True)
            if not logical.exists():logical.symlink_to(folder,target_is_directory=True)
            return dict(status='COMPLETE',arm=arm,path=str(folder),attempts=records,
                fallback_lr=config['optimizer']['lr']==.0001,checkpoint_sha256=r['checkpoints']['checkpoint_final_step.pt']['sha256'])
        failures=[json.loads(p.read_text()) for p in sorted((folder/'attempt_runtimes').glob('*.json'))]
        if not failures or failures[-1]['exception_type']!='FloatingPointError':
            raise RuntimeError('code/input/admission failure, not a numerical fallback: '+str(folder))
        if retry or base['optimizer']['lr']==.0001:
            return dict(status='UNTRAINABLE_AFTER_REGISTERED_FALLBACK',arm=arm,attempts=records)
        print(json.dumps(dict(event='T2V_REGISTERED_NUMERICAL_FALLBACK',arm=arm,lr=.0001,head_lr_unchanged=True)),flush=True)


def run(scope):
    started=time.perf_counter();names=['T2V_D1_s17','T2V_D3_s17'] if scope=='smoke' else list(ARMS)
    attempt=RUN/'training_launchers'/scope/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=True)
    while True:
        try:chosen,gpus,processes=audit_gpus(len(names),minimum_free_mib=20480);break
        except RuntimeError as e:
            print(json.dumps(dict(event='T2V_TRAIN_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    atomic_json(attempt/'admission.json',dict(selected=chosen,all_gpus=gpus,processes=processes,
        sharing_authorized=True,maximum_physical_GPUs=4,allocator_GiB=16,headroom_GiB=4))
    results={}
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        jobs={pool.submit(lane,name,gpu,scope):name for name,gpu in zip(names,chosen)}
        for future in as_completed(jobs):
            result=future.result();results[jobs[future]]=result
            atomic_json(RUN/(scope+'_status')/(jobs[future]+'.json'),result)
    result=dict(status='ALL_REGISTERED_ARMS_REPORTED',scope=scope,arms=results,
        actual_wall_seconds=time.perf_counter()-started,Val_inference_started=False)
    atomic_json(attempt/'completion.json',result);atomic_json(RUN/(scope+'_completion.json'),result)
    print(json.dumps(dict(event='T2V_TRAIN_PHASE_COMPLETE',**result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('smoke','full'),required=True);run(p.parse_args().scope)
