#!/usr/bin/env python3
"""Registered, memory-admitted PSR/paired-Naive student; preserves old outputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
METHODS={
    'NAIVE_PAIRED':('NAIVE','none'), 'UNIFORM_PAIRED':('UNIFORM','none'),
    'CONFIDENCE_PAIRED':('CONFIDENCE','none'), 'PSR_DISCOVERY':('PSR','none'),
    'PSR_GEOMETRY':('PSR','geometry'), 'PSR':('PSR','full'), 'NAIVE_TEMPORAL':('NAIVE','geometry'),
    'MEAN_TEACHER_MATCHED':('NAIVE','none'), 'PSR_PARENT_WEIGHT_CONTROL':('PSR','geometry'),
    'DIAG_FIT_OPT':('NAIVE','none'),'DIAG_FIT_OPT_POS':('NAIVE','none'),'DIAG_FIT_CAP':('NAIVE','none')}


def selected_checkpoint(folder,config=None):
    config=config or json.loads((Path(folder)/'config.json').read_text())
    return Path(folder)/('checkpoint_final_step.pt' if config.get('selection')=='final_step' else 'checkpoint_best_source_selected.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/uda/psr.yaml');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('interface','smoke','full'),default='smoke');p.add_argument('--resume',action='store_true');p.add_argument('--val-authorized',action='store_true')
    p.add_argument('--seed',type=int,help='registered student confirmation seed; immutable teacher samples are unchanged');a=p.parse_args()
    config=yaml.safe_load(a.config.read_text());policy,expanded=METHODS[a.method]
    if a.seed is not None:config['protocol']['seed']=a.seed
    config['method']=a.method;config['data']['pseudo_directory']=str(Path(config['data']['pseudo_directory']).parent/policy)
    config['data']['expanded_kind']=expanded
    if config['protocol']['target_split']=='val' and not a.val_authorized:raise PermissionError('explicit Val fallback required')
    out=a.output.resolve();registered=os.environ.get('PSR_PHYSICAL_GPU')
    # A queued peer may reach this method after another idle lane has claimed
    # it. Serialize that real overlap before CUDA admission/model allocation.
    out.parent.mkdir(parents=True,exist_ok=True)
    claim=(out.parent/(out.name+'.training.lock')).open('a');fcntl.flock(claim,fcntl.LOCK_EX)
    if a.scope=='full' and (out/'training_receipt.json').exists():
        previous=json.loads((out/'training_receipt.json').read_text())
        if previous['status']=='COMPLETE_FIXED_COVERAGE_PSR_STUDY' and previous['steps']==8973 and json.loads((out/'config.json').read_text())==config:
            print(json.dumps(dict(event='PRESERVED_COMPLETE_TRAINING_NO_GPU',method=a.method,optimizer_updates_this_attempt=0)),flush=True)
            raise SystemExit(0)
    from scripts.launch_uda_workers import audit_gpus
    from flex_tta.runtime.pf_receipts import atomic_json
    selected,gpus,processes=audit_gpus(1,minimum_free_mib=20480)
    gpu=next((g for g in gpus if str(g['physical_gpu'])==registered),None) if registered is not None else selected[0]
    if gpu is None or gpu['free_mib']<20480:raise RuntimeError('registered GPU no longer has16GiB budget plus4GiB headroom')
    stamp=time.time_ns();atomic_json(out.parent/(out.name+f'_admission_{stamp}.json'),dict(selected=gpu,all_gpus=gpus,processes=processes,
        sharing_authorized=True,maximum_simultaneous_task_GPUs=4,allocator_budget_mib=16384,headroom_mib=4096))
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],MLLM_PHYSICAL_GPU=str(gpu['physical_gpu']),MLLM_VAL_DIAGNOSTIC_AUTHORIZED='YES' if a.val_authorized else 'NO',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    print(json.dumps(dict(event='PSR_STUDENT_GPU_ADMITTED',method=a.method,physical_gpu=gpu['physical_gpu'],uuid=gpu['uuid'],pid=os.getpid())),flush=True)
    from flex_uda.mllm_training import run
    start=time.perf_counter()
    try:run(config,out,scope=a.scope,resume=a.resume)
    finally:atomic_json(out/'attempt_runtimes'/f'{stamp}.json',dict(method=a.method,scope=a.scope,resume=a.resume,physical_gpu=gpu['physical_gpu'],
        pid=os.getpid(),elapsed_seconds=time.perf_counter()-start,successful=sys.exc_info()[0] is None,exception_type=sys.exc_info()[0].__name__ if sys.exc_info()[0] else None))
