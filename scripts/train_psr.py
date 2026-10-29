#!/usr/bin/env python3
"""Registered, memory-admitted PSR/paired-Naive student; preserves old outputs."""
import argparse
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
    'MEAN_TEACHER_MATCHED':('NAIVE','none')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/uda/psr.yaml');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('interface','smoke','full'),default='smoke');p.add_argument('--resume',action='store_true');p.add_argument('--val-authorized',action='store_true');a=p.parse_args()
    config=yaml.safe_load(a.config.read_text());policy,expanded=METHODS[a.method]
    config['method']=a.method;config['data']['pseudo_directory']=str(Path(config['data']['pseudo_directory']).parent/policy)
    config['data']['expanded_kind']=expanded
    if config['protocol']['target_split']=='val' and not a.val_authorized:raise PermissionError('explicit Val fallback required')
    out=a.output.resolve();registered=os.environ.get('PSR_PHYSICAL_GPU')
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
