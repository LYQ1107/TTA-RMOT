#!/usr/bin/env python3
"""Fresh-admitted single-GPU MLLM distillation; never stop foreign processes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/uda/mllm_naive_distill.yaml')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=('interface','smoke','full'),default='interface')
    p.add_argument('--resume',action='store_true');p.add_argument('--diagnostic-val-authorized',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();config=yaml.safe_load(a.config.read_text())
    if config['protocol']['target_split']=='val' and not a.diagnostic_val_authorized:raise PermissionError('explicit Val diagnostic authorization required')
    from scripts.launch_uda_workers import audit_gpus
    from flex_tta.runtime.pf_receipts import atomic_json
    selected,all_gpus,processes=audit_gpus(1,minimum_free_mib=32768);g=selected[0]
    attempt=time.time_ns();admission=a.output.resolve().parent/(a.output.name+f'_admission_{attempt}.json')
    atomic_json(admission,dict(selected=g,all_gpus=all_gpus,existing_processes=processes,sharing_authorized=True,
        allocator_budget_mib=24576,admission_headroom_mib=8192,no_foreign_process_killed=True))
    os.environ.update(CUDA_VISIBLE_DEVICES=g['uuid'],MLLM_PHYSICAL_GPU=str(g['physical_gpu']),
        MLLM_VAL_DIAGNOSTIC_AUTHORIZED='YES' if a.diagnostic_val_authorized else 'NO',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    print(json.dumps(dict(event='GPU_ADMITTED',physical_gpu=g['physical_gpu'],uuid=g['uuid'],pid=os.getpid(),scope=a.scope)),flush=True)
    from flex_uda.mllm_training import run
    started=time.perf_counter()
    try:
        run(config,a.output,scope=a.scope,resume=a.resume)
    finally:
        atomic_json(a.output.resolve()/'attempt_runtimes'/f'{attempt}.json',dict(scope=a.scope,resume=a.resume,
            physical_gpu=g['physical_gpu'],gpu_uuid=g['uuid'],pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-started,successful=sys.exc_info()[0] is None,
            exception_type=sys.exc_info()[0].__name__ if sys.exc_info()[0] else None))
