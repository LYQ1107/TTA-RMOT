#!/usr/bin/env python3
"""Read-only complete-video evaluation of an independently selected student."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import os
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--method',default='MLLM_DISTILL')
    p.add_argument('--videos',nargs='+');p.add_argument('--parity-only',action='store_true');a=p.parse_args()
    from flex_tta.runtime.pf_receipts import atomic_json,file_sha
    a.output_root=a.output_root.resolve();folder=a.output_root/a.method
    folder.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();stamp=time.time_ns()
    claim=(folder/'inference.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX)
    lock_seconds=time.perf_counter()-begin
    receipt=folder/'run_metadata.json';audit=folder/'full_audit.json'
    if not a.parity_only and a.videos is None and receipt.exists() and audit.exists():
        old=json.loads(receipt.read_text())
        if old['status']=='SEALED' and old['valid_episodes']==425 and old['checkpoint_sha256']==file_sha(a.checkpoint):
            atomic_json(folder/'cli_inference_attempts'/f'{stamp}.json',dict(pid=os.getpid(),actual_GPU_calls=0,
                GPU_worker_seconds=0,CPU_lock_wait_seconds=lock_seconds,completed_full425_reused=True))
            print(json.dumps(dict(event='PRESERVED_COMPLETE_INFERENCE_NO_GPU',method=a.method)),flush=True)
            raise SystemExit(0)
    actual_start=time.perf_counter();visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    selected=[line for line in snapshot.splitlines() if visible in [s.strip() for s in line.split(',')[:2]]]
    if len(selected)!=1 or float(selected[0].split(',')[-1])<20480:raise RuntimeError('fresh evaluation GPU admission failed before model load')
    atomic_json(a.output_root/a.method/'fresh_gpu_admission.json',dict(selected=selected[0],snapshot=snapshot,pid=os.getpid(),required_free_mib=20480))
    from flex_uda.evaluation import evaluate
    try:evaluate(a.checkpoint,a.method,a.output_root,videos=a.videos,parity_only=a.parity_only)
    finally:atomic_json(folder/'cli_inference_attempts'/f'{stamp}.json',dict(pid=os.getpid(),actual_GPU_calls=1,
        GPU_worker_seconds=time.perf_counter()-actual_start,CPU_lock_wait_seconds=lock_seconds,
        completed_full425_reused=False,physical_gpu=os.environ.get('UDA_PHYSICAL_GPU'),successful=sys.exc_info()[0] is None))
