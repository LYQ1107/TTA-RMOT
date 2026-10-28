#!/usr/bin/env python3
"""Read-only complete-video evaluation of an independently selected student."""
import argparse
from pathlib import Path
import sys
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--method',default='MLLM_DISTILL')
    p.add_argument('--videos',nargs='+');p.add_argument('--parity-only',action='store_true');a=p.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    selected=[line for line in snapshot.splitlines() if visible in [s.strip() for s in line.split(',')[:2]]]
    if len(selected)!=1 or float(selected[0].split(',')[-1])<20480:raise RuntimeError('fresh evaluation GPU admission failed before model load')
    from flex_tta.runtime.pf_receipts import atomic_json
    a.output_root=a.output_root.resolve()
    atomic_json(a.output_root/'fresh_gpu_admission.json',dict(selected=selected[0],snapshot=snapshot,pid=os.getpid(),required_free_mib=20480))
    from flex_uda.evaluation import evaluate
    evaluate(a.checkpoint,a.method,a.output_root,videos=a.videos,parity_only=a.parity_only)
