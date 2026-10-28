#!/usr/bin/env python3
"""Read-only complete-video evaluation of an independently selected student."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize
from flex_uda.evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--method',default='MLLM_DISTILL')
    p.add_argument('--videos',nargs='+');p.add_argument('--parity-only',action='store_true');a=p.parse_args()
    evaluate(a.checkpoint,a.method,a.output_root,videos=a.videos,parity_only=a.parity_only)
