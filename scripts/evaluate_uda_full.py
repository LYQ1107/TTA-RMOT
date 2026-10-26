#!/usr/bin/env python3
"""Evaluate a source-val-selected UDA checkpoint; annotations are inaccessible."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.evaluation import evaluate
from flex_uda.training import METHODS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--videos',nargs='+')
    p.add_argument('--parity-only',action='store_true');a=p.parse_args()
    evaluate(a.checkpoint,a.method,a.output_root,videos=a.videos,parity_only=a.parity_only)
