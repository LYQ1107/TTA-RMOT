#!/usr/bin/env python3
"""Construct the pre-registered target-GT-blind 25video/425query sample."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
from flex_uda.teacher.gt_blind_sampler import build

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['smoke','full'],required=True)
    p.add_argument('--runtime-manifest',type=Path,default=ROOT/'manifests/dance_val_runtime.jsonl')
    a=p.parse_args();config=yaml.safe_load((ROOT/'configs/teacher/probe_protocol.yaml').read_text())
    name='smoke_manifest' if a.stage=='smoke' else 'manifest'
    result=build(a.runtime_manifest,ROOT/f'results/teacher_probe/{name}.jsonl',
        Path('/data2/user/TTA-RMOT-assets/teacher-p1')/f'staging_{a.stage}',
        config['smoke_anchor_quantiles' if a.stage=='smoke' else 'probe_anchor_quantiles'],config['seed'],config['candidate_limit'])
    print(json.dumps(result,indent=2))
