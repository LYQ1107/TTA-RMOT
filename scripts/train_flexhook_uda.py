#!/usr/bin/env python3
"""Independent single-GPU offline UDA; no DDP or target-GT training branch."""
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import compat.sitecustomize  # noqa: F401

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scope',choices=('plan','smoke','full'),default='plan')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    if args.scope=='plan':
        from flex_uda.training import SourceSupervision,TargetDanceDataset,build_plans
        source=SourceSupervision(ROOT/config['data']['source_train_manifest'])
        target=TargetDanceDataset(ROOT/config['data']['target_manifest'],split='val')
        src,tgt,coverage=build_plans(source,target,args.output)
        print(json.dumps(dict(source_windows=len(src),target_windows=len(tgt),source_coverage=coverage,
            epochs=config['train']['coverage_cycles'],steps_per_epoch=max(len(src),len(tgt))),indent=2))
    else:
        from flex_uda.training import train
        train(config,args.output,smoke=args.scope=='smoke',resume=args.resume)
