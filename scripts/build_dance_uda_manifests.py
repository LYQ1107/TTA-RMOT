#!/usr/bin/env python3
"""Unpack only missing Train image payloads, then create isolated manifests."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import ASSETS,build_target_manifests,extract_training_images
from flex_tta.runtime.pf_receipts import atomic_json

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-missing-train-images',action='store_true')
    parser.add_argument('--manifest-dir',type=Path,default=ROOT/'manifests')
    args=parser.parse_args();result={}
    if args.extract_missing_train_images:
        result['image_preparation']=extract_training_images(ASSETS/'Refer-Dance.zip')
        atomic_json(ROOT/'results/uda/assets/train_image_preparation.json',result['image_preparation'])
        print(json.dumps(dict(event='all40_training_video_images_ready'),ensure_ascii=False),flush=True)
    result['manifest_audit']=build_target_manifests(args.manifest_dir)
    atomic_json(ROOT/'results/uda/assets/dance_manifest_audit.json',result['manifest_audit'])
    print(json.dumps(result['manifest_audit'],ensure_ascii=False,indent=2))
