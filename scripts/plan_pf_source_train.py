#!/usr/bin/env python3
"""Print-only plan for optional *full source-train* FlexHook reproduction.

NEVER launches training, even when invoked with all paths. For this source-free
TTA study the released V2 checkpoint is already the primary source-trained
model; use this plan only if explicitly authorized to retrain on all 17 V2
source videos. Target Dance GT must never be referenced.
"""
from __future__ import annotations
import argparse
import json
import shlex
from pathlib import Path


def plan(vendor: Path, output: Path, nproc: int, batch_size: int, data_path: int):
    if nproc not in (2, 4) or batch_size < 1:
        raise ValueError("expected 2 or 4 source-training GPUs and valid batch size")
    cfg = vendor / "configs/train/train-kitti2.yaml"
    command = [
        "python", "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
        "--master_port", "19421", "main.py",
        "--cfg", "configs/train/train-kitti2.yaml", "--data-path", str(data_path),
        "--output", str(output), "--batch-size", str(batch_size),
        "--val-batch-size", "40", "--visual", "rope-swin-tiny",
        "--text", "roberta", "--pretrained", "src",
    ]
    return {
        "run_mode": "PRINT_ONLY_NO_TRAINING",
        "source_dataset": "Refer-KITTI-V2 complete official train 17 videos",
        "source_dataset_config": str(cfg),
        "test_dataset": "Refer-KITTI-V2 official test 4 videos / 862 expressions",
        "separate_target": "Refer-Dance validation 25 videos / 425 expressions, GT NEVER for TTA",
        "source_train_config_exists": cfg.is_file(),
        "distributed_processes": nproc,
        "per_gpu_batch": batch_size,
        "global_batch": nproc * batch_size,
        "official_reference_global_batch": 2 * 7,
        "official_matched_batch": (nproc == 2 and batch_size == 7),
        "warning": "Four-GPU global batch differs from authors' two-GPU 7x2 run; label as new source reproduction, not identical official optimizer schedule.",
        "cwd": str(vendor),
        "command_to_review_before_authorized_run": shlex.join(command),
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--nproc",type=int,choices=[2,4],default=4)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--data-path",type=int,default=12)
    p.add_argument("--save-plan",type=Path)
    args=p.parse_args(argv)
    receipt=plan(args.vendor,args.output_root,args.nproc,args.batch_size,args.data_path)
    if args.save_plan:
        args.save_plan.parent.mkdir(parents=True,exist_ok=True)
        args.save_plan.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
