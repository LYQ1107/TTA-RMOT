#!/usr/bin/env python3
"""Recover one FlexHook validation shard without a NCCL process group.

The upstream evaluator writes each DDP shard after inference and then waits at
an NCCL barrier.  If a slow external process delays one rank past the barrier
timeout, the completed shard can be retained and this script recomputes only
the missing shard with the same DistributedSampler partition.
"""

import argparse
import json
import os
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--val-batch-size", type=int, default=80)
    parser.add_argument("--log-rank", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.rank < args.world_size):
        raise ValueError("rank must be in [0, world_size)")

    vendor = str(Path(args.vendor).resolve())
    sys.path.insert(0, vendor)
    # The upstream launcher changes into vendor/FlexHook before constructing
    # the dataset; its relative pretrained paths must resolve the same way.
    os.chdir(vendor)
    os.environ.setdefault("LOCAL_RANK", "0")

    from config import get_config
    from data.mydataloader import RMOT_Dataset
    from logger import create_logger
    from models import build_model
    from utils import load_ckpt_my
    import main as flexhook_main

    cli = Namespace(
        cfg=args.cfg,
        opts=["data_root", args.data_root],
        batch_size=8,
        val_batch_size=args.val_batch_size,
        visual="rope-swin-tiny",
        text="roberta",
        dataset=None,
        freeze_text=False,
        freeze_visual=False,
        track_root=args.track_root,
        data_path=None,
        zip=False,
        cache_mode="part",
        pretrained=None,
        resume=args.checkpoint,
        accumulation_steps=None,
        use_checkpoint=False,
        disable_amp=False,
        amp_opt_level=None,
        output=args.output,
        tag=None,
        eval=True,
        throughput=False,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        lre=None,
    )
    config = get_config(cli)
    config.defrost()
    config.MODEL.PRETRAINED = args.checkpoint
    config.freeze()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = create_logger(
        str(output),
        dist_rank=args.rank if args.log_rank is None else args.log_rank,
        name=f"recovery_rank{args.rank}",
    )
    flexhook_main.logger = logger

    # RMOT_Dataset.__len__ calls dist.get_rank() only for a progress print.
    # This recovery process deliberately has no NCCL group, so provide the
    # recorded shard rank without changing sampler semantics.
    torch.distributed.get_rank = lambda: args.rank

    seed = config.SEED + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)

    dataset = RMOT_Dataset(mode="test", opt=config)
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.DATA.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
        persistent_workers=config.DATA.NUM_WORKERS > 0,
    )
    logger.info(
        "Recovering rank %d/%d: %d validation batches, batch size %d",
        args.rank,
        args.world_size,
        len(loader),
        config.DATA.VAL_BATCH_SIZE,
    )

    model = build_model(config).cuda()
    # Recent Transformers keeps these RoBERTa helper buffers as non-persistent
    # buffers.  DDP moves/synchronizes them in the original path; do it
    # explicitly for this single-process shard recovery.
    for module in model.modules():
        if hasattr(module, "token_type_ids"):
            module.token_type_ids = module.token_type_ids.cuda()
        if hasattr(module, "position_ids"):
            module.position_ids = module.position_ids.cuda()
    load_ckpt_my(config, model, logger)
    model.eval()

    class CudaTextLoader:
        def __init__(self, base_loader):
            self.base_loader = base_loader

        def __len__(self):
            return len(self.base_loader)

        def __iter__(self):
            for batch in self.base_loader:
                images, pes, bbox_gt, exp, expid, expma, video, obj, frame_id, last = batch
                yield (
                    images,
                    pes,
                    bbox_gt,
                    exp,
                    expid.cuda(non_blocking=True),
                    expma.cuda(non_blocking=True),
                    video,
                    obj,
                    frame_id,
                    last,
                )

    with torch.no_grad():
        result = flexhook_main.inference(config, CudaTextLoader(loader), model)

    destination = output / f"result_{args.rank}.json"
    temporary = output / f"result_{args.rank}.json.tmp"
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=1)
    os.replace(temporary, destination)
    logger.info("Wrote complete shard %s (%d bytes)", destination, destination.stat().st_size)


if __name__ == "__main__":
    main()
