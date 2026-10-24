#!/usr/bin/env python3
"""Materialize Dance prediction files from a merged FlexHook score JSON."""

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--track-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    vendor = str(Path(args.vendor).resolve())
    sys.path.insert(0, vendor)
    os.chdir(vendor)
    os.environ.setdefault("LOCAL_RANK", "0")

    from config import get_config
    from test_utils import generate_final_results_dance

    cli = Namespace(
        cfg=args.cfg,
        opts=["data_root", args.data_root],
        batch_size=None,
        val_batch_size=None,
        visual="rope-swin-tiny",
        text="roberta",
        dataset=None,
        freeze_text=None,
        freeze_visual=None,
        track_root=args.track_root,
        data_path=None,
        pretrained=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=None,
        disable_amp=None,
        amp_opt_level=None,
        output=str(Path(args.output).resolve()),
        tag=None,
        eval=True,
        throughput=False,
        fused_window_process=None,
        fused_layernorm=None,
        optim=None,
        lre=None,
        local_rank=0,
    )
    config = get_config(cli)
    with open(args.input) as handle:
        cls_dict = json.load(handle)

    generate_final_results_dance(
        config=config,
        cls_dict=cls_dict,
        data_dir=str(Path(args.data_root).resolve()),
        track_dir=str(Path(args.track_root).resolve()),
        save_dir=str(Path(args.output).resolve()),
    )
    print(f"materialized={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
