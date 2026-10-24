#!/usr/bin/env python3
"""Prove that the official Dance test loader does not open labels.json."""
from __future__ import annotations

import argparse
import builtins
import os
import sys
from argparse import Namespace
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, default=Path("vendor/FlexHook"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    args = parser.parse_args()
    vendor = args.vendor.resolve()
    sys.path.insert(0, str(vendor))
    os.chdir(vendor)
    os.environ.setdefault("LOCAL_RANK", "0")
    from config import get_config
    from data.mydataloader import RMOT_Dataset

    cli = Namespace(
        cfg=str(vendor / "configs/infer/dance.yaml"),
        opts=["data_root", str(args.data_root.resolve())],
        batch_size=8,
        val_batch_size=8,
        visual="rope-swin-tiny",
        text="roberta",
        dataset=None,
        freeze_text=False,
        freeze_visual=False,
        track_root=str(args.track_root.resolve()),
        data_path=None,
        zip=False,
        cache_mode="part",
        pretrained=None,
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        disable_amp=False,
        amp_opt_level=None,
        output=str(vendor / "gt_free_loader_test_output"),
        tag=None,
        eval=True,
        throughput=False,
        fused_window_process=False,
        fused_layernorm=False,
        optim=None,
        lre=None,
        local_rank=0,
    )
    config = get_config(cli)
    original_open = builtins.open

    def guarded_open(file, *open_args, **open_kwargs):
        if str(file).endswith("labels.json"):
            raise AssertionError(f"target GT opened by test loader: {file}")
        return original_open(file, *open_args, **open_kwargs)

    builtins.open = guarded_open
    try:
        dataset = RMOT_Dataset(mode="test", opt=config)
    finally:
        builtins.open = original_open
    print({"status": "PASS", "data_keys": len(dataset.data_keys), "videos": len(dataset.target_expressions)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
