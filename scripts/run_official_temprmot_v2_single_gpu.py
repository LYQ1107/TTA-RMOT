#!/usr/bin/env python3
"""Run the released TempRMOT ``inference.py`` Detector on the full seqmap.

The official Detector class and post-processing are imported from the pinned
source tree.  This entry point only removes the released eight-process
launcher so the official path can be executed on one selected GPU and with
the exact 862-entry V2 seqmap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _add_source_paths(source_root: Path) -> None:
    paths = [
        ROOT / "tools" / "temprmot_compat",
        Path("/data2/user/reference_envs/temprmot/site-packages"),
        Path("/data2/user/runtime_site_locatemot"),
        source_root / "models" / "ops",
        source_root,
    ]
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # This wrapper adds the compatibility path after Python startup, so the
    # interpreter will not have auto-imported sitecustomize. Import the
    # project file by absolute path because the shared runtime has its own
    # unrelated /usr/lib/python*/sitecustomize.py.
    import importlib.util
    compat_file = ROOT / "tools" / "temprmot_compat" / "sitecustomize.py"
    compat_spec = importlib.util.spec_from_file_location("_ttarmot_sitecustomize", compat_file)
    if compat_spec is None or compat_spec.loader is None:
        raise ImportError(f"cannot load TempRMOT compatibility shim: {compat_file}")
    compat_module = importlib.util.module_from_spec(compat_spec)
    compat_spec.loader.exec_module(compat_module)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/data2/user/reference_envs/temprmot/source"))
    parser.add_argument("--rmot-path", type=Path, default=Path("/data3/user/TTA-RMOT-assets/refer_kitti_v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seqmap", type=Path,
                        default=Path("/data2/user/reference_repos/TempRMOT/datasets/data_path/seqmap.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--checkpoint-id", type=int, default=60)
    parser.add_argument("--hist-len", type=int, default=5)
    parser.add_argument("--sampler-steps", type=int, nargs="+", default=[60, 80, 90])
    parser.add_argument("--sampler-lengths", type=int, nargs="+", default=[5, 5, 5, 5])
    parser.add_argument(
        "--extra-track-attn", action="store_true",
        help="opt into the upstream test shell's extra decoder attention; the released RK2 checkpoint does not contain its weights",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _add_source_paths(args.source_root.resolve())

    import torch
    from inference import Detector
    from main import get_args_parser, build_model
    from util.tool import load_model

    seq_nums: list[list[str]] = []
    for raw in args.seqmap.resolve().read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            video, expression_id = line.split("+", 1)
            seq_nums.append([video, f"{expression_id}.json"])
    if len(seq_nums) != 862:
        raise AssertionError(f"expected 862 official seqmap entries, found {len(seq_nums)}")

    source_args_parser = argparse.ArgumentParser(parents=[get_args_parser()])
    source_argv = [
        "--meta_arch", "temp_rmot", "--dataset_file", "e2e_rmot", "--with_box_refine",
        "--rmot_path", str(args.rmot_path.resolve()), "--resume", str(args.checkpoint.resolve()),
        "--output_dir", str(args.output_dir.resolve()), "--exp_name", "official_seqmap",
        "--query_interaction_layer", "QIM", "--update_query_pos", "--merger_dropout", "0",
        "--dropout", "0", "--random_drop", "0.1", "--fp_ratio", "0.3",
        "--sampler_steps", *[str(value) for value in args.sampler_steps],
        "--sampler_lengths", *[str(value) for value in args.sampler_lengths],
        "--hist_len", str(args.hist_len), "--use_checkpoint", "--device", "cuda",
    ]
    if args.extra_track_attn:
        source_argv.append("--extra_track_attn")
    source_args = source_args_parser.parse_args(source_argv)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    model, _criterion, _postprocessors = build_model(source_args)
    model = load_model(model, str(args.checkpoint.resolve()))
    model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    completed = 0
    for seq_num in seq_nums:
        detector = Detector(source_args, checkpoint_id=args.checkpoint_id, model=model, seq_num=seq_num)
        detector.detect()
        completed += 1
        if completed % 25 == 0:
            print(f"official_seqmap_completed={completed}/{len(seq_nums)}", flush=True)
    metadata = {
        "format": "temprmot-official-inference-v2",
        "status": "PREDICTION_COMPLETE",
        "source_root": str(args.source_root.resolve()),
        "source_commit": __import__("subprocess").check_output(["git", "-C", str(args.source_root.resolve()), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "seqmap": str(args.seqmap.resolve()),
        "seqmap_entry_count": len(seq_nums),
        "completed_entries": completed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "official_detector_class": "inference.Detector",
        "gt_written_after_each_prediction": True,
        "note": "This is the author's Detector/post-processing path; source-free TTA uses a separate GT-free runner.",
    }
    (args.output_dir.resolve() / "official_inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
