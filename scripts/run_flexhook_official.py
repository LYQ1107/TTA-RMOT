#!/usr/bin/env python3
"""Run the official FlexHook inference path on one GPU.

This runner keeps the official model, dataloader, checkpoint loader and
result writer, while making two protocol fixes required for a single-card
reproduction:

1. move token ids/masks to the model device before the Roberta forward;
2. use a GT-free runtime data/tracker view and only attach evaluator GT after
   prediction output has been sealed.

The external FlexHook checkout is never modified.  Raw predictions and model
outputs belong under ``/data3``; only the lightweight receipt is suitable for
the TTA-RMOT repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/data2/user/reference_repos/FlexHook"))
    parser.add_argument("--config", default="configs/infer/kitti2.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2-runtime"))
    parser.add_argument("--evaluation-data-root", type=Path,
                        default=Path("/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2"))
    parser.add_argument("--track-root", type=Path,
                        default=Path("/data3/user/TTA-RMOT-assets/flexhook/tracker_outputs/Temp-NeuralSORT-kitti2-runtime"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual", default="rope-swin-tiny")
    parser.add_argument("--text", default="roberta")
    parser.add_argument("--dataset", default="kitti-2")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="smoke only; do not generate/evaluate a partial result")
    parser.add_argument("--skip-trackeval", action="store_true")
    return parser.parse_args()


def _patch_video_split(dataset_name: str, videos: list[str] | None) -> None:
    if not videos:
        return
    from data import utils as data_utils
    requested = list(videos)
    known = set(data_utils.VIDEOS[dataset_name]["test"])
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"videos not in official {dataset_name} test split: {unknown}")
    data_utils.VIDEOS[dataset_name]["test"] = requested


def _make_scores(model: Any, loader: Any, torch: Any, logger: Any,
                 limit_batches: int, log_every: int) -> tuple[Any, int, float]:
    model.eval()
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    start = time.time()
    processed = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            images, pes, bbox_gt, exp, expid, expma, video, obj, frame_id, last = batch
            images = images.cuda(non_blocking=True)
            pes = pes.cuda(non_blocking=True)
            bbox_gt = bbox_gt.cuda(non_blocking=True)
            expid = expid.cuda(non_blocking=True)
            expma = expma.cuda(non_blocking=True)
            output, _ = model(images, pes, bbox_gt, expid, expma)
            if output.ndim == 4:
                output = output[:, -1, :, :]
            output = output.detach().cpu().numpy()
            for batch_index in range(output.shape[0]):
                last_count = int(last[batch_index])
                for expression_index in range(output.shape[1]):
                    if expression_index < output.shape[1] - last_count:
                        continue
                    scores[video[batch_index]][obj[batch_index]][frame_id[batch_index]][
                        exp[expression_index][batch_index]
                    ] = output[batch_index, expression_index].tolist()
            processed += 1
            if index % log_every == 0:
                logger.info(
                    f"FlexHook inference [{index}/{len(loader)}] "
                    f"elapsed={time.time()-start:.1f}s "
                    f"max_mem={torch.cuda.max_memory_allocated()/1024**2:.0f}MB"
                )
            if limit_batches and processed >= limit_batches:
                break
    return scores, processed, time.time() - start


def _run_trackeval(repo_root: Path, evaluation_data_root: Path, save_dir: Path) -> int:
    command = [
        sys.executable,
        str(repo_root / "TrackEval/scripts/run_mot_challenge.py"),
        "--METRICS", "HOTA",
        "--SEQMAP_FILE", str(repo_root / "seqmaps/kitti-2.txt"),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", str(evaluation_data_root / "gt_template"),
        "--TRACKERS_FOLDER", str(save_dir),
        "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--TRACKERS_TO_EVAL", str(save_dir),
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", "2",
        "--PLOT_CURVES", "False",
    ]
    return subprocess.run(command, cwd=str(repo_root), check=False).returncode


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    data_root = args.data_root.resolve()
    evaluation_data_root = args.evaluation_data_root.resolve()
    track_root = args.track_root.resolve()
    checkpoint = args.checkpoint.resolve()
    for required in [repo_root / args.config, checkpoint, data_root / "KITTI/training/image_02",
                     data_root / "expression", data_root / "labels.json", track_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(repo_root)
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    sys.path.insert(0, str(repo_root))

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if not torch.cuda.is_available():
        raise RuntimeError("FlexHook official runner requires CUDA")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)

    # Import official code only after the repository path and process group are ready.
    import main as official_main
    from data.mydataloader import RMOT_Dataset
    from models import build_model
    from logger import create_logger

    sys.argv = [
        "flexhook_official", "--cfg", str(args.config),
        "--visual", args.visual, "--text", args.text, "--eval",
        "--resume", str(checkpoint), "--track-root", str(track_root),
        "--val-batch-size", str(args.batch_size), "--output", str(output),
        "--dataset", args.dataset, "--data-path", "12",
    ]
    parsed, config = official_main.parse_option()
    config.defrost()
    config.data_root = str(data_root)
    config.track_root = str(track_root)
    config.DATA.VAL_BATCH_SIZE = args.batch_size
    config.DATA.NUM_WORKERS = args.num_workers
    config.MODEL.PRETRAINED = str(checkpoint)
    config.MODEL.RESUME = str(checkpoint)
    config.OUTPUT = str(output)
    config.freeze()

    _patch_video_split(args.dataset, args.videos)
    logger = create_logger(output_dir=str(output), dist_rank=0, name=f"{args.visual}+{args.text}")
    logger.info(json.dumps({
        "protocol": "official_flexhook_inference_single_card",
        "data_root": str(data_root),
        "evaluation_data_root": str(evaluation_data_root),
        "track_root": str(track_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "videos": args.videos,
        "limit_batches": args.limit_batches,
    }, ensure_ascii=False, sort_keys=True))

    dataset = RMOT_Dataset(mode="test", opt=config)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=config.DATA.VAL_BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS, pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False, persistent_workers=(config.DATA.NUM_WORKERS > 0),
    )
    logger.info(f"runtime_dataset_samples={len(dataset.data_keys)} batches={len(loader)}")

    torch.cuda.reset_peak_memory_stats()
    model = build_model(config).cuda().eval()
    official_main.load_ckpt_my(config, model, logger)
    scores, processed_batches, elapsed = _make_scores(
        model, loader, torch, logger, args.limit_batches, args.log_every
    )
    full = not args.limit_batches
    scores_path = output / ("prediction_scores.json" if full else "prediction_scores_partial.json")
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, ensure_ascii=False)

    receipt = {
        "format": "ttarmot-flexhook-official-inference-v1",
        "status": "PREDICTION_COMPLETE" if full else "SMOKE_COMPLETE",
        "official_repo": str(repo_root),
        "config": str(repo_root / args.config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "runtime_data_root": str(data_root),
        "evaluation_data_root": str(evaluation_data_root),
        "runtime_tracker_root": str(track_root),
        "videos": args.videos if args.videos else ["0005", "0011", "0013", "0019"],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "processed_batches": processed_batches,
        "runtime_dataset_samples": len(dataset.data_keys),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gt_used_during_inference": False,
        "future_frames_used": False,
        "prediction_scores": str(scores_path),
    }

    if full:
        # This is the protocol boundary: no evaluator GT is opened before this point.
        prediction_root = output / "results"
        official_main.generate_final_results(
            config=config,
            cls_dict=scores,
            data_dir=str(evaluation_data_root),
            track_dir=str(track_root),
            save_dir=str(prediction_root),
        )
        receipt["prediction_root"] = str(prediction_root)
        if args.skip_trackeval:
            receipt["evaluation_status"] = "SKIPPED_BY_FLAG"
        else:
            return_code = _run_trackeval(repo_root, evaluation_data_root, prediction_root)
            receipt["trackeval_return_code"] = return_code
            receipt["evaluation_status"] = "EVALUATION_COMPLETE" if return_code == 0 else "EVALUATION_FAILED"
            if return_code != 0:
                (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
                return return_code

    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
