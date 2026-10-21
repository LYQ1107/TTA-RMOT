#!/usr/bin/env python3
"""Audit whether a TempRMOT source run actually completed its budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_dataset_length(source_root: Path, rmot_path: Path, train_list: Path,
                           sampler_steps: list[int], sampler_lengths: list[int]) -> tuple[list[int], list[str]]:
    compat = ROOT / "tools" / "temprmot_compat"
    site = Path("/data2/user/reference_envs/temprmot/site-packages")
    runtime_site = Path("/data2/user/runtime_site_locatemot")
    ops = source_root / "models" / "ops"
    for value in (compat, site, runtime_site, ops, source_root):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    import argparse as argparse_module
    from datasets import build_dataset
    from main import get_args_parser

    parser = argparse_module.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args([
        "--meta_arch", "temp_rmot", "--dataset_file", "e2e_rmot",
        "--rmot_path", str(rmot_path), "--data_txt_path_train", str(train_list),
        "--sampler_steps", *[str(value) for value in sampler_steps],
        "--sampler_lengths", *[str(value) for value in sampler_lengths],
        "--sample_mode", "random_interval", "--sample_interval", "1",
        "--hist_len", "5",
    ])
    dataset = build_dataset("train", args)
    lengths: list[int] = []
    for epoch in range(60):
        dataset.set_epoch(epoch)
        lengths.append(int(len(dataset)))
    videos = sorted(path.split("/")[-1] for path in dataset.video_dict)
    return lengths, videos


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/source_training/temprmot_refer_kitti_v2_single_gpu.json")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/ShiftRMOT-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/source_training/temprmot_refer_kitti_v2_completion_audit.json")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_list = Path(config["official_train_list"]).resolve()
    source_root = Path(config["source_runtime_tree"]).resolve()
    rmot_path = Path("/data3/user/TTA-RMOT-assets/refer_kitti_v2")
    lengths, train_videos = _source_dataset_length(
        source_root, rmot_path, train_list,
        list(config["sampler_steps"]), list(config["sampler_lengths"]),
    )
    batch_size = int(config["batch_size"])
    expected_steps = sum(length // batch_size for length in lengths)
    checkpoint = run_root / "checkpoint.pth"
    command_file = run_root / "command.txt"
    checkpoint_info: dict[str, Any] = {"exists": checkpoint.is_file()}
    actual_steps = 0
    epochs_completed = 0
    if checkpoint.is_file():
        import torch
        package = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(package.get("epoch", -1))
        epochs_completed = epoch + 1
        checkpoint_info.update({
            "epoch_field": epoch,
            "epochs_completed": epochs_completed,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "has_model": isinstance(package.get("model"), dict),
            "has_optimizer": isinstance(package.get("optimizer"), dict),
            "has_scheduler": isinstance(package.get("lr_scheduler"), dict),
        })
        state_steps = []
        for state in package.get("optimizer", {}).get("state", {}).values():
            value = state.get("step")
            if value is not None:
                state_steps.append(int(value.item() if hasattr(value, "item") else value))
        actual_steps = max(state_steps, default=0)
        checkpoint_info["optimizer_state_step_max"] = actual_steps
    command_text = command_file.read_text(encoding="utf-8") if command_file.is_file() else ""
    exit_status = None
    for line in command_text.splitlines():
        if line.startswith("exit_status="):
            exit_status = int(line.split("=", 1)[1])
    base_episodes = []
    base_manifest = args.dataset_root.resolve() / "manifests/base_episodes.jsonl"
    if base_manifest.is_file():
        base_episodes = [json.loads(line) for line in base_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_expression_count = sum(
        bool(row.get("official_source_train_video")) for row in base_episodes
    )
    status = "COMPLETE" if (
        epochs_completed >= int(config["epochs_required"])
        and actual_steps >= expected_steps
        and checkpoint_info.get("has_model")
        and checkpoint_info.get("has_optimizer")
        and checkpoint_info.get("has_scheduler")
        and exit_status == 0
    ) else ("FAILED" if exit_status not in (None, 0) else "IN_PROGRESS_OR_INCOMPLETE")
    report = {
        "format": "temprmot-training-completion-audit-v1",
        "training_status": status,
        "source_dataset": config["source_dataset"],
        "official_train_split_hash": config["official_train_list_sha256"],
        "source_video_count": len(train_videos),
        "source_video_ids": train_videos,
        "source_expression_count": source_expression_count,
        "source_training_sample_count_expected": sum(lengths),
        "dataset_length_by_epoch": lengths,
        "batch_size": batch_size,
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "optimizer_steps_expected": expected_steps,
        "optimizer_steps_actual": actual_steps,
        "epochs_required": int(config["epochs_required"]),
        "epochs_completed": epochs_completed,
        "model_name": config["model"],
        "official_code_commit": config["official_code_commit"],
        "training_config_sha256": _sha256(config_path),
        "initial_checkpoint_sha256": config["initialization"]["sha256"],
        "checkpoint_final_path": str(checkpoint),
        "checkpoint": checkpoint_info,
        "command_file": str(command_file),
        "command_exit_status": exit_status,
        "train_dataset_coverage": {
            "video_count": len(train_videos), "video_ids": train_videos,
            "official_image_path_count": int(config["source_train_image_paths"]),
        },
        "validation_protocol": "official Refer-KITTI-V2 seqmap evaluation is a separate post-training step",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "note": "COMPLETE is emitted only after final checkpoint, optimizer/scheduler state, full epochs, expected steps, and clean exit are all present.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
