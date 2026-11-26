#!/usr/bin/env python3
"""Load the pinned source checkpoint and verify one real frame snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.baseline.temprmot_wrapper import TempRMOTConfig, TempRMOTWrapper  # noqa: E402


def preprocess(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 800.0 / min(height, width)
    if max(height, width) * scale > 1536.0:
        scale = 1536.0 / max(height, width)
    target_h, target_w = int(height * scale), int(width * scale)
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std, (height, width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/manifests/refer_kitti_v1_runtime_manifest.jsonl")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    row = json.loads(next(line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()))
    model_cfg = cfg["model"]
    wrapper = TempRMOTWrapper(TempRMOTConfig(
        source_root=Path(model_cfg["source_root"]), checkpoint=Path(model_cfg["checkpoint"]),
        text_root=Path(model_cfg["text_root"]), device="cuda:0",
    ))
    image, image_size = preprocess(Path(row["image_path"]))
    result = wrapper.infer_frame(image, row["query_text"], image_size, None, int(row["frame_index"]),
                                 f"sanity:{row['video_id']}:{row['expression_id']}", row["expression_id"], row["video_id"])
    snapshot = result.snapshot
    valid = snapshot.valid_mask
    box_wh = (snapshot.boxes_xyxy[:, 2:] - snapshot.boxes_xyxy[:, :2]).clamp_min(0)
    box_area = box_wh[:, 0] * box_wh[:, 1]
    valid_objectness = snapshot.objectness[valid]
    valid_source_probs = snapshot.source_probs[valid]
    valid_box_area = box_area[valid]
    payload = {
        "status": "complete", "provenance": wrapper.provenance,
        "video": row["video_id"], "expression": row["expression_id"], "frame_index": row["frame_index"],
        "query_text": row["query_text"], "snapshot_candidates": snapshot.count,
        "valid_candidates": int(snapshot.valid_mask.sum()),
        "output_candidates": len(result.track_instances),
        "feature_shape": list(snapshot.features.shape), "source_prob_min": float(snapshot.source_probs.min()),
        "source_prob_max": float(snapshot.source_probs.max()),
        "valid_objectness_min": float(valid_objectness.min()) if valid_objectness.numel() else None,
        "valid_objectness_max": float(valid_objectness.max()) if valid_objectness.numel() else None,
        "valid_source_prob_min": float(valid_source_probs.min()) if valid_source_probs.numel() else None,
        "valid_source_prob_max": float(valid_source_probs.max()) if valid_source_probs.numel() else None,
        "valid_box_area_min": float(valid_box_area.min()) if valid_box_area.numel() else None,
        "valid_box_area_max": float(valid_box_area.max()) if valid_box_area.numel() else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
