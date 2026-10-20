"""Only exact image-plane auxiliary cues; no appearance-class shortcuts."""
from __future__ import annotations

import torch


def image_left_right_evidence(boxes_xyxy: torch.Tensor, side: str) -> torch.Tensor:
    """Return a conservative mask for an image-center left/right condition."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=boxes_xyxy.device)
    center = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
    midpoint = 0.5 * (boxes_xyxy[:, 0].min() + boxes_xyxy[:, 2].max())
    if side == "left":
        return center < midpoint
    if side == "right":
        return center > midpoint
    raise ValueError(f"unsupported image-plane side: {side}")
