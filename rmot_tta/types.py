"""Small data contracts shared by the QCS-TTA runner and evaluator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class FrameSnapshot:
    """A detached pre-QIM candidate snapshot.

    The row order is intentionally not an identity contract.  ``candidate_uid``
    is present so that later alignment can make row correspondence explicit.
    """

    episode_id: str
    video_id: str
    expression_id: str
    query_text: str
    frame_index: int
    candidate_uid: tuple[str, ...]
    features: torch.Tensor
    source_logits: torch.Tensor
    source_probs: torch.Tensor
    boxes_xyxy: torch.Tensor
    objectness: torch.Tensor
    pred_track_id: torch.Tensor
    disappear_time: torch.Tensor
    image_size: tuple[int, int]
    valid_mask: torch.Tensor
    timestamp_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.candidate_uid)
        expected = {
            "features": (n, 256),
            "source_logits": (n,),
            "source_probs": (n,),
            "boxes_xyxy": (n, 4),
            "objectness": (n,),
            "pred_track_id": (n,),
            "disappear_time": (n,),
            "valid_mask": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be bool")
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"invalid image_size: {self.image_size}")

    @property
    def count(self) -> int:
        return len(self.candidate_uid)

    def detached_clone(self) -> "FrameSnapshot":
        """Return a snapshot that cannot alias model/runtime tensors."""
        tensor_names = (
            "features", "source_logits", "source_probs", "boxes_xyxy",
            "objectness", "pred_track_id", "disappear_time", "valid_mask",
        )
        payload = {name: getattr(self, name).detach().clone() for name in tensor_names}
        return FrameSnapshot(
            episode_id=self.episode_id,
            video_id=self.video_id,
            expression_id=self.expression_id,
            query_text=self.query_text,
            frame_index=self.frame_index,
            candidate_uid=tuple(self.candidate_uid),
            image_size=tuple(self.image_size),
            timestamp_sec=self.timestamp_sec,
            metadata=dict(self.metadata),
            **payload,
        )


@dataclass(frozen=True)
class PredictionFrame:
    """Post-source output with fixed boxes/IDs and a method-specific score."""

    frame_index: int
    boxes_xyxy: torch.Tensor
    track_ids: torch.Tensor
    objectness: torch.Tensor
    source_probs: torch.Tensor
    referring_probs: torch.Tensor
    referring_logits: torch.Tensor
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        n = int(self.track_ids.numel())
        for name in ("objectness", "source_probs", "referring_probs", "referring_logits"):
            if tuple(getattr(self, name).shape) != (n,):
                raise ValueError(f"{name} shape mismatch")
        if tuple(self.boxes_xyxy.shape) != (n, 4):
            raise ValueError("boxes_xyxy shape mismatch")


@dataclass(frozen=True)
class ProbeCandidate:
    candidate_uid: str
    box_xyxy: tuple[float, float, float, float]
    probability: float
    logit: float
    track_id: int


@dataclass(frozen=True)
class ProbeResult:
    query_type: str
    frame_index: int
    query_text: str
    candidates: tuple[ProbeCandidate, ...]
    elapsed_sec: float
    history_frames: tuple[int, ...]
    status: str = "OK"


@dataclass(frozen=True)
class AlignedCandidate:
    main_index: int
    main_uid: str
    probe_indices: dict[str, int | None]
    iou_by_query: dict[str, float]
    ambiguity_by_query: dict[str, str | None]
    r0: float | None
    req: float | None
    rcf: float | None
    alignment_quality: float
    status: str
