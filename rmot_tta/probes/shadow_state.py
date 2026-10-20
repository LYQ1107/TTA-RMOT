"""Explicitly isolated state for a frozen query replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ShadowState:
    track_instances: Any = None
    next_id_before: int = 0

    def reset(self) -> None:
        self.track_instances = None
        self.next_id_before = 0


class ShadowRuntime:
    """Owns a model copy and never borrows main-stream tracker state."""

    def __init__(self, wrapper: Any, query_type: str):
        self.wrapper = wrapper
        self.query_type = query_type
        self.state = ShadowState()

    def reset(self) -> None:
        self.state.reset()
        self.wrapper.reset_runtime()

    def step(self, image_tensor: Any, sentence: str, image_size: tuple[int, int], frame_index: int):
        result = self.wrapper.infer_frame(
            image_tensor=image_tensor,
            sentence=sentence,
            image_size=image_size,
            track_instances=self.state.track_instances,
            frame_index=frame_index,
            episode_id=f"shadow:{self.query_type}",
            expression_id=self.query_type,
        )
        self.state.track_instances = result.track_instances
        return result
