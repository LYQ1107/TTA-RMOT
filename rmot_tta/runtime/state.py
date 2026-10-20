"""Runtime bookkeeping kept separate from model and adaptation state."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class RawFrameRing:
    max_length: int = 4

    def __post_init__(self) -> None:
        self.items: deque = deque(maxlen=int(self.max_length))
        self.bytes_stored = 0

    def clear(self) -> None:
        self.items.clear()
        self.bytes_stored = 0

    def append(self, frame_index: int, image_tensor, image_size: tuple[int, int]) -> None:
        if len(self.items) == self.items.maxlen:
            old = self.items[0]
            self.bytes_stored -= int(old[1].numel() * old[1].element_size())
        image = image_tensor.detach().cpu().contiguous()
        self.items.append((int(frame_index), image, tuple(image_size)))
        self.bytes_stored += int(image.numel() * image.element_size())

    def as_list(self) -> list:
        return list(self.items)
