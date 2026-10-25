"""Causal Dance episode construction from images and fixed MOT rows."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


@dataclass(frozen=True)
class Observation:
    frame_id: int
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    raw_line: str


class DanceEpisode:
    """One video and one expression, with only causal tracker history."""

    def __init__(self, dataset_root: Path, tracker_root: Path, video: str, *, image_cache_size: int = 12):
        self.dataset_root = Path(dataset_root)
        self.video = video
        self.image_root = self.dataset_root / "DanceTrack" / "training" / "image_02" / video
        self.observations_by_frame: dict[int, dict[int, Observation]] = defaultdict(dict)
        self.observations_by_track: dict[int, list[Observation]] = defaultdict(list)
        tracker_path = Path(tracker_root) / f"{video}.txt"
        for number, raw in enumerate(tracker_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = raw.strip().split(",")
            if len(fields) != 10:
                raise ValueError(f"invalid tracker row {tracker_path}:{number}")
            frame_id = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, w, h = (float(fields[index]) for index in (2, 3, 4, 5))
            observation = Observation(frame_id, track_id, (x, y, x + w, y + h), raw.strip())
            self.observations_by_frame[frame_id][track_id] = observation
            self.observations_by_track[track_id].append(observation)
        self.observations_by_frame = {frame: dict(sorted(rows.items())) for frame, rows in sorted(self.observations_by_frame.items())}
        self.observations_by_track = {track: sorted(rows, key=lambda row: row.frame_id) for track, rows in self.observations_by_track.items()}
        self.track_index = {
            track: {observation.frame_id: index for index, observation in enumerate(rows)}
            for track, rows in self.observations_by_track.items()
        }
        self.frames = sorted(self.observations_by_frame)
        self.transform = T.Compose([
            T.Resize((224, 672)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.image_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.image_cache_size = int(image_cache_size)
        self.width, self.height = self._read_resolution()

    def _read_resolution(self) -> tuple[int, int]:
        path = self.image_root / f"{self.frames[0]:06d}.png"
        with Image.open(path) as image:
            return image.width, image.height

    def _image(self, frame_id: int) -> torch.Tensor:
        if frame_id in self.image_cache:
            value = self.image_cache.pop(frame_id)
            self.image_cache[frame_id] = value
            return value
        path = self.image_root / f"{frame_id:06d}.png"
        with Image.open(path) as image:
            value = self.transform(image.convert("RGB"))
        self.image_cache[frame_id] = value
        while len(self.image_cache) > self.image_cache_size:
            self.image_cache.popitem(last=False)
        return value

    def history(self, track_id: int, frame_id: int, length: int = 4) -> list[Observation]:
        rows = self.observations_by_track[int(track_id)]
        index = self.track_index[int(track_id)][int(frame_id)]
        selected = rows[max(0, index - length + 1): index + 1]
        if len(selected) < length:
            selected = [selected[0]] * (length - len(selected)) + selected
        return selected

    def current_candidates(self, frame_id: int, allowed_track_ids: set[int] | None = None):
        rows = self.observations_by_frame[int(frame_id)]
        for track_id in sorted(rows):
            if allowed_track_ids is None or track_id in allowed_track_ids:
                yield track_id, self.history(track_id, frame_id)

    def _grid(self, bbox: tuple[float, float, float, float]) -> torch.Tensor:
        x1, y1, x2, y2 = (bbox[index] / (self.width if index in (0, 2) else self.height) for index in range(4))
        grid_w = (x2 - x1) / 48.0
        grid_h = (y2 - y1) / 16.0
        x = torch.linspace(x1 + grid_w / 2, x2 - grid_w / 2, 48) * 2 - 1
        y = torch.linspace(y1 + grid_h / 2, y2 - grid_h / 2, 16) * 2 - 1
        x, y = torch.meshgrid(x, y, indexing="xy")
        return torch.stack([x, y])

    def geometry_inputs(self, candidates: list[tuple[int, list[Observation]]], device: torch.device):
        positions = []
        boxes = []
        track_ids = []
        for track_id, history in candidates:
            image_history_boxes = []
            image_history_positions = []
            for item in history:
                x1, y1, x2, y2 = item.bbox_xyxy
                nx1, nx2 = x1 / self.width, x2 / self.width
                ny1, ny2 = y1 / self.height, y2 / self.height
                image_history_boxes.append([(nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1])
                image_history_positions.append(self._grid(item.bbox_xyxy))
            positions.append(torch.stack(image_history_positions))
            boxes.append(torch.tensor(image_history_boxes, dtype=torch.float32))
            track_ids.append(int(track_id))
        return (
            torch.stack(positions).to(device, non_blocking=True),
            torch.stack(boxes).to(device, non_blocking=True),
            track_ids,
        )

    def batch_inputs(self, candidates: list[tuple[int, list[Observation]]], device: torch.device):
        images = [torch.stack([self._image(item.frame_id) for item in history])
                  for _, history in candidates]
        positions, boxes, ids = self.geometry_inputs(candidates, device)
        return torch.stack(images).to(device, non_blocking=True), positions, boxes, ids
