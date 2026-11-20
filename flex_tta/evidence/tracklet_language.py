"""Causal visual/tracklet evidence for clothing queries.

The evidence in this module is deliberately independent of FlexHook scores.
It uses only the current query text, the fixed track geometry and RGB pixels
inside the causal tracklet boxes.  It is a conservative development method:
only high-confidence color matches and high-confidence color mismatches are
returned as pseudo labels; ambiguous crops are marked unknown.
"""
from __future__ import annotations

import colorsys
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


_COLORS = {
    "black": (0.0, 0.0, 0.08),
    "white": (0.0, 0.0, 0.92),
    "gray": (0.0, 0.12, 0.55),
    "grey": (0.0, 0.12, 0.55),
    "brown": (0.07, 0.62, 0.40),
    "red": (0.0, 0.75, 0.60),
    "orange": (0.08, 0.75, 0.70),
    "yellow": (0.15, 0.75, 0.75),
    "green": (0.33, 0.65, 0.55),
    "blue": (0.60, 0.65, 0.60),
    "purple": (0.78, 0.60, 0.55),
}
_COLOR_WORDS = tuple(_COLORS)


def _hue_distance(first: np.ndarray, second: float) -> np.ndarray:
    return np.abs((first - second + 0.5) % 1.0 - 0.5)


def _query_specs(sentence: str) -> list[tuple[str, str]]:
    text = sentence.lower().replace("-", " ")
    if any(word in text for word in ("all in", "dressed all", "whole")):
        color = next((word for word in _COLOR_WORDS if re.search(rf"\b{word}\b", text)), None)
        return [(color, "full")] if color else []

    specs: list[tuple[str, str]] = []
    garment_pattern = r"(?:t\s*shirt|tshirt|shirt|pants|trousers|skirt|top)"
    for color in _COLOR_WORDS:
        for match in re.finditer(rf"\b{color}\s+({garment_pattern})\b", text):
            garment_word = match.group(1).replace(" ", "")
            garment = "bottom" if garment_word in {"pants", "trousers", "skirt"} else "top"
            specs.append((color, garment))
    if specs:
        return specs

    color = next((word for word in _COLOR_WORDS if re.search(rf"\b{word}\b", text)), None)
    if color is None:
        return []
    if any(word in text for word in ("pants", "trousers", "skirt")):
        return [(color, "bottom")]
    if any(word in text for word in ("shirt", "tshirt", "t shirt", "top")):
        return [(color, "top")]
    return []


class TrackletLanguageEvidence:
    """Compute conservative color evidence from a causal tracklet.

    Returned labels are ``1`` (supported positive), ``-1`` (supported
    negative), or ``0`` (unknown).  No source score, target annotation, or
    future observation is read.
    """

    def __init__(self, image_cache_size: int = 256):
        self.image_cache: OrderedDict[tuple[Path, int], np.ndarray] = OrderedDict()
        self.image_cache_size = int(image_cache_size)

    def _image(self, episode, frame_id: int) -> np.ndarray:
        key = (episode.image_root, int(frame_id))
        if key in self.image_cache:
            value = self.image_cache.pop(key)
            self.image_cache[key] = value
            return value
        path = episode.image_root / f"{int(frame_id):06d}.png"
        with Image.open(path) as image:
            value = np.asarray(image.convert("RGB"))
        self.image_cache[key] = value
        while len(self.image_cache) > self.image_cache_size:
            self.image_cache.popitem(last=False)
        return value

    @staticmethod
    def _crop_bounds(observation, width: int, height: int, garment: str | None):
        x1, y1, x2, y2 = observation.bbox_xyxy
        if garment == "top":
            y1, y2 = y1 + 0.18 * (y2 - y1), y1 + 0.60 * (y2 - y1)
        elif garment == "bottom":
            y1, y2 = y1 + 0.50 * (y2 - y1), y1 + 0.94 * (y2 - y1)
        elif garment == "full":
            y1, y2 = y1 + 0.12 * (y2 - y1), y1 + 0.95 * (y2 - y1)
        pad_x = 0.04 * max(1.0, x2 - x1)
        x1, x2 = x1 + pad_x, x2 - pad_x
        return (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(1, min(width, int(round(x2)))),
            max(1, min(height, int(round(y2)))),
        )

    def _color_score(self, image: np.ndarray, observation, color: str, garment: str | None) -> float:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._crop_bounds(observation, width, height, garment)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = Image.fromarray(image[y1:y2, x1:x2]).resize((32, 32), Image.Resampling.BILINEAR)
        rgb = np.asarray(crop, dtype=np.float32) / 255.0
        hsv = np.empty_like(rgb)
        # Vectorized RGB -> HSV conversion keeps the evidence inexpensive.
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        delta = maximum - minimum
        hsv[..., 2] = maximum
        hsv[..., 1] = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
        hue = np.zeros_like(maximum)
        nonzero = delta > 1e-6
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        red_mask = nonzero & (maximum == red)
        green_mask = nonzero & (maximum == green)
        blue_mask = nonzero & (maximum == blue)
        hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6.0
        hue[green_mask] = (blue[green_mask] - red[green_mask]) / delta[green_mask] + 2.0
        hue[blue_mask] = (red[blue_mask] - green[blue_mask]) / delta[blue_mask] + 4.0
        hsv[..., 0] = hue / 6.0
        target_h, target_s, target_v = _COLORS[color]
        value_error = np.abs(hsv[..., 2] - target_v)
        if color == "black":
            pixel_score = np.clip(1.0 - hsv[..., 2] / 0.34, 0.0, 1.0)
        elif color == "white":
            pixel_score = np.clip((hsv[..., 2] - 0.58) / 0.42, 0.0, 1.0) * np.clip(1.0 - hsv[..., 1] / 0.45, 0.0, 1.0)
        elif color in {"gray", "grey"}:
            pixel_score = np.clip(1.0 - hsv[..., 1] / 0.38, 0.0, 1.0) * np.exp(-value_error * value_error / 0.18)
        else:
            hue_score = np.exp(-_hue_distance(hsv[..., 0], target_h) ** 2 / 0.025)
            saturation_score = np.clip(hsv[..., 1] / max(target_s, 0.35), 0.0, 1.0)
            brightness_score = np.exp(-value_error * value_error / 0.30)
            pixel_score = hue_score * saturation_score * brightness_score
        # Robust aggregation suppresses a few background or skin pixels.
        return float(np.clip(np.quantile(pixel_score, 0.70), 0.0, 1.0))

    @staticmethod
    def _stability(history) -> float:
        if len(history) < 2:
            return 1.0
        displacements = []
        for previous, current in zip(history[:-1], history[1:]):
            px1, py1, px2, py2 = previous.bbox_xyxy
            cx1, cy1, cx2, cy2 = current.bbox_xyxy
            previous_center = ((px1 + px2) * 0.5, (py1 + py2) * 0.5)
            current_center = ((cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5)
            diagonal = max(1.0, ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5)
            displacements.append(((previous_center[0] - current_center[0]) ** 2
                                  + (previous_center[1] - current_center[1]) ** 2) ** 0.5 / diagonal)
        return float(np.exp(-np.median(displacements)))

    def score(self, episode, frame_id: int, track_id: int, sentence: str, history) -> tuple[float, float, int]:
        specs = _query_specs(sentence)
        if not specs:
            return 0.5, 0.0, 0
        per_spec_scores = []
        for color, garment in specs:
            image_scores = [self._color_score(self._image(episode, item.frame_id), item, color, garment) for item in history]
            per_spec_scores.append(float(np.mean(image_scores[-min(3, len(image_scores)):])))
        # A compositional query is supported only when every requested
        # garment attribute is visible; this avoids treating a white shirt as
        # a match for "white shirt and purple pants".
        color_score = float(min(per_spec_scores))
        stability = self._stability(history)
        reliability = float(np.clip(0.5 + 0.5 * stability, 0.5, 1.0))
        # The sign is entirely from RGB clothing evidence, not source logits.
        if color_score >= 0.62 and reliability >= 0.72:
            label = 1
        elif color_score <= 0.22 and reliability >= 0.72:
            label = -1
        else:
            label = 0
        return color_score, reliability, label

    def batch(self, episode, frame_id: int, candidates, sentence: str):
        scores, reliabilities, labels = [], [], []
        for track_id, history in candidates:
            score, reliability, label = self.score(episode, frame_id, track_id, sentence, history)
            scores.append(score)
            reliabilities.append(reliability)
            labels.append(label)
        return scores, reliabilities, labels
