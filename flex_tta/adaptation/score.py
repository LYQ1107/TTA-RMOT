"""Small source-free online adapters that never access target labels.

The adapter receives only the frozen referring margin and tracker identity at
the current frame.  It maintains a causal affine calibration (scale/bias) and
uses persistence of a track's own previous confidence as a pseudo-label.  The
tracker boxes and IDs are untouched.  A zero-update frozen adapter is provided
for apples-to-apples comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from flex_tta.types import OnlineStep


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class FrozenScoreAdapter:
    """Return the frozen source margin without state or adaptation."""

    def reset_episode(self) -> None:
        return None

    def step(self, frame_id: int, margins: dict[int, float]) -> OnlineStep:
        return OnlineStep(int(frame_id), dict(margins), 0, 1.0, 0.0)


@dataclass
class _Previous:
    margin: float
    frame_id: int


class CausalStabilityTTA:
    """Online affine calibration driven by temporally persistent tracks.

    ``confidence_high``/``confidence_low`` define pseudo-label anchors.  A
    positive anchor requires a candidate that was confidently positive in the
    immediately previous frame and remains present.  The negative rule is the
    symmetric low-confidence case.  Ambiguous or newly born tracks do not
    update the adapter.  This keeps target labels, future frames, and target
    evaluation files outside the runtime call path.
    """

    def __init__(self, *, lr: float = 1e-3, confidence_high: float = 0.8,
                 confidence_low: float = 0.2, max_gap: int = 1,
                 min_scale: float = 0.75, max_scale: float = 1.25,
                 max_abs_bias: float = 0.75):
        if not 0.5 < confidence_high < 1.0:
            raise ValueError("confidence_high must be in (0.5, 1)")
        if not 0.0 < confidence_low < 0.5:
            raise ValueError("confidence_low must be in (0, 0.5)")
        self.lr = float(lr)
        self.confidence_high = float(confidence_high)
        self.confidence_low = float(confidence_low)
        self.max_gap = int(max_gap)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_abs_bias = float(max_abs_bias)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.scale = 1.0
        self.bias = 0.0
        self.previous: dict[int, _Previous] = {}
        self.pseudo_updates = 0
        self.positive_updates = 0
        self.negative_updates = 0

    def _pseudo_target(self, previous_margin: float, current_margin: float) -> float | None:
        previous = _sigmoid(previous_margin)
        current = _sigmoid(current_margin)
        if previous >= self.confidence_high and current >= 0.5:
            return 1.0
        if previous <= self.confidence_low and current <= 0.5:
            return 0.0
        return None

    def step(self, frame_id: int, margins: dict[int, float]) -> OnlineStep:
        frame_id = int(frame_id)
        adapted: dict[int, float] = {}
        updates = 0
        for track_id, source_margin in sorted(margins.items()):
            source_margin = float(source_margin)
            calibrated = self.scale * source_margin + self.bias
            previous = self.previous.get(int(track_id))
            if previous is not None and frame_id - previous.frame_id <= self.max_gap:
                target = self._pseudo_target(previous.margin, calibrated)
                if target is not None:
                    probability = _sigmoid(calibrated)
                    error = probability - target
                    # One SGD step on BCE(scale * source + bias), then clamp
                    # the small adapter so it cannot override the source model.
                    self.scale -= self.lr * error * source_margin
                    self.bias -= self.lr * error
                    self.scale = min(self.max_scale, max(self.min_scale, self.scale))
                    self.bias = min(self.max_abs_bias, max(-self.max_abs_bias, self.bias))
                    calibrated = self.scale * source_margin + self.bias
                    updates += 1
                    self.pseudo_updates += 1
                    if target > 0.5:
                        self.positive_updates += 1
                    else:
                        self.negative_updates += 1
            adapted[int(track_id)] = calibrated
            self.previous[int(track_id)] = _Previous(calibrated, frame_id)

        # Stale track states cannot create pseudo labels after a long gap.
        self.previous = {
            track_id: item for track_id, item in self.previous.items()
            if frame_id - item.frame_id <= self.max_gap
        }
        return OnlineStep(frame_id, adapted, updates, self.scale, self.bias)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "scale": float(self.scale),
            "bias": float(self.bias),
            "pseudo_updates": int(self.pseudo_updates),
            "positive_updates": int(self.positive_updates),
            "negative_updates": int(self.negative_updates),
        }
class SourcePriorCalibration:
    """Apply a source-fixed temperature to the frozen margin.

    The source statistics are computed once on the source-domain score stream
    and are never estimated from Dance.  Since the released FlexHook decision
    is the sign of the two-class margin, a positive temperature is deliberately
    prediction-equivalent to FROZEN; this is a calibration control, not a
    hidden target-threshold search.
    """

    def __init__(self, *, source_mean: float, source_std: float):
        if source_std <= 0:
            raise ValueError("source_std must be positive")
        self.source_mean = float(source_mean)
        self.source_std = float(source_std)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.steps = 0

    def step(self, frame_id: int, margins: dict[int, float]) -> OnlineStep:
        self.steps += 1
        # Keep the source decision boundary at margin=0.  The mean is retained
        # as provenance for the source score population, while temperature is
        # the only applied transform so this control is exactly FROZEN at the
        # binary prediction level.
        adapted = {
            int(track_id): float(margin) / self.source_std
            for track_id, margin in margins.items()
        }
        # The returned scale/bias describe the fixed transform; no target
        # state is updated.
        return OnlineStep(int(frame_id), adapted, 0, 1.0 / self.source_std, 0.0)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "source_mean": self.source_mean,
            "source_std": self.source_std,
            "steps": int(self.steps),
        }


class QueryTrackletCalibration:
    """Conservative causal score smoothing using only each track's history.

    The source margin remains the anchor and the previous adapted margin is a
    bounded temporal cue.  No frame-level top-k or positive-rate assumption is
    introduced, so empty-target frames are allowed.  This is intentionally
    separate from ``CausalStabilityTTA``: it does not manufacture BCE
    pseudo-labels or update a global affine parameter.
    """

    def __init__(self, *, history_weight: float = 0.2, max_gap: int = 1):
        if not 0.0 <= history_weight < 1.0:
            raise ValueError("history_weight must be in [0, 1)")
        self.history_weight = float(history_weight)
        self.max_gap = int(max_gap)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.previous: dict[int, _Previous] = {}
        self.history_updates = 0

    def step(self, frame_id: int, margins: dict[int, float]) -> OnlineStep:
        frame_id = int(frame_id)
        adapted: dict[int, float] = {}
        updates = 0
        for track_id, source_margin in sorted(margins.items()):
            track_id = int(track_id)
            source_margin = float(source_margin)
            previous = self.previous.get(track_id)
            if previous is not None and frame_id - previous.frame_id <= self.max_gap:
                value = (1.0 - self.history_weight) * source_margin + self.history_weight * previous.margin
                updates += 1
            else:
                value = source_margin
            adapted[track_id] = value
            self.previous[track_id] = _Previous(value, frame_id)

        self.previous = {
            track_id: item for track_id, item in self.previous.items()
            if frame_id - item.frame_id <= self.max_gap
        }
        self.history_updates += updates
        return OnlineStep(frame_id, adapted, updates, 1.0, 0.0)

    def state_dict(self) -> dict[str, float | int]:
        return {
            "history_weight": self.history_weight,
            "max_gap": self.max_gap,
            "history_updates": int(self.history_updates),
        }
