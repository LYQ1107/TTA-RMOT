"""Sparse, frozen, self-query probe orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rmot_tta.query.query_spec import QuerySpec, QueryVariant
from rmot_tta.types import FrameSnapshot, ProbeCandidate, ProbeResult


@dataclass(frozen=True)
class ProbePacket:
    frame_index: int
    results: tuple[ProbeResult, ...]
    query_spec: QuerySpec
    calls: int
    elapsed_sec: float


class FrozenQueryProbe:
    """Replay a recent causal image window with one private wrapper per query."""

    def __init__(self, wrapper_factory: Callable[[str], Any], every_n_frames: int = 16,
                 dynamic_history_frames: int = 4, max_queries: int = 3):
        self.wrapper_factory = wrapper_factory
        self.every_n_frames = int(every_n_frames)
        self.dynamic_history_frames = int(dynamic_history_frames)
        self.max_queries = int(max_queries)
        self._runtimes: dict[str, Any] = {}
        self.calls = 0

    def reset(self) -> None:
        self._runtimes.clear()
        self.calls = 0

    def due(self, frame_index: int) -> bool:
        return frame_index == 0 or (frame_index % self.every_n_frames == 0)

    def _runtime(self, query_type: str):
        if query_type not in self._runtimes:
            from .shadow_state import ShadowRuntime
            self._runtimes[query_type] = ShadowRuntime(self.wrapper_factory(query_type), query_type)
        return self._runtimes[query_type]

    def run(self, ring: list[tuple[int, Any, tuple[int, int]]], spec: QuerySpec,
            variants: Iterable[QueryVariant], main_snapshot: FrameSnapshot) -> ProbePacket:
        selected = [variant for variant in variants if variant.query_type in {"q0", "q_eq", "q_cf"}]
        selected = selected[: self.max_queries]
        if not selected:
            return ProbePacket(main_snapshot.frame_index, (), spec, 0, 0.0)
        # Static expressions are evaluated on the current frame only.  A
        # dynamic/mixed expression replays the bounded history so every query
        # starts from the same cold shadow state and reaches the same final
        # timestamp.
        window = (ring[-self.dynamic_history_frames:]
                  if "dynamic" in spec.constraint_types or "mixed" in spec.constraint_types
                  else ring[-1:])
        started = time.perf_counter()
        results: list[ProbeResult] = []
        for variant in selected:
            runtime = self._runtime(variant.query_type)
            runtime.reset()
            final = None
            for frame_index, image_tensor, image_size in window:
                final = runtime.step(image_tensor, variant.text, image_size, frame_index)
            if final is None or final.snapshot is None:
                result = ProbeResult(variant.query_type, main_snapshot.frame_index, variant.text, (),
                                     time.perf_counter() - started, tuple(item[0] for item in window), "NO_CANDIDATES")
            else:
                snapshot = final.snapshot
                candidates = tuple(
                    ProbeCandidate(
                        candidate_uid=uid,
                        box_xyxy=tuple(float(value) for value in box),
                        probability=float(probability),
                        logit=float(logit),
                        track_id=int(track_id),
                    )
                    for uid, box, probability, logit, track_id in zip(
                        snapshot.candidate_uid,
                        snapshot.boxes_xyxy.detach().cpu().tolist(),
                        snapshot.source_probs.detach().cpu().tolist(),
                        snapshot.source_logits.detach().cpu().tolist(),
                        snapshot.pred_track_id.detach().cpu().tolist(),
                    ) if int(track_id) >= 0
                )
                result = ProbeResult(variant.query_type, main_snapshot.frame_index, variant.text, candidates,
                                     time.perf_counter() - started, tuple(item[0] for item in window), "OK")
            results.append(result)
            self.calls += 1
        return ProbePacket(main_snapshot.frame_index, tuple(results), spec, len(results), time.perf_counter() - started)
