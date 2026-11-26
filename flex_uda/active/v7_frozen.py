"""Frozen v7 arm: sealed margins/Stage-A arrays only, no model or PCD loading."""
import time
import numpy as np
from flex_uda.active.v6 import selection_fractions
from scripts.active_v6_preflight import stage_arrays


class FrozenFrameInputs:
    def __init__(self, stage_a, target, model, video, bank, queries, margins):
        self.bank, self.queries = bank, queries
        self.observations = np.asarray(bank['observations'])
        self.tracks = sorted({int(t) for t in self.observations[:, 1]})
        self.by_track = {t: np.flatnonzero(self.observations[:, 1] == t) for t in self.tracks}
        self.margin = np.stack([margins[q['start']:q['stop']] for q in queries])
        arrays, self.frame_index = stage_arrays(stage_a, target, model, video, len(queries))
        self.raw_presence = np.asarray(arrays['presence_logits'], dtype=float)
        self.utilities = arrays['expected_iou_utilities']; self.raw_k = arrays['raw_k']
        self.decode_seconds = 0.; self.score_records = []

    def decode(self, temperature, bias):
        begin = time.perf_counter()
        result = selection_fractions(self.observations, self.tracks, self.frame_index, self.margin,
            self.utilities, self.raw_presence, temperature=temperature, bias=bias, raw_k=self.raw_k)
        self.decode_seconds += time.perf_counter()-begin
        return result

    def refresh(self, state, temperature, bias, **unused):
        fractions, probability, _ = self.decode(temperature, bias)
        state.replace_fractions(fractions); state.refresh_presence(probability.max(1))
        state.replace_prior(scores=self.track_means())

    def track_means(self):
        return np.stack([self.margin[:, self.by_track[t]].mean(axis=1) for t in self.tracks], axis=1)
