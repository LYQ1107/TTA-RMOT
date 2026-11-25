"""Full-video image descriptors without changing the historical slice bank.

Current and anchor descriptors are gathered by physical observation. No
query-expanded full-video feature tensor is stored or kept in memory.
Every future ordinary opponent must use this same input interface.
"""
from functools import lru_cache
import json

import numpy as np
from flex_tta.runtime.pf_receipts import file_sha
from flex_uda.active.appearance_learning_inputs import (
    AppearanceBank, BASE_DIMENSION, APPEARANCE_DIMENSION, DIMENSION)
from scripts.a100_dino_appearance_collect import ANCHORS
from scripts.a100_dino_fullsix_collect import OUT, PLAN


class FullAppearanceBank(AppearanceBank):
    def __init__(self, target, video):
        self.target, self.video = target, video
        folder = OUT/target/video
        receipt = json.loads((folder/'receipt.json').read_text())
        assert receipt['input_plan_sha256'] == file_sha(PLAN)
        for name, sha in receipt['output_sha256'].items(): assert file_sha(folder/name) == sha
        self.features = np.load(folder/'features.npy', mmap_mode='r')
        self.valid = np.load(folder/'valid.npy')
        rows = np.load(folder/'selected_rows.npy'); obs = np.load(folder/'observations.npy')
        assert np.array_equal(rows, np.arange(len(rows)))
        assert self.features.shape == (len(rows), 768) and len(obs) == len(rows)
        assert np.load(folder/'processed.npy').all()
        self.by_observation = {tuple(map(int, r)): i for i, r in enumerate(obs)}
        self.by_source_row = {int(row): i for i, row in enumerate(rows)}
        assert len(self.by_observation) == len(rows)
        with np.load(ANCHORS/target/video/'anchor_histograms.npz') as saved:
            self.tracks = saved['tracks'].copy(); self.anchor_rows = saved['observations'].copy()
            self.present = np.asarray(saved['present'], dtype=bool)
        self.track_index = {int(t): i for i, t in enumerate(self.tracks)}
        anchors = np.zeros((len(self.tracks), 3, 768), dtype='float32')
        present = np.zeros((len(self.tracks), 3), dtype=bool)
        ti, slot = np.nonzero(self.present)
        indices = np.asarray([self.by_source_row[int(r)] for r in self.anchor_rows[ti, slot]], dtype=int)
        anchors[ti, slot] = self.normalized(indices)
        present[ti, slot] = self.valid[indices]
        coarse = anchors.sum(1)/np.maximum(present.sum(1, keepdims=True), 1)
        self.anchor_features = np.column_stack((anchors.reshape(len(self.tracks), -1), present)).astype('float32')
        self.coarse = coarse.astype('float32')

    def rows(self, frames, tracks):
        assert len(frames) == len(tracks)
        indices = np.asarray([self.by_observation[int(f), int(t)] for f, t in zip(frames, tracks)], dtype=int)
        ti = np.asarray([self.track_index[int(t)] for t in tracks], dtype=int)
        value = np.column_stack((self.anchor_features[ti], self.normalized(indices),
                                 self.valid[indices, None], self.coarse[ti])).astype('float32')
        assert value.shape == (len(frames), APPEARANCE_DIMENSION) and np.isfinite(value).all()
        return value


@lru_cache(maxsize=8)
def bank(target, video):
    return FullAppearanceBank(target, video)


def augment_paid(features, keys, target):
    assert features.shape == (len(keys), BASE_DIMENSION)
    extra = np.empty((len(keys), APPEARANCE_DIMENSION), dtype='float32')
    for video in sorted({r[0] for r in keys}):
        selected = [i for i, r in enumerate(keys) if r[0] == video]
        extra[selected] = bank(target, video).rows([int(keys[i][3]) for i in selected],
                                                 [int(keys[i][2]) for i in selected])
    value = np.column_stack((features, extra)).astype('float32')
    assert value.shape == (len(keys), DIMENSION) and np.array_equal(value[:, :BASE_DIMENSION], features)
    return value


def augment_dense(features, video, observation_rows):
    rows = np.asarray(observation_rows, dtype=int)
    assert features.shape == (len(rows), BASE_DIMENSION)
    extra = bank(video.target, video.video).rows(video.obs[rows, 0], video.obs[rows, 1])
    value = np.column_stack((features, extra)).astype('float32')
    assert np.array_equal(value[:, :BASE_DIMENSION], features)
    return value
