"""Append the same image-only appearance inputs to every adaptation learner.

The historical5508 coordinates remain byte-identical. New image descriptors
depend on physical candidate/frame only, including for repeated task queries.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from flex_tta.runtime.pf_receipts import file_sha
from scripts.a100_dino_appearance_collect import OUT,PLAN,ANCHORS

BASE_DIMENSION=5508
APPEARANCE_DIMENSION=3844
DIMENSION=BASE_DIMENSION+APPEARANCE_DIMENSION


@lru_cache(None)
def bank(target,video):
    return AppearanceBank(target,video)


class AppearanceBank:
    def __init__(self,target,video):
        self.target,self.video=target,video
        folder=OUT/target/video
        receipt=json.loads((folder/'receipt.json').read_text())
        assert receipt['input_plan_sha256']==file_sha(PLAN)
        for name,sha in receipt['output_sha256'].items():assert file_sha(folder/name)==sha
        self.features=np.load(folder/'features.npy',mmap_mode='r')
        self.valid=np.load(folder/'valid.npy')
        rows=np.load(folder/'selected_rows.npy');obs=np.load(folder/'observations.npy')
        assert self.features.shape==(len(rows),768) and len(obs)==len(rows)
        self.by_observation={tuple(map(int,r)):i for i,r in enumerate(obs)}
        self.by_source_row={int(row):i for i,row in enumerate(rows)}
        assert len(self.by_observation)==len(rows)
        with np.load(ANCHORS/target/video/'anchor_histograms.npz') as saved:
            self.tracks=saved['tracks'].copy();self.anchor_rows=saved['observations'].copy()
            self.present=np.asarray(saved['present'],dtype=bool)
        self.track_index={int(t):i for i,t in enumerate(self.tracks)}

    def normalized(self,indices):
        raw=np.asarray(self.features[indices],dtype='float32')
        scale=np.sqrt(np.square(raw).mean(axis=-1,keepdims=True)).clip(min=1e-8)
        value=raw/scale
        value[~self.valid[indices]]=0.
        return value

    def rows(self,frames,tracks):
        n=len(frames);assert n==len(tracks)
        current_index=np.asarray([self.by_observation[int(f),int(t)] for f,t in zip(frames,tracks)])
        current=self.normalized(current_index);valid=self.valid[current_index].astype('float32')[:,None]
        anchors=np.zeros((n,3,768),dtype='float32');present=np.zeros((n,3),dtype=bool)
        for i,track in enumerate(tracks):
            ti=self.track_index[int(track)]
            for slot in np.flatnonzero(self.present[ti]):
                row=int(self.anchor_rows[ti,slot])
                assert row in self.by_source_row,('Extend shared appearance bank; required anchor absent',self.target,self.video,track,row)
                index=self.by_source_row[row]
                anchors[i,slot]=self.normalized(np.asarray([index]))[0]
                present[i,slot]=self.valid[index]
        coarse=anchors.sum(1)/np.maximum(present.sum(1,keepdims=True),1)
        result=np.column_stack((anchors.reshape(n,-1),present,current,valid,coarse)).astype('float32')
        assert result.shape==(n,APPEARANCE_DIMENSION) and np.isfinite(result).all()
        return result


def augment_paid(features,keys,target):
    assert features.shape==(len(keys),BASE_DIMENSION)
    extra=np.empty((len(keys),APPEARANCE_DIMENSION),dtype='float32')
    for video in sorted({r[0] for r in keys}):
        rows=[i for i,r in enumerate(keys) if r[0]==video]
        extra[rows]=bank(target,video).rows([int(keys[i][3]) for i in rows],[int(keys[i][2]) for i in rows])
    result=np.column_stack((features,extra)).astype('float32')
    assert result.shape==(len(keys),DIMENSION) and np.array_equal(result[:,:BASE_DIMENSION],features)
    return result


def augment_dense(features,video,observation_rows):
    rows=np.asarray(observation_rows,dtype=int)
    assert features.shape==(len(rows),BASE_DIMENSION)
    extra=bank(video.target,video.video).rows(video.obs[rows,0],video.obs[rows,1])
    result=np.column_stack((features,extra)).astype('float32')
    assert np.array_equal(result[:,:BASE_DIMENSION],features)
    return result
