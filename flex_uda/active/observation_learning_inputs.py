"""Matched dense-observation inputs for conditional learning and ordinary KD.

Historical per-frame PCD is read on NAS, never overwritten or advertised as
byte-identical to the fresh A100 batch-eight features. Every head in this
study is refit on this same feature version. No target annotations enter.
"""
from collections import defaultdict
import json
import numpy as np
from flex_uda.a100.mechanism import HISTORY, PAIRS, metadata
from flex_uda.active.relational_features import geometry
from scripts.a100_anchor_observability import OUT as ANCHORS
from scripts.a100_dense_rgb_collect import OUT as DENSE_RGB
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA

DIMENSION = 1089  # 768 PCD + 312 colour + 3 observed-part flags + 6 geometry/time


class ObservationVideo:
    def __init__(self, pair, video):
        self.pair, self.video = pair, video
        model, self.target = PAIRS[pair]; m = metadata(self.target)
        self.bank = m['videos'][video]
        self.queries = [q for q in m['episodes'] if q['video_id'] == video]
        self.obs = np.asarray(self.bank['observations'], dtype=int)
        self.positions = {tuple(row):i for i,row in enumerate(self.obs.tolist())}
        assert len(self.positions) == len(self.obs)
        with np.load(ANCHORS/self.target/video/'anchor_histograms.npz') as a:
            self.tracks, self.sample_rows, self.present, hist = [a[k].copy() for k in
                ('tracks', 'observations', 'present', 'histograms')]
        lookup = {int(t):i for i,t in enumerate(self.tracks)}
        self.columns = np.asarray([lookup[int(t)] for _,t in self.obs])
        folder = HISTORY/'features'/self.target/model/'videos'/video
        old = json.loads((folder/'metadata.json').read_text())
        old_obs = np.load(folder/'observations.npy', mmap_mode='r')
        old_queries = {q['expression_id']:(i,q) for i,q in enumerate(old['queries'])}
        old_positions = {tuple(row):i for i,row in enumerate(old_obs.tolist())}
        self.old_rows = np.asarray([old_positions[tuple(row)] for row in self.obs.tolist()])
        self.old_queries = np.asarray([old_queries[q['expression_id']][0] for q in self.queries])
        assert all(old_queries[q['expression_id']][1]['query_text'] == q['query_text'] for q in self.queries)
        self.old_n = len(old_obs)
        self.pcd = np.load(folder/'pcd.npy', mmap_mode='r')
        self.index = np.load(folder/'index.npy', mmap_mode='r')
        assert self.pcd.dtype == np.float16 and self.pcd.shape == (len(old_queries)*self.old_n,768)
        coords, _, _ = geometry(self.bank, self.target)
        frames = self.obs[:,0]; first, last = int(frames.min()), int(frames.max())
        age = np.zeros(len(frames))
        for ti in range(len(self.tracks)):
            rows = np.flatnonzero(self.columns == ti); lo, hi = frames[rows].min(), frames[rows].max()
            age[rows] = (frames[rows]-lo)/max(int(hi-lo),1)
        self.geometry = np.c_[coords,(frames-first)/max(last-first,1),age].astype('float32')
        self.raw_bags = np.zeros((len(self.queries),len(self.tracks),3,DIMENSION),dtype='float32')
        ti, si = np.nonzero(self.present); rows = self.sample_rows[ti,si]
        colour = np.sqrt(hist[ti,si].clip(min=0)).reshape(len(rows),-1)
        valid = (hist[ti,si].sum(-1)>0).astype('float32')
        for qi in range(len(self.queries)):
            features = self._pcd_rows(np.full(len(rows),qi),rows)
            self.raw_bags[qi,ti,si] = np.c_[features,colour,valid,self.geometry[rows]]
        self.dense_hist = self.dense_valid = None

    def _pcd_rows(self, qi, rows):
        source_q = self.old_queries[qi]; ids = source_q*self.old_n+self.old_rows[rows]
        expected = np.c_[source_q,self.obs[rows]]
        assert np.array_equal(self.index[ids],expected), (self.pair,self.video,'PCD row mapping')
        return np.asarray(self.pcd[ids],dtype='float32')

    def dense_rows(self, qi, rows):
        if self.dense_hist is None:
            folder = DENSE_RGB/self.target/self.video
            assert (folder/'receipt.json').is_file(), 'Wait for this video dense RGB receipt.'
            assert np.array_equal(np.load(folder/'observations.npy'),self.obs)
            self.dense_hist = np.load(folder/'histograms.npy',mmap_mode='r')
            self.dense_valid = np.load(folder/'valid_parts.npy',mmap_mode='r')
        colour = np.sqrt(np.asarray(self.dense_hist[rows]).clip(min=0)).reshape(len(rows),-1)
        return np.c_[self._pcd_rows(qi,rows),colour,np.asarray(self.dense_valid[rows],dtype='float32'),self.geometry[rows]].astype('float32')

    def supervision(self, policy):
        """One raw mixture target per paid anchor, shared by every head.

        Exact request repeats add no examples. Distinct requested views at
        an identical anchor are averaged; no view is claimed a true label.
        CLIP is assigned to its requested anchor for all comparators, a
        common approximation rather than a true frame annotation.
        """
        doc = json.loads((ANSWERS/'policy_answers'/self.pair/self.video/(policy+'_answers.json')).read_text())
        assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
        seen = {}; grouped = defaultdict(list)
        view_sets = defaultdict(set)
        for item in doc['answers']:
            rid = item['request_id']; p = float(item['yes_probability'])
            if rid in seen:
                assert seen[rid] == p
                continue
            seen[rid] = p; qi,ti = int(item['local_query']),int(item['local_track'])
            assert self.queries[qi]['expression_id'] == item['query_id'] and int(self.tracks[ti]) == item['track_id']
            assert item['frozen_weight_sha256'] == TEACHER_SHA
            row = self.positions[int(item['frame_id']),int(item['track_id'])]
            slots = np.flatnonzero(self.present[ti] & (self.sample_rows[ti] == row))
            assert len(slots) == 1
            key = qi,ti,int(slots[0]); grouped[key].append(p); view_sets[key].add(item['effective_view'])
        keys = np.asarray(sorted(grouped),dtype=int)
        raw_y = np.asarray([np.mean(grouped[tuple(key)]) for key in keys],dtype='float32')
        bags = self.raw_bags[keys[:,0],keys[:,1]]
        masks = self.present[keys[:,1]]
        current = self.raw_bags[keys[:,0],keys[:,1],keys[:,2]]
        _,bag_group = np.unique(keys[:,:2],axis=0,return_inverse=True)
        return dict(bags=bags,present=masks,current=current,raw_target=raw_y,
            query_group=keys[:,0].copy(),bag_group=bag_group,
            report=dict(policy=policy,logical_requests=doc['logical_requested'],unique_requests=len(seen),
                observation_targets=len(keys),multiple_requested_views_at_same_anchor=sum(len(v)>1 for v in view_sets.values()),
                raw_view_mixture_shared_by_all_heads=True,CLIP_anchor_approximation=True))


def prefix_data(records, mean, std):
    bags = np.concatenate([r['bags'] for r in records]); present = np.concatenate([r['present'] for r in records])
    current = np.concatenate([r['current'] for r in records]); raw = np.concatenate([r['raw_target'] for r in records])
    qgroups=[];bgroups=[];qn=bn=0
    for r in records:
        q=r['query_group']; b=r['bag_group'];qgroups.append(q+qn);bgroups.append(b+bn)
        qn+=int(q.max())+1;bn+=int(b.max())+1
    bags=((bags-mean)/std).astype('float32');bags[~present]=0
    current=((current-mean)/std).astype('float32')
    group=np.concatenate(qgroups);bag_group=np.concatenate(bgroups)
    confidence=(1+3*np.abs(2*raw-1)).astype('float32')
    balanced=confidence/np.bincount(bag_group)[bag_group]
    return dict(bags=bags,present=present,current=current,raw_target=raw,
        target=np.clip(raw,.02,.98).astype('float32'),confidence=confidence,
        balanced_weight=balanced.astype('float32'),group=group.astype('int64'))
