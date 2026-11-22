"""Relocated, frozen inputs for the registered 2026-11-21 mechanism audit.

This module never opens ground truth. Offline truth lives in the audit script.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import numpy as np
from flex_uda.a100.runtime import ROOT, HISTORY, relocate

OUT = ROOT / 'a100_runs/mechanism_20261121'
OLD = ROOT / 'a100_runs/three_stage_20261120'
PAIRS = {'C1': ('V2-best', 'Dance'), 'C4': ('Dance-best', 'SportsMOT'),
         'C5': ('LaMOT-best', 'KITTI-V2')}
KEYS = {'Dance': 'dance', 'SportsMOT': 'sports', 'KITTI-V2': 'kitti'}


@lru_cache(None)
def metadata(target):
    p = ROOT / 'results/runs/multidomain_generality_20261101' / KEYS[target] / 'metadata.json'
    m = json.loads(p.read_text())
    restored = json.loads((ROOT/'configs/a100_paths.json').read_text())['datasets'][target]['video_roots']
    for video, bank in m['videos'].items():
        bank['image_root'] = restored.get(video, str(relocate(bank['image_root'])))
        bank.setdefault('frame_count', max(q['frame_count'] for q in m['episodes'] if q['video_id'] == video))
    return m


@lru_cache(None)
def history(pair):
    for version in ('redesign_v8p', 'redesign_v8'):
        p = HISTORY / version / 'active' / pair / 'MINUS_SLOW/rho_0.3'
        if (p / 'predictions_seal.json').is_file():
            return p
    raise FileNotFoundError('No sealed MINUS_SLOW for ' + pair)


@lru_cache(None)
def margin_bank(target, model):
    d = json.loads((ROOT / 'results/multidomain_generality/margin_paths.json').read_text())
    r = next(r for r in d['files'] if r['target'] == target and r['model'] == model and r['coverage'] == 'complete')
    return np.load(relocate(r['path']), mmap_mode='r')


@lru_cache(None)
def stage_index(target, model):
    execution = json.loads((ROOT / 'results/presence_active/redesign_v8p/execution.json').read_text())
    base = relocate(execution['stage_A_root']) / target / model
    offsets = {}; offset = 0
    for directory in sorted((HISTORY / 'features' / target / model / 'videos').iterdir()):
        md = json.loads((directory / 'metadata.json').read_text())
        n = len(md['queries']) * len(md['frames'])
        offsets[md['video_id']] = (offset, n, md['frames'])
        offset += n
    arrays = {n: np.load(base / (n + '.npy'), mmap_mode='r')
              for n in ('presence_logits', 'expected_iou_utilities', 'raw_k')}
    return offsets, arrays


class FrozenInputs:
    def __init__(self, pair, video):
        model, target = PAIRS[pair]; self.model, self.target = model, target
        m = metadata(target); self.bank = m['videos'][video]
        self.queries = [q for q in m['episodes'] if q['video_id'] == video]
        self.observations = np.asarray(self.bank['observations'], dtype=int)
        self.tracks = sorted(set(self.observations[:, 1].tolist()))
        self.by_track = {t: np.flatnonzero(self.observations[:, 1] == t) for t in self.tracks}
        canonical = OLD / 'stage2/canonical' / video / 'initial.npz'
        if pair == 'C4' and canonical.is_file():
            with np.load(canonical) as a:
                self.margin = a['margin'].copy(); self.raw_presence = a['raw_presence'].copy()
                self.utilities = a['utilities'].copy(); self.raw_k = a['raw_k'].copy()
            frames = np.unique(self.observations[:, 0])
        else:
            scores = margin_bank(target, model)
            self.margin = np.stack([scores[q['start']:q['stop']] for q in self.queries])
            offsets, arrays = stage_index(target, model)
            offset, size, frames = offsets[video]
            nq, nf = len(self.queries), len(frames)
            self.raw_presence = np.asarray(arrays['presence_logits'][offset:offset+size]).reshape(nq, nf).astype(float)
            a = arrays['expected_iou_utilities'][offset:offset+size]
            self.utilities = np.asarray(a).reshape((nq, nf) + a.shape[1:])
            self.raw_k = np.asarray(arrays['raw_k'][offset:offset+size]).reshape(nq, nf)
        self.frame_index = {int(t): i for i, t in enumerate(frames)}

    def decode(self, temperature=1., bias=0., margin=None):
        from flex_uda.active.v6 import selection_fractions
        return selection_fractions(self.observations, self.tracks, self.frame_index,
            self.margin if margin is None else margin, self.utilities, self.raw_presence,
            temperature=temperature, bias=bias, raw_k=self.raw_k)

    def track_means(self):
        return np.stack([self.margin[:, self.by_track[t]].mean(1) for t in self.tracks], 1)


def baseline_mask(pair, video):
    folder = OLD / 'stage2/MINUS/videos' / video if pair == 'C4' else history(pair) / 'videos' / video
    return np.load(folder / 'mask.npy')


def generation_guard():
    """Check GT names without resolving every image path through slow NAS FUSE."""
    import sys
    attempts = []
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0]))
        if p.name in ('gt.txt', 'labels.npy', 'GT_counts.npy', 'labels.json') or any(
                s in p.parts for s in ('gt_template', 'gt_templates', 'annotations_v1', 'offline_GT')):
            attempts.append(str(p)); raise PermissionError('GT is not a generation input: ' + str(p))
    sys.addaudithook(audit)
    return attempts
