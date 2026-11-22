"""Preserve the three original observations instead of averaging them early.

Preparation only: no teacher model, target labels or prediction selection.
The saved bank retains exactly the official boxes and Track IDs. It also checks
that re-averaging reproduces the earlier strong pooled-KD input features.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import HISTORY, PAIRS, generation_guard, metadata
from flex_uda.active.color_cohort import part_histograms
from scripts.a100_visual_classical_prepare import OUT

OBSERVATION = OUT / 'observation_alignment'
FRESH = OUT / 'cold_full_c4/C4_key_single_CLIP_KD_seed0'


def prepare(pair, only_video=None):
    guard = generation_guard()
    begin = time.perf_counter()
    model, target = PAIRS[pair]
    m = metadata(target)
    root = OBSERVATION / 'inputs' / pair
    receipts = []
    videos = [only_video] if only_video else sorted(m['videos'])
    for video in videos:
        folder = root / video
        if (folder / 'receipt.json').is_file():
            receipts.append(json.loads((folder / 'receipt.json').read_text()))
            continue
        started = time.perf_counter()
        bank = m['videos'][video]
        queries = [q for q in m['episodes'] if q['video_id'] == video]
        obs = np.asarray(bank['observations'], dtype=int)
        tracks = sorted(set(obs[:, 1].tolist()))
        boxes = np.asarray([[float(x) for x in line.split(',')[2:6]] for line in bank['lines']])
        samples, node_track, anchors, offsets = [], [], [], [0]
        for ti, track in enumerate(tracks):
            rows = np.flatnonzero(obs[:, 1] == track)
            selected = rows[np.unique(np.linspace(0, len(rows)-1, min(3, len(rows))).round().astype(int))]
            anchors.append(len(samples) + int(np.argmax(boxes[selected, 2] * boxes[selected, 3])))
            samples.extend(selected.tolist())
            node_track.extend([ti] * len(selected))
            offsets.append(len(samples))
        samples = np.asarray(samples, dtype=int)
        node_track = np.asarray(node_track, dtype=int)
        source = HISTORY / 'features' / target / model / 'videos' / video
        pcd = np.load(source / 'pcd.npy', mmap_mode='r')
        index = np.load(source / 'index.npy', mmap_mode='r')
        nobs, nq = len(obs), len(queries)
        assert pcd.shape == (nq*nobs, 768)
        per_query = []
        for qi in range(nq):
            ids = qi*nobs + samples
            assert np.array_equal(index[ids, 0], np.full(len(samples), qi))
            assert np.array_equal(index[ids, 1:], obs[samples])
            per_query.append(np.asarray(pcd[ids], dtype='float32'))
        node_pcd = np.stack(per_query)
        byframe = defaultdict(list)
        for ni, row in enumerate(samples):
            byframe[int(obs[row, 0])].append((ni, int(row)))
        hist = np.empty((len(samples), 3, 104), dtype=float)
        suffix = '.jpg' if target == 'SportsMOT' else '.png'
        for frame, items in sorted(byframe.items()):
            with Image.open(Path(bank['image_root']) / (f'{frame:06d}' + suffix)) as opened:
                image = opened.convert('RGB')
                for ni, row in items:
                    hist[ni] = part_histograms(image, boxes[row])
        pooled_pcd = np.stack([node_pcd[:, a:b].mean(1) for a, b in zip(offsets[:-1], offsets[1:])], axis=1)
        pooled_colour = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            value = hist[a:b]
            count = (value.sum(2) > 0).sum(0)
            mean = np.divide(value.sum(0), count[:, None], out=np.zeros_like(value[0]), where=count[:, None] > 0)
            pooled_colour.append(np.sqrt(mean))
        pooled_colour = np.asarray(pooled_colour, dtype='float32')
        if pair == 'C4':
            with np.load(FRESH / 'source' / video / 'sparse_inputs.npz') as a:
                reference_pcd, reference_colour, prior = a['pcd'], a['colours'], a['prior']
                assert list(a['tracks']) == tracks
            answer_path = FRESH / 'source' / video / 'answers.json'
        else:
            with np.load(OUT / 'inputs' / pair / video / 'inputs.npz') as a:
                reference_pcd, prior = a['features'], a['prior']
                assert list(a['tracks']) == tracks
            with np.load(OUT / 'color_features' / pair / video / 'features.npz') as a:
                reference_colour = a['colours']
            answer_path = OUT / 'causal_teacher' / pair / 'COVER' / (video + '.json')
        # Only the supplied selected answers are read, to check their actual
        # frame alignment; no unselected answer reservoir enters the bank.
        answers = json.loads(answer_path.read_text())['answers']
        for row in answers:
            ti, qi = int(row['local_track']), int(row['local_query'])
            assert int(row['track_id']) == tracks[ti]
            assert int(row['frame_id']) == int(obs[samples[anchors[ti]], 0])
            assert row['query_id'] == queries[qi]['expression_id']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'inputs.npz', pcd=node_pcd,
            colours=np.sqrt(hist).astype('float32'), pooled_pcd=pooled_pcd,
            pooled_colours=pooled_colour, prior=prior, tracks=tracks,
            samples=samples, frames=obs[samples, 0], node_track=node_track,
            offsets=offsets, anchors=anchors,
            query_ids=[q['expression_id'] for q in queries])
        receipt = dict(pair=pair, video=video, queries=nq, tracks=len(tracks),
            original_observations=nobs, retained_observations=len(samples),
            teacher_anchors_checked=len(answers), source_feature_path=str(source),
            pooled_PCD_max_abs_difference=float(np.max(np.abs(pooled_pcd.astype(float)-reference_pcd))),
            pooled_RGB_max_abs_difference=float(np.max(np.abs(pooled_colour.astype(float)-reference_colour))),
            actual_images=len(byframe), GT_open_attempts=len(guard),
            boxes_or_IDs_modified=False, new_teacher_calls=0,
            historical_features_prepaid=True, wall_seconds=time.perf_counter()-started)
        atomic_json(folder / 'receipt.json', receipt)
        receipts.append(receipt)
        print(json.dumps(dict(event='OBSERVATION_BANK_VIDEO', **receipt)), flush=True)
    assert not guard
    result = dict(status='UNLABELLED_OBSERVATION_BANK_READY', pair=pair,
        receipts=receipts, GT_open_attempts=0, new_teacher_calls=0,
        predictions_generated=False, wall_seconds=time.perf_counter()-begin)
    atomic_json(root / ('single_video_completion.json' if only_video else 'completion.json'), result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=['C4', 'C1'], required=True)
    p.add_argument('--video')
    args = p.parse_args()
    prepare(args.pair, args.video)
