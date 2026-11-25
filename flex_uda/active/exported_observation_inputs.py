"""Explicit exported PCD and real RGB inputs for a future matched execution.

No historical feature, anchor, answer or fitted-head store is opened here.
The caller supplies an export from its own execution. Reading an existing
export in an integration check does not measure a fresh cold deployment.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, metadata
from flex_uda.active.color_cohort import part_histograms
from flex_uda.active.joint_observation_inputs import trajectory_features
from flex_uda.active.relational_features import normalize_boxes

DIMENSION = 1101


def export_layout(pair, video, folder, *, allow_partial=False):
    """Resolve query-major rows against fixed candidate/query metadata only."""
    folder = Path(folder)
    source, target = PAIRS[pair]
    md = metadata(target); bank = md['videos'][video]
    queries = [q for q in md['episodes'] if q['video_id'] == video]
    exported = json.loads((folder/'metadata.json').read_text())
    seal = json.loads((folder/'seal.json').read_text())
    assert exported['video_id'] == seal['video_id'] == video
    assert exported['target'] == seal['target'] == target
    assert exported['model'] == seal['model'] == source
    assert seal['status'] in ('FEATURES_SEALED', 'SMOKE_PASS')
    observations = np.load(folder/'observations.npy', mmap_mode='r')
    original = np.asarray(bank['observations'], dtype=int)
    position = {tuple(map(int, row)): i for i, row in enumerate(original)}
    assert len(position) == len(original)
    rows = np.asarray([position[tuple(map(int, row))] for row in observations])
    assert len(set(rows.tolist())) == len(rows) and len(rows) > 0
    assert np.array_equal(observations, original[np.sort(rows)])
    complete = np.array_equal(observations, original)
    if not allow_partial:
        assert complete, 'Full execution requires every fixed candidate observation.'
        assert seal['status'] == 'FEATURES_SEALED'
    exported_queries = {q['expression_id']: (i, q) for i, q in enumerate(exported['queries'])}
    assert len(exported_queries) == len(exported['queries']) == len(queries)
    query_rows = []
    for query in queries:
        qi, old = exported_queries[query['expression_id']]
        assert old['query_text'] == query['query_text']
        query_rows.append(qi)
    assert seal['candidate_query_rows'] == len(observations)*len(queries)
    return dict(folder=folder, source=source, target=target, bank=bank, queries=queries,
        exported=exported, seal=seal, observations=observations,
        original=original, original_rows=rows, query_rows=np.asarray(query_rows),
        complete_fixed_candidate_coverage=complete)


def collect_export_rgb(pair, video, feature_folder, output_folder, *, allow_partial=False):
    """Compute query-independent descriptors from the actual images, once."""
    begin = time.perf_counter()
    layout = export_layout(pair, video, feature_folder, allow_partial=allow_partial)
    folder = Path(output_folder); folder.mkdir(parents=True, exist_ok=True)
    # A caller must choose a new output for a new execution; this routine
    # does not turn a previously paid RGB cache into a cold phase.
    assert not (folder/'receipt.json').exists(), 'RGB output is already complete.'
    observations = layout['observations']; bank = layout['bank']
    boxes = np.asarray([[float(v) for v in line.split(',')[2:6]] for line in bank['lines']])
    boxes = boxes[layout['original_rows']]
    groups = defaultdict(list)
    for row, (frame, _) in enumerate(observations):
        groups[int(frame)].append(row)
    hist = np.lib.format.open_memmap(folder/'histograms.npy', mode='w+', dtype='float32',
        shape=(len(observations), 3, 104))
    valid = np.zeros((len(observations), 3), dtype=bool)
    extension = '.jpg' if layout['target'] == 'SportsMOT' else '.png'
    decode = compute = 0.
    for frame, rows in sorted(groups.items()):
        stamp = time.perf_counter()
        with Image.open(Path(bank['image_root'])/f'{frame:06d}{extension}') as image:
            rgb = image.convert('RGB'); rgb.load()
        decode += time.perf_counter()-stamp; stamp = time.perf_counter()
        for row in rows:
            value = part_histograms(rgb, boxes[row]).astype('float32')
            hist[row] = value; valid[row] = value.sum(-1) > 0
        rgb.close(); compute += time.perf_counter()-stamp
    hist.flush()
    np.save(folder/'valid_parts.npy', valid)
    np.save(folder/'observations.npy', observations)
    receipt = dict(status='ACTUAL_IMAGES_TO_EXPLICIT_EXPORT_RGB_NO_GT', pair=pair, video=video,
        target=layout['target'], feature_export=str(Path(feature_folder).resolve()),
        observations=len(observations), image_frame_reads=len(groups),
        complete_fixed_candidate_coverage=layout['complete_fixed_candidate_coverage'],
        image_decode_wall_seconds=decode, histogram_wall_seconds=compute,
        phase_wall_seconds=time.perf_counter()-begin,
        RGB_actually_computed_in_this_call=True, neural_features_recomputed_in_this_call=False,
        historical_RGB_or_anchor_store_used=False, teacher_answers_or_fitted_models_used=False,
        no_full_cold_cost_or_quality_claim=True)
    atomic_json(folder/'receipt.json', receipt)
    return receipt


class ExportedJointObservationVideo:
    """The 1101D native learner interface, bound to explicit source inputs."""
    def __init__(self, pair, video, feature_folder, rgb_folder, *, allow_partial=False):
        begin = time.perf_counter()
        layout = export_layout(pair, video, feature_folder, allow_partial=allow_partial)
        self.pair, self.video, self.target = pair, video, layout['target']
        self.bank, self.queries = layout['bank'], layout['queries']
        self.obs = np.asarray(layout['observations'], dtype=int)
        self.positions = {tuple(row): i for i, row in enumerate(self.obs.tolist())}
        self.tracks = np.unique(self.obs[:, 1])
        lookup = {int(t): i for i, t in enumerate(self.tracks)}
        self.columns = np.asarray([lookup[int(t)] for _, t in self.obs])
        self.query_rows = layout['query_rows']; self.feature_n = len(self.obs)
        folder = Path(feature_folder)
        self.pcd = np.load(folder/'pcd.npy', mmap_mode='r')
        self.index = np.load(folder/'index.npy', mmap_mode='r')
        assert self.pcd.shape == (len(self.queries)*len(self.obs), 768)
        assert self.pcd.dtype == np.float16 and self.index.shape == (len(self.pcd), 3)
        rgb_folder = Path(rgb_folder)
        rgb = json.loads((rgb_folder/'receipt.json').read_text())
        assert rgb['RGB_actually_computed_in_this_call']
        assert rgb['pair'] == pair and rgb['video'] == video
        assert Path(rgb['feature_export']).resolve() == folder.resolve()
        assert np.array_equal(np.load(rgb_folder/'observations.npy'), self.obs)
        self.dense_hist = np.load(rgb_folder/'histograms.npy', mmap_mode='r')
        self.dense_valid = np.load(rgb_folder/'valid_parts.npy', mmap_mode='r')
        assert self.dense_hist.shape == (len(self.obs), 3, 104)
        assert self.dense_valid.shape == (len(self.obs), 3)
        xywh = np.asarray([[float(v) for v in line.split(',')[2:6]] for line in self.bank['lines']])
        xyxy = np.c_[xywh[:, :2], xywh[:, :2]+xywh[:, 2:]]
        coords = normalize_boxes(xyxy, layout['exported']['width'], layout['exported']['height'])
        original = layout['original']; frames = original[:, 0]
        age = np.zeros(len(original))
        for track in np.unique(original[:, 1]):
            rows = np.flatnonzero(original[:, 1] == track)
            lo, hi = frames[rows].min(), frames[rows].max()
            age[rows] = (frames[rows]-lo)/max(int(hi-lo), 1)
        full_geometry = np.c_[coords, (frames-frames.min())/max(int(frames.max()-frames.min()), 1), age].astype('float32')
        self.clip_stride = {'KITTI-V2': 1, 'Dance': 2, 'SportsMOT': 3}[self.target]
        trajectory = trajectory_features(original, full_geometry[:, :4], self.clip_stride).astype('float32')
        rows = layout['original_rows']
        self.geometry, self.trajectory = full_geometry[rows], trajectory[rows]
        self.sample_rows = np.zeros((len(self.tracks), 3), dtype=int)
        self.present = np.zeros((len(self.tracks), 3), dtype=bool)
        for ti, track in enumerate(self.tracks):
            rows = np.flatnonzero(self.obs[:, 1] == track)
            rows = rows[np.argsort(self.obs[rows, 0], kind='stable')]
            chosen = np.unique(np.linspace(0, len(rows)-1, min(3, len(rows))).round().astype(int))
            self.sample_rows[ti, :len(chosen)] = rows[chosen]
            self.present[ti, :len(chosen)] = True
        self.dimension = DIMENSION
        self.raw_bags = np.zeros((len(self.queries), len(self.tracks), 3, DIMENSION), dtype='float32')
        ti, si = np.nonzero(self.present); rows = self.sample_rows[ti, si]
        for qi in range(len(self.queries)):
            self.raw_bags[qi, ti, si] = self.dense_rows(np.full(len(rows), qi), rows)
        self.input_receipt = dict(pair=pair, video=video, feature_export=str(folder.resolve()),
            RGB_export=str(rgb_folder.resolve()), observations=len(self.obs), queries=len(self.queries),
            dimension=DIMENSION, complete_fixed_candidate_coverage=layout['complete_fixed_candidate_coverage'],
            partial_input_for_explicit_integration_check=not layout['complete_fixed_candidate_coverage'],
            CPU_input_binding_wall_seconds=time.perf_counter()-begin,
            prior_feature_answer_or_fitted_head_store_used=False,
            reading_export_is_not_new_neural_forward_or_cold_execution=True)

    def _pcd_rows(self, qi, rows):
        qi, rows = np.broadcast_arrays(np.asarray(qi, dtype=int), np.asarray(rows, dtype=int))
        source_q = self.query_rows[qi]; ids = source_q*self.feature_n+rows
        assert np.array_equal(self.index[ids], np.c_[source_q, self.obs[rows]])
        return np.asarray(self.pcd[ids], dtype='float32')

    def dense_rows(self, qi, rows):
        rows = np.asarray(rows, dtype=int)
        colour = np.sqrt(np.asarray(self.dense_hist[rows]).clip(min=0)).reshape(len(rows), -1)
        return np.c_[self._pcd_rows(qi, rows), colour,
            np.asarray(self.dense_valid[rows], dtype='float32'), self.trajectory[rows],
            self.geometry[rows]].astype('float32')
