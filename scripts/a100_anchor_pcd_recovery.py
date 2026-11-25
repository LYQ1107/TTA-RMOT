"""Pilot recovery of historical per-anchor features, without new GPU work.

Compare the three selected observations with the current A100 mean before
considering historical features usable. This script does not choose a
tolerance, alter either source, or make an RMOT-quality claim.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import HISTORY, PAIRS, generation_guard, metadata
from scripts.a100_six_direction import OUT as SIX, PUBLIC, source
from scripts.a100_anchor_observability import samples_for

REGISTRATION = PUBLIC/'anchor_PCD_recovery_registration_v1.json'
OUT = SIX/'anchor_PCD_recovery_20261123'


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    guard = generation_guard(); started = time.perf_counter()
    pair, video = plan['pair'], plan['video']; model, target = PAIRS[pair]
    bank = metadata(target)['videos'][video]
    queries = [q for q in metadata(target)['episodes'] if q['video_id'] == video]
    obs = np.asarray(bank['observations'], dtype=int); tracks, samples = samples_for(obs)
    folder = HISTORY/'features'/target/model/'videos'/video
    legacy = json.loads((folder/'metadata.json').read_text())
    old_obs = np.load(folder/'observations.npy', mmap_mode='r')
    pcd = np.load(folder/'pcd.npy', mmap_mode='r'); index = np.load(folder/'index.npy', mmap_mode='r')
    old_queries = {q['expression_id']:(i,q) for i,q in enumerate(legacy['queries'])}
    old_rows = {tuple(row):i for i,row in enumerate(old_obs.tolist())}
    assert len(old_queries) == len(legacy['queries']) and len(old_rows) == len(old_obs)
    assert pcd.dtype == np.float16 and pcd.shape == (len(old_queries)*len(old_obs), 768)
    values = np.zeros((len(queries), len(tracks), 3, 768), dtype='float16')
    present = np.zeros((len(tracks), 3), dtype=bool)
    for qi, query in enumerate(queries):
        old_qi, old_query = old_queries[query['expression_id']]
        assert old_query['query_text'] == query['query_text']
        for ti, rows in enumerate(samples):
            old = np.asarray([old_rows[tuple(obs[row])] for row in rows])
            source_rows = old_qi*len(old_obs)+old
            assert np.array_equal(index[source_rows,0], np.full(len(rows), old_qi))
            assert np.array_equal(index[source_rows,1:], obs[rows])
            values[qi,ti,:len(rows)] = pcd[source_rows]; present[ti,:len(rows)] = True
    recovered = np.stack([values[:,ti,:len(rows)].astype('float32').mean(1)
        for ti, rows in enumerate(samples)], axis=1)
    with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as current:
        assert np.array_equal(current['tracks'], tracks)
        current_mean = current['pcd'].copy()
    assert current_mean.shape == recovered.shape and np.isfinite(values).all()
    absolute = np.abs(current_mean.astype(float)-recovered)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT/'C2_0005_historical_anchors.npz', features=values, present=present,
        tracks=tracks, query_ids=[q['expression_id'] for q in queries])
    report = dict(status='COMPLETE_HISTORICAL_ANCHOR_PCD_RECOVERY_PILOT_NO_GT', pair=pair, video=video,
        feature_shape=list(values.shape), source_dtype=str(pcd.dtype), current_mean_dtype=str(current_mean.dtype),
        exact_mean_equal=bool(np.array_equal(current_mean, recovered)),
        mean_max_absolute_error=float(absolute.max()), mean_absolute_error=float(absolute.mean()),
        fraction_exact_mean_elements=float((absolute == 0).mean()),
        raw_feature_bytes=int(values.nbytes), selected_observations=int(present.sum()),
        historical_query_batch=64, current_A100_query_batch=8,
        historical_near_zero_query_isolation=True, current_frame_cache_has_no_such_isolation=True,
        no_recentring_or_threshold_based_reuse_performed=True,
        interpretation='An exact current-mean match can support further recovery checks; mismatch must stay visible. A mean match alone does not prove every historical anchor equals a fresh A100 forward.',
        historical_features_prepaid=True, costs_not_a_fresh_student_deployment=True,
        raw_recovered_features_not_to_Git=True, manual_GT_or_teacher_answers_or_predictions_read=0,
        new_GPU_forward_calls=0, wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(PUBLIC/'anchor_PCD_recovery_pilot_receipt.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
