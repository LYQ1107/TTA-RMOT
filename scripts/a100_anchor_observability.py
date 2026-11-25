"""Image-only audit of the three observations currently averaged per TrackID.

This describes available appearance variation. It cannot identify true ID
switches, semantic errors, or the benefit of a future memory mechanism.
The same raw descriptors must be available to ordinary KD in a later study.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from flex_uda.active.color_cohort import part_histograms
from scripts.a100_six_direction import OUT as SIX, PUBLIC, source

OUT = SIX/'anchor_observability_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'anchor_observability_registration_v1.json'
SUCCESS = 'PASS_THREE_ANCHOR_IMAGE_OBSERVABILITY_AUDIT_NO_GT'


def samples_for(observations):
    tracks = np.asarray(sorted(set(observations[:, 1])), dtype=int)
    rows = []
    for track in tracks:
        ids = np.flatnonzero(observations[:, 1] == track)
        selected = np.unique(np.linspace(0, len(ids)-1, min(3, len(ids))).round().astype(int))
        rows.append(ids[selected])
    return tracks, rows


def hellinger(left, right):
    """Distance between normalized observed histograms; missing is excluded."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    assert np.allclose(left.sum(-1), 1) and np.allclose(right.sum(-1), 1)
    return np.sqrt(.5*np.square(np.sqrt(left)-np.sqrt(right)).sum(-1))


def prepare():
    guard = generation_guard(); cases = []; totals = defaultdict(int)
    for target in ('Dance', 'KITTI-V2', 'SportsMOT'):
        for video, bank in sorted(metadata(target)['videos'].items()):
            obs = np.asarray(bank['observations'], dtype=int)
            tracks, rows = samples_for(obs)
            count = sum(map(len, rows)); frames = len(np.unique(obs[np.concatenate(rows), 0]))
            cases.append(dict(target=target, video=video, tracks=len(tracks), samples=count, frame_reads=frames))
            totals['videos'] += 1; totals['tracks'] += len(tracks)
            totals['samples'] += count; totals['image_frame_reads'] += frames
    assert not guard
    plan = dict(status='SEALED_THREE_ANCHOR_IMAGE_ONLY_PLAN', cases=cases, totals=dict(totals),
        same_fixed_three_anchor_row_indices_as_existing_PCD=True,
        images_only_no_teacher_answers_predictions_or_manual_GT=True,
        target_video_preprocessing_shared_across_two_source_checkpoints=True)
    atomic_json(OUT/'preparation.json', plan)
    return plan


def extract(case):
    target, video = case['target'], case['video']
    folder = OUT/target/video
    if (folder/'receipt.json').is_file():
        return json.loads((folder/'receipt.json').read_text())
    started = time.perf_counter(); bank = metadata(target)['videos'][video]
    obs = np.asarray(bank['observations'], dtype=int)
    boxes = np.asarray([list(map(float, line.split(',')[2:6])) for line in bank['lines']])
    tracks, rows = samples_for(obs); n = len(tracks)
    hist = np.zeros((n, 3, 3, 104), dtype=float)
    frames = np.full((n, 3), -1, dtype=int); row_ids = frames.copy()
    present = np.zeros((n, 3), dtype=bool); sample_boxes = np.zeros((n, 3, 4))
    jobs = defaultdict(list)
    for ti, chosen in enumerate(rows):
        for si, row in enumerate(chosen):
            frame = int(obs[row, 0]); frames[ti, si] = frame; row_ids[ti, si] = row
            sample_boxes[ti, si] = boxes[row]; present[ti, si] = True
            jobs[frame].append((ti, si, row))
    io_seconds = histogram_seconds = 0.
    extension = '.jpg' if target == 'SportsMOT' else '.png'
    for frame, entries in sorted(jobs.items()):
        stamp = time.perf_counter()
        with Image.open(Path(bank['image_root'])/f'{frame:06d}{extension}') as image:
            rgb = image.convert('RGB'); rgb.load()
        io_seconds += time.perf_counter()-stamp; stamp = time.perf_counter()
        for ti, si, row in entries:
            hist[ti, si] = part_histograms(rgb, boxes[row])
        rgb.close(); histogram_seconds += time.perf_counter()-stamp
    valid = hist.sum(-1) > 0
    coarse = np.sqrt(hist.sum(1)/np.maximum(valid.sum(1)[..., None], 1))
    comparisons = []
    for pair, (_, domain) in PAIRS.items():
        if domain != target:
            continue
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as saved:
            assert np.array_equal(saved['tracks'], tracks)
            old = saved['colours']; assert old.shape == coarse.shape
            error = float(np.max(np.abs(old-coarse)))
            recorded = bool(np.count_nonzero(old))
            # An explicitly uncomputed descriptor is absence, not a feature
            # reconstruction failure or a black-clothing observation.
            if recorded:
                assert error <= 1e-10, (pair, video, error)
            comparisons.append(dict(pair=pair, recorded_nonzero_RGB=recorded,
                coarse_mean_reconstruction_max_absolute_error=error))
    distances = defaultdict(list); gaps = defaultdict(list)
    part_names = ('whole', 'upper', 'lower')
    for ti, chosen in enumerate(rows):
        for kind, pairs in (
            ('adjacent_anchors', list(zip(range(len(chosen)-1), range(1, len(chosen))))),
            ('first_last', [(0, len(chosen)-1)] if len(chosen) >= 2 else [])):
            for left, right in pairs:
                for part, name in enumerate(part_names):
                    if valid[ti, left, part] and valid[ti, right, part]:
                        key = kind+'_'+name
                        distances[key].append(float(hellinger(hist[ti, left, part], hist[ti, right, part])))
                        gaps[key].append(int(frames[ti, right]-frames[ti, left]))
    # A deterministic descriptive control, not all quadratic TrackID pairs.
    # IDs are the provided candidates and can themselves split one real ID.
    for frame, entries in sorted(jobs.items()):
        ordered = sorted(entries, key=lambda item: tracks[item[0]])
        pairs = list(zip(ordered, ordered[1:]))
        if len(ordered) > 2:
            pairs.append((ordered[-1], ordered[0]))
        for (ti, si, _), (tj, sj, _) in pairs:
            assert ti != tj
            for part, name in enumerate(part_names):
                if valid[ti, si, part] and valid[tj, sj, part]:
                    key = 'same_frame_other_provided_ID_'+name
                    distances[key].append(float(hellinger(hist[ti, si, part], hist[tj, sj, part])))
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'anchor_histograms.npz', tracks=tracks, observations=row_ids,
        frames=frames, boxes=sample_boxes, present=present, histograms=hist.astype('float32'))
    np.savez_compressed(folder/'distance_samples.npz', **{k:np.asarray(v) for k, v in distances.items()},
        **{k+'_frame_gaps':np.asarray(v, dtype=int) for k, v in gaps.items()})
    receipt = dict(target=target, video=video, tracks=n, samples=int(present.sum()),
        image_frame_reads=len(jobs), valid_observed_parts=int(valid.sum()),
        missing_or_offscreen_observed_parts=int((present[:, :, None] & ~valid).sum()),
        coarse_comparisons=comparisons, distance_counts={k:len(v) for k, v in distances.items()},
        image_decode_wall_seconds=io_seconds, histogram_wall_seconds=histogram_seconds,
        worker_wall_seconds=time.perf_counter()-started, raw_arrays_not_to_Git=True)
    atomic_json(folder/'receipt.json', receipt)
    return receipt


def summarize(cases, receipts, guard, started):
    distributions = defaultdict(list)
    for case in cases:
        with np.load(OUT/case['target']/case['video']/'distance_samples.npz') as values:
            for key in values.files:
                distributions[case['target'], key].extend(values[key].tolist())
    summaries = {}
    levels = (0, .1, .25, .5, .75, .9, .95, 1)
    for (target, key), values in sorted(distributions.items()):
        summaries.setdefault(target, {})[key] = dict(count=len(values),
            quantiles={str(q):float(v) for q, v in zip(levels, np.quantile(values, levels))},
            mean=float(np.mean(values)))
    assert len(receipts) == 74 and not guard
    report = dict(status=SUCCESS, videos=len(receipts),
        totals={k:sum(r[k] for r in receipts) for k in ('tracks', 'samples', 'image_frame_reads',
            'valid_observed_parts', 'missing_or_offscreen_observed_parts')},
        max_recorded_coarse_reconstruction_error=max(c['coarse_mean_reconstruction_max_absolute_error']
            for r in receipts for c in r['coarse_comparisons'] if c['recorded_nonzero_RGB']),
        source_mean_comparisons=sum(len(r['coarse_comparisons']) for r in receipts),
        distributions=summaries, process_wall_seconds=time.perf_counter()-started,
        summed_image_decode_seconds=sum(r['image_decode_wall_seconds'] for r in receipts),
        summed_histogram_seconds=sum(r['histogram_wall_seconds'] for r in receipts),
        interpretation='Appearance distance is not an identity-switch label or a semantic error; three anchors miss interior changes.',
        controls='Same-frame different provided IDs, ordered cyclic neighbors only; not verified different true people.',
        cost_scope='Shared image-only descriptor preparation; no single-method cold deployment or HOTA gain claim.',
        future_comparison='Ordinary KD must receive the same per-anchor descriptors if a later method consumes them.',
        manual_target_GT_open_attempts=0, image_teacher_calls=0, new_network_forward_calls=0,
        goal_status='ACTIVE', research_goal_complete=False)
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'anchor_observability_receipt.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    return report


def main(prepare_only=False):
    registration = json.loads(REGISTRATION.read_text())
    for name, expected in registration['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    guard = generation_guard(); started = time.perf_counter(); plan = prepare()
    if prepare_only:
        return
    if (OUT/'completion.json').is_file():
        result = json.loads((OUT/'completion.json').read_text())
        atomic_json(STATE, dict(status=result['status'], pid=os.getpid(), completed=result['videos']))
        return
    receipts = []
    atomic_json(STATE, dict(status='RUNNING_IMAGE_ONLY_ANCHOR_OBSERVABILITY', pid=os.getpid(), completed=0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract, case) for case in plan['cases']]
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result())
            atomic_json(STATE, dict(status='RUNNING_IMAGE_ONLY_ANCHOR_OBSERVABILITY',
                pid=os.getpid(), completed=len(receipts), total=len(plan['cases']),
                last_completed=dict(target=receipts[-1]['target'], video=receipts[-1]['video']),
                manual_GT_open_attempts=len(guard), goal_status='ACTIVE'))
    report = summarize(plan['cases'], receipts, guard, started)
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=report['videos'], goal_status='ACTIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args.prepare_only)
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_THREE_ANCHOR_OBSERVABILITY', pid=os.getpid(),
            error=repr(error), goal_status='ACTIVE'))
        raise
