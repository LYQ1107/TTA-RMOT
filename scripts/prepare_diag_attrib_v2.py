#!/usr/bin/env python3
"""GT-free full matrices, fixed calibration, and historical sensitivity."""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_v2_common import *


def build_bank():
    path = CACHE / 'bank.json'
    if path.exists():
        metadata = json.loads(path.read_text())
        for name, sha in metadata['cache_files_sha256'].items():
            if file_sha(CACHE / name) != sha:
                raise ValueError('cached score bank changed')
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = runtime_rows()
    tracks = {v: load_tracker_lines(TRACKS, v) for v in sorted({q['video_id'] for q in rows})}
    n = sum(sum(map(len, tracks[q['video_id']].values())) for q in rows)
    if n != 3588445:
        raise ValueError('complete candidate universe differs')
    values = {name: np.lib.format.open_memmap(CACHE / (name + '.npy'), mode='w+', dtype=np.float64, shape=(n,)) for name in NAMES}
    groups = np.lib.format.open_memmap(CACHE / 'groups.npy', mode='w+', dtype=np.int32, shape=(n,))
    historical = np.lib.format.open_memmap(CACHE / 'historical.npy', mode='w+', dtype=bool, shape=(n,))
    protected = json.loads((ROOT / 'results/runs/diag_attrib_20261031_v1/protected_inputs.json').read_text())
    hashes = {r['margins_path']: r['margins_sha256'] for r in protected}
    sources = dict(MODELS, OFFICIAL_DANCE_COMPLETE=EVAL / 'OFFICIAL_DANCE_COMPLETE')
    missing = {(r['video'], r['frame'], r['track_id']) for r in json.loads((RUN / 'reference_reproduction.json').read_text())['missing_terminal_observations']}
    metadata, offset, group_id = [], 0, 0
    for row in rows:
        q = dict(row, start=offset, groups=[])
        by_frame = tracks[row['video_id']]
        if not set(by_frame).issubset(range(row['frame_indices']['start'], row['frame_indices']['stop'])):
            raise ValueError('runtime timeline does not cover every candidate')
        position = {}
        for t in range(row['frame_indices']['start'], row['frame_indices']['stop']):
            size = len(by_frame.get(t, {}))
            position[t] = (offset, offset+size, group_id)
            q['groups'].append([t, offset, offset+size, group_id])
            groups[offset:offset+size] = group_id
            historical[offset:offset+size] = [(row['video_id'], t, i) not in missing for i in by_frame.get(t, {})]
            offset += size
            group_id += 1
        q['stop'] = offset
        for model, source in sources.items():
            path = source / 'results' / row['video_id'] / row['expression_id'] / 'candidate_margins.jsonl.gz'
            sha = file_sha(path)
            if model != 'OFFICIAL_DANCE_COMPLETE' and sha != hashes[str(path)]:
                raise ValueError('sealed input margin changed')
            if model == 'OFFICIAL_DANCE_COMPLETE':
                seal = json.loads((path.parent / 'completion.json').read_text()) if (path.parent / 'completion.json').exists() else None
                # The new reference uses its explicit whole-prediction seal,
                # checked before and after this bank build below.
            seen = []
            with gzip.open(path, 'rt') as stream:
                for line in stream:
                    item = json.loads(line)
                    t, ids = item['frame'], item['ids']
                    if ids != sorted(by_frame[t]) or len(ids) != len(item['adapted']):
                        raise ValueError('score candidate alignment differs')
                    a, b, _ = position[t]
                    array = np.asarray(item['adapted'], dtype=np.float64)
                    if not np.isfinite(array).all():
                        raise ValueError('nonfinite margin')
                    values[model][a:b] = array
                    seen.append(t)
            if seen != sorted(by_frame):
                raise ValueError('score stream misses a candidate frame')
        metadata.append(q)
        print(json.dumps(dict(event='ATTRIB_V2_BANK_QUERY', video=row['video_id'], query=row['expression_id'])), flush=True)
    if offset != n or historical.sum() != 3576681:
        raise ValueError('registered full/historical support count differs')
    for array in [*values.values(), groups, historical]:
        array.flush()
    verify_seal('OFFICIAL_DANCE_COMPLETE')
    atomic_json(path := CACHE / 'bank.json', dict(episodes=metadata, candidate_query_rows=n,
        query_frame_groups=group_id, historical_candidate_query_rows=int(historical.sum()),
        margin_precision='original JSON values retained as float64; no float32 requantization',
        cache_files_sha256={p.name: file_sha(p) for p in CACHE.glob('*.npy')}))


def source_statistics():
    saved = RUN / 'source_statistics.json'
    if saved.exists():
        record = json.loads(saved.read_text())
        if file_sha(record['score_path']) != record['score_sha256']:
            raise ValueError('source scores changed')
        return record
    path = ROOT / 'results/runs/official_refer_kitti_v2_2gpu/result.json'
    source = json.loads(path.read_text())
    queries = defaultdict(list)
    for video, by_id in source.items():
        for i, by_frame in by_id.items():
            for t, by_query in by_frame.items():
                for sentence, pair in by_query.items():
                    if len(pair) != 2 or not all(map(math.isfinite, pair)):
                        raise ValueError('invalid source two-class logits')
                    queries[video, sentence].append((int(t), int(i), float(pair[1])-float(pair[0])))
    statistics, all_values, zoff, zcausal = [], [], [], []
    for (video, sentence), data in sorted(queries.items()):
        data.sort()
        frames = np.asarray([r[0] for r in data])
        values = np.asarray([r[2] for r in data], dtype=np.float64)
        mu, sigma = float(values.mean()), float(values.std())
        if sigma > 0:
            zoff.append(-mu/sigma)
        unique, first, count, cmu, csigma = causal_moments(frames, values)
        valid = (count > 0) & (csigma > 0)
        if valid.any():
            zcausal.append(float((-cmu[valid]/csigma[valid]).mean()))
        all_values.append(values)
        statistics.append(dict(video=video, query_text=sentence, candidates=len(values),
            positives=int((values > 0).sum()), mean=mu, std=sigma, causal_valid_frames=int(valid.sum())))
    flat = np.concatenate(all_values)
    if len(source) != 4 or len(statistics) != 862 or len(flat) != 2355528 or int((flat > 0).sum()) != 750834:
        raise ValueError('source official rate statistics do not reproduce preregistered input')
    if not zoff or not zcausal:
        raise ValueError('source has no usable standardization statistics')
    record = dict(score_path=str(path), score_sha256=file_sha(path), source_videos=4, source_queries=862,
        source_candidate_rows=len(flat), source_positive_rows=int((flat > 0).sum()),
        emission_rate=float((flat > 0).mean()), mean=float(flat.mean()),
        z0_offline=float(np.mean(zoff)), z0_causal=float(np.mean(zcausal)),
        offline_valid_queries=len(zoff), causal_valid_queries=len(zcausal), queries=statistics,
        source_scope='official source validation scores; NOT 17-video Train statistics', target_GT_reads=0)
    atomic_json(saved, record)
    print(json.dumps(dict(event='ATTRIB_V2_SOURCE_STATS', **{k:v for k,v in record.items() if k != 'queries'})), flush=True)
    return record


def teacher_counts():
    folder = Path('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE')
    seal = json.loads((folder / 'seal.json').read_text())
    path = folder / 'soft_labels.jsonl'
    if file_sha(path) != seal['files_sha256']['soft_labels.jsonl']:
        raise ValueError('Naive8500 labels changed')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    seen, counts = {}, defaultdict(lambda: [0, 0])
    for row in rows:
        if not row['valid'] or row.get('teacher_prob') is None or not math.isfinite(row['teacher_prob']):
            continue
        key = row['video_id'], row['query_id'], row['frame_id'], row['track_id']
        if key in seen:
            if seen[key] != row['teacher_prob']:
                raise ValueError('conflicting duplicate teacher judgment')
            continue
        seen[key] = row['teacher_prob']
        count = counts[key[:2]]
        count[0] += 1
        count[1] += row['teacher_prob'] >= .95
    if len(rows) != 8500 or len(seen) != 8500 or sum(v[1] for v in counts.values()) != 656:
        raise ValueError('registered Naive8500/POS656 judgment set differs')
    return counts, dict(path=str(path), sha256=file_sha(path), valid_unique_judgments=len(seen), positives=656)


def generate():
    gate = require_prereg()
    attempts = deny_gt()
    start, cpu = time.perf_counter(), time.process_time()
    build_bank()
    metadata, scores, groups, historical = load_bank()
    lines = candidate_lines(metadata)
    n, g = len(groups), metadata['query_frame_groups']
    ranks = {m: rank_within_groups(scores[m], groups, g) for m in NAMES}
    counts = {m: np.bincount(groups, weights=np.asarray(scores[m]) > 0, minlength=g).astype(np.int32) for m in NAMES}
    produced, equivalence = [], []
    for rank in NAMES:
        for count in NAMES:
            name = matrix_name(rank, count)
            mask = ranks[rank] < counts[count][groups]
            write_selection(name, mask, metadata, lines)
            produced.append(name)
            if rank == count:
                original = EVAL / 'OFFICIAL_DANCE_COMPLETE' if rank == 'OFFICIAL_DANCE_COMPLETE' else MODELS[rank]
                assert_equivalent(name, original)
                equivalence.append(dict(method=name, reference=str(original), status='ALL425_ROWS_EXACTLY_EQUAL'))
            elif rank == 'FROZEN' and count in OLD_TAGS:
                original = OLD / 'evaluation/merged' / ('FROZEN_AT_' + OLD_TAGS[count] + '_COUNT')
                assert_equivalent(name, original)
                equivalence.append(dict(method=name, reference=str(original), status='ALL425_ROWS_EXACTLY_EQUAL'))
            elif count == 'FROZEN' and rank in OLD_TAGS:
                original = OLD / 'evaluation/merged' / (OLD_TAGS[rank] + '_AT_FROZEN_COUNT')
                assert_equivalent(name, original)
                equivalence.append(dict(method=name, reference=str(original), status='ALL425_ROWS_EXACTLY_EQUAL'))
    # Six further historical-support predictions; the official historical
    # diagonal was already freshly evaluated in the reference stage.
    hcounts = {m: np.bincount(groups, weights=(np.asarray(scores[m]) > 0) & historical, minlength=g).astype(np.int32) for m in NAMES}
    for rank in NAMES:
        if rank != 'OFFICIAL_DANCE_COMPLETE':
            name = historical_name(rank)
            write_selection(name, (np.asarray(scores[rank]) > 0) & historical, metadata, lines,
                            protocol='HISTORICAL_SUPPORT_SENSITIVITY_NOT_FOR_VERDICT')
            produced.append(name)
    for rank in ('D1', 'D3'):
        name = historical_name(rank, 'OFFICIAL_DANCE_COMPLETE')
        restricted_rank = rank_within_groups(scores[rank], groups, g, historical)
        mask = historical & (restricted_rank < hcounts['OFFICIAL_DANCE_COMPLETE'][groups])
        write_selection(name, mask, metadata, lines, protocol='HISTORICAL_SUPPORT_SENSITIVITY_NOT_FOR_VERDICT')
        produced.append(name)
    frozen = np.asarray(scores['FROZEN'])
    source = source_statistics()
    teacher, teacher_audit = teacher_counts()
    baseline_masks = {name: np.zeros(n, dtype=bool) for name in BASELINES}
    global_mask, global_boundary = top_budget(frozen, round_budget(source['emission_rate'], n))
    baseline_masks['B1_SRC_RATE_GLOBAL'][:] = global_mask
    baseline_masks['B3_MEAN_GLOBAL'][:] = frozen-frozen.mean()+source['mean'] > 0
    boundary, fallbacks, teacher_ratios = [], [], []
    for q in metadata['episodes']:
        a, b = q['start'], q['stop']
        values = frozen[a:b]
        native = values > 0
        mask, detail = top_budget(values, round_budget(source['emission_rate'], len(values)))
        baseline_masks['B1_SRC_RATE_QUERY'][a:b] = mask
        boundary.append(dict(video=q['video_id'], expression=q['expression_id'], baseline='B1_SRC_RATE_QUERY', **detail))
        mu, sigma = float(values.mean()), float(values.std())
        if sigma > 0:
            baseline_masks['B2_Z_OFFLINE'][a:b] = (values-mu)/sigma >= source['z0_offline']
        else:
            baseline_masks['B2_Z_OFFLINE'][a:b] = native
            fallbacks.append(dict(baseline='B2_Z_OFFLINE', video=q['video_id'], expression=q['expression_id'],
                                  reason='zero_variance', candidates=b-a, frames=q['frame_count'], selected=int(native.sum())))
        times = np.concatenate([np.repeat(t, end-begin) for t, begin, end, _ in q['groups']])
        unique, first, count, cmu, csigma = causal_moments(times, values)
        valid = (count > 0) & (csigma > 0)
        ends = np.r_[first[1:], len(values)]
        causal = native.copy()
        for j in np.flatnonzero(valid):
            causal[first[j]:ends[j]] = (values[first[j]:ends[j]]-cmu[j])/csigma[j] >= source['z0_causal']
        baseline_masks['B2_Z_CAUSAL'][a:b] = causal
        fallback_mask = np.repeat(~valid, ends-first)
        if fallback_mask.any():
            fallbacks.append(dict(baseline='B2_Z_CAUSAL', video=q['video_id'], expression=q['expression_id'],
                reason='no_history_or_zero_variance', candidates=int(fallback_mask.sum()), frames=int((~valid).sum()),
                selected=int(native[fallback_mask].sum())))
        baseline_masks['B3_MEAN_QUERY'][a:b] = values-mu+source['mean'] > 0
        judged, positive = teacher.get((q['video_id'], q['expression_id']), [0, 0])
        if judged:
            rate = (positive/judged)*(judged/len(values))
            mask, detail = top_budget(values, round_budget(rate, len(values)))
            baseline_masks['B4_TEACHER_RATE'][a:b] = mask
            baseline_masks['B4b_TEACHER_GATE'][a:b] = native if positive else False
            boundary.append(dict(video=q['video_id'], expression=q['expression_id'], baseline='B4_TEACHER_RATE', **detail))
        else:
            for name in ('B4_TEACHER_RATE', 'B4b_TEACHER_GATE'):
                baseline_masks[name][a:b] = native
                fallbacks.append(dict(baseline=name, video=q['video_id'], expression=q['expression_id'],
                    reason='no_valid_teacher_judgment', candidates=b-a, frames=q['frame_count'], selected=int(native.sum())))
        teacher_ratios.append(dict(video=q['video_id'], expression=q['expression_id'], judged=judged,
            positives=positive, candidates=len(values), emission_rate=positive/len(values) if judged else None))
    for name, mask in baseline_masks.items():
        write_selection(name, mask, metadata, lines)
        produced.append(name)
    # D0 is alignment-only and never enters ranking/count matrix decisions.
    old_d0 = OLD / 'evaluation/merged/NAIVE_PAIRED'
    chosen = {}
    for q in metadata['episodes']:
        path = old_d0 / 'results' / q['video_id'] / q['expression_id'] / 'predict.txt'
        chosen[q['video_id'], q['expression_id']] = path.read_text().splitlines()
    seal_plain('D0_NAIVE', metadata['episodes'], chosen)
    assert_equivalent('D0_NAIVE', old_d0)
    equivalence.append(dict(method='D0_NAIVE', reference=str(old_d0), status='ALL425_ROWS_EXACTLY_EQUAL'))
    produced.append('D0_NAIVE')
    audit = dict(status='ALL_GT_FREE_PREDICTIONS_SEALED', prereg=gate, methods=produced,
        GT_free_target_open_attempts=attempts, GPU_calls=0, optimizer_steps=0, teacher_calls=0,
        source_statistics_path=str(RUN / 'source_statistics.json'), teacher=teacher_audit,
        equivalent_old_results=equivalence, global_rate_boundary=global_boundary,
        quantile_boundaries=boundary, fallback_records=fallbacks, teacher_query_rates=teacher_ratios,
        CPU_wall_seconds=time.perf_counter()-start, CPU_process_seconds=time.process_time()-cpu,
        prediction_seals={m:file_sha(EVAL / m / 'predictions_seal.json') for m in produced})
    atomic_json(RUN / 'GT_free_predictions.json', audit)
    print(json.dumps(dict(event='ATTRIB_V2_GT_FREE_COMPLETE', methods=len(produced),
                         wall=audit['CPU_wall_seconds'], target_GT_open_attempts=attempts)), flush=True)


if __name__ == '__main__':
    generate()
