"""Shared numerical/I/O definitions for frozen-score attribution v2. No GT IO."""
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_tta.dance_io import load_tracker_lines
from scripts.diag_attrib_audit import (OLD, OFFICIAL, TRACKS, DATA, MODELS,
                                     deny_gt, runtime_rows, prediction, expected_rows)

RUN = ROOT / 'results/runs/diag_attrib_20261031_v2'
PUBLIC = ROOT / 'results/diag_attrib/v2'
EVAL = RUN / 'evaluation'
CACHE = RUN / 'score_bank'
NAMES = ['FROZEN', 'D1', 'D2', 'D3', 'OFFICIAL_DANCE_COMPLETE']
BASELINES = ['B1_SRC_RATE_GLOBAL', 'B1_SRC_RATE_QUERY', 'B2_Z_OFFLINE', 'B2_Z_CAUSAL',
             'B3_MEAN_GLOBAL', 'B3_MEAN_QUERY', 'B4_TEACHER_RATE', 'B4b_TEACHER_GATE']
OLD_TAGS = {'D1': 'DIAG_FIT_OPT', 'D2': 'DIAG_FIT_OPT_POS', 'D3': 'DIAG_FIT_CAP'}
HISTORICAL_HOTA = dict(FROZEN=8.218, D1=29.702, D2=22.844, D3=27.768)


def require_prereg():
    gate = json.loads((RUN / 'prereg_publication.json').read_text())
    if not gate['remote_verified'] or file_sha(ROOT / 'results/diag_attrib/PREREG_v2.md') != gate['prereg_sha256']:
        raise ValueError('v2 must be published unchanged before computing')
    if file_sha(ROOT / 'results/diag_attrib/PREREG.md') != 'd064ca9e7762fa5ff5701fc4c66f033d64ce680a617bac1980b6153ec9a589b6':
        raise ValueError('v1 preregistration changed')
    return gate


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_new(path, text):
    """Idempotent generated artifact, never overwrite a differing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('different existing generated output: ' + str(path))
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(text)
        temporary.replace(path)


def seal_plain(method, rows, selected_by_query, *, margins=None, protocol='GT_FREE_FROZEN_SCORE_DIAGNOSTIC'):
    folder = EVAL / method
    records = []
    for row in rows:
        video, query = row['video_id'], row['expression_id']
        path = folder / 'results' / video / query / 'predict.txt'
        chosen = selected_by_query[video, query]
        write_new(path, ''.join(line + '\n' for line in chosen))
        item = dict(video=video, expression=query, prediction_path=str(path), prediction_sha256=file_sha(path),
                    selected_rows=len(chosen), frames=row['frame_count'])
        if margins is not None:
            margin = path.parent / 'candidate_margins.jsonl.gz'
            if not margin.exists():
                with gzip.open(margin, 'wt') as stream:
                    for record in margins[video, query]:
                        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
            item.update(margins_path=str(margin), margins_sha256=file_sha(margin))
        records.append(item)
    manifest = dict(status='SEALED', method=method, episodes=records, episode_count=len(records),
                    prediction_rows=sum(e['selected_rows'] for e in records), protocol=protocol,
                    immutable_MOT_rows=True, optimizer_steps=0, GPU_calls=0)
    if len(records) != 425:
        raise ValueError('all425 episodes required')
    manifest_path = folder / 'predictions_seal.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('different prediction seal on recovery')
    else:
        atomic_json(manifest_path, manifest)
    atomic_json(folder / 'full_audit.json', dict(status='PASS', valid_episodes=425, unique_videos=25,
        method=method, selected_rows=manifest['prediction_rows'], episodes=records,
        tracker_geometry_changed=False, tracker_ids_changed=False))
    return manifest


def verify_seal(method):
    manifest = json.loads((EVAL / method / 'predictions_seal.json').read_text())
    if manifest['episode_count'] != 425:
        raise ValueError('incomplete prediction seal')
    for row in manifest['episodes']:
        if file_sha(row['prediction_path']) != row['prediction_sha256']:
            raise ValueError('prediction changed after sealing')
        if 'margins_path' in row and file_sha(row['margins_path']) != row['margins_sha256']:
            raise ValueError('margin changed after sealing')
    return manifest


def matrix_name(rank, count):
    return 'M_' + rank + '__' + count


def historical_name(rank, count=None):
    return 'HIST_' + rank + '__' + (count or rank)


def round_budget(rate, n):
    return min(n, max(0, int(math.floor(rate*n + .5))))


def top_budget(values, count):
    """Global canonical-order tie-break; indices already video/query/frame/ID."""
    values = np.asarray(values)
    count = min(len(values), max(0, int(count)))
    order = np.argsort(-values, kind='stable')
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:count]] = True
    threshold = float(values[order[count-1]]) if count else None
    ties = (values == threshold) if count else np.zeros(len(values), dtype=bool)
    return mask, dict(selected=count, threshold=threshold, boundary_ties=int(ties.sum()),
                      selected_boundary_ties=int((ties & mask).sum()))


def causal_moments(frames, values, width=100):
    """Only preceding numerical frames [t-width,t-1], never current/future."""
    frames, values = np.asarray(frames), np.asarray(values, dtype=np.float64)
    unique, first = np.unique(frames, return_index=True)
    lower = np.searchsorted(frames, unique-width, side='left')
    prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
    squares = np.r_[0., np.cumsum(values*values, dtype=np.float64)]
    count = first-lower
    mu = np.divide(prefix[first]-prefix[lower], count, out=np.zeros(len(first)), where=count > 0)
    second = np.divide(squares[first]-squares[lower], count, out=np.zeros(len(first)), where=count > 0)
    sigma = np.sqrt(np.maximum(0., second-mu*mu))
    return unique, first, count, mu, sigma


def rank_within_groups(values, groups, group_count, allowed=None):
    """Ranks exactly equivalent to v1 (-margin,numeric ID) inside frames."""
    values, groups = np.asarray(values), np.asarray(groups)
    indices = np.arange(len(values)) if allowed is None else np.flatnonzero(allowed)
    # Global canonical index breaks exact score ties; IDs are sorted per frame.
    order = indices[np.lexsort((indices, -values[indices], groups[indices]))]
    counts = np.bincount(groups[indices], minlength=group_count)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ranks = np.full(len(values), len(values), dtype=np.int32)
    ranks[order] = np.arange(len(order))-starts[groups[order]]
    return ranks


def load_bank():
    metadata = json.loads((CACHE / 'bank.json').read_text())
    arrays = {name: np.load(CACHE / (name + '.npy'), mmap_mode='r') for name in NAMES}
    groups = np.load(CACHE / 'groups.npy', mmap_mode='r')
    allowed = np.load(CACHE / 'historical.npy', mmap_mode='r')
    return metadata, arrays, groups, allowed


def candidate_lines(metadata):
    tracks = {v: load_tracker_lines(TRACKS, v) for v in sorted({q['video_id'] for q in metadata['episodes']})}
    lines = []
    for q in metadata['episodes']:
        for t, _, _, _ in q['groups']:
            lines.extend(tracks[q['video_id']].get(t, {}).values())
    if len(lines) != metadata['candidate_query_rows']:
        raise ValueError('candidate universe changed')
    return lines


def write_selection(name, mask, metadata, lines, *, protocol='GT_FREE_FROZEN_SCORE_DIAGNOSTIC'):
    selected = {}
    for q in metadata['episodes']:
        a, b = q['start'], q['stop']
        selected[q['video_id'], q['expression_id']] = [lines[a+i] for i in np.flatnonzero(mask[a:b])]
    result = seal_plain(name, metadata['episodes'], selected, protocol=protocol)
    print(json.dumps(dict(event='ATTRIB_V2_PREDICTION_SEALED', method=name, rows=result['prediction_rows'])), flush=True)
    return result


def assert_equivalent(method, original):
    """GT-free row proof, without reading/copying any evaluation metrics."""
    original = Path(original)
    manifest = verify_seal(method)
    for row in manifest['episodes']:
        previous = original / 'results' / row['video'] / row['expression'] / 'predict.txt'
        if prediction(row['prediction_path']) != prediction(previous):
            raise ValueError('metrics reuse without exact row equivalence: ' + method)
    return manifest


def reuse_metrics(method, original):
    """Evaluator-only copy after complete row equality; old results immutable."""
    original = Path(original)
    assert_equivalent(method, original)
    for name in ('pedestrian_summary.txt', 'pedestrian_detailed.csv'):
        source = original / 'results' / name
        target = EVAL / method / 'results' / name
        if target.exists():
            if file_sha(source) != file_sha(target):
                raise ValueError('different reused official CSV')
        else:
            shutil.copy2(source, target)
    atomic_json(EVAL / method / 'official_completion.json', dict(status='PASS', exit_code=0,
        reused_exact_row_equivalent_reference=str(original), new_TrackEval_job=False, worker_wall_seconds=0,
        prediction_seal_sha256=file_sha(EVAL / method / 'predictions_seal.json'),
        summary_sha256=file_sha(EVAL / method / 'results/pedestrian_summary.txt'),
        detailed_sha256=file_sha(EVAL / method / 'results/pedestrian_detailed.csv')))
