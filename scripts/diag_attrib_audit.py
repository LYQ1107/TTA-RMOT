#!/usr/bin/env python3
"""GT-free, read-only input audit for preregistered D-FIT attribution.

This is not an evaluator and never invents scores for absent observations.
Missing official margins require genuine model inference before proceeding.
"""
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

RUN = ROOT / 'results/runs/diag_attrib_20261031_v1'
PUBLIC = ROOT / 'results/diag_attrib'
OLD = ROOT / 'results/runs/diag_fit_20261030_v1'
OFFICIAL = ROOT / 'results/runs/flexhook_dance_to_dance_official_detr_neuralsort_corrected_tokenizer'
TRACKS = Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs')
DATA = Path('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')
MODELS = {'FROZEN': ROOT / 'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE',
          'D1': OLD / 'evaluation/merged/DIAG_FIT_OPT',
          'D2': OLD / 'evaluation/merged/DIAG_FIT_OPT_POS',
          'D3': OLD / 'evaluation/merged/DIAG_FIT_CAP'}


def deny_gt():
    attempts = []
    def hook(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        resolved = path.resolve()
        if any(p.is_relative_to(DATA) and (p.name in ('gt.txt', 'labels.json') or
               any(x in p.parts for x in ('gt_template', 'expression', 'labels_with_ids')))
               for p in (path, resolved)):
            attempts.append(str(path))
            raise PermissionError('attribution prediction may not read target GT/annotation: ' + str(path))
    sys.addaudithook(hook)
    return attempts


def require_prereg():
    gate = json.loads((RUN / 'prereg_publication.json').read_text())
    if not gate['remote_verified'] or file_sha(PUBLIC / 'PREREG.md') != gate['prereg_sha256']:
        raise ValueError('publish immutable preregistration before computing')
    return gate


def runtime_rows():
    rows = [json.loads(line) for line in (ROOT / 'manifests/dance_val_runtime.jsonl').read_text().splitlines() if line]
    if len(rows) != 425 or len({(r['video_id'], r['expression_id']) for r in rows}) != 425:
        raise ValueError('runtime must contain all 425 unique episodes')
    return sorted(rows, key=lambda r: (r['video_id'], r['expression_id']))


def prediction(path):
    """Canonical numeric MOT rows; preserve original raw files separately."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = tuple(map(float, line.split(',')))
        key = (int(values[0]), int(values[1]))
        if len(values) != 10 or key in out:
            raise ValueError('invalid or duplicate prediction row: ' + str(path))
        out[key] = values
    return out


def expected_rows(tracks, chosen):
    return {(t, i): tuple(map(float, tracks[t][i].split(','))) for t, i in chosen}


def audit():
    gate = require_prereg()
    attempts = deny_gt()
    start = time.perf_counter()
    rows = runtime_rows()
    banks = {v: load_tracker_lines(TRACKS, v) for v in sorted({r['video_id'] for r in rows})}
    seal = json.loads((OLD / 'predictions_seal.json').read_text())
    expected = {e['margins_path']: e['margins_sha256'] for m in seal['methods'].values() for e in m['episodes']}
    protected = []
    for m in seal['methods'].values():
        if file_sha(m['run_metadata']) != m['run_metadata_sha256']:
            raise ValueError('sealed run metadata changed')
    results = {}
    for name, folder in MODELS.items():
        count = total = frames = 0
        records = []
        for row in rows:
            video, expression = row['video_id'], row['expression_id']
            directory = folder / 'results' / video / expression
            margin, pred = directory / 'candidate_margins.jsonl.gz', directory / 'predict.txt'
            receipt = json.loads((directory / 'completion.json').read_text())
            actual_sha = file_sha(margin)
            if actual_sha != receipt['margins_sha256'] or file_sha(pred) != receipt['prediction_sha256']:
                raise ValueError('sealed episode changed: ' + str(directory))
            if name != 'FROZEN' and actual_sha != expected[str(margin)]:
                raise ValueError('D-FIT aggregate seal differs: ' + str(margin))
            selected = []
            seen = []
            with gzip.open(margin, 'rt') as stream:
                for line in stream:
                    item = json.loads(line)
                    t, ids, values = int(item['frame']), item['ids'], item['adapted']
                    if ids != sorted(banks[video][t]) or len(ids) != len(values) or not all(map(math.isfinite, values)):
                        raise ValueError('candidate alignment or finite margin failure')
                    seen.append(t)
                    selected.extend((t, i) for i, m in zip(ids, values) if m > 0)
                    total += len(ids)
                    frames += 1
            if seen != sorted(banks[video]):
                raise ValueError('incomplete score timeline')
            if prediction(pred) != expected_rows(banks[video], selected):
                raise ValueError('margin>0 does not reproduce native: ' + name)
            count += len(selected)
            records.append(dict(video=video, expression=expression, rows=len(selected),
                                prediction_path=str(pred), prediction_sha256=file_sha(pred),
                                margins_path=str(margin), margins_sha256=actual_sha))
        if total != 3588445:
            raise ValueError('complete bank is required')
        results[name] = dict(status='PASS', episodes=425, candidates=total, active_query_frames=frames,
                             prediction_rows=count, margin_positive_rows_exactly_reproduce_native=True)
        protected.extend(dict(model=name, **r) for r in records)
        print(json.dumps(dict(event='ATTRIB_NATIVE_AUDIT_PASS', model=name, **results[name])), flush=True)
    source_path = OFFICIAL / 'result.json'
    original_sha = file_sha(source_path)
    if original_sha != 'de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110':
        raise ValueError('official historical score SHA differs')
    source = json.loads(source_path.read_text())
    missing = []
    official_count = observed = 0
    details = []
    by_video = defaultdict(list)
    for row in rows:
        by_video[row['video_id']].append(row)
    for video, queries in by_video.items():
        tracks = banks[video]
        available = {(int(t), int(i)) for i, history in source[video].items() for t in history}
        universe = {(t, i) for t, values in tracks.items() for i in values}
        if available - universe:
            raise ValueError('official scores include non-bank candidates')
        for t, i in sorted(universe - available):
            missing.append(dict(video=video, frame=t, track_id=i))
        for row in queries:
            selected = []
            for t, i in sorted(available):
                pair = source[video][str(i)][str(t)][row['query_text']]
                if len(pair) != 2 or not all(map(math.isfinite, pair)):
                    raise ValueError('invalid two-class official logit')
                if pair[1] - pair[0] > 0:
                    selected.append((t, i))
            old_pred = OFFICIAL / 'results' / video / row['expression_id'] / 'predict.txt'
            if prediction(old_pred) != expected_rows(tracks, selected):
                raise ValueError('observed official scores already fail historical row reproduction')
            official_count += len(selected)
        observed += 17 * len(available)
        details.append(dict(video=video, MOT_rows=len(universe), observed_rows=len(available), missing_rows=len(universe-available)))
    if official_count != 254118:
        raise ValueError('official 254118 native count failed')
    atomic_json(RUN / 'official_missing_observations.json', missing)
    atomic_json(RUN / 'protected_inputs.json', protected)
    status = 'OFFICIAL_MARGIN_SUPPLEMENT_REQUIRED' if missing else 'ALL_NATIVE_MARGIN_REPRODUCTIONS_PASS'
    report = dict(status=status, prereg=gate, models=results,
        official=dict(score_path=str(source_path), score_sha256=original_sha, observed_candidate_query_rows=observed,
            required_candidate_query_rows=3588445, missing_MOT_observations=len(missing),
            missing_candidate_query_rows=17*len(missing), historical_rows=official_count,
            observed_scores_reproduce_all425_historical_predictions=True, per_video=details),
        target_GT_open_attempts=attempts, optimizer_steps=0, GPU_calls=0,
        CPU_wall_seconds=time.perf_counter()-start, original_files_written=0)
    atomic_json(RUN / 'input_audit.json', report)
    atomic_json(PUBLIC / 'input_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    audit()
