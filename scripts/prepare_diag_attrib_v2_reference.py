#!/usr/bin/env python3
"""Seal complete official reference; GT-free, no new GPU, no optimization."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_v2_common import *


def prepare():
    gate = require_prereg()
    attempts = deny_gt()
    start, cpu = time.perf_counter(), time.process_time()
    old_stop = json.loads((ROOT / 'results/diag_attrib/official_reproduction.json').read_text())
    path = OFFICIAL / 'result.json'
    if file_sha(path) != old_stop['original_score_sha256']:
        raise ValueError('original official logits changed')
    if file_sha(Path(old_stop['checkpoint'])) != old_stop['checkpoint_sha256']:
        raise ValueError('official checkpoint changed')
    source = json.loads(path.read_text())
    supplement = {}
    for entry in old_stop['supplements']:
        path = Path(entry['path'])
        if file_sha(path) != entry['sha256']:
            raise ValueError('sealed supplement changed')
        for e in json.loads(path.read_text())['entries']:
            key = (e['video'], e['expression'], e['frame'], e['track_id'])
            if key in supplement:
                raise ValueError('duplicate supplement key')
            supplement[key] = e
    rows = runtime_rows()
    queries = defaultdict(list)
    for row in rows:
        queries[row['video_id']].append(row)
    chosen, historical, margins = {}, {}, {}
    missing, new_positive, known_count = [], [], 0
    for video, expressions in queries.items():
        tracks = load_tracker_lines(TRACKS, video)
        all_keys = {(t, i) for t, by_id in tracks.items() for i in by_id}
        known = {(int(t), int(i)) for i, by_t in source[video].items() for t in by_t}
        last = {i: max(t for t, j in all_keys if j == i) for _, i in all_keys}
        absent = all_keys-known
        if known-all_keys or any(t != last[i] for t, i in absent):
            raise ValueError('missing official observations are not exactly terminal MOT observations')
        if {k for k in all_keys if k[0] == last[k[1]]} != absent:
            raise ValueError('historical support must equal bank minus all terminal observations')
        missing.extend(dict(video=video, frame=t, track_id=i) for t, i in sorted(absent))
        known_count += len(known)
        for row in expressions:
            query, sentence = row['expression_id'], row['query_text']
            key = (video, query)
            native, old_rows, stream = [], [], []
            for t, by_id in tracks.items():
                ids, values, raw_logits = [], [], []
                for i in by_id:
                    if (t, i) in known:
                        pair = source[video][str(i)][str(t)][sentence]
                    else:
                        pair = supplement[video, query, t, i]['logits']
                    if len(pair) != 2 or not all(map(math.isfinite, pair)):
                        raise ValueError('invalid official logits')
                    margin = float(pair[1])-float(pair[0])
                    ids.append(i); values.append(margin); raw_logits.append(pair)
                    if margin > 0:
                        native.append(by_id[i])
                        if (t, i) in known:
                            old_rows.append(by_id[i])
                        else:
                            new_positive.append(dict(video=video, expression=query, frame=t, track_id=i, margin=margin))
                stream.append(dict(frame=t, ids=ids, adapted=values, two_class_logits=raw_logits))
            old_pred = OFFICIAL / 'results' / video / query / 'predict.txt'
            if prediction(old_pred) != expected_rows(tracks, [(int(float(r.split(',')[0])), int(float(r.split(',')[1]))) for r in old_rows]):
                raise ValueError('observed-support native official differs from historical prediction')
            chosen[key], historical[key], margins[key] = native, old_rows, stream
    if known_count != 210393 or len(missing) != 692 or len(supplement) != 11764 or len(new_positive) != 792:
        raise ValueError('registered coverage/added-native-row counts differ')
    complete = seal_plain('OFFICIAL_DANCE_COMPLETE', rows, chosen, margins=margins)
    hist = seal_plain(historical_name('OFFICIAL_DANCE_COMPLETE'), rows, historical,
                      protocol='HISTORICAL_SUPPORT_SENSITIVITY_NOT_FOR_VERDICT')
    if complete['prediction_rows'] != 254910 or hist['prediction_rows'] != 254118:
        raise ValueError('official complete/historical native count mismatch')
    report = dict(status='PASS', prereg=gate, complete_native_rows=254910, historical_native_rows=254118,
        observed_MOT_rows=210393, complete_MOT_rows=211085, missing_terminal_observations=missing,
        added_positive_rows=new_positive, added_rows_count=792, historical_all425_rows_exactly_equal=True,
        all_added_rows_are_from_692_missing_terminal_observations=True, original_score_sha256=old_stop['original_score_sha256'],
        supplements=old_stop['supplements'], complete_seal=str(EVAL / 'OFFICIAL_DANCE_COMPLETE/predictions_seal.json'),
        complete_seal_sha256=file_sha(EVAL / 'OFFICIAL_DANCE_COMPLETE/predictions_seal.json'),
        historical_seal=str(EVAL / historical_name('OFFICIAL_DANCE_COMPLETE') / 'predictions_seal.json'),
        target_GT_open_attempts=attempts, GPU_calls=0, optimizer_steps=0,
        CPU_wall_seconds=time.perf_counter()-start, CPU_process_seconds=time.process_time()-cpu)
    atomic_json(RUN / 'reference_reproduction.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing_terminal_observations','added_positive_rows','supplements')}, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
