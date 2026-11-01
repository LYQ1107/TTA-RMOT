"""Read-only official score banks and preregistered rank/select definitions."""
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import multidomain_diagnostics as previous
from scripts.diag_attrib_v2_common import write_csv, write_new, top_budget, rank_within_groups, round_budget

RUN = ROOT / 'results/runs/decompose_20261102'
PUBLIC = ROOT / 'results/decompose'
OLD = previous.RUN
OLD_PUBLIC = previous.PUBLIC
COUNT = ROOT / 'results/runs/count_structure_20261031'
TARGETS = ('KITTI-V2', 'SportsMOT', 'Dance')
MODELS = ('V2-best', 'Dance-best', 'LaMOT-best')
PAIRS = [('C1', 'V2-best', 'Dance'), ('C2', 'Dance-best', 'KITTI-V2'),
         ('C3', 'V2-best', 'SportsMOT'), ('C4', 'Dance-best', 'SportsMOT'),
         ('C5', 'LaMOT-best', 'KITTI-V2'), ('C6', 'LaMOT-best', 'Dance')]
ORACLES = ('G0_GLOBAL', 'G1_QUERY', 'G2_V_MEAN', 'G3_F_MEAN', 'G4_F_FRAME', 'TRACK_ORACLE')
CALIBRATORS = ('B1_SRC_RATE_GLOBAL', 'B1_SRC_RATE_QUERY', 'B2_Z_OFFLINE',
               'B3_MEAN_GLOBAL', 'B3_MEAN_QUERY', 'C1_TRACK_MEAN')
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
PREREG_SHA = '457c068a5d2587ac8b6eeceda5514ac98115d56fb736eb0dec612e1880c20b8d'
WORDS = {
    'MOTION': 'moving|move|turning|turn|parking|parked|standing|stand|walking|walk|running|run|stopped|stop|faster|slower|same direction|opposite direction|counter direction|braking|accelerating|jumping|serving|dancing',
    'SPATIAL': 'left|right|front|behind|back|near|far|ahead|middle|side|beside|closest',
    'APPEARANCE': 'black|white|red|blue|green|yellow|gray|grey|silver|brown|purple|orange|pink|dark|light|shirt|t-shirt|tshirt|pants|skirt|clothes|wearing|dressed|jersey|uniform',
    'ROLE': 'team|libero|goalkeeper|referee|setter|coach',
}


def require_plan():
    p = json.loads((RUN / 'prereg_publication.json').read_text())
    if not p['remote_verified'] or file_sha(PUBLIC / 'PREREG.md') != PREREG_SHA:
        raise ValueError('Publish the unchanged literal preregistration before computation')
    return p


def types(text):
    text = ' '.join(text.lower().split())
    found = [k for k, words in WORDS.items() if any(re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', text)
                                                       for w in words.split('|'))]
    return found or ['OTHER']


def load_target(target):
    meta, _, groups, _ = previous.inputs(target)
    files = json.loads((OLD_PUBLIC / 'margin_paths.json').read_text())['files']
    records = {r['model']: r for r in files if r['target'] == target and r['coverage'] == 'complete'}
    if set(records) != set(MODELS):
        raise ValueError('Need all three sealed official score banks')
    arrays = {}
    for m, r in records.items():
        if file_sha(r['path']) != r['sha256']:
            raise ValueError('Changed official margin bank: ' + m)
        arrays[m] = np.load(r['path'], mmap_mode='r')
        if len(arrays[m]) != len(groups) or not np.isfinite(arrays[m]).all():
            raise ValueError('Invalid score coverage')
    return meta, arrays, groups, records


def video_queries(meta):
    out = defaultdict(list)
    for q in meta['episodes']:
        out[q['video_id']].append(q)
    return {v: sorted(q, key=lambda x: x['expression_id']) for v, q in sorted(out.items())}


def physical(meta, target, video):
    bank = meta['videos'][video]
    rows = []
    for raw in bank['lines']:
        f = list(map(float, raw.split(',')))
        rows.append(dict(frame_id=int(f[0]), track_id=int(f[1]), bbox_xywh=f[2:6], MOT_row=raw))
    return rows


def guard():
    return previous.guard()


def sandbox(read_roots, write_roots):
    from flex_uda.teacher.isolation import restrict_filesystem
    roots = [ROOT / d for d in ('scripts', 'flex_tta', 'flex_uda', 'configs')]
    roots += [sys.prefix, sys.base_prefix, '/usr', '/lib', '/lib64', '/bin', '/etc', '/proc', '/sys']
    return restrict_filesystem([*roots, *read_roots], [*write_roots, '/dev', '/tmp'])


def folder(combo, name):
    return RUN / 'predictions' / combo / name


def seal(combo, name, mask, meta, oracle=False):
    out = folder(combo, name)
    rows = []
    for q in meta['episodes']:
        v, e = q['video_id'], q['expression_id']
        bank = meta['videos'][v]
        chosen = [bank['lines'][j] for j in np.flatnonzero(mask[q['start']:q['stop']])]
        p = out / 'results' / v / e / 'predict.txt'
        write_new(p, ''.join(x + '\n' for x in chosen))
        rows.append(dict(video=v, expression=e, prediction_path=str(p), prediction_sha256=file_sha(p), selected_rows=len(chosen)))
    result = dict(status='SEALED_BEFORE_EVALUATION', combo=combo, method=name, oracle=oracle,
                  role='ORACLE_ONLY_NOT_A_METHOD' if oracle else 'GT_FREE_SCORE_CALIBRATION_DIAGNOSTIC',
                  episodes=rows, episode_count=len(rows), prediction_rows=sum(e['selected_rows'] for e in rows),
                  GT_open_attempts=None if oracle else 0, teacher_calls=0, optimizer_updates=0, immutable_MOT=True)
    atomic_json(out / 'predictions_seal.json', result)
    print(json.dumps(dict(event='DECOMPOSE_PREDICTION_SEALED', combo=combo, method=name, rows=result['prediction_rows'])), flush=True)
    return result


def verify(combo, name):
    s = json.loads((folder(combo, name) / 'predictions_seal.json').read_text())
    for e in s['episodes']:
        if file_sha(e['prediction_path']) != e['prediction_sha256']:
            raise ValueError('Changed sealed prediction')
    return s
