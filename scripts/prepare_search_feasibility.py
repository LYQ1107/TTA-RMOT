"""Project only sealed decompose probes, teacher answers and official margins."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *
from scripts import decompose_common as dc


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    proof = require_prereg(); attempts = dc.guard(); start = time.perf_counter()
    dest = RUN / 'runtime'; dest.mkdir(parents=True, exist_ok=True)
    old = dc.RUN / 'probe'; plan = json.loads((old / 'plan.json').read_text())
    seal = json.loads((old / 'seal.json').read_text())
    if seal['status'] != 'ALL_PAIRED_PROBES_SEALED_BEFORE_GT': raise ValueError('Unsealed decompose probes')
    checked(dict(path=plan['manifest_path'], sha256=plan['manifest_sha256']))
    checked(dict(path=plan['videos_path'], sha256=plan['videos_sha256']))
    inputs = [record(old / 'plan.json'), record(old / 'seal.json'), record(plan['manifest_path']), record(plan['videos_path'])]
    raw = {}
    for i, worker in enumerate(seal['workers']):
        if worker['GT_open_attempts'] or not worker['isolation']['kernel_read_whitelist']:
            raise ValueError('Old teacher GT isolation failure')
        if worker['outputs_sha256'] != RAW_SHAS[i]: raise ValueError('Not the 48dfb245 sealed teacher output')
        path = checked(dict(path=worker['outputs_path'], sha256=RAW_SHAS[i])); inputs.append(record(path))
        for r in json_rows(path):
            key = r['target'], r['method'], r['video_id'], r['frame_id'], r['track_id'], r['query_id']
            if key in raw: raise ValueError('Duplicate paired answer')
            p = float(r['yes_probability'])
            if not np.isfinite(p) or not 0 <= p <= 1: raise ValueError('Invalid sealed teacher probability')
            raw[key] = p
    frames = list(json_rows(plan['manifest_path']))
    if len(raw) != sum(len(f['queries']) * len(f['candidates']) * 3 for f in frames):
        raise ValueError('Incomplete teacher answer coverage')
    data = {t: dc.load_target(t) for t in TARGETS}
    for r in plan['margin_banks']:
        checked(r); inputs.append(record(r['path']))
        if r['sha256'] != data[r['target']][3][r['model']]['sha256']: raise ValueError('Model bank differs from decompose')
    qindex = {t: {(q['video_id'], q['expression_id']): q for q in d[0]['episodes']} for t, d in data.items()}
    pos = {(t, v): {(r['frame_id'], r['track_id']): j for j, r in enumerate(dc.physical(d[0], t, v))}
           for t, d in data.items() for v in d[0]['videos']}
    rows = []; excluded = Counter(); excluded_pairs = Counter()
    for f in frames:
        target, video, frame = f['target'], f['video_id'], f['frame_id']; cs = f['candidates']
        if len(cs) < 2:
            excluded[target] += len(f['queries']); excluded_pairs[target] += len(cs) * len(f['queries']); continue
        ids = [c['track_id'] for c in cs]
        if len(set(ids)) != len(ids): raise ValueError('Repeated candidate physical ID')
        for q in f['queries']:
            query = q['query_id']; key = list_key(target, video, frame, query)
            offset = qindex[target][video, query]['start']
            margins = {m: [float(data[target][1][m][offset + pos[target, video][frame, i]]) for i in ids] for m in MODELS}
            probabilities = {name: [raw[target, method, video, frame, i, query] for i in ids]
                             for name, method in zip(TEACHERS, ('CROP', 'CLIP', 'FRAME_BOX'))}
            rows.append(dict(list_id=key, target=target, video=video, frame=frame, query=query, N=len(cs),
                             subseed=sub_seed(key), track_ids=ids, boxes=[c['bbox_xywh'] for c in cs],
                             margins=margins, probabilities=probabilities))
    rows.sort(key=lambda r: (r['target'], r['video'], r['frame'], r['query']))
    if len({r['list_id'] for r in rows}) != len(rows): raise ValueError('Duplicate probe list')
    output = dest / 'lists.jsonl'
    with output.open('w') as stream:
        for row in rows: stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    inventory = []
    for t in TARGETS:
        rr = [r for r in rows if r['target'] == t]; n = [r['N'] for r in rr]
        inventory.append(dict(target=t, lists=len(rr), videos=len({r['video'] for r in rr}),
            N_mean=float(np.mean(n)), N_p50=float(np.percentile(n, 50)), N_p90=float(np.percentile(n, 90)),
            N_min=min(n), N_max=max(n), candidate_query_pairs=sum(n),
            excluded_N_lt_2_lists=excluded[t], excluded_candidate_query_pairs=excluded_pairs[t]))
    if attempts: raise PermissionError('Preparation tried opening GT')
    result = dict(status='PURE_RUNTIME_SEALED', runtime=record(output), inputs=inputs, inventory=inventory,
                  decompose_source_commit='48dfb245dd4fe9b58f42dc1e37425a5e4095e8a5',
                  prereg_publication=proof, GT_open_attempts=0, teacher_calls=0, model_calls=0, GPU_hours=0,
                  CPU_wall_seconds=time.perf_counter()-start)
    dump(dest / 'seal.json', result); dump(PUBLIC / 'input_inventory.json', result)
    print(json.dumps(dict(event='SEARCH_RUNTIME_SEALED', inventory=inventory, GT_open_attempts=0)), flush=True)


if __name__ == '__main__': main()
