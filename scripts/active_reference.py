"""Read-only old teacher gates + sealed margins: Gate-Reassign reference only.

C1 must reproduce the historical FROZEN result byte-for-byte. C6 uses the same
sealed gates on LaMOT margins; it is a new CPU-derived reference, NOT a new
teacher run or a previously measured C6 HOTA. Historical artifacts stay intact.
"""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, target_metadata, sealed_margins, runtime_guard
from flex_uda.presence.predictions import seal_mask
from flex_tta.runtime.pf_receipts import file_sha


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--pair', choices=['C1', 'C6'], required=True); a = p.parse_args()
    access = runtime_guard()
    old = ROOT / 'results/runs/gated_decode_20261101'
    provenance = json.loads((old / 'gates_seal.json').read_text())
    if provenance['GT_open_attempts'] or file_sha(provenance['gate_path']) != provenance['gates_sha256']:
        raise ValueError('Historical gate seal mismatch')
    with open(provenance['gate_path']) as stream: gates = list(csv.DictReader(stream))
    present = {(r['video'], r['expression']): r['predicted_present'] == 'True'
               for r in gates if r['dataset'] == 'Dance' and r['gate_mode'] == 'normal'}
    if len(present) != 425: raise ValueError('Need all425 old normal gate decisions')
    model = 'V2-best' if a.pair == 'C1' else 'LaMOT-best'
    cost_path = ROOT / 'results/gated_decode/teacher_costs.json'
    inherited_cost = json.loads(cost_path.read_text())['by_dataset_mode']['Dance_normal']
    meta = target_metadata('Dance'); margins, record = sealed_margins('Dance', model)
    mask = np.zeros(len(margins), dtype=bool)
    from scripts.count_structure_common import track_mean
    for video, bank in sorted(meta['videos'].items()):
        queries = sorted([q for q in meta['episodes'] if q['video_id'] == video], key=lambda q: q['expression_id'])
        _, inverse = np.unique(np.asarray(bank['observations'])[:, 1], return_inverse=True)
        n = int(inverse.max())+1
        means = np.stack([track_mean(margins[q['start']:q['stop']], inverse, n) for q in queries])
        allowed = np.asarray([present[video, q['expression_id']] for q in queries])
        if not allowed.any(): continue
        winners = means.argmax(0); rejected = ~allowed[winners]
        winners[rejected] = np.where(allowed[:, None], means, -np.inf).argmax(0)[rejected]
        for q, query in enumerate(queries): mask[query['start']:query['stop']] = winners[inverse] == q
    extra = dict(pair=a.pair, rho=None, margin_input=record, video_transductive=True,
        teacher=dict(logical_calls=inherited_cost['requested_calls'], new_calls=0, cache_hits=0, forward_seconds=0.,
            equivalent_forward_seconds=inherited_cost['forward_seconds'], inherited_teacher_calls=inherited_cost['requested_calls'],
            cost_semantics='same old425 normal gates; forward0 incremental, inherited equivalent cost is not zero',
            inherited_cost_path=str(cost_path), inherited_cost_sha256=file_sha(cost_path)),
        gates=dict(path=provenance['gate_path'], sha256=provenance['gates_sha256']),
        historical_reuse=a.pair == 'C1', newly_derived_CPU_reference=a.pair == 'C6',
        LoRA_GPU_seconds=0., new_teacher_calls=0)
    if a.pair == 'C1':
        prior = old / 'evaluation/FROZEN__TG_X0_REASSIGN'
        seal = json.loads((prior / 'predictions_seal.json').read_text())
        import hashlib
        episodes = {(e['video'], e['expression']): e for e in seal['episodes']}
        for query in meta['episodes']:
            episode = episodes[query['video_id'], query['expression_id']]
            lines = meta['videos'][query['video_id']]['lines']
            text = ''.join(lines[i]+'\n' for i in np.flatnonzero(mask[query['start']:query['stop']]))
            if hashlib.sha256(text.encode()).hexdigest() != episode['prediction_sha256'] or file_sha(episode['prediction_path']) != episode['prediction_sha256']:
                raise ValueError('C1 Gate-Reassign historical row parity failed')
        extra['historical_reference'] = str(prior)
    out = RUN / 'active' / a.pair / 'GATE_REASSIGN_REF'
    receipt = seal_mask(out, meta, mask, target='Dance', model=model, method='GATE_REASSIGN_REF', GT_open_attempts=len(access), extra=extra)
    print(json.dumps(dict(status=receipt['status'], path=str(out), rows=receipt['prediction_rows'], new_teacher_calls=0)), flush=True)


if __name__ == '__main__': main()
