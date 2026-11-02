"""Separate GT-free presence decode and post-seal official RMOT evaluation."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata, sealed_margins
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

FPS = {'KITTI-V2': 10, 'Dance': 20, 'SportsMOT': 25}


def decode(target, model_name, batch_size=256):
    import torch
    from torch.utils.data import DataLoader
    from flex_uda.presence.data import FeatureUnits, collate_units
    from flex_uda.presence.head import PresenceHead
    from flex_uda.presence.predictions import seal_mask
    from scripts.multidomain_runtime import guard
    from scripts.diag_pf_countmatched import topk_ids
    attempts = guard(); start = time.perf_counter()
    root = RUN / 'features' / target / model_name
    data = FeatureUnits(root, labels=False)
    head_path = RUN / 'heads' / model_name / 'checkpoint_best_source.pt'
    state = torch.load(head_path, map_location='cpu', weights_only=False)
    model = PresenceHead().eval(); model.load_state_dict(state['head'])
    # CPU decode is intentional; leaves the two admitted GPU slots available.
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_units)
    presence = np.empty(len(data), dtype='float32'); probabilities = np.empty((len(data), 11), dtype='float32')
    with torch.no_grad():
        for batch in loader:
            p, k = model(**{n: batch[n] for n in ('pcd', 'margin', 'box', 'sent', 'padding_mask')})
            presence[batch['index'].numpy()] = p.numpy()
            probabilities[batch['index'].numpy()] = k.softmax(-1).numpy()
    raw_k = np.where(presence >= 0, np.maximum(1, probabilities.argmax(1)), 0)
    smooth_k = np.empty_like(raw_k)
    units_by_query = defaultdict(list)
    for j, unit in enumerate(data.units): units_by_query[unit[0], unit[1]].append(j)
    radius = round(FPS[target]/2)
    for ids in units_by_query.values():
        frames = np.asarray([data.units[i][2] for i in ids])
        values = raw_k[ids]
        for j, i in enumerate(ids):
            # A frame-number window, not a window of cached candidate frames.
            lo, hi = np.searchsorted(frames, [frames[j]-radius, frames[j]+radius], side='left')
            hi = np.searchsorted(frames, frames[j]+radius, side='right')
            smooth_k[i] = int(np.median(values[lo:hi]))
    meta = target_metadata(target); margins, ref = sealed_margins(target, model_name)
    episode_index = {(q['video_id'], q['expression_id']): q for q in meta['episodes']}
    masks = dict(PRES_TOPK=np.zeros(len(margins), dtype=bool), PRES_TOPK_SMOOTH=np.zeros(len(margins), dtype=bool))
    for j, unit in enumerate(data.units):
        shard, qi, frame, a, b, ba, bb, _ = unit
        _, local = data.shards[shard]
        q = episode_index[local['video_id'], local['queries'][qi]['expression_id']]
        obs = meta['videos'][q['video_id']]['observations'][ba:bb]
        ids = [i for _, i in obs]; scores = margins[q['start']+ba:q['start']+bb]
        for method, counts in [('PRES_TOPK', raw_k), ('PRES_TOPK_SMOOTH', smooth_k)]:
            chosen = topk_ids(ids, dict(zip(ids, scores)), int(counts[j]))
            masks[method][q['start']+ba:q['start']+bb] = [i in chosen for i in ids]
    folder = RUN / 'stage_A' / target / model_name; folder.mkdir(parents=True, exist_ok=True)
    for name, values in [('presence_logits', presence), ('count_probs', probabilities), ('raw_k', raw_k), ('smooth_k', smooth_k)]:
        np.save(folder / (name+'.npy'), values)
    extra = dict(head_checkpoint=str(head_path), head_checkpoint_sha256=file_sha(head_path),
                 margin_reference=ref, teacher_calls=0, test_time_optimizer_updates=0)
    for name, mask in masks.items():
        seal_mask(folder / name, meta, mask, target=target, model=model_name, method=name,
                  GT_open_attempts=len(attempts), extra=extra)
    atomic_json(folder / 'decode_receipt.json', dict(status='SEALED_BEFORE_GT', target=target, model=model_name,
        units=len(data), GT_open_attempts=len(attempts), future_frames_in_smoothing=True,
        source_validation_early_stopping=True, wall_seconds=time.perf_counter()-start,
        files_sha256={p.name: file_sha(p) for p in folder.glob('*.npy')}))


def evaluate_folder(folder, target, *, sports418=False):
    from flex_uda.presence.predictions import verify_predictions
    from scripts.multidomain_runtime import VENDOR, SPORTS_GT_PROTOCOL
    from scripts.multidomain_diagnostics import target_root
    from scripts.pf_diag_common import official_metrics
    receipt = verify_predictions(folder); folder = Path(folder)
    if (folder / 'official_completion.json').exists():
        return json.loads((folder / 'official_completion.json').read_text())
    root = target_root(target)
    seq = {'Dance': 'dance', 'KITTI-V2': 'kitti-2', 'SportsMOT': 'SportsMOT'}[target]
    seqmap = VENDOR / 'seqmaps' / (seq+'.txt')
    if sports418:
        missing = set(json.loads((ROOT / 'results/multidomain_generality/sports_template_inventory.json').read_text())['missing_GT'])
        seqmap = folder / 'official418_seqmap.txt'
        all_seq = [s.strip() for s in (VENDOR / 'seqmaps/SportsMOT.txt').read_text().splitlines() if s.strip()]
        seqmap.write_text('\n'.join(s for s in all_seq if s not in missing)+'\n')
    for row in receipt['episodes']:
        link = folder / 'results' / row['video'] / row['expression'] / 'gt.txt'
        if not link.exists(): link.symlink_to(root / 'gt_template' / row['video'] / row['expression'] / 'gt.txt')
    command = [sys.executable, './TrackEval/scripts/run_mot_challenge.py', '--METRICS', 'HOTA', 'CLEAR', 'Identity',
        '--SEQMAP_FILE', str(seqmap), '--SKIP_SPLIT_FOL', 'True', '--GT_FOLDER', str(root),
        '--TRACKERS_FOLDER', str(folder / 'results'), '--GT_LOC_FORMAT', '{gt_folder}{video_id}/{expression_id}/gt.txt',
        '--TRACKERS_TO_EVAL', str(folder / 'results'), '--USE_PARALLEL', 'True', '--NUM_PARALLEL_CORES', '2', '--PLOT_CURVES', 'False']
    start = time.perf_counter()
    with (folder / 'trackeval.log').open('w') as handle:
        code = subprocess.call(command, cwd=VENDOR, stdout=handle, stderr=subprocess.STDOUT)
    if code: raise RuntimeError('Official TrackEval failed: ' + str(folder / 'trackeval.log'))
    metrics = official_metrics(folder / 'results')
    result = dict(status='PASS', target=target, metrics=metrics, GT_protocol=SPORTS_GT_PROTOCOL if target == 'SportsMOT' else 'OFFICIAL_UNCHANGED',
        sports418=sports418, wall_seconds=time.perf_counter()-start,
        detailed_sha256=file_sha(folder / 'results/pedestrian_detailed.csv'),
        prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'))
    if not sports418 and int(metrics['Dets']) != receipt['prediction_rows']:
        raise ValueError('TrackEval row count differs from sealed predictions')
    atomic_json(folder / 'official_completion.json', result)
    return result


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=list(IN_DOMAIN), required=True)
    p.add_argument('--model', choices=['V2-best', 'Dance-best', 'LaMOT-best'], required=True)
    p.add_argument('--phase', choices=['decode', 'evaluate'], required=True)
    a = p.parse_args()
    if a.phase == 'decode': decode(a.target, a.model)
    else:
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            print(json.dumps(evaluate_folder(RUN / 'stage_A' / a.target / a.model / method, a.target)), flush=True)


if __name__ == '__main__': main()
