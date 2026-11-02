"""Separate GT-free presence decode and post-seal official RMOT evaluation."""
from argparse import ArgumentParser
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata, sealed_margins
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

FPS = {'KITTI-V2': 10, 'Dance': 20, 'SportsMOT': 25}


def decode(target, model_name, batch_size=256, *, head_checkpoint=None, output_dir=None):
    import torch
    from torch.utils.data import DataLoader
    from flex_uda.presence.data import FeatureUnits, collate_units
    from flex_uda.presence.residual import ARCHITECTURE, head_from_checkpoint
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from flex_uda.presence.runtime import runtime_guard
    from scripts.diag_pf_countmatched import topk_ids
    attempts = runtime_guard(); start = time.perf_counter()
    folder = Path(output_dir) if output_dir else RUN / 'stage_A' / target / model_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.decode.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    saved = folder / 'decode_receipt.json'
    if saved.exists():
        receipt = json.loads(saved.read_text())
        for name, sha in receipt['files_sha256'].items():
            if file_sha(folder / name) != sha: raise ValueError('Changed completed presence decode')
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'): verify_predictions(folder / method)
        print(json.dumps(dict(status='ALREADY_SEALED_NO_RERUN', target=target, model=model_name)), flush=True)
        return receipt
    head_path = Path(head_checkpoint) if head_checkpoint else RUN / 'heads' / model_name / 'checkpoint_best_source.pt'
    state = torch.load(head_path, map_location='cpu', weights_only=False)
    residual = state.get('architecture') == ARCHITECTURE
    root = RUN / 'features' / target / model_name
    data = FeatureUnits(root, labels=False, track_ids=residual)
    model = head_from_checkpoint(state).eval()
    # CPU decode is intentional; leaves the two admitted GPU slots available.
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_units)
    cap = max(u[6]-u[5] for u in data.units) if residual else 10
    presence = np.empty(len(data), dtype='float32'); probabilities = np.zeros((len(data), cap+1), dtype='float32')
    joint = np.zeros_like(probabilities) if residual else None
    raw_k = np.empty(len(data), dtype='int64')
    with torch.no_grad():
        for batch in loader:
            names = ('pcd', 'margin', 'box', 'sent', 'padding_mask') + (('track_id',) if residual else ())
            output = model(**{n: batch[n] for n in names})
            if residual:
                p, k = output['presence'], output['logits']
                prob = k.softmax(-1)
                ix = batch['index'].numpy()
                joint[ix, :prob.shape[1]] = prob.numpy()
                conditional = torch.zeros_like(prob)
                if prob.shape[1] > 1: conditional[:, 1:] = k[:, 1:].softmax(-1)
                # Empty sets have no conditional positive count distribution.
                conditional[batch['padding_mask'].all(1)] = 0
                probabilities[ix, :prob.shape[1]] = conditional.numpy()
                raw_k[ix] = k.argmax(1).numpy()
            else:
                p, k = output
                probabilities[batch['index'].numpy()] = k.softmax(-1).numpy()
            presence[batch['index'].numpy()] = p.numpy()
    if not residual: raw_k = np.where(presence >= 0, np.maximum(1, probabilities.argmax(1)), 0)
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
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in [('presence_logits', presence), ('count_probs', probabilities), ('raw_k', raw_k), ('smooth_k', smooth_k)]:
        np.save(folder / (name+'.npy'), values)
    if residual: np.save(folder / 'joint_count_probs.npy', joint)
    extra = dict(head_checkpoint=str(head_path), head_checkpoint_sha256=file_sha(head_path),
                 margin_reference=ref, teacher_calls=0, test_time_optimizer_updates=0,
                 source_learning_rate=state.get('learning_rate',3e-4))
    if residual:
        extra.update(head_architecture=ARCHITECTURE, count_decoder='joint_MAP_0_to_n',
            presence_definition='marginal_P(count>0); MAP count can disagree with probability>=0.5',
            conditional_count_probabilities=True, count_cap=None)
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
    folder = Path(folder).resolve()
    lock = (folder / '.evaluate.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    receipt = verify_predictions(folder)
    if (folder / 'official_completion.json').exists():
        return json.loads((folder / 'official_completion.json').read_text())
    root = target_root(target)
    historical = Path(receipt['historical_reference']) if receipt.get('historical_reference') else None
    if historical is not None:
        previous = json.loads((historical / 'predictions_seal.json').read_text())
        index = {(r['video'], r['expression']): r for r in previous['episodes']}
        for row in receipt['episodes']:
            old = index[row['video'], row['expression']]
            if row['prediction_sha256'] != old['prediction_sha256'] or file_sha(old['prediction_path']) != old['prediction_sha256']:
                raise ValueError('Historical reuse is not byte-identical')
        old_completion = json.loads((historical / 'official_completion.json').read_text())
        if file_sha(historical / 'results/pedestrian_detailed.csv') != old_completion['detailed_sha256']:
            raise ValueError('Historical official sufficient statistics changed')
        for filename in ('pedestrian_summary.txt', 'pedestrian_detailed.csv'):
            shutil.copy2(historical / 'results' / filename, folder / 'results' / filename)
        metrics = official_metrics(folder / 'results')
        if int(metrics['Dets']) != receipt['prediction_rows']: raise ValueError('Historical reuse row count mismatch')
        result = dict(status='PASS', target=target, metrics=metrics, GT_protocol='OFFICIAL_UNCHANGED',
            sports418=False, wall_seconds=0., reused_after_all_rows_equal=True, historical_reference=str(historical),
            detailed_sha256=file_sha(folder / 'results/pedestrian_detailed.csv'),
            prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'))
        atomic_json(folder / 'official_completion.json', result)
        return result
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
    previous_log = folder / 'trackeval.log'
    if previous_log.exists():
        archive = folder / 'trackeval_attempts'; archive.mkdir(exist_ok=True)
        previous_log.rename(archive / (str(time.time_ns())+'.log'))
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
    p.add_argument('--head-checkpoint', type=Path); p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    if a.phase == 'decode': decode(a.target, a.model, head_checkpoint=a.head_checkpoint, output_dir=a.output_dir)
    else:
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            print(json.dumps(evaluate_folder((a.output_dir or RUN / 'stage_A' / a.target / a.model) / method, a.target)), flush=True)


if __name__ == '__main__': main()
