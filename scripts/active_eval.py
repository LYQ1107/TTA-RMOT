"""Independent post-seal official evaluation, quality and video bootstrap."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/FlexHook/TrackEval'))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata, sealed_margins
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def offline_labels(target):
    """Own-source validation already matched the SAME complete tracker bank.

    Only this offline evaluator may reuse those numerical labels. The target
    runtime explicitly rejects labels.npy and GT_counts.npy as well as raw GT.
    """
    meta = target_metadata(target)
    truth = {}; by_video = {}
    root = RUN / 'features' / target / IN_DOMAIN[target] / 'videos'
    for directory in sorted(root.iterdir()):
        m = json.loads((directory / 'metadata.json').read_text())
        observations = np.load(directory / 'observations.npy')
        y = np.load(directory / 'labels.npy').reshape(len(m['queries']), -1)
        counts = np.load(directory / 'GT_counts.npy')
        for q, query in enumerate(m['queries']):
            truth[m['video_id'], query['expression_id']] = dict(labels=y[q], counts=counts[q], frames=m['frames'])
        by_video[m['video_id']] = dict(observations=observations, metadata=m)
    if len(truth) != len(meta['episodes']): raise ValueError('Incomplete independent evaluation truth')
    return meta, truth, by_video


def official_stats(folder):
    from scripts.analyze_pf_video_uncertainty import HOTA, unpack, sufficient, pooled_hota
    path = Path(folder) / 'results/pedestrian_detailed.csv'
    with path.open() as stream: rows = list(csv.DictReader(stream))
    metric = HOTA({'PRINT_CONFIG': False})
    by = {r['seq']: unpack(r, metric) for r in rows if r['seq'] != 'COMBINED'}
    videos = sorted({key.split('+')[0] for key in by})
    stats = sufficient([metric.combine_sequences({key: value for key, value in by.items() if key.startswith(video+'+')}) for video in videos])
    exact = float(pooled_hota(stats, np.ones(len(videos))))
    combined = next(r for r in rows if r['seq'] == 'COMBINED')
    if not np.isclose(exact, 100*float(combined['HOTA___AUC']), atol=1e-8, rtol=0):
        raise ValueError('Video-cluster statistics do not reproduce official pooling')
    return dict(videos=videos, stats=stats, rows=by, HOTA_unrounded=exact)


def summarize(folder, target):
    from scripts.presence_eval_offline import evaluate_folder
    folder = Path(folder); seal = verify_predictions(folder)
    evaluated = evaluate_folder(folder, target)
    meta, truth, videos = offline_labels(target)
    stats = official_stats(folder)
    from scripts.analyze_pf_video_uncertainty import HOTA
    metric = HOTA({'PRINT_CONFIG': False})
    truth = {(v, q): value for (v, q), value in truth.items() if v+'+'+q in stats['rows']}
    present = {v+'+'+q: stats['rows'][v+'+'+q] for (v, q), t in truth.items() if t['counts'].sum()}
    grouped = metric.combine_sequences(present)
    evaluated_episodes = [row for row in seal['episodes'] if (row['video'], row['expression']) in truth]
    empty_rows = sum(row['selected_rows'] for row in evaluated_episodes if not truth[row['video'], row['expression']]['counts'].sum())
    result = dict(target=target, model=seal['model'], method=seal['method'], rho=seal.get('rho'),
        **{k: evaluated['metrics'][k] for k in ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW')},
        HOTA_unrounded=stats['HOTA_unrounded'], prediction_rows=sum(row['selected_rows'] for row in evaluated_episodes),
        query_empty_prediction_rows=empty_rows, query_empty_FP=empty_rows,
        Present_HOTA=100*float(np.mean(grouped['HOTA'])), query_present_episodes=len(present),
        query_empty_episodes=len(truth)-len(present), teacher_calls=seal.get('teacher', {}).get('logical_calls', 0),
        teacher_forward_GPU_hours=seal.get('teacher', {}).get('forward_seconds', 0)/3600,
        teacher_equivalent_forward_GPU_hours=seal.get('teacher', {}).get('equivalent_forward_seconds', 0)/3600,
        LoRA_GPU_hours=seal.get('LoRA_GPU_seconds', 0)/3600,
        rescore_GPU_hours=seal.get('rescore_GPU_seconds', 0)/3600,
        GT_open_attempts=seal['GT_open_attempts'], future_frames=seal.get('video_transductive', seal['method'].endswith('SMOOTH')),
        path=str(folder), prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'))
    if seal['method'].startswith('PRES_TOPK') and len(truth) == len(meta['episodes']):
        decoded = folder.parent
        plogits = np.load(decoded / 'presence_logits.npy'); raw_k = np.load(decoded / ('smooth_k.npy' if seal['method'].endswith('SMOOTH') else 'raw_k.npy'))
        offset = 0; correct = count_error = no_candidates = 0; units = 0
        observable_correct = observable_error = 0
        native_margin, _ = sealed_margins(target, seal['model'])
        query_index = {(q['video_id'], q['expression_id']):q for q in meta['episodes']}
        native_correct = native_count_error = native_observable_correct = native_observable_error = 0
        head_tp = head_fn = head_fp = head_tn = native_tp = native_fn = native_fp = native_tn = 0
        overflow_units = 0
        for video in sorted(videos):
            md = videos[video]['metadata']
            frames = videos[video]['observations'][:, 0]
            observed_frames = set(frames)
            for q in md['queries']:
                t = truth[video, q['expression_id']]; n = len(t['frames'])
                head_present = plogits[offset:offset+n] >= 0; scene_present = t['counts'] > 0
                correct += int((head_present == scene_present).sum())
                count_error += float(np.abs(raw_k[offset:offset+n]-t['counts']).sum())
                positive_counts = dict(zip(*np.unique(frames[t['labels']], return_counts=True)))
                accessible = np.asarray([positive_counts.get(frame, 0) for frame in t['frames']])
                observable_correct += int(((plogits[offset:offset+n] >= 0) == (accessible > 0)).sum())
                observable_error += float(np.abs(raw_k[offset:offset+n]-accessible).sum())
                query = query_index[video, q['expression_id']]
                source_scores = native_margin[query['start']:query['stop']]
                selected_counts = dict(zip(*np.unique(frames[source_scores > 0], return_counts=True)))
                native_k = np.asarray([selected_counts.get(frame, 0) for frame in t['frames']])
                native_present = native_k > 0
                native_correct += int((native_present == scene_present).sum())
                native_count_error += float(np.abs(native_k-t['counts']).sum())
                native_observable_correct += int((native_present == (accessible > 0)).sum())
                native_observable_error += float(np.abs(native_k-accessible).sum())
                head_tp += int((head_present & scene_present).sum()); head_fn += int((~head_present & scene_present).sum())
                head_fp += int((head_present & ~scene_present).sum()); head_tn += int((~head_present & ~scene_present).sum())
                native_tp += int((native_present & scene_present).sum()); native_fn += int((~native_present & scene_present).sum())
                native_fp += int((native_present & ~scene_present).sum()); native_tn += int((~native_present & ~scene_present).sum())
                overflow_units += int((accessible > 10).sum())
                no_candidates += sum(frame not in observed_frames for frame in t['frames'])
                units += n; offset += n
        if offset != len(plogits): raise ValueError('Presence evaluation unit order differs')
        result.update(frame_query_units=units, presence_accuracy_head=correct/units,
                      count_MAE_head=count_error/units, presence_accuracy_observable=observable_correct/units,
                      count_MAE_observable=observable_error/units, zero_candidate_units=no_candidates,
                      native_presence_accuracy=native_correct/units, native_count_MAE=native_count_error/units,
                      native_presence_accuracy_observable=native_observable_correct/units,
                      native_count_MAE_observable=native_observable_error/units,
                      head_presence_TP=head_tp, head_presence_FN=head_fn, head_presence_FP=head_fp, head_presence_TN=head_tn,
                      native_presence_TP=native_tp, native_presence_FN=native_fn, native_presence_FP=native_fp, native_presence_TN=native_tn,
                      supported_count_above_cap10_units=overflow_units,
                      scene_vs_observable='head/scene fields use official GT count; observable fields use matched-positive candidate count')
        source_domain = next(t for t, name in IN_DOMAIN.items() if name == seal['model'])
        seen = set()
        for directory in (RUN / 'features/source_train' / source_domain / seal['model'] / 'videos').iterdir():
            m = json.loads((directory / 'metadata.json').read_text())
            seen.update(' '.join(q['query_text'].lower().split()) for q in m['queries'])
        groups = dict(SEEN_EXACT_PHRASE={}, UNSEEN_EXACT_PHRASE={})
        for query in meta['episodes']:
            key = query['video_id']+'+'+query['expression_id']
            if key not in stats['rows']: continue
            group = 'SEEN_EXACT_PHRASE' if ' '.join(query['query_text'].lower().split()) in seen else 'UNSEEN_EXACT_PHRASE'
            groups[group][key] = stats['rows'][key]
        for group, sequences in groups.items():
            result[group+'_episodes'] = len(sequences)
            result[group+'_HOTA'] = 100*float(np.mean(metric.combine_sequences(sequences)['HOTA'])) if sequences else None
    atomic_json(folder / 'offline_summary.json', result)
    return result


def sports418(folder):
    """Keep the v4 GT content, restrict ONLY the sequence set to old418 IDs."""
    import copy
    from scripts.presence_eval_offline import evaluate_folder
    folder = Path(folder); original = verify_predictions(folder)
    missing = set(json.loads((ROOT / 'results/multidomain_generality/sports_template_inventory.json').read_text())['missing_GT'])
    out = folder / 'sensitivity_418'; out.mkdir(exist_ok=True)
    receipt = copy.deepcopy(original); receipt['episodes'] = []
    for row in original['episodes']:
        if row['video']+'+'+row['expression'] in missing: continue
        path = out / 'results' / row['video'] / row['expression'] / 'predict.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(): path.symlink_to(row['prediction_path'])
        receipt['episodes'].append(dict(row, prediction_path=str(path)))
    receipt['episode_count'] = len(receipt['episodes'])
    receipt['prediction_rows'] = sum(row['selected_rows'] for row in receipt['episodes'])
    if receipt['episode_count'] != 418: raise ValueError('Sports sensitivity must keep exactly418 official sequence IDs')
    receipt['original_full466_seal_sha256'] = file_sha(folder / 'predictions_seal.json')
    receipt['sensitivity_protocol'] = 'same v4 content, official418 IDs only; not old inconsistent GT templates'
    atomic_json(out / 'predictions_seal.json', receipt)
    return evaluate_folder(out, 'SportsMOT', sports418=True)


def bootstrap(first, second, target):
    a, b = official_stats(first), official_stats(second)
    if a['videos'] != b['videos']: raise ValueError('Different video clusters')
    from scripts.analyze_pf_video_uncertainty import paired_clusters
    n = len(a['videos']); delta = float(paired_clusters(a['stats'], b['stats'], np.ones(n)))
    if target == 'KITTI-V2': return dict(delta_HOTA=delta, clusters=n, CI95=None, note='4 videos; point estimate only')
    weights = np.random.default_rng(0).multinomial(n, np.ones(n)/n, size=2000)
    samples = paired_clusters(a['stats'], b['stats'], weights)
    return dict(delta_HOTA=delta, clusters=n, samples=2000, seed=0, CI95=np.percentile(samples, [2.5, 97.5]).tolist(),
                unit='video, all correlated expressions and frames kept together', pooling='official TP-weighted HOTA')


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True); p.add_argument('--target', choices=list(IN_DOMAIN), required=True)
    p.add_argument('--compare', type=Path)
    p.add_argument('--sports418', action='store_true')
    a = p.parse_args(); result = summarize(a.folder, a.target)
    if a.sports418:
        if a.target != 'SportsMOT': raise ValueError('418 sensitivity applies only to SportsMOT')
        result['sports418'] = sports418(a.folder)
    if a.compare: result['comparison'] = bootstrap(a.folder, a.compare, a.target)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
