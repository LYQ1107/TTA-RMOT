"""Strong pooled KD versus anchor-aligned, appearance-conditioned inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.color_cohort import augmented_features, part_mask
from flex_uda.active.observation_memory import observation_edges, anchored_solution, interpolate_nodes
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_benchmark_scope import inputs
from scripts.a100_visual_classical_observation_prepare import OBSERVATION
from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions


def run(pair, seed):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, target = PAIRS[pair]
    m = metadata(target)
    videos = sorted(m['videos'])
    root = OBSERVATION / f'seed{seed}' / pair
    if (root / 'completion.json').is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    moments = {k: dict(total=np.zeros(1080), square=np.zeros(1080), rows=0) for k in ('POOL', 'NODE')}
    bx = {k: [] for k in moments}
    bz, by, receipts, start = [], [], [], 0
    stream = root / 'stream.pt'
    if stream.is_file():
        state = torch.load(stream, map_location='cpu', weights_only=False)
        moments, bx, bz, by = state['moments'], state['bx'], state['bz'], state['by']
        receipts, start = state['receipts'], state['next_video']
    methods = None
    for vi, video in enumerate(videos):
        if vi < start:
            continue
        started = time.perf_counter()
        _, z, answer_path, context, columns, gates = inputs(pair, video, m)
        with np.load(OBSERVATION / 'inputs' / pair / video / 'inputs.npz') as a:
            bank = {k: a[k].copy() for k in a.files}
        texts = [q['query_text'] for q in context.queries]
        geometry, _, parsed = augmented_features(bank['pooled_pcd'], bank['pooled_colours'], texts)
        node_geometry, _, node_parsed = augmented_features(bank['pcd'], bank['colours'], texts)
        raw = dict(POOL=np.concatenate([bank['pooled_pcd'], parsed], axis=2).astype('float32'),
            NODE=np.concatenate([bank['pcd'], node_parsed], axis=2).astype('float32'))
        sources = dict(POOL=z, NODE=z[:, bank['node_track']])
        statistics = {}
        for label in moments:
            current = raw[label].reshape(-1, 1080)
            moment = moments[label]
            moment['total'] += current.sum(0, dtype=np.float64)
            moment['square'] += np.square(current.astype(float)).sum(0)
            moment['rows'] += len(current)
            mean = moment['total']/moment['rows']
            std = np.maximum(np.sqrt(np.maximum(moment['square']/moment['rows']-mean**2, 0)), 1e-3)
            statistics[label] = mean, std
        data = json.loads(answer_path.read_text())
        teacher = np.full(z.shape, np.nan)
        for row in data['answers']:
            teacher[int(row['local_query']), int(row['local_track'])] = row['yes_probability']
        known = np.isfinite(teacher)
        qi, ti = np.nonzero(known)
        bx['POOL'].append(raw['POOL'][known])
        bx['NODE'].append(raw['NODE'][qi, bank['anchors'][ti]])
        bz.append(z[known])
        by.append(teacher[known])
        y, source = np.concatenate(by), np.concatenate(bz)
        scores, fits = {}, {}
        for key, label, soft in [('POOLED_SOFT', 'POOL', True), ('POOLED_HARD', 'POOL', False), ('ANCHOR', 'NODE', True)]:
            mean, std = statistics[label]
            fitted, receipt, head = fit_head(np.concatenate(bx[label]), source, np.clip(y, .02, .98),
                raw[label].reshape(-1, 1080), sources[label].ravel(), mean, std, soft=soft, seed=seed)
            scores[key] = {point: value.reshape(sources[label].shape) for point, value in fitted.items()}
            fits[key] = receipt
            torch.save(head, root / (key + '_latest_head.pt'))
        pooled = scores['POOLED_SOFT']['CONVERGED']
        pooled_mean = .5*(scores['POOLED_SOFT']['400']+pooled)
        local_score = dict(POOLED_SOFT=pooled, POOLED_SOFT400=scores['POOLED_SOFT']['400'],
            POOLED_MEAN=pooled_mean, POOLED_HARD400=scores['POOLED_HARD']['400'],
            POOLED_HARD_FINAL=scores['POOLED_HARD']['CONVERGED'], POOLED_GRAPH=pooled.copy(),
            POOLED_MEAN_GRAPH=pooled_mean.copy())
        node = scores['ANCHOR']['CONVERGED']
        node_mean = .5*(scores['ANCHOR']['400']+node)
        node_scores = dict(ANCHOR_KD=node, ANCHOR_MEAN=node_mean)
        for name in ('ANCHOR_NO_GRAPH', 'ANCHOR_VISUAL_ONLY', 'ANCHOR_TEMPORAL_ONLY', 'ANCHOR_FULL',
                     'CONSTANT_TEMPORAL', 'MASS_MATCHED_CONSTANT_TEMPORAL'):
            node_scores[name] = node.copy()
        node_scores['ANCHOR_MEAN_FULL'] = node_mean.copy()
        graph_receipts = []
        for q, text in enumerate(texts):
            required = part_mask(text) > 0
            if not required.any():
                continue
            valid = np.all(bank['pooled_colours'][:, required].sum(2) > 0, axis=1)
            local_score['POOLED_GRAPH'][q] = refine_distilled_prior(pooled[q], geometry[q], teacher[q],
                graph=True, source_weight=.1, node_mask=valid)
            local_score['POOLED_MEAN_GRAPH'][q] = refine_distilled_prior(pooled_mean[q], geometry[q], teacher[q],
                graph=True, source_weight=.1, node_mask=valid)
            visible = np.all(bank['colours'][:, required].sum(2) > 0, axis=1)
            weights, receipt = observation_edges(node_geometry[q], bank['node_track'], bank['offsets'], visible)
            anchor_answers = np.full(node.shape[1], np.nan)
            anchor_answers[bank['anchors'][known[q]]] = teacher[q, known[q]]
            for name, matrix in weights.items():
                node_scores[name][q] = anchored_solution(node[q], anchor_answers, matrix)
            node_scores['ANCHOR_MEAN_FULL'][q] = anchored_solution(node_mean[q], anchor_answers, weights['ANCHOR_FULL'])
            graph_receipts.append(dict(query=q, **receipt))
        obs = np.asarray(context.bank['observations'])
        expanded = {name: value[:, columns] for name, value in local_score.items()}
        for name, value in node_scores.items():
            full = interpolate_nodes(value, bank['frames'], bank['offsets'], obs, bank['tracks'])
            fallback = pooled_mean if name in ('ANCHOR_MEAN', 'ANCHOR_MEAN_FULL') else pooled
            for q, text in enumerate(texts):
                if not part_mask(text).any():
                    full[q] = fallback[q, columns]
            expanded[name] = full
        exclusive = exclusive_group(texts, target)
        methods = list(expanded)
        for name, value in expanded.items():
            mask = plain(value, exclusive)
            for q, gate in gates.items():
                mask[q] &= gate
            folder = root / name
            local = folder / 'videos' / video
            local.mkdir(parents=True, exist_ok=True)
            np.save(local / 'mask.npy', mask)
            # Interpolated decisions use float64. Preserve their actual
            # precision so the subsequent common CountSelf projection does
            # not acquire new ties from serialization alone.
            np.save(local / 'scores.npy', value)
            immediate_predictions(folder, context, mask)
        receipt = dict(video=video, index=vi, seed=seed, fits=fits, teacher_examples=len(y),
            current_teacher_examples=int(known.sum()), graph_receipts=graph_receipts,
            GT_open_attempts=len(guard), boxes_or_IDs_modified=False, wall_seconds=time.perf_counter()-started)
        atomic_json(root / 'videos' / video / 'receipt.json', receipt)
        receipts.append(receipt)
        state = dict(next_video=vi+1, moments=moments, bx=bx, bz=bz, by=by, receipts=receipts, methods=methods)
        temporary = stream.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(stream)
        print(json.dumps(dict(event='OBSERVATION_PREFIX_VIDEO', pair=pair, seed=seed,
            completed=vi+1, total=len(videos), video=video, teacher_examples=len(y))), flush=True)
    if methods is None:
        methods = state['methods']
    records = []
    for name in methods:
        folder = root / name
        mask = np.zeros(m['candidate_query_rows'], dtype=bool)
        for video in videos:
            local = np.load(folder / 'videos' / video / 'mask.npy')
            for q, episode in enumerate(q for q in m['episodes'] if q['video_id'] == video):
                mask[episode['start']:episode['stop']] = local[q]
        seal = seal_mask(folder, m, mask, target=target, model=model, method='OBSERVATION_V1_'+name,
            GT_open_attempts=len(guard), extra=dict(seed=seed, current_video_transductive=True,
                frozen_seed0_QA=True, neural_seed_not_new_domain=True, same_official_boxes_and_IDs=True,
                static_clothing_only_new_mechanism=True, new_teacher_calls=0,
                historical_features_prepaid=True, not_cold_deployment_cost=True))
        records.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root / 'completion.json', dict(status='SEALED_OBSERVATION_PREFIX', pair=pair, seed=seed,
        methods=records, GT_open_attempts=0, new_teacher_calls=0, receipts=receipts,
        wall_seconds=time.perf_counter()-begin, research_goal_still_active=True,
        implementation_sha256={str(p.relative_to(ROOT)): file_sha(p) for p in
            (Path(__file__), ROOT/'flex_uda/active/observation_memory.py')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=['C4', 'C1'], required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args.pair, args.seed)
