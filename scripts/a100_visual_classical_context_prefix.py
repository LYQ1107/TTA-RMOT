"""Registered relative-context study and sparse-budget graph specificity."""
import argparse
import hashlib
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
from flex_uda.active.context_cohort import context_arrays, context_weights, relative_positions
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_benchmark_scope import inputs, FRESH
from scripts.a100_visual_classical_budget_prefix import BUDGET
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions

CONTEXT = OUT / 'context_cohort'


def permuted(weights, visible, video, query):
    ids = np.flatnonzero(visible)
    key = f'{video}:{query}:context_specificity:0'
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    order = np.arange(len(weights))
    order[ids] = np.random.default_rng(seed).permutation(ids)
    return weights[np.ix_(order, order)]


def run(pair, seed):
    guard = generation_guard()
    model, target = PAIRS[pair]
    m = metadata(target)
    videos = sorted(m['videos'])
    root = CONTEXT / f'seed{seed}' / pair
    if (root / 'completion.json').is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    original = BUDGET / f'rho0.1_seed{seed}' / pair
    assert json.loads((original / 'completion.json').read_text())['GT_open_attempts'] == 0
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    moments = {k: dict(total=np.zeros(1088), square=np.zeros(1088), rows=0) for k in ('MASKED', 'CONTEXT')}
    bx, bz, by, receipts = {k: [] for k in moments}, [], [], []
    start, methods = 0, None
    stream = root / 'stream.pt'
    if stream.is_file():
        state = torch.load(stream, map_location='cpu', weights_only=False)
        moments, bx, bz, by = state['moments'], state['bx'], state['bz'], state['by']
        receipts, start = state['receipts'], state['next_video']
    begin = time.perf_counter()
    for vi, video in enumerate(videos):
        if vi < start:
            continue
        started = time.perf_counter()
        raw, z, _, context, columns, gates = inputs(pair, video, m)
        texts = [q['query_text'] for q in context.queries]
        path = FRESH / 'source' / video / 'sparse_inputs.npz' if pair == 'C4' else OUT / 'color_features' / pair / video / 'features.npz'
        with np.load(path) as a:
            colours = a['colours']
        geometry, _, _ = augmented_features(raw[:, :, :768], colours, texts)
        data = json.loads((BUDGET / 'evidence/rho0.1' / pair / video / 'answers.json').read_text())
        teacher = np.full(z.shape, np.nan)
        for row in data['answers']:
            teacher[int(row['local_query']), int(row['local_track'])] = row['yes_probability']
        known = np.isfinite(teacher)
        bz.append(z[known])
        by.append(teacher[known])
        first = [int(np.flatnonzero(columns == ti)[0]) for ti in range(z.shape[1])]
        old_mean = np.load(original / 'SOFT_MEAN_400_FINAL/videos' / video / 'scores.npy')[:, first]
        # The original budget worker assigned refined doubles into its
        # float32 head array before decoding. Reproduce that precision here.
        original_graph = old_mean.copy()
        predicted = {name: old_mean.copy().astype(float) for name in ('ORIGINAL_COMPLETE', 'ORIGINAL_PERMUTED')}
        for q, text in enumerate(texts):
            required = part_mask(text) > 0
            if not required.any():
                continue
            visible = np.all(colours[:, required].sum(2) > 0, axis=1)
            matrices = context_weights(geometry[q], np.zeros((z.shape[1], 8)), visible, spatial=False)
            original_graph[q] = refine_distilled_prior(old_mean[q], geometry[q], teacher[q], source_weight=.1,
                weights_override=matrices['FULL'])
            for name, weights in [('ORIGINAL_COMPLETE', matrices['COMPLETE']),
                ('ORIGINAL_PERMUTED', permuted(matrices['FULL'], visible, video, q))]:
                predicted[name][q] = refine_distilled_prior(old_mean[q], geometry[q], teacher[q], source_weight=.1,
                    weights_override=weights)
        exclusive = exclusive_group(texts, target)
        check = plain(original_graph[:, columns], exclusive)
        for q, gate in gates.items():
            check[q] &= gate
        old_mask = np.load(original / 'MEAN_GRAPH/videos' / video / 'mask.npy')
        changed = int(np.count_nonzero(check != old_mask))
        assert changed == 0, ('Original graph regression', pair, seed, video, changed)
        fits = {}
        if pair == 'C4':
            positions = relative_positions(context.bank)
            contextual, contextual_geometry, parts, spatial = context_arrays(raw[:, :, :768], colours, texts, positions)
            arrays = dict(MASKED=np.concatenate([raw, np.zeros((*z.shape, 8), dtype='float32')], axis=2),
                CONTEXT=contextual)
            statistics = {}
            for label, value in arrays.items():
                current = value.reshape(-1, 1088)
                moment = moments[label]
                moment['total'] += current.sum(0, dtype=np.float64)
                moment['square'] += np.square(current.astype(float)).sum(0)
                moment['rows'] += len(current)
                mean = moment['total']/moment['rows']
                std = np.maximum(np.sqrt(np.maximum(moment['square']/moment['rows']-mean**2, 0)), 1e-3)
                statistics[label] = mean, std
                bx[label].append(value[known])
            y, source = np.concatenate(by), np.concatenate(bz)
            heads = {}
            for name, label, soft in [('MASKED', 'MASKED', True), ('SOFT', 'CONTEXT', True), ('HARD', 'CONTEXT', False)]:
                mean, std = statistics[label]
                if len(y):
                    fitted, receipt, head = fit_head(np.concatenate(bx[label]), source, np.clip(y, .02, .98),
                        arrays[label].reshape(-1, 1088), z.ravel(), mean, std, soft=soft, seed=seed)
                    torch.save(head, root / (name+'_latest_head.pt'))
                else:
                    fitted = {point: expit(z.ravel()) for point in ('400', '4000', 'CONVERGED')}
                    receipt = dict(status='NO_TEACHER_EXAMPLES_SOURCE_FALLBACK', teacher_examples=0, steps=0)
                heads[name] = {point: value.reshape(z.shape) for point, value in fitted.items()}
                fits[name] = receipt
            average = .5*(heads['SOFT']['400']+heads['SOFT']['CONVERGED'])
            predicted.update(MASKED_MEAN=.5*(heads['MASKED']['400']+heads['MASKED']['CONVERGED']),
                CONTEXT_SOFT400=heads['SOFT']['400'], CONTEXT_SOFT_FINAL=heads['SOFT']['CONVERGED'],
                CONTEXT_MEAN=average, CONTEXT_HARD400=heads['HARD']['400'],
                CONTEXT_HARD_FINAL=heads['HARD']['CONVERGED'])
            for name in ('CONTEXT_ANCHORS', 'CONTEXT_VISUAL', 'CONTEXT_FULL', 'CONTEXT_COMPLETE', 'CONTEXT_PERMUTED'):
                predicted[name] = average.copy().astype(float)
            for q, text in enumerate(texts):
                required = parts[q] > 0
                if not required.any():
                    continue
                visible = np.all(colours[:, required].sum(2) > 0, axis=1)
                matrices = context_weights(contextual_geometry[q], positions, visible, spatial=bool(spatial[q]))
                for name, weights in [('CONTEXT_ANCHORS', np.zeros_like(matrices['FULL'])),
                    ('CONTEXT_VISUAL', matrices['VISUAL']), ('CONTEXT_FULL', matrices['FULL']),
                    ('CONTEXT_COMPLETE', matrices['COMPLETE']),
                    ('CONTEXT_PERMUTED', permuted(matrices['FULL'], visible, video, q))]:
                    predicted[name][q] = refine_distilled_prior(average[q], contextual_geometry[q], teacher[q],
                        source_weight=.1, weights_override=weights)
        methods = list(predicted)
        for name, value in predicted.items():
            expanded = value[:, columns]
            mask = plain(expanded, exclusive)
            for q, gate in gates.items():
                mask[q] &= gate
            folder = root / name
            local = folder / 'videos' / video
            local.mkdir(parents=True, exist_ok=True)
            np.save(local / 'mask.npy', mask)
            np.save(local / 'scores.npy', expanded)
            immediate_predictions(folder, context, mask)
        receipt = dict(video=video, index=vi, seed=seed, teacher_examples=sum(map(len, by)),
            current_teacher_examples=int(known.sum()), fits=fits, original_graph_changed_rows=changed,
            GT_open_attempts=len(guard), wall_seconds=time.perf_counter()-started)
        atomic_json(root / 'videos' / video / 'receipt.json', receipt)
        receipts.append(receipt)
        state = dict(next_video=vi+1, moments=moments, bx=bx, bz=bz, by=by, receipts=receipts, methods=methods)
        temporary = stream.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(stream)
        print(json.dumps(dict(event='CONTEXT_PREFIX_VIDEO', pair=pair, seed=seed, completed=vi+1,
            total=len(videos), video=video, teacher_examples=sum(map(len, by)))), flush=True)
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
        seal = seal_mask(folder, m, mask, target=target, model=model, method='CONTEXT_COHORT_'+name,
            GT_open_attempts=len(guard), extra=dict(seed=seed, teacher_fraction=.1,
                same_frozen_QA=True, new_teacher_calls=0, same_official_boxes_and_IDs=True,
                current_video_transductive=True, no_future_feedback=True, quality_replay_not_cold_cost=True))
        records.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root / 'completion.json', dict(status='SEALED_CONTEXT_COHORT_PREFIX', pair=pair,
        seed=seed, logical_teacher_calls=sum(map(len, by)), methods=records, GT_open_attempts=0,
        original_graph_regression_changed_rows=sum(r['original_graph_changed_rows'] for r in receipts),
        receipts=receipts, new_teacher_calls=0, wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE',
        implementation_sha256={str(p.relative_to(ROOT)): file_sha(p) for p in
            (Path(__file__), ROOT/'flex_uda/active/context_cohort.py')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=['C4', 'C1'], required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.pair, a.seed)
