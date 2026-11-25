"""Fixed nested teacher-budget curves with fully fitted matched KD controls."""
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
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_benchmark_scope import inputs, FRESH
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain, immediate_predictions

BUDGET = OUT / 'sparse_evidence'
METHODS = ('SOFT_400', 'SOFT_FINAL', 'SOFT_MEAN_400_FINAL', 'HARD_400', 'HARD_FINAL',
           'NO_GRAPH', 'GRAPH', 'MEAN_GRAPH')


def prepare_evidence(pair, fraction):
    """Opaque frozen broker: seal questions, then export only selected answers.

    This evidence preparation is not a cold teacher execution. Adaptation only
    opens its smaller shard and never the unused full-budget answer reservoir.
    """
    guard = generation_guard()
    m = metadata(PAIRS[pair][1])
    root = BUDGET / 'evidence' / f'rho{fraction:g}' / pair
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'completion.json').is_file():
        return
    full_plan = None
    if pair == 'C1':
        full_plan = json.loads((OUT / 'evidence' / pair / 'acquisition_plan.json').read_text())['positions']
    receipts = []
    for video, bank in sorted(m['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in m['episodes'] if q['video_id'] == video]
        tracks, counts = np.unique(obs[:, 1], return_counts=True)
        budget = int(np.floor(fraction*len(queries)*int((counts >= 3).sum())))
        quota = np.full(len(queries), budget//len(queries))
        quota[:budget % len(queries)] += 1
        if pair == 'C4':
            plan_path = FRESH / 'source' / video / 'questions_before_answers.json'
            all_questions = [(int(r['local_query']), int(r['local_track']))
                             for r in json.loads(plan_path.read_text())['requests']]
            source = plan_path.with_name('answers.json')
        else:
            plan_path = OUT / 'evidence' / pair / 'acquisition_plan.json'
            all_questions = [tuple(map(int, r)) for r in full_plan[video]['COVER']]
            source = OUT / 'causal_teacher' / pair / 'COVER' / (video+'.json')
        questions = []
        for qi, number in enumerate(quota):
            ordered = [(q, t) for q, t in all_questions if q == qi]
            assert len(ordered) >= number
            questions.extend(ordered[:number])
        assert len(questions) == budget and len(set(questions)) == budget
        folder = root / video
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / 'questions_before_answers.json', dict(pair=pair, video=video,
            fraction=fraction, positions=questions, quota=quota.tolist(), source_question_plan=str(plan_path),
            source_plan_sha256=file_sha(plan_path), response_blind=True, GT_open_attempts=len(guard)))
        # The fixed question subset was written before opening the responses.
        data = json.loads(source.read_text())
        lookup = {(int(r['local_query']), int(r['local_track'])): r for r in data['answers']}
        answers = [lookup[q, t] for q, t in questions]
        atomic_json(folder / 'answers.json', dict(pair=pair, video=video, fraction=fraction,
            answers=answers, logical_calls=len(answers), GT_open_attempts=len(guard),
            new_teacher_calls=0, opaque_frozen_broker_preparation_not_deployment_cost=True))
        receipts.append(dict(video=video, questions=budget, eligible_pairs=len(queries)*int((counts>=3).sum())))
    assert not guard
    atomic_json(root / 'completion.json', dict(status='SEALED_NESTED_EVIDENCE', pair=pair,
        fraction=fraction, videos=receipts, logical_calls=sum(r['questions'] for r in receipts),
        GT_open_attempts=0, new_teacher_calls=0))


def run(pair, fraction, seed):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, target = PAIRS[pair]
    m = metadata(target)
    videos = sorted(m['videos'])
    evidence = BUDGET / 'evidence' / f'rho{fraction:g}' / pair
    assert (evidence / 'completion.json').is_file()
    root = BUDGET / f'rho{fraction:g}_seed{seed}' / pair
    if (root / 'completion.json').is_file():
        return
    root.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    total, square, image_rows = np.zeros(1080), np.zeros(1080), 0
    bx, bz, by, receipts, start = [], [], [], [], 0
    stream = root / 'stream.pt'
    if stream.is_file():
        state = torch.load(stream, map_location='cpu', weights_only=False)
        total, square, image_rows = state['total'], state['square'], state['image_rows']
        bx, bz, by, receipts, start = state['bx'], state['bz'], state['by'], state['receipts'], state['next_video']
    for vi, video in enumerate(videos):
        if vi < start:
            continue
        started = time.perf_counter()
        raw, z, _, context, columns, gates = inputs(pair, video, m)
        if pair == 'C4':
            with np.load(FRESH / 'source' / video / 'sparse_inputs.npz') as a:
                colours = a['colours']
        else:
            with np.load(OUT / 'color_features' / pair / video / 'features.npz') as a:
                colours = a['colours']
        texts = [q['query_text'] for q in context.queries]
        geometry, _, _ = augmented_features(raw[:, :, :768], colours, texts)
        current = raw.reshape(-1, 1080)
        total += current.sum(0, dtype=np.float64)
        square += np.square(current.astype(float)).sum(0)
        image_rows += len(current)
        mean = total/image_rows
        std = np.maximum(np.sqrt(np.maximum(square/image_rows-mean**2, 0)), 1e-3)
        data = json.loads((evidence / video / 'answers.json').read_text())
        teacher = np.full(z.shape, np.nan)
        for row in data['answers']:
            teacher[int(row['local_query']), int(row['local_track'])] = row['yes_probability']
        known = np.isfinite(teacher)
        bx.append(raw[known])
        bz.append(z[known])
        by.append(teacher[known])
        x, source, y = np.concatenate(bx), np.concatenate(bz), np.concatenate(by)
        predicted, fits = {}, {}
        for label, soft in [('SOFT', True), ('HARD', False)]:
            if len(y):
                score, fit, head = fit_head(x, source, np.clip(y, .02, .98), current, z.ravel(),
                    mean, std, soft=soft, seed=seed)
                torch.save(head, root / (label+'_latest_head.pt'))
            else:
                score = {point: expit(z.ravel()) for point in ('400', '4000', 'CONVERGED')}
                fit = dict(status='NO_TEACHER_EXAMPLES_SOURCE_FALLBACK', steps=0, teacher_examples=0)
            fits[label] = fit
            predicted[label+'_400'] = score['400'].reshape(z.shape)
            predicted[label+'_FINAL'] = score['CONVERGED'].reshape(z.shape)
        kd = predicted['SOFT_FINAL']
        avg = .5*(predicted['SOFT_400']+kd)
        predicted['SOFT_MEAN_400_FINAL'] = avg
        predicted['NO_GRAPH'] = kd.copy()
        predicted['GRAPH'] = kd.copy()
        predicted['MEAN_GRAPH'] = avg.copy()
        for q, text in enumerate(texts):
            required = part_mask(text) > 0
            if not required.any():
                continue
            visible = np.all(colours[:, required].sum(2) > 0, axis=1)
            for name, prior, graph in [('NO_GRAPH', kd, False), ('GRAPH', kd, True), ('MEAN_GRAPH', avg, True)]:
                predicted[name][q] = refine_distilled_prior(prior[q], geometry[q], teacher[q],
                    graph=graph, source_weight=.1, node_mask=visible)
        exclusive = exclusive_group(texts, target)
        for name in METHODS:
            expanded = predicted[name][:, columns]
            mask = plain(expanded, exclusive)
            for q, gate in gates.items():
                mask[q] &= gate
            folder = root / name
            local = folder / 'videos' / video
            local.mkdir(parents=True, exist_ok=True)
            np.save(local / 'mask.npy', mask)
            np.save(local / 'scores.npy', expanded.astype('float32'))
            immediate_predictions(folder, context, mask)
        receipt = dict(video=video, index=vi, seed=seed, fraction=fraction,
            teacher_examples=len(y), current_teacher_examples=int(known.sum()), fits=fits,
            GT_open_attempts=len(guard), wall_seconds=time.perf_counter()-started)
        atomic_json(root / 'videos' / video / 'receipt.json', receipt)
        receipts.append(receipt)
        state = dict(next_video=vi+1, total=total, square=square, image_rows=image_rows,
            bx=bx, bz=bz, by=by, receipts=receipts)
        temporary = stream.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(stream)
        print(json.dumps(dict(event='SPARSE_BUDGET_VIDEO', pair=pair, fraction=fraction, seed=seed,
            completed=vi+1, total=len(videos), video=video, teacher_examples=len(y))), flush=True)
    records = []
    for name in METHODS:
        folder = root / name
        mask = np.zeros(m['candidate_query_rows'], dtype=bool)
        for video in videos:
            local = np.load(folder / 'videos' / video / 'mask.npy')
            for q, episode in enumerate(q for q in m['episodes'] if q['video_id'] == video):
                mask[episode['start']:episode['stop']] = local[q]
        seal = seal_mask(folder, m, mask, target=target, model=model, method='SPARSE_BUDGET_'+name,
            GT_open_attempts=len(guard), extra=dict(seed=seed, teacher_fraction=fraction,
                logical_teacher_calls=sum(map(len, by)), new_teacher_calls=0,
                fixed_recorded_seed0_question_order=True, same_QA_in_all_matched_methods=True,
                no_future_feedback=True, current_video_transductive=True,
                frozen_features_and_QA_prepaid=True, not_cold_deployment_cost=True))
        records.append(dict(method=name, rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root / 'completion.json', dict(status='SEALED_SPARSE_BUDGET_PREFIX', pair=pair,
        seed=seed, fraction=fraction, logical_teacher_calls=sum(map(len, by)), methods=records,
        receipts=receipts, GT_open_attempts=0, new_teacher_calls=0,
        wall_seconds=time.perf_counter()-begin, research_goal_still_active=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', choices=['C4', 'C1'], required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prepare-evidence', action='store_true')
    args = p.parse_args()
    if args.prepare_evidence:
        prepare_evidence(args.pair, args.fraction)
    else:
        run(args.pair, args.fraction, args.seed)
