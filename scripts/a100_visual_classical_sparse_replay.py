"""Fixed fresh full-stream QA, with only source prior/gate sparsified.

Quality-only intervention: full-reference PCD/RGB and identical actual teacher
answers are used on both sides, while this side uses genuinely sparse source
fractions and interpolated presence. This is not a cold deployment cost claim.
"""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit, logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata, generation_guard
from flex_uda.active.color_cohort import augmented_features, part_mask
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.acquisition import expression_types
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT

METHODS = ('RAW_KD_400', 'RAW_KD', 'RAW_COHORT', 'CLIP_KD_400', 'CLIP_KD',
           'CLIP_COHORT', 'HARD_KD_400', 'HARD_KD')


def run():
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions, plain
    gt = generation_guard();visible = set();forbidden = []
    full = OUT / 'cold_full_c4/C4_fresh_full_stream_seed0'
    sparse = OUT / 'cold_full_c4/C4_fresh_track_three_seed0'
    root = OUT / 'cold_full_c4/C4_sparse_fixedQA_seed0'
    state_path = OUT / 'supervisor/sparse_replay_state.json'
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]));flags = args[2] if len(args) > 2 else 0
        if flags & (os.O_WRONLY | os.O_RDWR):
            return
        try:
            parts = path.relative_to(OUT).parts
        except ValueError:
            return
        reason = None
        if parts and parts[0] in ('analysis', 'evidence', 'offline_GT'):
            reason = 'aggregate future evidence or evaluation-derived artifacts'
        if len(parts) > 1 and parts[0] == 'cold_full_c4' and parts[1] in (full.name, sparse.name):
            if len(parts) < 4 or parts[2] != 'source' or parts[3] not in visible:
                reason = 'future source/teacher video or another trained head'
        if reason:
            forbidden.append(dict(path=str(path), reason=reason))
            raise PermissionError(reason + ': ' + str(path))
    sys.addaudithook(audit)
    if (root / 'completion.json').is_file():
        return
    root.mkdir(parents=True, exist_ok=True);start = time.perf_counter()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = False
    m = metadata('SportsMOT');videos = sorted(m['videos'])
    bx = [];bz = [];by = [];receipts = [];next_video = 0
    moment_sum = np.zeros(1080);moment_square = np.zeros(1080);image_rows = 0
    checkpoint = root / 'stream.pt'
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        bx, bz, by, receipts = state['x'], state['z'], state['y'], state['receipts']
        moment_sum, moment_square, image_rows = state['sum'], state['square'], state['image_rows']
        next_video = state['next_video'];visible.update(videos[:next_video])
    for vi, video in enumerate(videos):
        if vi < next_video:
            continue
        visible.add(video)
        atomic_json(state_path, dict(status='WAITING_FOR_BOTH_ACTUAL_VIDEO_STREAMS', video=video, index=vi, pid=os.getpid()))
        while not all((p / 'source' / video / 'receipt.json').is_file() for p in (full, sparse)):
            for parent in ('cold_full_state.json', 'cold_sparse_state.json'):
                path = OUT / 'supervisor' / parent
                if path.is_file() and json.loads(path.read_text()).get('status', '').startswith('FAILED'):
                    raise RuntimeError('Source stream failed: ' + parent)
            time.sleep(15)
        began = time.perf_counter();bank = m['videos'][video];observations = np.asarray(bank['observations'])
        queries = [q for q in m['episodes'] if q['video_id'] == video]
        with np.load(full / 'source' / video / 'sparse_inputs.npz') as values:
            original, colour, tracks = values['pcd'], values['colours'], values['tracks']
        with np.load(sparse / 'source' / video / 'sparse_inputs.npz') as values:
            assert np.array_equal(tracks, values['tracks'])
            source = np.clip(logit(np.clip(values['prior'], .05, .95)), -2, 2)
            feature_delta = float(np.max(np.abs(original - values['pcd'])))
            colour_delta = float(np.max(np.abs(colour - values['colours'])))
        geometry, _, selected = augmented_features(original, colour, [q['query_text'] for q in queries])
        raw = np.concatenate([original, selected], axis=2).astype('float32');current = raw.reshape(-1, 1080)
        moment_sum += current.sum(0, dtype=np.float64);moment_square += np.square(current.astype(float)).sum(0)
        image_rows += len(current);mean = moment_sum / image_rows
        std = np.maximum(np.sqrt(np.maximum(moment_square / image_rows - mean ** 2, 0)), 1e-3)
        teacher = np.full(source.shape, np.nan)
        answers = json.loads((full / 'source' / video / 'answers.json').read_text())
        assert answers['GT_open_attempts'] == 0
        for answer in answers['answers']:
            teacher[answer['local_query'], answer['local_track']] = answer['yes_probability']
        known = np.isfinite(teacher);bx.append(raw[known]);bz.append(source[known]);by.append(teacher[known])
        x, z, y = np.concatenate(bx), np.concatenate(bz), np.concatenate(by)
        predictions = {};fits = {}
        for label in ('RAW', 'CLIP', 'HARD'):
            targets = np.clip(y, .02, .98) if label == 'CLIP' else y
            scores, fit, _ = fit_head(x, z, targets, current, source.ravel(), mean, std, soft=label != 'HARD', seed=0)
            fits[label] = fit
            predictions[label + '_KD_400'] = scores['400'].reshape(source.shape)
            predictions[label + '_KD'] = scores['CONVERGED'].reshape(source.shape)
            if label != 'HARD':
                result = predictions[label + '_KD'].copy()
                for qi, query in enumerate(queries):
                    part = part_mask(query['query_text']) > 0
                    if part.any():
                        result[qi] = refine_distilled_prior(result[qi], geometry[qi], teacher[qi], source_weight=.1,
                                                           node_mask=np.all(colour[:, part].sum(2) > 0, axis=1))
                predictions[label + '_COHORT'] = result
        record = json.loads((sparse / 'source' / video / 'cache_receipt.json').read_text())
        scored_frames = record['records'][0]['frames']
        presence = np.load(sparse / 'source' / video / 'presence_logits.npy')
        columns = np.asarray([{int(t): j for j, t in enumerate(tracks)}[int(t)] for _, t in observations])
        context = SimpleNamespace(bank=bank, queries=queries)
        for method in METHODS:
            score = predictions[method][:, columns];mask = plain(score, False)
            for qi, query in enumerate(queries):
                if 'MOTION' in expression_types(query['query_text']):
                    mask[qi] &= expit(np.interp(observations[:, 0], scored_frames, presence[qi])) >= .5
            folder = root / 'methods' / method;destination = folder / 'videos' / video
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / 'scores.npy', score.astype('float32'));np.save(destination / 'mask.npy', mask)
            immediate_predictions(folder, context, mask)
        receipt = dict(video=video, index=vi, visible_teacher_videos=sorted(visible), teacher_examples=len(y),
                       full_vs_sparse_actual_PCD_max_delta=feature_delta, RGB_max_delta=colour_delta,
                       full_reference_features_used_on_both_sides=True, fits=fits, GT_open_attempts=len(gt),
                       wall_seconds=time.perf_counter() - began)
        atomic_json(root / 'videos' / video / 'receipt.json', receipt);receipts.append(receipt)
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(dict(next_video=vi + 1, x=bx, z=bz, y=by, sum=moment_sum, square=moment_square,
                        image_rows=image_rows, receipts=receipts), temporary);temporary.replace(checkpoint)
        print(json.dumps(dict(event='SPARSE_FIXED_QA_VIDEO_COMPLETE', video=video, completed=vi+1,
                              PCD_max_delta=feature_delta, teacher_examples=len(y))), flush=True)
    seals = []
    for method in METHODS:
        folder = root / 'methods' / method;mask = np.zeros(m['candidate_query_rows'], dtype=bool)
        for video in videos:
            bits = np.load(folder / 'videos' / video / 'mask.npy')
            for qi, q in enumerate([q for q in m['episodes'] if q['video_id'] == video]):
                mask[q['start']:q['stop']] = bits[qi]
        seal = seal_mask(folder, m, mask, target='SportsMOT', model='Dance-best', method=method,
                         GT_open_attempts=len(gt), extra=dict(fixed_actual_full_stream_QA=True,
                         source_prior_and_motion_gate_only_intervention=True,
                         no_future_teacher_feedback=True, quality_only_not_live_cost=True))
        seals.append(dict(method=method, rows=seal['prediction_rows']))
    assert not gt and not forbidden
    atomic_json(root / 'completion.json', dict(status='SEALED_SPARSE_SOURCE_FIXED_QA', methods=seals,
                receipts=receipts, GT_open_attempts=0, forbidden_read_attempts=0,
                new_teacher_calls=0, logical_teacher_examples=len(np.concatenate(by)),
                same_reference_PCD_and_RGB=True, wall_seconds=time.perf_counter()-start,
                quality_control_not_cold_deployment=True, all_development_data=True))
    atomic_json(state_path, dict(status='SEALED_SPARSE_SOURCE_FIXED_QA', pid=os.getpid()))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        atomic_json(OUT / 'supervisor/sparse_replay_state.json', dict(status='FAILED_SPARSE_REPLAY_REQUIRES_ROOT_CAUSE', pid=os.getpid()))
        raise
