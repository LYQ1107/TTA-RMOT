"""Seed0 matched joint-observation study, with all six settings before GT.

GPU 0/1 may continue teacher collection. The user's 2026-11-23 instruction
also authorizes small-head coexistence on GPU 2/3; existing jobs are retained.
Quality replay and shared-card training times are not cold deployment costs.
"""
import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_six_direction import OUT as SIX, PUBLIC
from scripts.a100_joint_observation_collect import OUT as COLLECTION, STATE as COLLECTION_STATE

OUT = SIX/'joint_observation_quality_20261123'
PLAN = PUBLIC/'joint_observation_quality_registration_v1.json'
STATE = OUT/'supervisor/state.json'
PRIMARY = 'FULL_CONDITIONAL_REFERENCE_DESIGN_FINAL'
SUCCESS = 'PASS_JOINT_OBSERVATION_SIX_SETTING_EXECUTION_GOAL_ACTIVE'


def recipes():
    rows = {}
    for policy in ('DESIGN', 'UNIFORM'):
        for conditional in (True, False):
            for reference in (True, False):
                name = 'FULL_'+('CONDITIONAL' if conditional else 'LIKELIHOOD')+'_'+('REFERENCE' if reference else 'LOCAL')+'_'+policy
                rows[name] = dict(family='structured', conditional=conditional,
                                  reference=reference, policy=policy)
        for label, risk in (('JOINT', 'joint'), ('MARGINAL', 'marginal')):
            rows[label+'_KD_'+policy] = dict(family='generic', risk=risk, policy=policy)
        for confidence in (False, True):
            for hard in (False, True):
                name = 'COARSE_'+('CONF' if confidence else 'UNIT')+'_'+('HARD' if hard else 'SOFT')+'_'+policy
                rows[name] = dict(family='coarse', confidence=confidence, hard=hard, policy=policy)
    rows['BINARY_CONDITIONAL_REFERENCE_DESIGN'] = dict(
        family='structured', conditional=True, reference=True, policy='DESIGN', binary=True)
    return rows


RECIPES = recipes()
METHODS = tuple(name+'_'+snapshot for name, recipe in RECIPES.items()
                for snapshot in (('FINAL', 'MEAN') if recipe['family'] != 'structured' else ('FINAL',)))
COMPARATORS = tuple(m for m in METHODS if m != PRIMARY)


def check_registration():
    plan = json.loads(PLAN.read_text())
    assert plan['neural_seeds'] == [0] and plan['primary'] == PRIMARY
    assert plan['recipes'] == RECIPES and tuple(plan['methods']) == METHODS
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    return plan


def infer_video(video, inputs, models, mean, std, local, device):
    import torch
    from torch.nn import functional as F
    from flex_uda.active.joint_observation_fit import probabilities

    nq, no = len(video.queries), len(video.obs)
    scores = {name: np.lib.format.open_memmap(local/(name+'_scores.npy'), mode='w+',
                                             dtype='float64', shape=(nq, no)) for name in METHODS}
    anchors = ((video.raw_bags-mean)/std).astype('float32')
    anchors[:, ~video.present] = 0
    pool = ((inputs['pool_values']-mean)/std).astype('float32')
    pool[:, ~inputs['pool_present']] = 0
    device = torch.device(device)
    # Ordinary track-mean models summarize all distinct original context and
    # teacher-pool observations, not just the old three context anchors.
    coarse = ((inputs['coarse']-mean)/std).astype('float32')
    with torch.no_grad():
        cc = torch.as_tensor(coarse.reshape(-1, video.dimension), device=device)
        for name, (final, early) in models.items():
            if RECIPES[name]['family'] != 'coarse':
                continue
            z = final(cc).squeeze(-1).double()
            ze = early(cc).squeeze(-1).double()
            p = torch.logaddexp(F.logsigmoid(z), F.logsigmoid(ze))-np.log(2.)
            n = torch.logaddexp(F.logsigmoid(-z), F.logsigmoid(-ze))-np.log(2.)
            scores[name+'_FINAL'][:] = z.cpu().numpy().reshape(coarse.shape[:2])[:, video.columns]
            scores[name+'_MEAN'][:] = (p-n).cpu().numpy().reshape(coarse.shape[:2])[:, video.columns]
        for begin in range(0, nq*no, 1024):
            flat = np.arange(begin, min(begin+1024, nq*no))
            qi, rows = flat//no, flat % no
            ti = video.columns[rows]
            current = ((video.dense_rows(qi, rows)-mean)/std).astype('float32')
            for name, (final, early) in models.items():
                if RECIPES[name]['family'] == 'coarse':
                    continue
                total = None
                early_total = None
                for slot in range(3):
                    valid = inputs['pool_present'][ti, slot]
                    if not valid.any():
                        continue
                    data = dict(anchors=torch.as_tensor(anchors[qi[valid], ti[valid]], device=device),
                                present=torch.as_tensor(video.present[ti[valid]], device=device),
                                first=torch.as_tensor(current[valid], device=device),
                                second=torch.as_tensor(pool[qi[valid], ti[valid], slot], device=device))
                    logq = probabilities(final, data)
                    marginal = torch.stack((torch.logsumexp(logq[:, (0, 2)], -1),
                                            torch.logsumexp(logq[:, (1, 3)], -1)), -1)
                    if total is None:
                        total = np.full((len(flat), 2), -np.inf)
                    total[valid] = np.logaddexp(total[valid], marginal.cpu().numpy())
                    if RECIPES[name]['family'] == 'generic':
                        loge = probabilities(early, data)
                        marg_e = torch.stack((torch.logsumexp(loge[:, (0, 2)], -1),
                                              torch.logsumexp(loge[:, (1, 3)], -1)), -1)
                        if early_total is None:
                            early_total = np.full((len(flat), 2), -np.inf)
                        early_total[valid] = np.logaddexp(early_total[valid], marg_e.cpu().numpy())
                # Every track has >=1 actual partner. Equal reference-pool
                # averaging cancels from log(p_yes)-log(p_no); no clip/saturation.
                assert total is not None and np.isfinite(total).all()
                scores[name+'_FINAL'].reshape(-1)[flat] = total[:, 0]-total[:, 1]
                if early_total is not None:
                    mixed = np.logaddexp(total, early_total)-np.log(2.)
                    scores[name+'_MEAN'].reshape(-1)[flat] = mixed[:, 0]-mixed[:, 1]
    for values in scores.values():
        values.flush()
        assert np.isfinite(values).all()
    return scores


def run(pair):
    plan = check_registration()
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.joint_observation_study import video_records, prefix_arrays, fit_prefix, make_model
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions

    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['GPU_memory_fraction'], device=0)
    use_verified_local_features()
    model_name, target = PAIRS[pair]
    md = metadata(target)
    root = OUT/pair/'seed0'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file():
        return
    started = time.perf_counter()
    previous, receipts, next_video = {}, [], 0
    stream = root/'stream.pt'
    if stream.is_file():
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        previous, receipts, next_video = (saved[k] for k in ('previous', 'receipts', 'next_video'))
    dimension = 1101
    total, square, number = np.zeros(dimension), np.zeros(dimension), 0
    records = {p: [] for p in ('DESIGN', 'UNIFORM')}
    for vi, video in enumerate(sorted(md['videos'])):
        stamp = time.perf_counter()
        v = JointObservationVideo(pair, video)
        common = video_records(v, COLLECTION)
        total += common['norm_sum']
        square += common['norm_square']
        number += common['norm_rows']
        mean = total/number
        std = np.sqrt(np.maximum(square/number-mean**2, 0)).clip(min=1e-3)
        for policy in records:
            records[policy].append(common['records'][policy])
        if vi < next_video:
            continue
        atomic_json(root/'progress.json', dict(stage='FITTING_OWN_PREFIX_NO_GT', video=video,
                    completed_videos=vi, total_videos=len(md['videos']), completed_arms=0))
        prepared = {p: prefix_arrays(records[p], mean, std) for p in records}
        models, fits, new_previous = {}, {}, {}
        local = root/'videos'/video
        local.mkdir(parents=True, exist_ok=True)
        for name, recipe in RECIPES.items():
            path = local/'fits'/(name+'.pt')
            if path.is_file():
                saved = torch.load(path, map_location='cpu', weights_only=False)
                checkpoint, fit_receipt = saved['checkpoint'], saved['checkpoint']['receipt']
                assert checkpoint['recipe'] == recipe
                assert np.array_equal(checkpoint['mean'], mean) and np.array_equal(checkpoint['std'], std)
                final = make_model(recipe, dimension, checkpoint['width']).cuda().eval()
                final.load_state_dict(checkpoint['model'])
                early = make_model(recipe, dimension, checkpoint['width']).cuda().eval()
                early.load_state_dict(saved['early'])
            else:
                final, early, checkpoint, fit_receipt = fit_prefix(
                    prepared[recipe['policy']], recipe, mean, std,
                    previous=previous.get(name), device='cuda:0')
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                torch.save(dict(checkpoint=checkpoint,
                    early={k: value.detach().cpu().clone() for k, value in early.state_dict().items()}), temporary)
                temporary.replace(path)
            models[name] = final, early
            fits[name], new_previous[name] = fit_receipt, checkpoint
            atomic_json(root/'progress.json', dict(stage='FITTING_OWN_PREFIX_NO_GT', video=video,
                        completed_videos=vi, total_videos=len(md['videos']), last_arm=name,
                        completed_arms=len(models), teacher_KL=fit_receipt['metrics']['excess_KL']))
        # Keep each actual deployed video head and coordinates, not merely a
        # final all-video model that would leak future answers into the past.
        atomic_json(local/'heads.json', dict(video=video, own_prefix=True,
                    model_checkpoints={name: str(local/'fits'/(name+'.pt')) for name in RECIPES}))
        prediction_start = time.perf_counter()
        scores = infer_video(v, common, models, mean, std, local, 'cuda:0')
        for name, values in scores.items():
            bits = np.asarray(values) > 0
            folder = root/name/'videos'/video
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder/'mask.npy', bits)
            immediate_predictions(root/name, type('Context', (), dict(bank=v.bank, queries=v.queries))(), bits)
        torch.cuda.synchronize()
        receipt = dict(video=video, index=vi, fits=fits,
                       evidence={p: common['records'][p]['report'] for p in records},
                       normalization_rows=number, past_and_current_video_only=True,
                       prediction_output_wall_seconds=time.perf_counter()-prediction_start,
                       wall_seconds=time.perf_counter()-stamp, GT_open_attempts=len(guard),
                       quality_replay_not_cold_deployment=True,
                       GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(local/'receipt.json', receipt)
        receipts.append(receipt)
        temporary = stream.with_suffix('.tmp')
        torch.save(dict(previous=new_previous, receipts=receipts, next_video=vi+1), temporary)
        temporary.replace(stream)
        previous = new_previous
        del prepared, models, scores, common, v
        torch.cuda.empty_cache()
    methods = []
    for name in METHODS:
        mask = np.zeros(md['candidate_query_rows'], dtype=bool)
        for video in sorted(md['videos']):
            bits = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in md['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = bits[qi]
        row = seal_mask(root/name, md, mask, target=target, model=model_name, method=name,
                        GT_open_attempts=len(guard), extra=dict(seed=0, own_policy_prefix_only=True,
                        current_video_transductive=True, quality_replay_not_cold_deployment=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_JOINT_OBSERVATION_RAW_NO_GT',
                pair=pair, seed=0, methods=methods, receipts=receipts, execution=execution,
                GT_open_attempts=0, wall_seconds=time.perf_counter()-started, goal_status='ACTIVE'))


def counts(pair):
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard()
    source_model, target = PAIRS[pair]
    md = metadata(target)
    root = OUT/pair/'seed0'
    destination = root/'countmatched'
    if (destination/'completion.json').is_file():
        return
    pairs = [(rank, donor) for other in COMPARATORS for rank, donor in ((other, PRIMARY), (PRIMARY, other))]
    masks = {p: np.zeros(md['candidate_query_rows'], dtype=bool) for p in pairs}
    for video, bank in sorted(md['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        scores = {n: np.load(root/'videos'/video/(n+'_scores.npy'), mmap_mode='r') for n in METHODS}
        bits = {n: np.load(root/n/'videos'/video/'mask.npy') for n in METHODS}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            for qi, query in enumerate(queries):
                order = {n: ids[np.lexsort((obs[ids, 1], -np.asarray(s[qi, ids])))] for n, s in scores.items()}
                cardinality = {n: int(bits[n][qi, ids].sum()) for n in METHODS}
                for name in METHODS:
                    expected = order[name][:cardinality[name]]
                    assert np.array_equal(np.sort(expected), ids[bits[name][qi, ids]]), (pair, video, name, 'CountSelf')
                for (rank, donor), mask in masks.items():
                    mask[query['start']+order[rank][:cardinality[donor]]] = True
    methods = []
    for (rank, donor), mask in masks.items():
        name = rank+'__AT_'+donor+'_COUNT'
        row = seal_mask(destination/name, md, mask, target=target, model=source_model, method=name,
                        GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=donor,
                        counts_equal_per_query_frame=True, counts_from_GT=False,
                        diagnostic_not_deployment=True, float64_stable_logits=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    atomic_json(destination/'completion.json', dict(status='SEALED_JOINT_OBSERVATION_COUNTS_NO_GT',
                methods=methods, GT_open_attempts=0, CountSelf_identical_masks_for_all_methods=True,
                CountSelf_aliases={n+'__AT_'+n+'_COUNT': str(root/n) for n in METHODS}))


def ready(pair):
    _, target = PAIRS[pair]
    return all((COLLECTION/'policy_answers'/pair/video/(policy+'_answers.json')).is_file()
               for video in metadata(target)['videos'] for policy in ('DESIGN', 'UNIFORM'))


def main():
    plan = check_registration()
    started, last = time.perf_counter(), 0.
    pending, active, finished, failure = list(PAIRS), {}, [], None
    def state(status):
        view = {}
        for gpu, (_, _, job) in active.items():
            path = Path(job['folder'])/'progress.json'
            view[str(gpu)] = dict(job, progress=json.loads(path.read_text()) if path.is_file() else {})
        row = dict(status=status, pid=os.getpid(), active=view, pending=pending,
                   finished=finished, failure=failure, wall_seconds=time.perf_counter()-started,
                   neural_seeds=[0], goal_status='ACTIVE')
        atomic_json(STATE, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    while pending or active:
        changed = False
        for gpu, (proc, handle, job) in list(active.items()):
            if proc.poll() is None:
                continue
            handle.close()
            del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():
                failure = dict(job=job, exit_code=proc.returncode,
                               log=Path(job['log']).read_text()[-6000:])
            else:
                finished.append(job)
            changed = True
        for pair in list(pending):
            if (OUT/pair/'seed0/completion.json').is_file():
                pending.remove(pair)
                finished.append(dict(pair=pair, reused_sealed=True))
                changed = True
        if COLLECTION_STATE.is_file():
            collection = json.loads(COLLECTION_STATE.read_text())
            if collection.get('status', '').startswith('FAILED_'):
                failure = dict(teacher_collection=collection)
        eligible = [p for p in pending if ready(p)] if not failure else []
        candidates = [g for g in (2, 3, 0, 1) if g not in active]
        if eligible and candidates:
            output = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, candidates)),
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]): int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                # Explicit new user authorization permits small-head shared
                # use on 2/3. Nothing is killed, reset, or made exclusive.
                limit = plan['shared_GPU_preflight_max_MiB'] if gpu in (2, 3) else 512
                if not eligible or memory[gpu] >= limit:
                    continue
                pair = eligible.pop(0)
                pending.remove(pair)
                log = OUT/'supervisor'/(pair+'_seed0.log')
                handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--pair', pair], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                             OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
                active[gpu] = proc, handle, dict(pair=pair, pid=proc.pid, physical_GPU=gpu,
                    folder=str(OUT/pair/'seed0'), log=str(log), preflight_memory_MiB=memory[gpu],
                    shared_card_authorization=gpu in (2, 3))
                changed = True
        if changed or time.perf_counter()-last >= 45:
            state('RUNNING_MATCHED_JOINT_OBSERVATION_SIX_SETTINGS_NO_GT' if not failure else
                  'PRESERVING_HEALTHY_JOINT_WORKERS_AFTER_FAILURE')
            last = time.perf_counter()
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(5)
    state('ALL_SIX_RAW_SEALED_GENERATING_DUAL_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(counts, PAIRS))
    tasks = []
    for pair in PAIRS:
        root = OUT/pair/'seed0'
        for folder in (root, root/'countmatched'):
            doc = json.loads((folder/'completion.json').read_text())
            assert doc['GT_open_attempts'] == 0
            tasks.extend((str(folder/r['method']), pair) for r in doc['methods'])
    expected = 6*(len(METHODS)+2*len(COMPARATORS))
    assert len(tasks) == len(set(tasks)) == expected
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=expected,
                raw_methods=6*len(METHODS), CountSelf_aliases=6*len(METHODS),
                GT_open_attempts=0, primary=PRIMARY, source_sha256=plan['source_sha256']))
    state('ALL_JOINT_RAW_AND_DUAL_COUNTS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    atomic_json(OUT/'official.json', dict(methods=expected, results=results))
    state(SUCCESS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=tuple(PAIRS))
    args = parser.parse_args()
    if args.pair:
        folder = OUT/args.pair/'seed0'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'condition.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run(args.pair)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                main()
            except Exception as error:
                old = json.loads(STATE.read_text()) if STATE.is_file() else {}
                atomic_json(STATE, dict(old, status='FAILED_JOINT_QUALITY_REQUIRES_ROOT_CAUSE',
                                       error=repr(error), goal_status='ACTIVE'))
                raise
