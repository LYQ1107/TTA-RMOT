"""Ordinary KD with raw soft targets or hard labels and unit weights.

The existing confidence-weighted, clipped-target comparators remain intact.
This supplement tests a basic alternative without assuming that those
weighting choices improve ordinary distillation. It adds no teacher data,
backbone calls, new research mechanism, or GT-based model selection.
"""
import argparse
import concurrent.futures
import faulthandler
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
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.observation_learning_inputs import ObservationVideo, prefix_data, DIMENSION
from scripts.a100_conditional_six_setting import OUT as ORIGINAL
from scripts.a100_conditional_coarse_baseline import OUT as COARSE, coarse_inputs, supervision as coarse_supervision
from scripts.a100_conditional_shared_control import OUT as GENERIC
from scripts.a100_conditional_logit_precision import OUT as REFERENCE, STATE as REFERENCE_STATE, mean_logit
from scripts.a100_observation_local_cache import CACHE, FILES
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS

OUT = SIX/'conditional_standard_kd_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'conditional_standard_kd_registration_v1.json'
STEM = 'conditional_standard_kd'
PRIMARY = 'FULL_FINAL'
SUCCESS = 'PASS_CONDITIONAL_STANDARD_KD_SIX_SETTINGS_GOAL_ACTIVE'
DENSE = ('STD_FRAME_SOFT', 'STD_FRAME_HARD', 'STD_FLAT_SOFT', 'STD_FLAT_HARD')
COARSE_ARMS = ('STD_COARSE_SOFT', 'STD_COARSE_HARD')
ARMS = (*DENSE, *COARSE_ARMS)
METHODS = tuple(a+'_'+s for a in ARMS for s in ('FINAL', 'MEAN'))
CONTRASTS = {'FULL_vs_'+b:(PRIMARY, b) for b in METHODS}
for b in METHODS:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b] = (PRIMARY+'__AT_'+b+'_COUNT', b+'__AT_'+b+'_COUNT')


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def standard_targets(data):
    """Leave inputs/groups/raw answers unchanged; remove target heuristics."""
    raw = np.asarray(data['raw_target'], dtype='float32')
    assert np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all()
    return dict(data, target=raw.copy(), confidence=np.ones_like(raw), balanced_weight=np.ones_like(raw))


def use_verified_cache():
    receipt = json.loads((CACHE/'completion.json').read_text())
    assert receipt['status'] == 'PASS_BYTE_IDENTICAL_HISTORICAL_OBSERVATION_LOCAL_CACHE'
    assert receipt['files'] == 592
    assert receipt['aggregate_sha256'] == 'd21792b9ac72fd1574a54b2dfc8e76481ddb054b38d79de91cef78d217cc4ae8'
    from flex_uda.active import observation_learning_inputs as inputs
    inputs.HISTORY = CACHE  # Only this process; no live source or wrapper edit.
    return receipt


def check_registration():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    return plan


def run(pair, seed):
    import torch
    from flex_uda.active.observation_learning_fit import fit, forward
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    check_registration()
    guard = generation_guard()
    cache = use_verified_cache()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    started = time.perf_counter()
    source, target = PAIRS[pair]
    md = metadata(target)
    root = OUT/pair/f'seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file():
        return
    stream = root/'stream.pt'
    previous, receipts, next_video = {}, [], 0
    if stream.is_file():
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        previous, receipts, next_video = [saved[k] for k in ('previous', 'receipts', 'next_video')]
    total = np.zeros(DIMENSION)
    square = total.copy()
    count = 0
    coarse_total, coarse_square, coarse_count = total.copy(), total.copy(), 0
    records, coarse_records = [], []
    for vi, video in enumerate(sorted(md['videos'])):
        stamp = time.perf_counter()
        folder = CACHE/'features'/target/source/'videos'/video
        for name in FILES:
            path = folder/name
            proof = json.loads(path.with_name(path.name+'.copy_receipt.json').read_text())
            assert proof['source_read_and_independent_local_read_agree'] and path.stat().st_size == proof['bytes']
        inputs = ObservationVideo(pair, video)
        flat = inputs.raw_bags[:, inputs.present].reshape(-1, DIMENSION).astype(float)
        total += flat.sum(0)
        square += np.square(flat).sum(0)
        count += len(flat)
        mean = total/count
        std = np.maximum(np.sqrt(np.maximum(square/count-mean**2, 0)), 1e-3)
        records.append(inputs.supervision('DESIGN'))
        coarse_values = coarse_inputs(inputs)
        coarse_records.append(coarse_supervision(inputs, coarse_values))
        cf = coarse_values.reshape(-1, DIMENSION).astype(float)
        coarse_total += cf.sum(0)
        coarse_square += np.square(cf).sum(0)
        coarse_count += len(cf)
        cm = coarse_total/coarse_count
        cs = np.maximum(np.sqrt(np.maximum(coarse_square/coarse_count-cm**2, 0)), 1e-3)
        if vi < next_video:
            continue
        data = standard_targets(prefix_data(records, mean, std))
        cd = standard_targets(prefix_data(coarse_records, cm, cs))
        heads, fits, new_previous = {}, {}, {}
        for arm in ARMS:
            coarse = arm in COARSE_ARMS
            recipe = 'FRAME_'+arm.rsplit('_', 1)[1] if coarse else arm.removeprefix('STD_')
            head, early, checkpoint, receipt = fit(cd if coarse else data, recipe, seed=seed,
                mean=cm if coarse else mean, std=cs if coarse else std, previous=previous.get(arm))
            assert receipt['target_fit_gate_pass'], (pair, video, arm, receipt['metrics'])
            receipt = dict(receipt, ordinary_variant='unit_weight_raw_probability_or_hard_label',
                confidence_weights_all_one=True, soft_targets_not_clipped=True)
            heads[arm] = head, early, checkpoint['kind']
            fits[arm] = receipt
            new_previous[arm] = checkpoint
            atomic_json(root/'progress.json', dict(completed=vi, total=len(md['videos']), video=video,
                stage='FITTING', last_arm=arm, completed_arms=len(heads), fit_gate_pass=True))
        local = root/'videos'/video
        local.mkdir(parents=True, exist_ok=True)
        nq, no = len(inputs.queries), len(inputs.obs)
        logits = {a:{t:np.lib.format.open_memmap(local/(a+'_'+t+'_logits.npy'), mode='w+',
            dtype='float32', shape=(nq, no)) for t in ('FINAL', 'EARLY')} for a in DENSE}
        bits = {n:np.lib.format.open_memmap(local/(n+'_mask.npy'), mode='w+', dtype='bool', shape=(nq, no)) for n in METHODS}
        coarse_z = {}
        with torch.no_grad():
            cc = torch.as_tensor(((coarse_values.reshape(-1, DIMENSION)-cm)/cs).astype('float32'), device='cuda:0')
            for arm in COARSE_ARMS:
                head, early, kind = heads[arm]
                for tail, module in (('FINAL', head), ('EARLY', early)):
                    coarse_z[arm+'_'+tail] = forward(module, kind, None, None, cc).cpu().numpy().reshape(coarse_values.shape[:2])
        np.savez_compressed(local/'coarse_logits.npz', tracks=inputs.tracks, **coarse_z)
        normalized = ((inputs.raw_bags-mean)/std).astype('float32')
        normalized[:, ~inputs.present] = 0
        with torch.no_grad():
            for a in range(0, nq*no, 2048):
                flat = np.arange(a, min(a+2048, nq*no))
                qi, rows = flat//no, flat % no
                ti = inputs.columns[rows]
                current = ((inputs.dense_rows(qi, rows)-mean)/std).astype('float32')
                bb = torch.as_tensor(normalized[qi, ti], device='cuda:0')
                mm = torch.as_tensor(inputs.present[ti], device='cuda:0')
                cc = torch.as_tensor(current, device='cuda:0')
                values = {}
                for arm in DENSE:
                    head, early, kind = heads[arm]
                    f = forward(head, kind, bb, mm, cc).cpu().numpy()
                    e = forward(early, kind, bb, mm, cc).cpu().numpy()
                    logits[arm]['FINAL'].reshape(-1)[flat] = f
                    logits[arm]['EARLY'].reshape(-1)[flat] = e
                    values[arm] = f, e
                for arm in COARSE_ARMS:
                    values[arm] = coarse_z[arm+'_FINAL'][qi, ti], coarse_z[arm+'_EARLY'][qi, ti]
                for arm, (f, e) in values.items():
                    mean_bits = f.astype(float)+e.astype(float) > 0
                    assert np.array_equal(mean_logit(f, e)>0, mean_bits), (pair, video, arm)
                    bits[arm+'_FINAL'].reshape(-1)[flat] = f > 0
                    bits[arm+'_MEAN'].reshape(-1)[flat] = mean_bits
        for arrays in logits.values():
            for value in arrays.values():
                value.flush()
        for name, mask in bits.items():
            mask.flush()
            destination = root/name/'videos'/video
            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination/'mask.npy', mask)
            immediate_predictions(root/name, type('Context', (), dict(bank=inputs.bank, queries=inputs.queries))(), mask)
        torch.cuda.synchronize()
        receipt = dict(video=video, index=vi, fits=fits, teacher_evidence=records[-1]['report'],
            GT_open_attempts=len(guard), state_normalization_rows=count, coarse_normalization_rows=coarse_count,
            current_and_past_only=True, all_confidence_weights_one=True, soft_target_clipping=False,
            wall_seconds=time.perf_counter()-stamp, quality_replay_not_cold_cost=True)
        atomic_json(local/'receipt.json', receipt)
        receipts.append(receipt)
        previous, next_video = new_previous, vi+1
        temp = stream.with_suffix('.tmp')
        torch.save(dict(previous=previous, receipts=receipts, next_video=next_video), temp)
        temp.replace(stream)
        del heads, logits, bits, data, cd
        torch.cuda.empty_cache()
    methods = []
    for name in METHODS:
        mask = np.zeros(md['candidate_query_rows'], bool)
        for video in sorted(md['videos']):
            values = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in md['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = values[qi]
        row = seal_mask(root/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(seed=seed, no_future_answers_or_normalization=True, current_video_transductive=True,
                unquantized_logit_threshold=True, raw_soft_or_hard_targets_unit_weights=True, no_source_logit_residual=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_STANDARD_RAW_UNIT_WEIGHT_KD_NO_GT', methods=methods,
        receipts=receipts, target_fit_failures=[], GT_open_attempts=0, wall_seconds=time.perf_counter()-started,
        cache_aggregate_sha256=cache['aggregate_sha256'], no_new_teacher_or_backbone_calls=True, goal_status='ACTIVE'))


def counts(record):
    from collections import Counter
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard()
    pair = record['pair']
    source, target = PAIRS[pair]
    md = metadata(target)
    root = Path(record['folder'])
    reference = REFERENCE/pair/f"seed{record['seed']}"
    destination = root/'countmatched'
    if (destination/'completion.json').is_file():
        return str(destination)
    pairs = [(PRIMARY, PRIMARY)]+[p for b in METHODS for p in ((b, PRIMARY), (PRIMARY, b), (b, b))]
    masks = {p:np.zeros(md['candidate_query_rows'], bool) for p in pairs}
    ties, nontrivial = Counter(), Counter()
    for video, bank in sorted(md['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        local = root/'videos'/video
        zf = {a:np.load(local/(a+'_FINAL_logits.npy'), mmap_mode='r') for a in DENSE}
        ze = {a:np.load(local/(a+'_EARLY_logits.npy'), mmap_mode='r') for a in DENSE}
        with np.load(local/'coarse_logits.npz') as data:
            lookup = {int(t):i for i, t in enumerate(data['tracks'])}
            columns = np.asarray([lookup[int(t)] for _, t in obs])
            for arm in COARSE_ARMS:
                zf[arm] = data[arm+'_FINAL'][:, columns]
                ze[arm] = data[arm+'_EARLY'][:, columns]
        primary_z = np.load(reference/'videos'/video/(PRIMARY+'_logits.npy'), mmap_mode='r')
        bits = {n:np.load((reference if n == PRIMARY else root)/n/'videos'/video/'mask.npy', mmap_mode='r') for n in (*METHODS, PRIMARY)}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            for qi, q in enumerate(queries):
                values = {a+'_FINAL':np.asarray(zf[a][qi, ids], dtype=float) for a in ARMS}
                values.update({a+'_MEAN':mean_logit(zf[a][qi, ids], ze[a][qi, ids]) for a in ARMS})
                values[PRIMARY] = np.asarray(primary_z[qi, ids], dtype=float)
                order = {n:ids[np.lexsort((obs[ids, 1], -v))] for n, v in values.items()}
                sorted_values = {n:np.sort(v)[::-1] for n, v in values.items()}
                number = {n:int(bits[n][qi, ids].sum()) for n in (*METHODS, PRIMARY)}
                for rank, donor in pairs:
                    n = number[donor]
                    masks[rank, donor][q['start']+order[rank][:n]] = True
                    if 0 < n < len(ids):
                        name = rank+'__AT_'+donor+'_COUNT'
                        nontrivial[name] += 1
                        ties[name] += int(sorted_values[rank][n-1] == sorted_values[rank][n])
    methods = []
    for (rank, donor), mask in masks.items():
        name = rank+'__AT_'+donor+'_COUNT'
        row = seal_mask(destination/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(rank_method=rank, count_method=donor, counts_from_GT=False, counts_equal_per_query_frame=True,
                ordering_before_sigmoid_quantization=True, primary_from_fixed_precision_cohort=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    changed = {}
    for name in (*METHODS, PRIMARY):
        old = json.loads(((reference if name == PRIMARY else root)/name/'predictions_seal.json').read_text())
        new = json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        lookup = {(r['video'], r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name] = sum(lookup[r['video'], r['expression']] != r['prediction_sha256'] for r in new['episodes'])
    assert not guard and not any(changed.values()), changed
    atomic_json(destination/'completion.json', dict(status='SEALED_STANDARD_KD_DUAL_COUNTS_NO_GT', methods=methods,
        CountSelf_changed_query_files=changed, actual_exact_rank_cutoff_ties=dict(ties), nontrivial_cutoffs=dict(nontrivial), GT_open_attempts=0))
    return str(destination)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    data, statistics, summary, error = {}, {}, {}, 0.
    for record in conditions():
        root = Path(record['folder'])
        slug = record['pair']+f"_seed{record['seed']}"
        raw = json.loads((root/'completion.json').read_text())
        cnt = json.loads((root/'countmatched/completion.json').read_text())
        paths = {r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in cnt['methods']})
        paths[PRIMARY] = REFERENCE/record['pair']/f"seed{record['seed']}"/PRIMARY
        data[slug] = {n:read_stats(p) for n, p in paths.items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']
            error = max(error, abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'], numerator=num.tolist(), denominator=den.tolist())
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in (*METHODS, PRIMARY)},
            target_fit_failures=raw['target_fit_failures'], exact_rank_cutoff_ties=cnt['actual_exact_rank_cutoff_ties'],
            CountSelf_changed_query_files=cnt['CountSelf_changed_query_files'])
    repeated = {p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in (*METHODS, PRIMARY)},
        comparisons={label:comparison([data[p+f'_seed{s}'][a] for s in SEEDS], [data[p+f'_seed{s}'][b] for s in SEEDS])
            for label, (a, b) in CONTRASTS.items()}) for p in PAIRS}
    assert error < 1e-8 and len(statistics) == 900, (error, len(statistics))
    result = dict(status='STANDARD_RAW_UNIT_WEIGHT_KD_SIX_SETTINGS_ANALYZED', primary=PRIMARY,
        primary_cohort='Unchanged registered FULL_FINAL from logit-precision study, chosen before quality',
        conditions=summary, repeated=repeated, HOTA_parent_sum_max_error=error,
        all_own_and_required_parent_outputs_sealed_before_this_GT=True,
        standard_variant_changes_both_confidence_weights_and_soft_target_clipping=True,
        not_a_pure_confidence_ablation=True, all_data_development=True, independent_blind_validation=False,
        quality_replay_not_cold_cost=True, no_new_teacher_information_or_backbone_calls=True,
        not_a_new_innovation=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics, separators=(',', ':'))+'\n')


def main():
    check_registration()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_LOGIT_PRECISION_GPU_STAGE_RELEASE', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        earlier = json.loads(REFERENCE_STATE.read_text()) if REFERENCE_STATE.is_file() else {}
        if earlier.get('status', '').startswith('FAILED_'):
            raise RuntimeError('Resolve registered logit-precision study first')
        if earlier.get('completed', 0) == 18 and not earlier.get('active'):
            break
        time.sleep(15)
    pending, done, active, failure = conditions(), [], {}, None
    def state(status):
        atomic_json(STATE, dict(status=status, pid=os.getpid(), completed=len(done), finished=done, pending=pending,
            active={str(g):j for g, (_, _, j) in active.items()}, failure=failure, goal_status='ACTIVE'))
    while pending or active:
        for gpu, (proc, handle, job) in list(active.items()):
            if proc.poll() is None:
                continue
            handle.close()
            del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():
                failure = dict(job=job, exit_code=proc.returncode, last_log=Path(job['log']).read_text()[-8000:])
            else:
                done.append(job)
        free = [g for g in (0, 1, 2, 3) if g not in active]
        if pending and free and not failure:
            output = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, free)),
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in free:
                if not pending or memory[gpu] >= 512:
                    continue
                job = pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():
                    done.append(dict(job, reused_sealed=True))
                    continue
                log = STATE.parent/(job['pair']+'_seed'+str(job['seed'])+'.log')
                handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--pair', job['pair'], '--seed', str(job['seed'])],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4'))
                active[gpu] = proc, handle, dict(job, pid=proc.pid, log=str(log), physical_GPU=gpu)
        state('RUNNING_STANDARD_RAW_UNIT_WEIGHT_KD_NO_GT' if not failure else 'PRESERVING_HEALTHY_STANDARD_KD_AFTER_FAILURE')
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(5)
    state('ALL18_STANDARD_KD_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(counts, conditions()))
    for parent, expected in ((ORIGINAL, 1692), (COARSE, 378), (GENERIC, 162), (REFERENCE, 1116)):
        path = parent/'all_predictions_sealed_before_GT.json'
        while not path.is_file():
            time.sleep(15)
        assert json.loads(path.read_text())['methods'] == expected
    tasks = []
    for record, count_root in zip(conditions(), roots):
        for folder in (Path(record['folder']), Path(count_root)):
            doc = json.loads((folder/'completion.json').read_text())
            assert doc['GT_open_attempts'] == 0
            tasks.extend((str(folder/r['method']), record['pair']) for r in doc['methods'])
    assert len(tasks) == 882
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=882, required_parent_studies=[1692, 378, 162, 1116], GT_open_attempts=0))
    state('ALL882_STANDARD_KD_AND_REQUIRED_PARENT_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    while not (REFERENCE/'official.json').is_file():
        time.sleep(15)
    atomic_json(OUT/'official.json', dict(methods=882, results=results))
    analyze()
    state(SUCCESS)


if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', choices=tuple(PAIRS))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.pair:
        import fcntl
        root = OUT/args.pair/f'seed{args.seed}'
        root.mkdir(parents=True, exist_ok=True)
        with (root/'condition.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run(args.pair, args.seed)
    else:
        try:
            main()
        except Exception as error:
            old = json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_STANDARD_KD_REQUIRES_ROOT_CAUSE', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, old)
            raise
