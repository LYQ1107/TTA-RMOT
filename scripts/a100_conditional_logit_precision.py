"""Matched ordinary replays with ranking before sigmoid quantization.

Preserve the completed/running studies. Refit the fixed FULL and ordinary
recipes without new data, retain logits at every actual prefix, and reuse
the generic product control's already registered saved logits. This is a
numerical fairness supplement, not a new semantic-learning mechanism.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.observation_learning_inputs import ObservationVideo, prefix_data, DIMENSION
from scripts.a100_conditional_six_setting import OUT as PARENT
from scripts.a100_conditional_coarse_baseline import OUT as COARSE, coarse_inputs, supervision as coarse_supervision
from scripts.a100_conditional_shared_control import OUT as GENERIC, STATE as GENERIC_STATE
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS

OUT = SIX/'conditional_logit_precision_20261123'; STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'conditional_logit_precision_registration_v1.json'
STEM = 'conditional_logit_precision'; PRIMARY = 'FULL_FINAL'
SUCCESS = 'PASS_CONDITIONAL_LOGIT_PRECISION_SIX_SETTINGS_GOAL_ACTIVE'
DENSE = ('FRAME_SOFT', 'FRAME_HARD', 'FLAT_SOFT', 'FLAT_HARD', 'FULL')
COARSE_ARMS = ('COARSE_SOFT', 'COARSE_HARD')
ARMS = (*DENSE, *COARSE_ARMS, 'LOCAL_TWO')
METHODS = tuple(a+'_'+s for a in ARMS for s in ('FINAL', 'MEAN'))
BASELINES = tuple(n for n in METHODS if n != PRIMARY)
CONTRASTS = {'FULL_vs_'+b:(PRIMARY, b) for b in BASELINES}
for b in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b] = (PRIMARY+'__AT_'+b+'_COUNT', b+'__AT_'+b+'_COUNT')
CONTRASTS['FULL_REPLICA_vs_ORIGINAL_RAW'] = (PRIMARY, 'ORIGINAL_FULL_FINAL')


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def mean_logit(final, early):
    """Float64 log odds of (sigmoid(final)+sigmoid(early))/2.

Compute both probability and complement in log space. Converting a
rounded mean probability back with logit would already have lost ranks.
"""
    f, e = np.asarray(final, dtype=float), np.asarray(early, dtype=float)
    positive = np.logaddexp(-np.logaddexp(0., -f), -np.logaddexp(0., -e))
    negative = np.logaddexp(-np.logaddexp(0., f), -np.logaddexp(0., e))
    return positive-negative


def mathematical_checks():
    z = np.asarray([-100., -30., -20., -3., 0., 3., 20., 30., 100.])
    error = float(np.max(np.abs(mean_logit(z, z)-z)))
    pair = mean_logit(np.asarray([20., 30.]), np.asarray([20., 30.]))
    rounded = (1/(1+np.exp(-np.asarray([20., 30.])))).astype('float32')
    assert np.array_equal(rounded, [1., 1.]) and pair[1] > pair[0]
    left = np.asarray([-12., -1., 1., 12.]); right = np.asarray([2., 3., -3., -2.])
    probability = .5*(1/(1+np.exp(-left))+1/(1+np.exp(-right)))
    exact = np.log(probability)-np.log1p(-probability)
    moderate_error = float(np.max(np.abs(mean_logit(left, right)-exact)))
    assert error < 1e-12 and moderate_error < 1e-12
    assert np.array_equal(mean_logit(left, right)>0, left+right>0)
    return dict(mean_equal_logits_identity_max_error=error,
        moderate_probability_formula_max_error=moderate_error,
        distinct_large_logits_preserved=pair.tolist(), rounded_float32_probabilities=rounded.tolist(),
        mean_threshold_uses_sign_of_final_plus_early_logit=True,
        limitations='Finite float64 precision still exists; actual remaining cutoff ties are recorded.')


def run(pair, seed):
    import torch
    from flex_uda.active.observation_learning_fit import fit, forward
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    guard = generation_guard(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.cuda.set_device(0)
    started = time.perf_counter(); source, target = PAIRS[pair]; md = metadata(target)
    root = OUT/pair/f'seed{seed}'; root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file(): return
    generic_root = GENERIC/pair/f'seed{seed}'
    generic_done = json.loads((generic_root/'completion.json').read_text()); assert generic_done['GT_open_attempts'] == 0
    stream = root/'stream.pt'; previous = {}; receipts = []; next_video = 0
    if stream.is_file():
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        previous, receipts, next_video = [saved[k] for k in ('previous', 'receipts', 'next_video')]
    total = np.zeros(DIMENSION); square = total.copy(); count = 0
    coarse_total = total.copy(); coarse_square = total.copy(); coarse_count = 0
    records = []; coarse_records = []
    for vi, video in enumerate(sorted(md['videos'])):
        stamp = time.perf_counter(); inputs = ObservationVideo(pair, video)
        flat = inputs.raw_bags[:, inputs.present].reshape(-1, DIMENSION).astype(float)
        total += flat.sum(0); square += np.square(flat).sum(0); count += len(flat)
        mean = total/count; std = np.maximum(np.sqrt(np.maximum(square/count-mean**2, 0)), 1e-3)
        records.append(inputs.supervision('DESIGN'))
        coarse_values = coarse_inputs(inputs); coarse_records.append(coarse_supervision(inputs, coarse_values))
        cf = coarse_values.reshape(-1, DIMENSION).astype(float)
        coarse_total += cf.sum(0); coarse_square += np.square(cf).sum(0); coarse_count += len(cf)
        cm = coarse_total/coarse_count; cs = np.maximum(np.sqrt(np.maximum(coarse_square/coarse_count-cm**2, 0)), 1e-3)
        if vi < next_video: continue
        data = prefix_data(records, mean, std); cd = prefix_data(coarse_records, cm, cs)
        heads = {}; fits = {}; new_previous = {}
        for arm in (*DENSE, *COARSE_ARMS):
            coarse = arm in COARSE_ARMS; recipe = 'FRAME_'+arm.split('_')[1] if coarse else arm
            head, early, checkpoint, receipt = fit(cd if coarse else data, recipe, seed=seed,
                mean=cm if coarse else mean, std=cs if coarse else std, previous=previous.get(arm))
            heads[arm] = head, early, checkpoint['kind']; fits[arm] = receipt; new_previous[arm] = checkpoint
            atomic_json(root/'progress.json', dict(completed=vi, total=len(md['videos']), video=video,
                stage='FITTING', last_arm=arm, completed_arms=len(heads), fit_gate_pass=receipt['target_fit_gate_pass']))
        local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
        nq, no = len(inputs.queries), len(inputs.obs)
        dense_z = {a:{t:np.lib.format.open_memmap(local/(a+'_'+t+'_logits.npy'), mode='w+', dtype='float32', shape=(nq, no))
            for t in ('FINAL', 'EARLY')} for a in DENSE}
        probabilities = {a+'_'+t:np.lib.format.open_memmap(local/(a+'_'+t+'_scores.npy'), mode='w+', dtype='float32', shape=(nq, no))
            for a in DENSE for t in ('FINAL', 'MEAN')}
        bits = {n:np.lib.format.open_memmap(local/(n+'_mask.npy'), mode='w+', dtype='bool', shape=(nq, no)) for n in METHODS}
        coarse_z = {}; coarse_p = {}; normalized_coarse = ((coarse_values.reshape(-1, DIMENSION)-cm)/cs).astype('float32')
        with torch.no_grad():
            cc = torch.as_tensor(normalized_coarse, device='cuda:0')
            for arm in COARSE_ARMS:
                head, early, kind = heads[arm]
                f = forward(head, kind, None, None, cc); e = forward(early, kind, None, None, cc)
                coarse_z[arm+'_FINAL'] = f.cpu().numpy().reshape(coarse_values.shape[:2])
                coarse_z[arm+'_EARLY'] = e.cpu().numpy().reshape(coarse_values.shape[:2])
                coarse_p[arm+'_FINAL'] = torch.sigmoid(f).cpu().numpy().reshape(coarse_values.shape[:2])
                coarse_p[arm+'_MEAN'] = .5*(torch.sigmoid(f)+torch.sigmoid(e)).cpu().numpy().reshape(coarse_values.shape[:2])
        np.savez_compressed(local/'coarse_logits.npz', tracks=inputs.tracks, **coarse_z, **{k+'_probability':v for k,v in coarse_p.items()})
        generic_z = {t:np.load(generic_root/'videos'/video/('LOCAL_TWO_'+t+'_logits.npy'), mmap_mode='r') for t in ('FINAL', 'EARLY')}
        old_p = {a+'_'+t:np.load(PARENT/pair/f'seed{seed}'/'videos'/video/(a+'_'+t+'_scores.npy'), mmap_mode='r')
            for a in DENSE for t in ('FINAL', 'MEAN')}
        audit = {n:dict(max_probability_difference_from_original=0., changed_original_raw_decisions=0,
            replay_rounded_vs_original_rounded_decisions=0, logit_vs_own_rounded_probability_decisions=0)
            for n in probabilities}
        coarse_audit = {}
        with np.load(COARSE/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as old_coarse:
            assert np.array_equal(old_coarse['tracks'], inputs.tracks)
            for arm in COARSE_ARMS:
                for tail in ('FINAL', 'MEAN'):
                    name = arm+'_'+tail; value = coarse_p[name]; old = old_coarse[name]
                    raw_bits = coarse_z[arm+'_FINAL'] > 0 if tail == 'FINAL' else coarse_z[arm+'_FINAL'].astype(float)+coarse_z[arm+'_EARLY'].astype(float) > 0
                    coarse_audit[name] = dict(scope='query_track_scores_not_expanded_dense_rows',
                        max_probability_difference_from_original=float(np.max(np.abs(value-old))),
                        replay_rounded_vs_original_rounded_decisions=int(((value>.5)!=(old>.5)).sum()),
                        logit_vs_own_rounded_probability_decisions=int((raw_bits!=(value>.5)).sum()),
                        changed_original_raw_decisions=int((raw_bits!=(old>.5)).sum()))
        normalized = ((inputs.raw_bags-mean)/std).astype('float32'); normalized[:, ~inputs.present] = 0
        with torch.no_grad():
            for a in range(0, nq*no, 2048):
                flat = np.arange(a, min(a+2048, nq*no)); qi, rows = flat//no, flat % no; ti = inputs.columns[rows]
                current = ((inputs.dense_rows(qi, rows)-mean)/std).astype('float32')
                bb = torch.as_tensor(normalized[qi, ti], device='cuda:0')
                mm = torch.as_tensor(inputs.present[ti], device='cuda:0'); cc = torch.as_tensor(current, device='cuda:0')
                values = {}; pvalues = {}
                for arm in DENSE:
                    head, early, kind = heads[arm]
                    f = forward(head, kind, bb, mm, cc); e = forward(early, kind, bb, mm, cc)
                    zf, ze = f.cpu().numpy(), e.cpu().numpy(); values[arm] = zf, ze
                    pvalues[arm+'_FINAL'] = torch.sigmoid(f).cpu().numpy()
                    pvalues[arm+'_MEAN'] = .5*(torch.sigmoid(e)+torch.sigmoid(f)).cpu().numpy()
                    dense_z[arm]['FINAL'].reshape(-1)[flat] = zf; dense_z[arm]['EARLY'].reshape(-1)[flat] = ze
                values['LOCAL_TWO'] = np.asarray(generic_z['FINAL'][qi, rows]), np.asarray(generic_z['EARLY'][qi, rows])
                for arm in COARSE_ARMS:
                    values[arm] = coarse_z[arm+'_FINAL'][qi, ti], coarse_z[arm+'_EARLY'][qi, ti]
                for arm, (zf, ze) in values.items():
                    final_bits = zf > 0; mean_bits = zf.astype(float)+ze.astype(float) > 0
                    # Expose any double cancellation at the threshold rather
                    # than silently using it to change a selected count.
                    assert np.array_equal(mean_logit(zf, ze)>0, mean_bits), (pair, video, arm, 'mean-logit precision exhausted')
                    bits[arm+'_FINAL'].reshape(-1)[flat] = final_bits
                    bits[arm+'_MEAN'].reshape(-1)[flat] = mean_bits
                for name, value in pvalues.items():
                    probabilities[name].reshape(-1)[flat] = value
                    old = np.asarray(old_p[name][qi, rows]); new_bits = np.asarray(bits[name].reshape(-1)[flat])
                    audit[name]['max_probability_difference_from_original'] = max(audit[name]['max_probability_difference_from_original'], float(np.max(np.abs(value-old))))
                    audit[name]['changed_original_raw_decisions'] += int((new_bits != (old > .5)).sum())
                    audit[name]['replay_rounded_vs_original_rounded_decisions'] += int(((value>.5)!=(old>.5)).sum())
                    audit[name]['logit_vs_own_rounded_probability_decisions'] += int((new_bits != (value > .5)).sum())
        for arrays in dense_z.values():
            for value in arrays.values(): value.flush()
        for value in probabilities.values(): value.flush()
        for name, mask in bits.items():
            mask.flush(); destination = root/name/'videos'/video; destination.mkdir(parents=True, exist_ok=True)
            np.save(destination/'mask.npy', mask)
            immediate_predictions(root/name, type('Context', (), dict(bank=inputs.bank, queries=inputs.queries))(), mask)
        torch.cuda.synchronize()
        receipt = dict(video=video, index=vi, fits=fits, teacher_evidence=records[-1]['report'],
            original_probability_replica_audit=audit, coarse_probability_replica_audit=coarse_audit, GT_open_attempts=len(guard),
            exact_original_heads_not_assumed=True, generic_actual_own_prefix_logits_reused=True,
            state_normalization_rows=count, coarse_normalization_rows=coarse_count,
            current_and_past_only=True, wall_seconds=time.perf_counter()-stamp, quality_replay_not_cold_cost=True)
        receipts.append(receipt); previous = new_previous; next_video = vi+1
        temp = stream.with_suffix('.tmp'); torch.save(dict(previous=previous, receipts=receipts, next_video=next_video), temp); temp.replace(stream)
        del heads, bits, probabilities, dense_z, data, cd
        torch.cuda.empty_cache()
    methods = []
    for name in METHODS:
        mask = np.zeros(md['candidate_query_rows'], bool)
        for video in sorted(md['videos']):
            values = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in md['episodes'] if q['video_id'] == video): mask[q['start']:q['stop']] = values[qi]
        row = seal_mask(root/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(seed=seed, no_future_answers_or_normalization=True, current_video_transductive=True,
                unquantized_logit_threshold=True, identical_registered_learning_recipes=True, no_source_logit_residual=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    failures = [dict(video=r['video'], arm=a, metrics=f['metrics']) for r in receipts for a,f in r['fits'].items() if not f['target_fit_gate_pass']]
    atomic_json(root/'completion.json', dict(status='SEALED_UNQUANTIZED_LOGIT_REPLICA_RAW_NO_GT', methods=methods,
        receipts=receipts, target_fit_failures=failures, reused_generic_target_fit_failures=generic_done['target_fit_failures'],
        GT_open_attempts=0, wall_seconds=time.perf_counter()-started, no_new_teacher_or_backbone_calls=True, goal_status='ACTIVE'))


def video_logits(root, pair, seed, video, obs):
    final = {}; early = {}
    for arm in DENSE:
        final[arm] = np.load(root/'videos'/video/(arm+'_FINAL_logits.npy'), mmap_mode='r')
        early[arm] = np.load(root/'videos'/video/(arm+'_EARLY_logits.npy'), mmap_mode='r')
    for arm in ('LOCAL_TWO',):
        base = GENERIC/pair/f'seed{seed}'/'videos'/video
        final[arm] = np.load(base/(arm+'_FINAL_logits.npy'), mmap_mode='r')
        early[arm] = np.load(base/(arm+'_EARLY_logits.npy'), mmap_mode='r')
    with np.load(root/'videos'/video/'coarse_logits.npz') as data:
        lookup = {int(t):i for i,t in enumerate(data['tracks'])}; columns = np.asarray([lookup[int(t)] for _,t in obs])
        for arm in COARSE_ARMS:
            final[arm] = data[arm+'_FINAL'][:, columns]; early[arm] = data[arm+'_EARLY'][:, columns]
    return final, early


def counts(record):
    from collections import Counter
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard(); pair = record['pair']; source, target = PAIRS[pair]; md = metadata(target)
    root = Path(record['folder']); destination = root/'countmatched'
    if (destination/'completion.json').is_file(): return str(destination)
    pairs = [(PRIMARY, PRIMARY)]+[p for b in BASELINES for p in ((b, PRIMARY), (PRIMARY, b), (b, b))]
    masks = {p:np.zeros(md['candidate_query_rows'], bool) for p in pairs}; ties = Counter(); nontrivial = Counter()
    for video, bank in sorted(md['videos'].items()):
        obs = np.asarray(bank['observations']); queries = [q for q in md['episodes'] if q['video_id'] == video]
        final, early = video_logits(root, pair, record['seed'], video, obs)
        bits = {n:np.load(root/n/'videos'/video/'mask.npy', mmap_mode='r') for n in METHODS}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            for qi, q in enumerate(queries):
                values = {a+'_FINAL':np.asarray(final[a][qi, ids], dtype=float) for a in ARMS}
                values.update({a+'_MEAN':mean_logit(final[a][qi, ids], early[a][qi, ids]) for a in ARMS})
                order = {n:ids[np.lexsort((obs[ids, 1], -v))] for n,v in values.items()}
                sorted_values = {n:np.sort(v)[::-1] for n,v in values.items()}
                count = {n:int(bits[n][qi, ids].sum()) for n in METHODS}
                for rank, donor in pairs:
                    n = count[donor]; masks[rank, donor][q['start']+order[rank][:n]] = True
                    if 0 < n < len(ids):
                        name = rank+'__AT_'+donor+'_COUNT'; nontrivial[name] += 1
                        ties[name] += int(sorted_values[rank][n-1] == sorted_values[rank][n])
    methods = []
    for (rank, donor), mask in masks.items():
        name = rank+'__AT_'+donor+'_COUNT'
        row = seal_mask(destination/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(rank_method=rank, count_method=donor, counts_from_GT=False, counts_equal_per_query_frame=True,
                ordering_before_sigmoid_quantization=True, mean_order_from_float64_log_probability_odds=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    changed = {}
    for name in METHODS:
        old = json.loads((root/name/'predictions_seal.json').read_text())
        new = json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        lookup = {(r['video'], r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name] = sum(lookup[r['video'], r['expression']] != r['prediction_sha256'] for r in new['episodes'])
    assert not guard and not any(changed.values()), changed
    atomic_json(destination/'completion.json', dict(status='SEALED_UNQUANTIZED_LOGIT_COUNTS_NO_GT', methods=methods,
        CountSelf_changed_query_files=changed, actual_exact_rank_cutoff_ties=dict(ties), nontrivial_cutoffs=dict(nontrivial), GT_open_attempts=0))
    return str(destination)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    data = {}; statistics = {}; summary = {}; error = 0.
    for record in conditions():
        root = Path(record['folder']); slug = record['pair']+f"_seed{record['seed']}"
        raw = json.loads((root/'completion.json').read_text()); cnt = json.loads((root/'countmatched/completion.json').read_text())
        paths = {r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in cnt['methods']})
        paths['ORIGINAL_FULL_FINAL'] = PARENT/record['pair']/f"seed{record['seed']}"/PRIMARY
        data[slug] = {n:read_stats(p) for n,p in paths.items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']; error = max(error, abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'], numerator=num.tolist(), denominator=den.tolist())
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in (*METHODS, 'ORIGINAL_FULL_FINAL')},
            target_fit_failures=raw['target_fit_failures'], reused_generic_target_fit_failures=raw['reused_generic_target_fit_failures'],
            exact_rank_cutoff_ties=cnt['actual_exact_rank_cutoff_ties'], CountSelf_changed_query_files=cnt['CountSelf_changed_query_files'])
    repeated = {p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in (*METHODS, 'ORIGINAL_FULL_FINAL')},
        comparisons={label:comparison([data[p+f'_seed{s}'][a] for s in SEEDS], [data[p+f'_seed{s}'][b] for s in SEEDS])
            for label,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    assert error < 1e-8 and len(statistics) == 1134, (error, len(statistics))
    result = dict(status='LOGIT_PRECISION_MATCHED_TEACHER_BASELINES_ANALYZED', primary=PRIMARY,
        primary_recipe_unchanged='FULL; numerical precision replica fixed before parent quality', conditions=summary, repeated=repeated,
        HOTA_parent_sum_max_error=error, all_own_and_required_parent_outputs_sealed_before_this_GT=True,
        all_data_development=True, independent_blind_validation=False, quality_replay_not_cold_cost=True,
        no_new_teacher_information_or_backbone_calls=True, not_a_new_innovation=True,
        replicas_not_assumed_identical_to_old_heads=True, latent_factors_not_semantically_identified=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics, separators=(',', ':'))+'\n')


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_GENERIC_CONTROL_GPU_STAGE_RELEASE', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        earlier = json.loads(GENERIC_STATE.read_text()) if GENERIC_STATE.is_file() else {}
        if earlier.get('status', '').startswith('FAILED_'): raise RuntimeError('Resolve registered generic-product control first')
        if earlier.get('completed', 0) == 18 and not earlier.get('active'): break
        time.sleep(15)
    pending = conditions(); done = []; active = {}; failure = None
    def state(status):
        atomic_json(STATE, dict(status=status, pid=os.getpid(), completed=len(done), finished=done, pending=pending,
            active={str(g):j for g,(_,_,j) in active.items()}, failure=failure, goal_status='ACTIVE'))
    while pending or active:
        for gpu,(proc,handle,job) in list(active.items()):
            if proc.poll() is None: continue
            handle.close(); del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():
                failure = dict(job=job, exit_code=proc.returncode, last_log=Path(job['log']).read_text()[-8000:])
            else: done.append(job)
        free = [g for g in (0,1,2,3) if g not in active]
        if pending and free and not failure:
            output = subprocess.check_output(['nvidia-smi','-i',','.join(map(str,free)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in free:
                if not pending or memory[gpu] >= 512: continue
                job = pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file(): done.append(dict(job,reused_sealed=True)); continue
                log = STATE.parent/(job['pair']+'_seed'+str(job['seed'])+'.log'); handle = log.open('a',buffering=1)
                proc = subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'))
                active[gpu] = proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu)
        state('RUNNING_MATCHED_LOGIT_PRECISION_REPLICAS_NO_GT' if not failure else 'PRESERVING_HEALTHY_LOGIT_REPLICAS_AFTER_FAILURE')
        if failure and not active: raise RuntimeError(json.dumps(failure))
        if pending or active: time.sleep(5)
    state('ALL18_LOGIT_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool: roots = list(pool.map(counts, conditions()))
    for parent in (PARENT, COARSE, GENERIC):
        while not (parent/'all_predictions_sealed_before_GT.json').is_file(): time.sleep(15)
    tasks = []
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            doc = json.loads((folder/'completion.json').read_text()); assert doc['GT_open_attempts'] == 0
            tasks.extend((str(folder/r['method']),record['pair']) for r in doc['methods'])
    assert len(tasks) == 1116
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=1116,required_parent_studies=[1692,378,162],GT_open_attempts=0))
    state('ALL1116_LOGIT_AND_REQUIRED_PARENT_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool: results = dict(pool.map(evaluate_one,tasks))
    while not (PARENT/'official.json').is_file(): time.sleep(15)
    atomic_json(OUT/'official.json',dict(methods=1116,results=results)); analyze(); state(SUCCESS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pair',choices=tuple(PAIRS)); parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--math-check',action='store_true'); args = parser.parse_args()
    if args.math_check: print(json.dumps(mathematical_checks()))
    elif args.pair:
        import fcntl
        root = OUT/args.pair/f'seed{args.seed}'; root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock: fcntl.flock(lock,fcntl.LOCK_EX); run(args.pair,args.seed)
    else:
        try: main()
        except Exception as error:
            old = json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_LOGIT_PRECISION_REPLAY_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,old); raise
