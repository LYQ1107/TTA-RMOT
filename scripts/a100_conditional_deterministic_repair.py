"""Revalidate the unchanged observation-learning mechanisms deterministically.

Keep the original cohorts. Repair execution before interpreting the shared
scope/profile/policy effects, add paired ordinary heads, and compare the
already adequate fixed ordinary references without GT-based promotion.
"""
import argparse
from collections import Counter
import concurrent.futures
import hashlib
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
from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
from scripts.a100_conditional_six_setting import OUT as ORIGINAL
from scripts.a100_conditional_coarse_baseline import OUT as COARSE, coarse_inputs, supervision as coarse_supervision
from scripts.a100_conditional_shared_control import OUT as GENERIC
from scripts.a100_conditional_logit_precision import OUT as PRECISION, mean_logit, video_logits as precision_logits
from scripts.a100_conditional_standard_kd import OUT as STANDARD, standard_targets, use_verified_cache
from scripts.a100_conditional_repeatability import OUT as REPEATABILITY, STATE as REPEATABILITY_STATE, SUCCESS as REPEATABILITY_SUCCESS
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS

OUT = SIX/'conditional_deterministic_repair_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'conditional_deterministic_repair_registration_v1.json'
STEM = 'conditional_deterministic_repair'
PRIMARY = 'FULL_FINAL'
SUCCESS = 'PASS_DETERMINISTIC_CONDITIONAL_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
CORE = ('FULL', 'DIRECT_POINT', 'DIRECT_BALANCED', 'DIRECT_PROFILE', 'CONDITIONAL_BALANCED',
    'LOCAL_TWO', 'FULL_PRIOR', 'FULL_UNIFORM', 'FULL_COVERAGE')
PAIRED = ('PAIRED_FLAT_SOFT', 'PAIRED_COARSE_HARD')
OWN = tuple(a+'_FINAL' for a in CORE)+tuple(a+'_'+t for a in PAIRED for t in ('FINAL', 'MEAN'))
WEIGHTED = tuple(a+'_'+t for a in ('FRAME_SOFT', 'FRAME_HARD', 'FLAT_SOFT', 'FLAT_HARD', 'COARSE_SOFT', 'COARSE_HARD') for t in ('FINAL', 'MEAN'))
REFERENCES = WEIGHTED+tuple('STD_'+n for n in WEIGHTED)
METHODS = OWN+REFERENCES
BASELINES = tuple(n for n in METHODS if n != PRIMARY)
CONTRASTS = {'FULL_vs_'+b:(PRIMARY, b) for b in BASELINES}
for b in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b] = (PRIMARY+'__AT_'+b+'_COUNT', b+'__AT_'+b+'_COUNT')
CONTRASTS.update(CONDITIONAL_STRUCTURE=(PRIMARY, 'DIRECT_PROFILE_FINAL'),
    QUERY_PROFILE=(PRIMARY, 'CONDITIONAL_BALANCED_FINAL'),
    EXPOSURE_BALANCE=('DIRECT_BALANCED_FINAL', 'DIRECT_POINT_FINAL'),
    PROFILE_ON_DIRECT=('DIRECT_PROFILE_FINAL', 'DIRECT_BALANCED_FINAL'),
    SHARED_SCOPE=(PRIMARY, 'LOCAL_TWO_FINAL'))
PARENTS = ((ORIGINAL, 1692), (COARSE, 378), (GENERIC, 162), (PRECISION, 1116), (STANDARD, 882))


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def projection_pairs():
    return [(PRIMARY, PRIMARY)]+[p for b in OWN if b != PRIMARY for p in ((b, PRIMARY), (PRIMARY, b), (b, b))]+[
        p for b in REFERENCES for p in ((b, PRIMARY), (PRIMARY, b))]


def reference_root(name, pair, seed):
    return (STANDARD if name.startswith('STD_') else PRECISION)/pair/f'seed{seed}'


def check_registration():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    return plan


def fit_key(data, recipe, previous, mean, std):
    h = hashlib.sha256(recipe.encode())
    for key in ('bags', 'present', 'current', 'raw_target', 'target', 'group', 'balanced_weight', 'confidence'):
        h.update(np.ascontiguousarray(data[key]).tobytes())
    h.update(np.asarray(mean).tobytes())
    h.update(np.asarray(std).tobytes())
    if previous is not None:
        for key, value in sorted(previous['model'].items()):
            h.update(key.encode())
            h.update(value.numpy().tobytes())
        h.update(np.asarray(previous['mean']).tobytes())
        h.update(np.asarray(previous['std']).tobytes())
    return h.hexdigest()


def run(pair, seed):
    check_registration()
    execution = configure_before_cuda()
    import torch
    from flex_uda.active import observation_learning_fit as learning
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    assert 'LOCAL_TWO' not in learning.RECIPES
    learning.RECIPES['LOCAL_TWO'] = ('local_two_factor', True, True, False)
    guard = generation_guard()
    use_verified_cache()
    torch.set_num_threads(4)
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
    ct, cq, cn = total.copy(), total.copy(), 0
    records = {p:[] for p in ('DESIGN', 'PRIOR', 'UNIFORM', 'COVERAGE_ONLY')}
    coarse_records = []
    for vi, video in enumerate(sorted(md['videos'])):
        stamp = time.perf_counter()
        inputs = ObservationVideo(pair, video)
        flat = inputs.raw_bags[:, inputs.present].reshape(-1, DIMENSION).astype(float)
        total += flat.sum(0)
        square += np.square(flat).sum(0)
        count += len(flat)
        mean = total/count
        std = np.maximum(np.sqrt(np.maximum(square/count-mean**2, 0)), 1e-3)
        for policy in records:
            records[policy].append(inputs.supervision(policy))
        coarse_values = coarse_inputs(inputs)
        coarse_records.append(coarse_supervision(inputs, coarse_values))
        cf = coarse_values.reshape(-1, DIMENSION).astype(float)
        ct += cf.sum(0)
        cq += np.square(cf).sum(0)
        cn += len(cf)
        cm = ct/cn
        cs = np.maximum(np.sqrt(np.maximum(cq/cn-cm**2, 0)), 1e-3)
        if vi < next_video:
            continue
        prepared = {p:prefix_data(rows, mean, std) for p, rows in records.items()}
        unit = standard_targets(prepared['DESIGN'])
        coarse_unit = standard_targets(prefix_data(coarse_records, cm, cs))
        heads, fits, new_previous, cache = {}, {}, {}, {}
        actual_fits = 0
        for arm in (*CORE, *PAIRED):
            policy = {'FULL_PRIOR':'PRIOR', 'FULL_UNIFORM':'UNIFORM', 'FULL_COVERAGE':'COVERAGE_ONLY'}.get(arm, 'DESIGN')
            paired = arm in PAIRED
            coarse = arm == 'PAIRED_COARSE_HARD'
            data = coarse_unit if coarse else unit if paired else prepared[policy]
            mu, sigma = (cm, cs) if coarse else (mean, std)
            recipe = 'FRAME_HARD' if coarse else 'FLAT_SOFT' if paired else 'FULL' if arm.startswith('FULL_') else arm
            old = previous.get(arm)
            key = fit_key(data, recipe, old, mu, sigma)
            if key in cache:
                owner, model, early, checkpoint, receipt = cache[key]
                receipt = dict(receipt, identical_own_prefix_fit_reused_from=owner)
            else:
                model, early, checkpoint, receipt = learning.fit(data, recipe, seed=seed, mean=mu, std=sigma, previous=old)
                actual_fits += 1
                cache[key] = arm, model, early, checkpoint, receipt
            assert receipt['target_fit_gate_pass'], (pair, video, arm, receipt['metrics'])
            fits[arm] = dict(receipt, deterministic_execution=True, raw_unit_weight_ordinary=paired,
                actual_policy=policy, source_recipe_unchanged=True)
            heads[arm] = model, early, checkpoint['kind']
            new_previous[arm] = checkpoint
            atomic_json(root/'progress.json', dict(completed=vi, total=len(md['videos']), video=video,
                stage='FITTING', last_arm=arm, completed_arms=len(heads), fit_gate_pass=True))
        local = root/'videos'/video
        local.mkdir(parents=True, exist_ok=True)
        nq, no = len(inputs.queries), len(inputs.obs)
        dense_arms = (*CORE, 'PAIRED_FLAT_SOFT')
        logits = {arm:{tail:np.lib.format.open_memmap(local/(arm+'_'+tail+'_logits.npy'), mode='w+',
            dtype='float32', shape=(nq, no)) for tail in (('FINAL', 'EARLY') if arm in PAIRED else ('FINAL',))} for arm in dense_arms}
        bits = {name:np.lib.format.open_memmap(local/(name+'_mask.npy'), mode='w+', dtype='bool', shape=(nq, no)) for name in OWN}
        with torch.no_grad():
            cc = torch.as_tensor(((coarse_values.reshape(-1, DIMENSION)-cm)/cs).astype('float32'), device='cuda:0')
            model, early, kind = heads['PAIRED_COARSE_HARD']
            coarse_z = {tail:learning.forward(module, kind, None, None, cc).cpu().numpy().reshape(coarse_values.shape[:2])
                for tail, module in (('FINAL', model), ('EARLY', early))}
        np.savez_compressed(local/'coarse_logits.npz', tracks=inputs.tracks, **coarse_z)
        normalized = ((inputs.raw_bags-mean)/std).astype('float32')
        normalized[:, ~inputs.present] = 0
        with torch.no_grad():
            for a in range(0, nq*no, 2048):
                ids = np.arange(a, min(a+2048, nq*no))
                qi, rows = ids//no, ids % no
                ti = inputs.columns[rows]
                bb = torch.as_tensor(normalized[qi, ti], device='cuda:0')
                mm = torch.as_tensor(inputs.present[ti], device='cuda:0')
                cc = torch.as_tensor(((inputs.dense_rows(qi, rows)-mean)/std).astype('float32'), device='cuda:0')
                for arm in dense_arms:
                    model, early, kind = heads[arm]
                    z = learning.forward(model, kind, bb, mm, cc).cpu().numpy()
                    logits[arm]['FINAL'].reshape(-1)[ids] = z
                    bits[arm+'_FINAL'].reshape(-1)[ids] = z > 0
                    if arm in PAIRED:
                        e = learning.forward(early, kind, bb, mm, cc).cpu().numpy()
                        logits[arm]['EARLY'].reshape(-1)[ids] = e
                        selected = z.astype(float)+e.astype(float) > 0
                        assert np.array_equal(mean_logit(z, e)>0, selected)
                        bits[arm+'_MEAN'].reshape(-1)[ids] = selected
                f, e = coarse_z['FINAL'][qi, ti], coarse_z['EARLY'][qi, ti]
                selected = f.astype(float)+e.astype(float) > 0
                assert np.array_equal(mean_logit(f, e)>0, selected)
                bits['PAIRED_COARSE_HARD_FINAL'].reshape(-1)[ids] = f > 0
                bits['PAIRED_COARSE_HARD_MEAN'].reshape(-1)[ids] = selected
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
        receipt = dict(video=video, index=vi, fits=fits, actual_head_fits=actual_fits,
            exact_identical_fit_reuse=len(heads)-actual_fits, execution=execution, GT_open_attempts=len(guard),
            normalization_rows=count, coarse_normalization_rows=cn, current_and_past_only=True,
            teacher_evidence={p:rows[-1]['report'] for p, rows in records.items()},
            wall_seconds=time.perf_counter()-stamp, quality_replay_not_cold_cost=True)
        atomic_json(local/'receipt.json', receipt)
        receipts.append(receipt)
        previous, next_video = new_previous, vi+1
        temp = stream.with_suffix('.tmp')
        torch.save(dict(previous=previous, receipts=receipts, next_video=next_video), temp)
        temp.replace(stream)
        del prepared, heads, cache, logits, bits, unit, coarse_unit
        torch.cuda.empty_cache()
    methods = []
    for name in OWN:
        mask = np.zeros(md['candidate_query_rows'], bool)
        for video in sorted(md['videos']):
            values = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in md['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = values[qi]
        row = seal_mask(root/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(seed=seed, execution=execution, no_future_answers_or_normalization=True,
                current_video_transductive=True, unquantized_logit_threshold=True, no_source_logit_residual=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_DETERMINISTIC_CONDITIONAL_REPAIR_RAW_NO_GT', methods=methods,
        receipts=receipts, target_fit_failures=[], GT_open_attempts=0, actual_head_fits=sum(r['actual_head_fits'] for r in receipts),
        wall_seconds=time.perf_counter()-started, no_new_teacher_or_backbone_calls=True, goal_status='ACTIVE'))


def own_vectors(root, video, obs):
    local = root/'videos'/video
    values = {a+'_FINAL':(np.load(local/(a+'_FINAL_logits.npy'), mmap_mode='r'), None) for a in CORE}
    f = np.load(local/'PAIRED_FLAT_SOFT_FINAL_logits.npy', mmap_mode='r')
    e = np.load(local/'PAIRED_FLAT_SOFT_EARLY_logits.npy', mmap_mode='r')
    values['PAIRED_FLAT_SOFT_FINAL'] = f, None
    values['PAIRED_FLAT_SOFT_MEAN'] = f, e
    with np.load(local/'coarse_logits.npz') as data:
        lookup = {int(t):i for i, t in enumerate(data['tracks'])}
        columns = np.asarray([lookup[int(t)] for _, t in obs])
        f, e = data['FINAL'][:, columns], data['EARLY'][:, columns]
    values['PAIRED_COARSE_HARD_FINAL'] = f, None
    values['PAIRED_COARSE_HARD_MEAN'] = f, e
    return values


def reference_vectors(pair, seed, video, obs):
    root = PRECISION/pair/f'seed{seed}'
    final, early = precision_logits(root, pair, seed, video, obs)
    values = {name:(final[name.rsplit('_', 1)[0]], None if name.endswith('_FINAL') else early[name.rsplit('_', 1)[0]]) for name in WEIGHTED}
    local = STANDARD/pair/f'seed{seed}'/'videos'/video
    final = {a:np.load(local/(a+'_FINAL_logits.npy'), mmap_mode='r') for a in ('STD_FRAME_SOFT', 'STD_FRAME_HARD', 'STD_FLAT_SOFT', 'STD_FLAT_HARD')}
    early = {a:np.load(local/(a+'_EARLY_logits.npy'), mmap_mode='r') for a in final}
    with np.load(local/'coarse_logits.npz') as data:
        lookup = {int(t):i for i, t in enumerate(data['tracks'])}
        columns = np.asarray([lookup[int(t)] for _, t in obs])
        for arm in ('STD_COARSE_SOFT', 'STD_COARSE_HARD'):
            final[arm] = data[arm+'_FINAL'][:, columns]
            early[arm] = data[arm+'_EARLY'][:, columns]
    values.update({name:(final[name.rsplit('_', 1)[0]], None if name.endswith('_FINAL') else early[name.rsplit('_', 1)[0]]) for name in REFERENCES if name.startswith('STD_')})
    return values


def selected_scores(vectors, qi, ids):
    return {name:np.asarray(f[qi, ids], dtype=float) if e is None else mean_logit(f[qi, ids], e[qi, ids])
        for name, (f, e) in vectors.items()}


def counts(record):
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard()
    pair, seed = record['pair'], record['seed']
    source, target = PAIRS[pair]
    md = metadata(target)
    root = Path(record['folder'])
    destination = root/'countmatched'
    if (destination/'completion.json').is_file():
        return str(destination)
    pairs = projection_pairs()
    assert len(pairs) == 85
    masks = {p:np.zeros(md['candidate_query_rows'], bool) for p in pairs}
    ties, nontrivial = Counter(), Counter()
    for video, bank in sorted(md['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        vectors = own_vectors(root, video, obs)
        vectors.update(reference_vectors(pair, seed, video, obs))
        assert set(vectors) == set(METHODS)
        bits = {n:np.load((reference_root(n, pair, seed) if n in REFERENCES else root)/n/'videos'/video/'mask.npy', mmap_mode='r') for n in METHODS}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            for qi, q in enumerate(queries):
                values = selected_scores(vectors, qi, ids)
                order = {n:ids[np.lexsort((obs[ids, 1], -v))] for n, v in values.items()}
                ordered = {n:np.sort(v)[::-1] for n, v in values.items()}
                numbers = {n:int(bits[n][qi, ids].sum()) for n in METHODS}
                for rank, donor in pairs:
                    n = numbers[donor]
                    masks[rank, donor][q['start']+order[rank][:n]] = True
                    if 0 < n < len(ids):
                        name = rank+'__AT_'+donor+'_COUNT'
                        nontrivial[name] += 1
                        ties[name] += int(ordered[rank][n-1] == ordered[rank][n])
    methods = []
    for (rank, donor), mask in masks.items():
        name = rank+'__AT_'+donor+'_COUNT'
        row = seal_mask(destination/name, md, mask, target=target, model=source, method=name, GT_open_attempts=len(guard),
            extra=dict(rank_method=rank, count_method=donor, counts_from_GT=False, counts_equal_per_query_frame=True,
                ordering_before_sigmoid_quantization=True, deterministic_core_execution=True,
                ordinary_reference_fixed=rank in REFERENCES or donor in REFERENCES))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    changed = {}
    for name in OWN:
        old = json.loads((root/name/'predictions_seal.json').read_text())
        new = json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        lookup = {(r['video'], r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name] = sum(lookup[r['video'], r['expression']] != r['prediction_sha256'] for r in new['episodes'])
    assert not guard and not any(changed.values()), changed
    atomic_json(destination/'completion.json', dict(status='SEALED_DETERMINISTIC_REPAIR_DUAL_COUNTS_NO_GT', methods=methods,
        CountSelf_changed_query_files=changed, actual_exact_rank_cutoff_ties=dict(ties), nontrivial_cutoffs=dict(nontrivial),
        frozen_ordinary_reference_own_counts_reused_only_at_analysis=True, GT_open_attempts=0))
    return str(destination)


def statistical_paths(record):
    root = Path(record['folder'])
    raw = json.loads((root/'completion.json').read_text())
    count = json.loads((root/'countmatched/completion.json').read_text())
    paths = {r['method']:root/r['method'] for r in raw['methods']}
    paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
    for name in REFERENCES:
        parent = reference_root(name, record['pair'], record['seed'])
        paths[name] = parent/name
        self_name = name+'__AT_'+name+'_COUNT'
        paths[self_name] = parent/'countmatched'/self_name
    assert len(paths) == 146
    return paths


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    data, statistics, summary, error = {}, {}, {}, 0.
    for record in conditions():
        root = Path(record['folder'])
        slug = record['pair']+f"_seed{record['seed']}"
        raw = json.loads((root/'completion.json').read_text())
        count = json.loads((root/'countmatched/completion.json').read_text())
        data[slug] = {n:read_stats(p) for n, p in statistical_paths(record).items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']
            error = max(error, abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'], numerator=num.tolist(), denominator=den.tolist())
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},
            target_fit_failures=raw['target_fit_failures'], exact_rank_cutoff_ties=count['actual_exact_rank_cutoff_ties'],
            CountSelf_changed_query_files=count['CountSelf_changed_query_files'])
    repeated = {}
    for pair in PAIRS:
        keys = [pair+f'_seed{s}' for s in SEEDS]
        interaction = [data[k]['FULL_FINAL']['HOTA']-data[k]['CONDITIONAL_BALANCED_FINAL']['HOTA']-
            data[k]['DIRECT_PROFILE_FINAL']['HOTA']+data[k]['DIRECT_BALANCED_FINAL']['HOTA'] for k in keys]
        repeated[pair] = dict(mean_HOTA={n:float(np.mean([data[k][n]['HOTA'] for k in keys])) for n in METHODS},
            comparisons={label:comparison([data[k][a] for k in keys], [data[k][b] for k in keys]) for label, (a, b) in CONTRASTS.items()},
            factorization_profile_interaction=dict(per_seed=interaction, mean=float(np.mean(interaction)), CI95=None))
    assert error < 1e-8 and len(statistics) == 2628, (error, len(statistics))
    result = dict(status='DETERMINISTIC_CONDITIONAL_REPAIR_SIX_SETTINGS_ANALYZED', primary=PRIMARY,
        primary_recipe_unchanged=True, conditions=summary, repeated=repeated, HOTA_parent_sum_max_error=error,
        execution_repair_not_a_new_semantic_innovation=True, paired_ordinary_deterministic_methods=list(OWN[-4:]),
        other_ordinary_references_frozen_not_all_paired_execution_ablation=True,
        all_own_and_required_parent_outputs_sealed_before_this_GT=True,
        all_data_development=True, independent_blind_validation=False,
        quality_replay_not_cold_cost=True, latent_factors_not_semantically_identified=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics, separators=(',', ':'))+'\n')


def main():
    check_registration()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_CONTROLLED_REPEATABILITY_AND_OLD_GPU_STAGE_RELEASE', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        proof = json.loads(REPEATABILITY_STATE.read_text()) if REPEATABILITY_STATE.is_file() else {}
        if proof.get('status', '').startswith('FAILED_'):
            raise RuntimeError('Resolve the controlled repeatability diagnostic first')
        if proof.get('status') == REPEATABILITY_SUCCESS:
            break
        time.sleep(15)
    diagnosis = json.loads((REPEATABILITY/'analysis.json').read_text())
    assert diagnosis['deterministic_repeatability_observed']
    for parent, _ in PARENTS:
        while True:
            path = parent/'supervisor/state.json'
            older = json.loads(path.read_text()) if path.is_file() else {}
            if older.get('status', '').startswith('FAILED_'):
                raise RuntimeError('Resolve the registered parent study first: '+str(parent))
            if older.get('completed', 0) == 18 and not older.get('active'):
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
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4'))
                active[gpu] = proc, handle, dict(job, pid=proc.pid, log=str(log), physical_GPU=gpu)
        state('RUNNING_DETERMINISTIC_CONDITIONAL_REPAIR_NO_GT' if not failure else 'PRESERVING_HEALTHY_DETERMINISTIC_REPAIR_WORKERS')
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(5)
    state('ALL18_DETERMINISTIC_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(counts, conditions()))
    for parent, expected in PARENTS:
        marker = parent/'all_predictions_sealed_before_GT.json'
        while not marker.is_file():
            time.sleep(15)
        proof = json.loads(marker.read_text())
        assert proof['methods'] == expected and proof['GT_open_attempts'] == 0
    tasks = []
    for record, count_root in zip(conditions(), roots):
        for folder in (Path(record['folder']), Path(count_root)):
            proof = json.loads((folder/'completion.json').read_text())
            assert proof['GT_open_attempts'] == 0
            tasks.extend((str(folder/r['method']), record['pair']) for r in proof['methods'])
        for parent in (PRECISION, STANDARD):
            proof = json.loads((parent/record['pair']/f"seed{record['seed']}"/'countmatched/completion.json').read_text())
            assert proof['GT_open_attempts'] == 0 and not any(proof['CountSelf_changed_query_files'].values())
    assert len(tasks) == 1764
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=1764,
        required_parent_studies=[n for _, n in PARENTS], GT_open_attempts=0))
    state('ALL1764_DETERMINISTIC_AND_REQUIRED_PARENT_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    atomic_json(OUT/'official.json', dict(methods=1764, results=results))
    # Existing owners evaluate their own reference folders. Waiting for their
    # completion markers avoids duplicate writers and preserves GT isolation.
    state('ALL1764_OWN_OFFICIAL_DONE_AWAITING_FIXED_REFERENCE_STATISTICS')
    for record in conditions():
        for name in REFERENCES:
            parent = reference_root(name, record['pair'], record['seed'])
            for folder in (parent/name, parent/'countmatched'/(name+'__AT_'+name+'_COUNT')):
                while not (folder/'official_completion.json').is_file():
                    time.sleep(15)
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
            old.update(status='FAILED_DETERMINISTIC_CONDITIONAL_REPAIR_REQUIRES_ROOT_CAUSE', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, old)
            raise
