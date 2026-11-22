"""Isolate shared support from a generic two-factor task parameterization.

The control receives the same context and current features and has only
one extra width-sized row of parameters. Its first factor can depend on
the current observation. All original numerical source files stay intact.
"""
import argparse
import concurrent.futures
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
from flex_uda.active.observation_learning_inputs import ObservationVideo, prefix_data, DIMENSION
from scripts.a100_conditional_six_setting import OUT as PARENT, PRIMARY
from scripts.a100_conditional_coarse_baseline import STATE as COARSE_STATE
from scripts.a100_six_direction import OUT as SIX, PUBLIC, SEEDS

OUT = SIX/'conditional_shared_support_control_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'conditional_shared_control_registration_v1.json'
STEM = 'conditional_shared_control'
SUCCESS = 'PASS_CONDITIONAL_SHARED_SUPPORT_CONTROL_SIX_SETTINGS_GOAL_ACTIVE'
RECIPE = 'LOCAL_TWO_PROFILE'
METHODS = ('LOCAL_TWO_FINAL', 'LOCAL_TWO_MEAN')
CONTRASTS = {'FULL_vs_'+b:(PRIMARY, b) for b in METHODS}
for b in METHODS:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b] = (PRIMARY+'__AT_'+PRIMARY+'_COUNT', b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_CONTROL_COUNT_vs_'+b] = (PRIMARY+'__AT_'+b+'_COUNT', b+'__AT_'+b+'_COUNT')


def conditions():
    return [dict(pair=p, seed=s, folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def nesting_check():
    import torch
    from flex_uda.active.conditional_observation import BagContextHead
    torch.manual_seed(1701)
    d, w = 9, 8
    shared = BagContextHead(d, w, 'conditional').double()
    local = BagContextHead(d, w, 'local_two_factor').double()
    # Nonzero factors test the real embedding, rather than just equal zero
    # initialization. The local factor's extra current-observation row is 0.
    with torch.no_grad():
        for parameter in shared.parameters():
            parameter.normal_(0, .2)
    old = shared.state_dict(); new = local.state_dict()
    for name in old:
        if name != 'bag_head.weight':
            new[name].copy_(old[name])
    new['bag_head.weight'].zero_()
    new['bag_head.weight'][:, w:].copy_(old['bag_head.weight'])
    local.load_state_dict(new)
    bags = torch.randn(7, 3, d, dtype=torch.float64)
    masks = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0],
        [1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    current = torch.randn(7, d, dtype=torch.float64)
    with torch.no_grad():
        error = float((shared(bags, masks, current)-local(bags, masks, current)).abs().max())
        fresh = [BagContextHead(d, w, kind).double() for kind in ('conditional', 'local_two_factor')]
        initial_error = max(float((torch.sigmoid(m(bags, masks, current))-.5).abs().max()) for m in fresh)
    # Constructor constants are stored in float32 before this replay casts
    # to double. The common 0.5 initialization has that original rounding.
    assert error < 1e-12 and initial_error < 1e-7, (error, initial_error)
    sizes = {k:sum(p.numel() for p in BagContextHead(DIMENSION, 32, k).parameters())
        for k in ('conditional', 'local_two_factor')}
    assert sizes['local_two_factor']-sizes['conditional'] == 32
    return dict(nonzero_shared_function_embedding_max_error=error,
        equal_initial_task_probability_error=initial_error, width32_parameter_counts=sizes,
        control_nests_shared_function_by_zeroing_only_extra_current_row=True,
        no_semantic_factor_identification_or_HOTA_guarantee=True)


def run(pair, seed):
    import torch
    from flex_uda.active import observation_learning_fit as learning
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    # This is an explicit new recipe in this worker's own Python process.
    # No source or existing recipe used by another process is changed.
    assert RECIPE not in learning.RECIPES
    learning.RECIPES[RECIPE] = ('local_two_factor', True, True, False)
    guard = generation_guard(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0); started = time.perf_counter()
    source, target = PAIRS[pair]; md = metadata(target)
    root = OUT/pair/f'seed{seed}'; root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').is_file():
        return
    stream = root/'stream.pt'; previous = None; receipts = []; next_video = 0
    if stream.is_file():
        saved = torch.load(stream, map_location='cpu', weights_only=False)
        previous, receipts, next_video = [saved[k] for k in ('previous', 'receipts', 'next_video')]
    total = np.zeros(DIMENSION); square = total.copy(); count = 0; records = []
    for vi, video in enumerate(sorted(md['videos'])):
        stamp = time.perf_counter(); inputs = ObservationVideo(pair, video)
        values = inputs.raw_bags[:, inputs.present].reshape(-1, DIMENSION).astype(float)
        total += values.sum(0); square += np.square(values).sum(0); count += len(values)
        mean = total/count; std = np.maximum(np.sqrt(np.maximum(square/count-mean**2, 0)), 1e-3)
        evidence = inputs.supervision('DESIGN'); records.append(evidence)
        if vi < next_video:
            continue
        data = prefix_data(records, mean, std)
        atomic_json(root/'progress.json', dict(completed=vi, total=len(md['videos']), video=video, stage='FITTING'))
        model, early, checkpoint, fit_receipt = learning.fit(data, RECIPE, seed=seed,
            mean=mean, std=std, previous=previous)
        local = root/'videos'/video; local.mkdir(parents=True, exist_ok=True)
        nq, no = len(inputs.queries), len(inputs.obs)
        scores = {n:np.lib.format.open_memmap(local/(n+'_scores.npy'), mode='w+', dtype='float32', shape=(nq, no)) for n in METHODS}
        logits = {n:np.lib.format.open_memmap(local/(n+'_logits.npy'), mode='w+', dtype='float32', shape=(nq, no))
            for n in ('LOCAL_TWO_FINAL', 'LOCAL_TWO_EARLY')}
        normalized = ((inputs.raw_bags-mean)/std).astype('float32'); normalized[:, ~inputs.present] = 0
        with torch.no_grad():
            for a in range(0, nq*no, 2048):
                flat = np.arange(a, min(a+2048, nq*no)); qi, rows = flat//no, flat % no; ti = inputs.columns[rows]
                current = ((inputs.dense_rows(qi, rows)-mean)/std).astype('float32')
                bb = torch.as_tensor(normalized[qi, ti], device='cuda:0')
                mm = torch.as_tensor(inputs.present[ti], device='cuda:0'); cc = torch.as_tensor(current, device='cuda:0')
                final_z = learning.forward(model, 'local_two_factor', bb, mm, cc)
                early_z = learning.forward(early, 'local_two_factor', bb, mm, cc)
                final = torch.sigmoid(final_z).cpu().numpy(); first = torch.sigmoid(early_z).cpu().numpy()
                scores['LOCAL_TWO_FINAL'].reshape(-1)[flat] = final
                scores['LOCAL_TWO_MEAN'].reshape(-1)[flat] = .5*(first+final)
                logits['LOCAL_TWO_FINAL'].reshape(-1)[flat] = final_z.cpu().numpy()
                logits['LOCAL_TWO_EARLY'].reshape(-1)[flat] = early_z.cpu().numpy()
        for value in logits.values():
            value.flush()
        for name, score in scores.items():
            score.flush(); bits = np.asarray(score) > .5
            destination = root/name/'videos'/video; destination.mkdir(parents=True, exist_ok=True)
            np.save(destination/'mask.npy', bits)
            immediate_predictions(root/name, type('Context', (), dict(bank=inputs.bank, queries=inputs.queries))(), bits)
        torch.cuda.synchronize()
        receipt = dict(video=video, index=vi, fit=fit_receipt, teacher_evidence=evidence['report'],
            historical_PCD_and_dense_RGB_version_same_as_parent=True,
            current_and_past_only_normalization_rows=count, GT_open_attempts=len(guard),
            wall_seconds=time.perf_counter()-stamp, quality_replay_not_cold_cost=True)
        receipts.append(receipt); previous = checkpoint; next_video = vi+1
        temp = stream.with_suffix('.tmp')
        torch.save(dict(previous=previous, receipts=receipts, next_video=next_video), temp); temp.replace(stream)
        atomic_json(root/'progress.json', dict(completed=next_video, total=len(md['videos']), video=video,
            stage='VIDEO_SEALED_NO_GT', last_arm=RECIPE, completed_arms=1, fit_gate_pass=fit_receipt['target_fit_gate_pass']))
        del scores, logits, model, early, data
        torch.cuda.empty_cache()
    methods = []
    for name in METHODS:
        mask = np.zeros(md['candidate_query_rows'], bool)
        for video in sorted(md['videos']):
            bits = np.load(root/name/'videos'/video/'mask.npy')
            for qi, q in enumerate(q for q in md['episodes'] if q['video_id'] == video):
                mask[q['start']:q['stop']] = bits[qi]
        row = seal_mask(root/name, md, mask, target=target, model=source, method=name,
            GT_open_attempts=len(guard), extra=dict(seed=seed, no_source_logit_residual=True,
                current_video_transductive=True, no_future_answers_or_normalization=True))
        methods.append(dict(method=name, rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json', dict(status='SEALED_LOCAL_TWO_FACTOR_CONTROL_NO_GT',
        methods=methods, receipts=receipts, GT_open_attempts=0,
        target_fit_failures=[dict(video=r['video'], metrics=r['fit']['metrics']) for r in receipts if not r['fit']['target_fit_gate_pass']],
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE'))


def counts(record):
    # Use the original CountMatched implementation and tie rule verbatim.
    # These module globals belong only to this new child process.
    from scripts import a100_conditional_six_setting as parent
    root = Path(record['folder']); original = PARENT/record['pair']/f"seed{record['seed']}"
    for destination, source in [(root/PRIMARY, original/PRIMARY)]+[
        (root/'videos'/v/(PRIMARY+'_scores.npy'), original/'videos'/v/(PRIMARY+'_scores.npy'))
        for v in metadata(PAIRS[record['pair']][1])['videos']]:
        if not destination.is_symlink():
            assert not destination.exists(), destination
            destination.symlink_to(source, target_is_directory=source.is_dir())
        assert destination.resolve() == source.resolve()
    parent.METHODS = (PRIMARY, *METHODS); parent.BASELINES = METHODS
    return parent.counts(record)


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
        paths[PRIMARY] = PARENT/record['pair']/f"seed{record['seed']}"/PRIMARY
        data[slug] = {name:read_stats(path) for name, path in paths.items()}
        for name, row in data[slug].items():
            num, den = row['parent_stats']
            error = max(error, abs(float(pooled_hota((num, den), np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name] = dict(parents=row['parent_videos'], HOTA=row['HOTA'], numerator=num.tolist(), denominator=den.tolist())
        summary[slug] = dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in (PRIMARY, *METHODS)},
            target_fit_failures=raw['target_fit_failures'], CountSelf_changed_query_files=cnt['CountSelf_changed_query_files'])
    repeated = {p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in (PRIMARY, *METHODS)},
        comparisons={label:comparison([data[p+f'_seed{s}'][a] for s in SEEDS], [data[p+f'_seed{s}'][b] for s in SEEDS])
            for label, (a, b) in CONTRASTS.items()}) for p in PAIRS}
    assert error < 1e-8 and len(statistics) == 180, (error, len(statistics))
    result = dict(status='SHARED_SUPPORT_VS_GENERIC_PRODUCT_CONTROL_ANALYZED', primary=PRIMARY,
        control_primary='LOCAL_TWO_FINAL', conditions=summary, repeated=repeated,
        HOTA_parent_sum_max_error=error, shared_parent_primary=True,
        all_own_and_parent_outputs_sealed_before_this_GT=True,
        all_data_development=True, independent_blind_validation=False,
        quality_replay_not_cold_cost=True, latent_factors_not_semantically_identified=True,
        no_new_teacher_information=True, source_files_of_original_studies_unchanged=True, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result, separators=(',', ':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics, separators=(',', ':'))+'\n')


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_COARSE_ORDINARY_GPU_STAGE_RELEASE', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        earlier = json.loads(COARSE_STATE.read_text()) if COARSE_STATE.is_file() else {}
        if earlier.get('status', '').startswith('FAILED_'):
            raise RuntimeError('Resolve the already registered coarse ordinary supplement first')
        if earlier.get('completed', 0) == 18 and not earlier.get('active'):
            break
        time.sleep(15)
    pending = conditions(); done = []; active = {}; failure = None
    def state(status):
        atomic_json(STATE, dict(status=status, pid=os.getpid(), completed=len(done), finished=done,
            pending=pending, active={str(g):j for g, (_, _, j) in active.items()}, failure=failure, goal_status='ACTIVE'))
    while pending or active:
        for gpu, (proc, handle, job) in list(active.items()):
            if proc.poll() is None:
                continue
            handle.close(); del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():
                failure = dict(job=job, exit_code=proc.returncode, last_log=Path(job['log']).read_text()[-8000:])
            else:
                done.append(job)
        free = [g for g in (0, 1, 2, 3) if g not in active]
        if pending and free and not failure:
            text = subprocess.check_output(['nvidia-smi', '-i', ','.join(map(str, free)),
                '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()}
            for gpu in free:
                if not pending or memory[gpu] >= 512:
                    continue
                job = pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():
                    done.append(dict(job, reused_sealed=True)); continue
                log = STATE.parent/(job['pair']+'_seed'+str(job['seed'])+'.log'); handle = log.open('a', buffering=1)
                proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--pair', job['pair'], '--seed', str(job['seed'])],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4'))
                active[gpu] = proc, handle, dict(job, pid=proc.pid, log=str(log), physical_GPU=gpu)
        state('RUNNING_SHARED_SUPPORT_CONTROL_NO_GT' if not failure else 'PRESERVING_HEALTHY_SHARED_CONTROL_WORKERS_AFTER_FAILURE')
        if failure and not active:
            raise RuntimeError(json.dumps(failure))
        if pending or active:
            time.sleep(5)
    state('ALL18_SHARED_CONTROL_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(counts, conditions()))
    while not (PARENT/'all_predictions_sealed_before_GT.json').is_file():
        time.sleep(15)
    tasks = []
    for record, count_root in zip(conditions(), roots):
        for folder in (Path(record['folder']), Path(count_root)):
            row = json.loads((folder/'completion.json').read_text()); assert row['GT_open_attempts'] == 0
            tasks.extend((str(folder/r['method']), record['pair']) for r in row['methods'])
    assert len(tasks) == 162
    atomic_json(OUT/'all_predictions_sealed_before_GT.json', dict(methods=162, parent_methods=1692, GT_open_attempts=0))
    state('ALL162_SHARED_CONTROL_AND1692_PARENT_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    while not (PARENT/'official.json').is_file():
        time.sleep(15)
    atomic_json(OUT/'official.json', dict(methods=162, results=results)); analyze(); state(SUCCESS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pair', choices=tuple(PAIRS)); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nesting-check', action='store_true'); args = parser.parse_args()
    if args.nesting_check:
        print(json.dumps(nesting_check()))
    elif args.pair:
        import fcntl
        folder = OUT/args.pair/f'seed{args.seed}'; folder.mkdir(parents=True, exist_ok=True)
        with (folder/'condition.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); run(args.pair, args.seed)
    else:
        try:
            main()
        except Exception as error:
            old = json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_SHARED_SUPPORT_CONTROL_REQUIRES_ROOT_CAUSE', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, old); raise
