"""Full 148-unit 3-innovation study for UNIFIED_RSI_RMOT_V2 (IWAL-Debiased I1+I2+I3).

Addresses active selection bias empirically diagnosed from UNIFIED_RSI_RMOT_V1:
1. Classical IWAL (Beygelzimer et al. ICML 2009 / Farquhar et al. ICLR 2021):
   - Stage 0 (35% budget): Pure visual-clique coverage via sequential rank-1
     Sherman-Morrison DeltaLev + confident PRESENCE_COVER only when max(p0) > 0.65.
   - Stages 1-3 (65% budget): Strictly positive mixture proposal (eps=0.30 uniform floor,
     temp=2.0) logging exact propensities & bounded Horvitz-Thompson IWAL weights
     w_i = clip((1/|V_t|)/q_t(k_t), 0.35, 2.50) BEFORE calling frozen Qwen3-VL-2B.
2. IWAL-debiased S_id, S_ctx, and balanced OOF-LOTO + stricter correction-only R1/R2 gate
   ((lev > 0.15) & (|p* - mu| > 0.08), transduced_weight=0.18).
3. Seven fair comparators (including IWAL-weighted ordinary KD ACTIVE_IWAL_SOFT_POOLED
   so ordinary KD is never disadvantaged!), 2,960 outputs sealed under GT_open_attempts==0,
   followed by verified 0.0-error 20-worker fast_evaluation_reuse + independent reproduction.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
V1_CACHE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_rmot_20261128/primary/teacher_cache/actual_inputs.sqlite'
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_rmot_v2_20261128/primary'
PLAN = OUT / 'registration.json'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'unified_rsi_rmot_v2_20261128'
PRIMARY = 'UNIFIED_RSI_RMOT_V2'
UNIFORM_ORDINARY_CONTROLS = ('UNIFORM_PARENT_SOFT_KD', 'UNIFORM_PARENT_POOLED')
ACTIVE_ORDINARY_CONTROLS = ('ACTIVE_SOFT_KD', 'ACTIVE_HARD_KD', 'ACTIVE_SOFT_POOLED', 'ACTIVE_IWAL_SOFT_POOLED')
ORDINARY_CONTROLS = UNIFORM_ORDINARY_CONTROLS + ACTIVE_ORDINARY_CONTROLS
SHARED_CONTROLS = ('ACTIVE_SHARED_TRANSDUCT',)
COMPARATORS = ORDINARY_CONTROLS + SHARED_CONTROLS
NEW_RAW_METHODS = (PRIMARY, 'ACTIVE_SOFT_KD', 'ACTIVE_HARD_KD', 'ACTIVE_SOFT_POOLED',
                   'ACTIVE_IWAL_SOFT_POOLED', 'ACTIVE_SHARED_TRANSDUCT')
SOURCES = (
    'flex_uda/active/dual_transduct_coevol.py',
    'flex_uda/active/dual_transduct_runtime.py',
    'flex_uda/active/dual_transduct_v2.py',
    'flex_uda/active/dual_transduct_v3.py',
    'flex_uda/active/dual_transduct_v3_runtime.py',
    'flex_uda/active/unified_rsi_core.py',
    'flex_uda/active/unified_rsi_runtime.py',
    'flex_uda/active/unified_rsi_v2_core.py',
    'flex_uda/active/unified_rsi_v2_runtime.py',
    'flex_uda/active/fullscope_visual_runtime.py',
    'flex_uda/teacher/teacher_cache.py',
    'flex_uda/teacher/decompose_inputs.py',
    'scripts/a100_unified_rsi_v2_contract.py',
    'scripts/a100_unified_rsi_v2_primary.py',
    'scripts/a100_unified_rsi_v2_statistics.py',
    'scripts/a100_factor_rsi_primary_quality.py',
    'flex_uda/presence/predictions.py',
    'flex_uda/a100/fast_evaluation_reuse.py',
)


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    if PLAN.exists():
        raise FileExistsError(PLAN)
    parent = json.loads((BASE / 'registration.json').read_text())
    contract = ROOT / 'a100_runs/visual_classical_20261121/supervisor/unified_rsi_v2_contract_20261128/result.json'
    assert json.loads(contract.read_text())['status'] == 'PASS_UNIFIED_RSI_RMOT_V2_MATHEMATICAL_CONTRACT'
    old_provider = PUBLIC / 'fullscope_visual_six_20261126_statistics.json'
    assert old_provider.is_file()

    # Copy isolated exact-input Qwen cache (preferring enriched V1 cache, never mutating V1/BASE)
    src_cache = V1_CACHE if V1_CACHE.is_file() else (BASE / 'teacher_cache/actual_inputs.sqlite')
    dst_cache = OUT / 'teacher_cache/actual_inputs.sqlite'
    dst_cache.parent.mkdir(parents=True, exist_ok=True)
    if src_cache.is_file() and not dst_cache.exists():
        # Safe SQLite online backup so WAL frames are cleanly consolidated
        import sqlite3
        src_conn = sqlite3.connect(f'file:{src_cache}?mode=ro', uri=True)
        dst_conn = sqlite3.connect(dst_cache)
        with dst_conn:
            src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()

    cases = []
    paths = {BASE / 'registration.json', BASE / 'all4588_sealed_before_GT.json', contract, old_provider}
    for old in parent['cases']:
        case = dict(old)
        folder = BASE / 'units' / case['pair'] / case['evaluated_video']
        case.update(
            parent=str(folder),
            original_scores=str(folder / 'predictions/UNIFORM_SOFT_KD288/scores.npy'),
            original_mask=str(folder / 'predictions/UNIFORM_SOFT_KD288/mask.npy'),
            original_pooled_scores=str(folder / 'predictions/UNIFORM_POOLED/scores.npy'),
            original_pooled_mask=str(folder / 'predictions/UNIFORM_POOLED/mask.npy'),
        )
        for k in ('original_scores', 'original_mask', 'original_pooled_scores', 'original_pooled_mask',
                  'observations', 'temporal_queries', 'auxiliary_path'):
            paths.add(Path(case[k]))
        paths.update((folder / 'unlabeled_normalizer.npz', folder / 'generation_complete.json'))
        cases.append(case)
    assert len(cases) == 148 and len({c['pair'] for c in cases}) == 6

    source_sha = {name: file_sha(ROOT / name) for name in SOURCES}
    plan = dict(
        status='PREREGISTERED_ALL148_UNIFIED_RSI_RMOT_V2_PRIMARY_BEFORE_FITS',
        primary=PRIMARY,
        uniform_ordinary_controls=UNIFORM_ORDINARY_CONTROLS,
        active_ordinary_controls=ACTIVE_ORDINARY_CONTROLS,
        ordinary_controls=ORDINARY_CONTROLS,
        shared_controls=SHARED_CONTROLS,
        comparators=COMPARATORS,
        new_raw_methods=NEW_RAW_METHODS,
        old_statistics_provider=old_provider.name,
        old_statistics_provider_sha256=file_sha(old_provider),
        frozen_teacher_weight_sha256='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956',
        cases=cases,
        source_sha256=source_sha,
        input_sha256={str(p): file_sha(p) for p in sorted(paths)},
        contract_sha256=file_sha(contract),
        width=256, dropout=0.10, learning_rate=8e-4, weight_decay=1e-3,
        rounds=3, oof_max_steps=2400, r0_max_steps=4800, coevol_max_steps=1200,
        ordinary_max_steps=4800, target_excess_ce=0.01,
        k_nn=3, dist_gate_grid=[0.40, 0.55, 0.75, 1.00],
        eps_uniform=0.30, challenger_temp=2.0,
        leverage_threshold=0.15, correction_delta_threshold=0.08, transduced_weight=0.18,
        kappa=6.0, base_decay=0.94,
        seed=0, GPUs=[0, 1, 2, 3], GPU_memory_fraction=0.35,
        existing_GPU2_service_preserved=True,
        new_outputs_per_unit=20, total_new_outputs=2960,
        whole_setting_HOTA_records=132, paired_effects=126,
        eval_workers=20,
        all_data_DEV=True, manual_GT_never_used_for_adaptation=True,
        goal_status='ACTIVE', goal_complete=False,
    )
    assert not guard
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN, plan)
    compact = {k: v for k, v in plan.items() if k not in ('cases', 'input_sha256')}
    compact.update(
        video_units=148,
        setting_video_counts={p: sum(c['pair'] == p for c in cases) for p in sorted({c['pair'] for c in cases})},
    )
    atomic_json(PUBLIC / (STEM + '_registration.json'), compact)
    print(json.dumps(dict(status=plan['status'], units=148, new_outputs=2960, pins=len(source_sha))), flush=True)


def read_plan():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT / name) == sha, name
    plan['registration_sha256'] = file_sha(PLAN)
    return plan


def probability_scores(probability):
    import numpy as np
    p = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return (np.log(np.maximum(p, np.finfo(float).tiny)) -
            np.log1p(-np.minimum(p, np.nextafter(1., 0.)))).astype(np.float32)


def run_unit(case, dispatcher, gpu, plan, guard):
    import numpy as np
    from scipy.special import expit
    from flex_uda.a100.mechanism import PAIRS
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.dual_transduct_runtime import FastVideoFeatureAccessor
    from flex_uda.active.dual_transduct_v3_runtime import apply_shared_transduct_v3_readout
    from flex_uda.active.unified_rsi_v2_runtime import (
        fit_active_ordinary_controls_v2, learn_unified_rsi_rmot_v2,
    )
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from scripts.a100_factor_rsi_primary_quality import projection, subset_metadata

    folder = OUT / 'units' / case['pair'] / case['evaluated_video']
    folder.mkdir(parents=True, exist_ok=True)
    finished = folder / 'generation_complete.json'
    if finished.exists():
        done = json.loads(finished.read_text())
        assert done['registration_sha256'] == plan['registration_sha256']
        assert len(done['tasks']) == 20 and done['GT_open_attempts'] == 0
        for task in done['tasks']:
            verify_predictions(task['folder'])
        return done

    started = time.perf_counter()
    shutil.copyfile(Path(case['parent']) / 'unlabeled_normalizer.npz', folder / 'unlabeled_normalizer.npz')
    context = Context(case, folder, dispatcher, gpu)
    accessor = FastVideoFeatureAccessor(context)

    learned = learn_unified_rsi_rmot_v2(context, case, plan, folder / 'primary', device='cuda:0', accessor=accessor)
    ord_started = time.perf_counter()
    act_scores, act_receipts = fit_active_ordinary_controls_v2(
        context, learned['events'], plan, device='cuda:0', accessor=accessor)
    ord_seconds = time.perf_counter() - ord_started

    pred_started = time.perf_counter()
    parent_soft = np.load(case['original_scores'])
    parent_pool = np.load(case['original_pooled_scores'])
    parent_soft_mask = np.load(case['original_mask'])
    parent_pool_mask = np.load(case['original_pooled_mask'])
    assert np.array_equal(parent_soft > 0, parent_soft_mask)
    assert np.array_equal(parent_pool > 0, parent_pool_mask)

    def pool_track(sc):
        out = np.empty_like(sc)
        for track in np.unique(context.video.obs[:, 1]):
            at = np.flatnonzero(context.video.obs[:, 1] == track)
            out[:, at] = probability_scores(expit(sc[:, at].astype(float)).mean(1))[:, None]
        return out

    act_pooled = pool_track(act_scores['ACTIVE_SOFT_KD'])
    act_iwal_pooled = pool_track(act_scores['ACTIVE_IWAL_SOFT_KD'])
    act_shared_prob = apply_shared_transduct_v3_readout(act_scores['ACTIVE_IWAL_SOFT_KD'], context, learned, plan)

    scores = {
        PRIMARY: probability_scores(learned['trans']['frame_prob']),
        'UNIFORM_PARENT_SOFT_KD': parent_soft,
        'UNIFORM_PARENT_POOLED': parent_pool,
        'ACTIVE_SOFT_KD': act_scores['ACTIVE_SOFT_KD'],
        'ACTIVE_HARD_KD': act_scores['ACTIVE_HARD_KD'],
        'ACTIVE_SOFT_POOLED': act_pooled,
        'ACTIVE_IWAL_SOFT_POOLED': act_iwal_pooled,
        'ACTIVE_SHARED_TRANSDUCT': probability_scores(act_shared_prob),
    }
    masks = {name: val > 0 for name, val in scores.items()}
    md = subset_metadata(case['target'], case['evaluated_video'])
    tasks = []

    def seal(name, mask, extra):
        dest = folder / 'predictions' / name
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'mask.npy', mask)
        seal_mask(dest, md, mask.flatten(), target=case['target'], model=PAIRS[case['pair']][0],
                  method=name, GT_open_attempts=len(guard),
                  extra=dict(seed=0, registration_sha256=plan['registration_sha256'], **extra))
        tasks.append(dict(pair=case['pair'], video=case['evaluated_video'], method=name, folder=str(dest)))

    for name in NEW_RAW_METHODS:
        assert np.isfinite(scores[name]).all()
        dest = folder / 'predictions' / name
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'scores.npy', scores[name])
        seal(name, masks[name], dict(same_budget_active_iwal=True))

    for rank, count in [(PRIMARY, c) for c in COMPARATORS] + [(c, PRIMARY) for c in COMPARATORS]:
        name = rank + '__AT_' + count + '_COUNT'
        mask = projection(scores[rank], masks[count], context.video.obs)
        seal(name, mask, dict(rank_method=rank, count_method=count, counts_from_GT=False, counts_equal_per_query_frame=True))

    assert len(tasks) == 20 and not guard
    receipt = dict(
        status='SEALED_ALL20_UNIFIED_RSI_RMOT_V2_VIDEO_OUTPUTS',
        pair=case['pair'], video=case['evaluated_video'],
        tasks=tasks, registration_sha256=plan['registration_sha256'],
        GT_open_attempts=0, neural_seed=0,
        primary_fit=learned['receipt'], ordinary_fits=act_receipts,
        ordinary_phase_wall_seconds=ord_seconds,
        prediction_seal_phase_wall_seconds=time.perf_counter() - pred_started,
        total_wall_seconds=time.perf_counter() - started,
        prepaid_front_and_aux_not_fresh_cold_cost=True,
    )
    atomic_json(finished, receipt)
    print(json.dumps(dict(status=receipt['status'], pair=case['pair'], video=case['evaluated_video'],
                          GPU=gpu, outputs=20, sec=round(receipt['total_wall_seconds'], 1))), flush=True)
    return receipt


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.fullscope_visual_runtime import TeacherDispatcher

    guard = generation_guard()
    plan = read_plan()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(float(plan['GPU_memory_fraction']), 0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    (OUT / 'teacher_cache').mkdir(parents=True, exist_ok=True)
    dispatcher = TeacherDispatcher(ROOT, OUT / 'teacher_cache/actual_inputs.sqlite')
    ordered = sorted(plan['cases'], key=lambda c: (-int(c['logical_budget']), c['pair'], c['evaluated_video']))
    completed = []
    try:
        for case in ordered:
            folder = OUT / 'units' / case['pair'] / case['evaluated_video']
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / 'generation_complete.json').exists():
                continue
            with (folder / 'unit.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (folder / 'generation_complete.json').exists():
                    continue
                atomic_json(folder / 'assignment.json', dict(
                    GPU=gpu, pid=os.getpid(), registration_sha256=plan['registration_sha256']))
                done = run_unit(case, dispatcher, gpu, plan, guard)
                completed.append(dict(pair=case['pair'], video=case['evaluated_video'], outputs=len(done['tasks'])))
                torch.cuda.empty_cache()
    finally:
        sealed_teacher = dispatcher.close()
        assert not guard
        atomic_json(OUT / 'supervisor' / f'GPU{gpu}_completion.json', dict(
            status='COMPLETE_UNIFIED_RSI_RMOT_V2_GPU_WORKER', GPU=gpu, execution=execution,
            teacher_final=sealed_teacher, completed=completed, GT_open_attempts=len(guard),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            registration_sha256=plan['registration_sha256'],
        ))


def evaluate_and_collect(plan, tasks):
    from flex_uda.a100.fast_evaluation_reuse import evaluate_many
    from scripts.a100_unified_rsi_v2_statistics import collect, reproduce
    assert len(tasks) == 2960
    summary = evaluate_many(tasks, max_workers=int(plan.get('eval_workers', 20)), use_cache=True)
    print(json.dumps(summary), flush=True)
    collect()
    reproduce()


def main():
    plan = read_plan()
    sup = OUT / 'supervisor'
    sup.mkdir(parents=True, exist_ok=True)
    lock = (sup / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    barrier = OUT / 'all2960_sealed_before_GT.json'
    if not barrier.exists():
        def drain(process, path):
            with path.open('a', buffering=1) as stream:
                for line in process.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
            return process.wait()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for gpu in (0, 1, 2, 3):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                proc = subprocess.Popen(
                    [sys.executable, '-u', __file__, '--worker', str(gpu)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                futures[pool.submit(drain, proc, sup / f'GPU{gpu}_{proc.pid}.log')] = gpu
                atomic_json(sup / f'GPU{gpu}_launch.json', dict(
                    GPU=gpu, pid=proc.pid, registration_sha256=plan['registration_sha256'], time_unix=time.time()))
                print(json.dumps(dict(status='STARTED_FULL_UNIFIED_RSI_RMOT_V2', GPU=gpu, pid=proc.pid)), flush=True)
            failed = []
            for fut in concurrent.futures.as_completed(futures):
                code = fut.result()
                gpu = futures[fut]
                print(json.dumps(dict(status='UNIFIED_RSI_RMOT_V2_WORKER_EXIT', GPU=gpu, exit_code=code)), flush=True)
                if code:
                    failed.append(dict(GPU=gpu, exit_code=code))
        if failed:
            atomic_json(sup / f'failure_{int(time.time())}.json', dict(failed=failed, preserve_completed_units=True))
            raise RuntimeError(failed)
        from flex_uda.presence.predictions import verify_predictions
        tasks = []
        for case in plan['cases']:
            path = OUT / 'units' / case['pair'] / case['evaluated_video'] / 'generation_complete.json'
            done = json.loads(path.read_text())
            assert done['registration_sha256'] == plan['registration_sha256'] and done['GT_open_attempts'] == 0
            for task in done['tasks']:
                verify_predictions(task['folder'])
            tasks.extend(done['tasks'])
        assert len(tasks) == 2960
        atomic_json(barrier, dict(
            status='ALL2960_UNIFIED_RSI_RMOT_V2_OUTPUTS_SEALED_BEFORE_GT',
            tasks=tasks, GT_open_attempts=0, registration_sha256=plan['registration_sha256'],
        ))
    sealed = json.loads(barrier.read_text())
    assert sealed['GT_open_attempts'] == 0 and sealed['registration_sha256'] == plan['registration_sha256']
    read_plan()
    print(json.dumps(dict(status='ALL2960_SEALED_START_FAST_20WORKER_OFFICIAL_EVAL', outputs=2960)), flush=True)
    evaluate_and_collect(plan, sealed['tasks'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker is not None:
        worker(args.worker)
    else:
        main()
