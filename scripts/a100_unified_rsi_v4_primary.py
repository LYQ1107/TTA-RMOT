"""Full 148-unit 3-innovation study for UNIFIED_RSI_RMOT_V4 (Two-Stream Convex OOF + Godel-Machine RSI).

Replays the exact 55,019 sealed V3 Hybrid Challenger active teacher responses (0 new Qwen forwards, 0 GT).
Produces 14 new outputs per video unit (2,072 total) and automatically runs 20-worker 0.0-error official
evaluation, whole-setting collection, and independent public-only reproduction.
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
V3_OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_rmot_v3_20261128/primary'
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_rmot_v4_20261128/primary'
PLAN = OUT / 'registration.json'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'unified_rsi_rmot_v4_20261128'
PRIMARY = 'UNIFIED_RSI_RMOT_V4'
UNIFORM_ORDINARY_CONTROLS = ('UNIFORM_PARENT_SOFT_KD', 'UNIFORM_PARENT_POOLED')
ACTIVE_ORDINARY_CONTROLS = ('ACTIVE_SOFT_KD', 'ACTIVE_HARD_KD', 'ACTIVE_SOFT_POOLED')
ORDINARY_CONTROLS = UNIFORM_ORDINARY_CONTROLS + ACTIVE_ORDINARY_CONTROLS
SHARED_CONTROLS = ('ACTIVE_SHARED_TRANSDUCT',)
COMPARATORS = ORDINARY_CONTROLS + SHARED_CONTROLS
NEW_RAW_METHODS = (PRIMARY, 'ACTIVE_SHARED_TRANSDUCT')
SOURCES = (
    'flex_uda/active/dual_transduct_coevol.py',
    'flex_uda/active/dual_transduct_runtime.py',
    'flex_uda/active/dual_transduct_v2.py',
    'flex_uda/active/dual_transduct_v3.py',
    'flex_uda/active/dual_transduct_v3_runtime.py',
    'flex_uda/active/unified_rsi_core.py',
    'flex_uda/active/unified_rsi_runtime.py',
    'flex_uda/active/unified_rsi_v4_runtime.py',
    'flex_uda/active/fullscope_visual_runtime.py',
    'scripts/a100_unified_rsi_v4_contract.py',
    'scripts/a100_unified_rsi_v4_primary.py',
    'scripts/a100_unified_rsi_v4_statistics.py',
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
    contract = ROOT / 'a100_runs/visual_classical_20261121/supervisor/unified_rsi_v4_contract_20261128/result.json'
    assert json.loads(contract.read_text())['status'] == 'PASS_UNIFIED_RSI_RMOT_V4_MATHEMATICAL_CONTRACT'
    old_provider = PUBLIC / 'fullscope_visual_six_20261126_statistics.json'
    v3_provider = PUBLIC / 'unified_rsi_rmot_v3_20261128_statistics.json'
    assert old_provider.is_file() and v3_provider.is_file()

    cases = []
    paths = {
        BASE / 'registration.json', BASE / 'all4588_sealed_before_GT.json',
        V3_OUT / 'all2516_sealed_before_GT.json',
        contract, old_provider, v3_provider,
    }
    for old in parent['cases']:
        case = dict(old)
        folder = BASE / 'units' / case['pair'] / case['evaluated_video']
        v3_folder = V3_OUT / 'units' / case['pair'] / case['evaluated_video']
        case.update(
            parent=str(folder),
            original_scores=str(folder / 'predictions/UNIFORM_SOFT_KD288/scores.npy'),
            original_mask=str(folder / 'predictions/UNIFORM_SOFT_KD288/mask.npy'),
            original_pooled_scores=str(folder / 'predictions/UNIFORM_POOLED/scores.npy'),
            original_pooled_mask=str(folder / 'predictions/UNIFORM_POOLED/mask.npy'),
            v3_active_answers=str(v3_folder / 'active_answers.jsonl'),
            v3_active_soft_scores=str(v3_folder / 'predictions/ACTIVE_SOFT_KD/scores.npy'),
            v3_active_soft_mask=str(v3_folder / 'predictions/ACTIVE_SOFT_KD/mask.npy'),
            v3_active_hard_scores=str(v3_folder / 'predictions/ACTIVE_HARD_KD/scores.npy'),
            v3_active_hard_mask=str(v3_folder / 'predictions/ACTIVE_HARD_KD/mask.npy'),
            v3_active_pooled_scores=str(v3_folder / 'predictions/ACTIVE_SOFT_POOLED/scores.npy'),
            v3_active_pooled_mask=str(v3_folder / 'predictions/ACTIVE_SOFT_POOLED/mask.npy'),
        )
        for k in ('original_scores', 'original_mask', 'original_pooled_scores', 'original_pooled_mask',
                  'v3_active_answers', 'v3_active_soft_scores', 'v3_active_soft_mask',
                  'v3_active_hard_scores', 'v3_active_hard_mask',
                  'v3_active_pooled_scores', 'v3_active_pooled_mask',
                  'observations', 'temporal_queries', 'auxiliary_path'):
            paths.add(Path(case[k]))
        paths.update((folder / 'unlabeled_normalizer.npz', folder / 'generation_complete.json'))
        cases.append(case)
    assert len(cases) == 148 and len({c['pair'] for c in cases}) == 6

    source_sha = {name: file_sha(ROOT / name) for name in SOURCES}
    plan = dict(
        status='PREREGISTERED_ALL148_UNIFIED_RSI_RMOT_V4_PRIMARY_BEFORE_FITS',
        primary=PRIMARY,
        uniform_ordinary_controls=UNIFORM_ORDINARY_CONTROLS,
        active_ordinary_controls=ACTIVE_ORDINARY_CONTROLS,
        ordinary_controls=ORDINARY_CONTROLS,
        shared_controls=SHARED_CONTROLS,
        comparators=COMPARATORS,
        new_raw_methods=NEW_RAW_METHODS,
        old_statistics_provider=old_provider.name,
        old_statistics_provider_sha256=file_sha(old_provider),
        v3_statistics_provider=v3_provider.name,
        v3_statistics_provider_sha256=file_sha(v3_provider),
        frozen_teacher_weight_sha256='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956',
        fixed_v3_active_replay_zero_new_qwen=True,
        cases=cases,
        source_sha256=source_sha,
        input_sha256={str(p): file_sha(p) for p in sorted(paths)},
        contract_sha256=file_sha(contract),
        width=256, dropout=0.10, learning_rate=8e-4, weight_decay=1e-3,
        rounds=3, oof_max_steps=2400, r0_max_steps=4800, coevol_max_steps=1200,
        target_excess_ce=0.01,
        k_nn=3, dist_gate_grid=[0.40, 0.55, 0.75, 1.00],
        eb_graph_delta_0=0.020,
        leverage_threshold=0.15, transduced_weight=0.18,
        kappa=6.0, base_decay=0.94,
        seed=0, GPUs=[0, 1, 2, 3], GPU_memory_fraction=0.20,
        existing_GPU2_service_preserved=True,
        new_outputs_per_unit=14, total_new_outputs=2072,
        whole_setting_HOTA_records=114, paired_effects=108,
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
    print(json.dumps(dict(status=plan['status'], units=148, new_outputs=2072, pins=len(source_sha))), flush=True)


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


def run_unit(case, gpu, plan, guard):
    import numpy as np
    from flex_uda.a100.mechanism import PAIRS
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.dual_transduct_runtime import FastVideoFeatureAccessor
    from flex_uda.active.unified_rsi_v4_runtime import (
        apply_shared_transduct_v4_readout,
        learn_unified_rsi_rmot_v4,
    )
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from scripts.a100_factor_rsi_primary_quality import projection, subset_metadata

    folder = OUT / 'units' / case['pair'] / case['evaluated_video']
    folder.mkdir(parents=True, exist_ok=True)
    finished = folder / 'generation_complete.json'
    if finished.exists():
        done = json.loads(finished.read_text())
        assert done['registration_sha256'] == plan['registration_sha256']
        assert len(done['tasks']) == 14 and done['GT_open_attempts'] == 0
        for task in done['tasks']:
            verify_predictions(task['folder'])
        return done

    started = time.perf_counter()
    shutil.copyfile(Path(case['parent']) / 'unlabeled_normalizer.npz', folder / 'unlabeled_normalizer.npz')
    context = Context(case, folder, None, gpu)
    accessor = FastVideoFeatureAccessor(context)

    learned = learn_unified_rsi_rmot_v4(context, case, plan, folder / 'primary', device='cuda:0', accessor=accessor)

    pred_started = time.perf_counter()
    parent_soft = np.load(case['original_scores'])
    parent_pool = np.load(case['original_pooled_scores'])
    assert np.array_equal(parent_soft > 0, np.load(case['original_mask']))
    assert np.array_equal(parent_pool > 0, np.load(case['original_pooled_mask']))

    act_soft = np.load(case['v3_active_soft_scores'])
    act_hard = np.load(case['v3_active_hard_scores'])
    act_pool = np.load(case['v3_active_pooled_scores'])
    assert np.array_equal(act_soft > 0, np.load(case['v3_active_soft_mask']))
    assert np.array_equal(act_hard > 0, np.load(case['v3_active_hard_mask']))
    assert np.array_equal(act_pool > 0, np.load(case['v3_active_pooled_mask']))

    act_shared_prob = apply_shared_transduct_v4_readout(act_soft, context, learned, plan)

    scores = {
        PRIMARY: probability_scores(learned['trans']['frame_prob']),
        'UNIFORM_PARENT_SOFT_KD': parent_soft,
        'UNIFORM_PARENT_POOLED': parent_pool,
        'ACTIVE_SOFT_KD': act_soft,
        'ACTIVE_HARD_KD': act_hard,
        'ACTIVE_SOFT_POOLED': act_pool,
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
        seal(name, masks[name], dict(same_budget_active_v3_replay=True))

    for rank, count in [(PRIMARY, c) for c in COMPARATORS] + [(c, PRIMARY) for c in COMPARATORS]:
        name = rank + '__AT_' + count + '_COUNT'
        mask = projection(scores[rank], masks[count], context.video.obs)
        seal(name, mask, dict(rank_method=rank, count_method=count, counts_from_GT=False, counts_equal_per_query_frame=True))

    assert len(tasks) == 14 and not guard
    receipt = dict(
        status='SEALED_ALL14_UNIFIED_RSI_RMOT_V4_VIDEO_OUTPUTS',
        pair=case['pair'], video=case['evaluated_video'],
        tasks=tasks, registration_sha256=plan['registration_sha256'],
        GT_open_attempts=0, neural_seed=0,
        primary_fit=learned['receipt'],
        prediction_seal_phase_wall_seconds=time.perf_counter() - pred_started,
        total_wall_seconds=time.perf_counter() - started,
        fixed_v3_active_replay_zero_new_qwen=True,
    )
    atomic_json(finished, receipt)
    print(json.dumps(dict(status=receipt['status'], pair=case['pair'], video=case['evaluated_video'],
                          GPU=gpu, outputs=14, sec=round(receipt['total_wall_seconds'], 1))), flush=True)
    return receipt


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard

    guard = generation_guard()
    plan = read_plan()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(float(plan['GPU_memory_fraction']), 0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

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
                done = run_unit(case, gpu, plan, guard)
                completed.append(dict(pair=case['pair'], video=case['evaluated_video'], outputs=len(done['tasks'])))
                torch.cuda.empty_cache()
    finally:
        assert not guard
        atomic_json(OUT / 'supervisor' / f'GPU{gpu}_completion.json', dict(
            status='COMPLETE_UNIFIED_RSI_RMOT_V4_GPU_WORKER', GPU=gpu, execution=execution,
            completed=completed, GT_open_attempts=len(guard),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            registration_sha256=plan['registration_sha256'],
        ))


def evaluate_and_collect(plan, tasks):
    from flex_uda.a100.fast_evaluation_reuse import evaluate_many
    from scripts.a100_unified_rsi_v4_statistics import collect, reproduce
    assert len(tasks) == 2072
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
    barrier = OUT / 'all2072_sealed_before_GT.json'

    if not barrier.exists():
        procs = []
        for gpu in plan['GPUs']:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
            proc = subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(gpu)], env=env)
            procs.append((gpu, proc))
            atomic_json(sup / f'GPU{gpu}_launch.json', dict(GPU=gpu, pid=proc.pid, started_unix=time.time()))
            print(json.dumps(dict(status='STARTED_FULL_UNIFIED_RSI_RMOT_V4', GPU=gpu, pid=proc.pid)), flush=True)
        codes = {}
        for gpu, proc in procs:
            codes[gpu] = proc.wait()
            print(json.dumps(dict(status='UNIFIED_RSI_RMOT_V4_WORKER_EXIT', GPU=gpu, exit_code=codes[gpu])), flush=True)
        assert all(c == 0 for c in codes.values()), codes

        from flex_uda.presence.predictions import verify_predictions
        tasks = []
        for case in plan['cases']:
            done = json.loads((OUT / 'units' / case['pair'] / case['evaluated_video'] / 'generation_complete.json').read_text())
            assert done['registration_sha256'] == plan['registration_sha256'] and done['GT_open_attempts'] == 0
            tasks.extend(done['tasks'])
        assert len(tasks) == 2072
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda t: verify_predictions(t['folder']), tasks))
        atomic_json(barrier, dict(
            status='ALL2072_UNIFIED_RSI_RMOT_V4_SEALED_BEFORE_GT',
            registration_sha256=plan['registration_sha256'],
            GT_open_attempts=0, tasks=tasks,
        ))
    else:
        tasks = json.loads(barrier.read_text())['tasks']

    print(json.dumps(dict(status='ALL2072_SEALED_START_FAST_20WORKER_OFFICIAL_EVAL', outputs=len(tasks))), flush=True)
    evaluate_and_collect(plan, tasks)


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
