"""Real-input GT-free 6-setting engineering closure for DUAL_TRANSDUCT_COEVOL.

Runs on the metadata-first video in EACH of C1..C6 with 0 GT, 0 masks, 0 HOTA.
Verifies exact track-invariance of X_id, fast LOTO observer non-increase,
dual-student co-evolution convergence, regularized ordinary baseline fits,
shared-transduct readout finiteness, and wall-clock/memory safety on GPU 0..3.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/dual_transduct_20261127/engineering'
PLAN = OUT / 'registration.json'


def make_plan():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    parent = json.loads((BASE / 'registration.json').read_text())
    cases, seen = [], set()
    for old in parent['cases']:
        if old['pair'] in seen:
            continue
        seen.add(old['pair'])
        c = dict(old)
        folder = BASE / 'units' / c['pair'] / c['evaluated_video']
        c.update(
            parent=str(folder),
            original_scores=str(folder / 'predictions/UNIFORM_SOFT_KD288/scores.npy'),
            uniform_answers=str(folder / 'uniform_answers.jsonl'),
            uniform_draws=str(folder / 'uniform_draws.json'),
        )
        cases.append(c)
    assert len(cases) == 6 and not guard
    plan = dict(
        status='PREREGISTERED_6SETTING_ENGINEERING_NO_GT',
        cases=cases,
        width=256, dropout=0.2, learning_rate=3e-4, weight_decay=2e-2,
        rounds=3, steps_per_round=[500, 300, 300],
        k_nn=5, kappa=4.0, base_decay=0.85,
        seed=0, GPUs=[0, 1, 2, 3], GPU_memory_fraction=0.20,
        no_GT_masks_or_HOTA=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN, plan)
    return plan


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.dual_transduct_coevol import split_shared_features_np
    from flex_uda.active.dual_transduct_runtime import (
        FastVideoFeatureAccessor,
        learn_dual_transduct_coevol,
        fit_regularized_ordinary_controls,
        apply_shared_transduct_readout,
    )
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    plan = json.loads(PLAN.read_text())
    results = []
    for i, case in enumerate(plan['cases']):
        if i % 4 != gpu:
            continue
        started = time.perf_counter()
        folder = OUT / 'units' / case['pair'] / case['evaluated_video']
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(case['parent']) / 'unlabeled_normalizer.npz', folder / 'unlabeled_normalizer.npz')
        context = Context(case, folder, None, gpu)
        accessor = FastVideoFeatureAccessor(context)
        # Verify exact bit-identical match between FastVideoFeatureAccessor and LazyFeatures
        check_ids = np.unique(np.linspace(0, context.n * context.nq - 1, min(64, context.n * context.nq)).astype(np.int64))
        slow_x = context.features[check_ids]
        fast_x = accessor.full_rows(check_ids)
        assert float(np.max(np.abs(slow_x - fast_x))) == 0.0, 'Fast accessor mismatch!'
        slow_id, slow_dyn = split_shared_features_np(slow_x)
        assert float(np.max(np.abs(slow_dyn - accessor.dyn_rows(check_ids)))) == 0.0
        x_id_all = accessor.all_track_id()
        q_chk, r_chk = np.divmod(check_ids, context.n)
        t_chk = context.video.columns[r_chk]
        assert float(np.max(np.abs(slow_id - x_id_all[q_chk, t_chk]))) == 0.0
        learned = learn_dual_transduct_coevol(context, case, plan, folder / 'primary', device='cuda:0', accessor=accessor)
        reg_scores, reg_receipts = fit_regularized_ordinary_controls(context, case, plan, device='cuda:0', accessor=accessor)
        parent_scores = np.load(case['original_scores'])
        p_shared = apply_shared_transduct_readout(parent_scores, context, learned)
        r_shared = apply_shared_transduct_readout(reg_scores['REG_SOFT_KD'], context, learned)
        for arr in (learned['trans']['frame_prob'], p_shared, r_shared):
            assert arr.shape == (context.nq, context.n) and np.isfinite(arr).all()
            assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0
        rec = dict(
            pair=case['pair'], video=case['evaluated_video'], GPU=gpu,
            protocol=learned['protocol'],
            x_id_track_invariant_exact=True,
            learning_wall_seconds=learned['receipt']['wall_seconds'],
            total_wall_seconds=time.perf_counter() - started,
            rounds=learned['receipt']['rounds'],
            reg_receipts=reg_receipts,
            GT_open_attempts=len(guard),
        )
        atomic_json(folder / 'engineering_done.json', rec)
        results.append(rec)
        torch.cuda.empty_cache()
    assert not guard
    atomic_json(OUT / f'GPU{gpu}_done.json', dict(GPU=gpu, results=results, GT_open_attempts=0))


def main():
    plan = make_plan()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {}
        for gpu in (0, 1, 2, 3):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            proc = subprocess.Popen(
                [sys.executable, '-u', __file__, '--worker', str(gpu)],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            def drain(p):
                for line in p.stdout:
                    print(line, end='', flush=True)
                return p.wait()
            futs[pool.submit(drain, proc)] = gpu
        for fut in concurrent.futures.as_completed(futs):
            code = fut.result()
            if code != 0:
                raise RuntimeError(f'GPU{futs[fut]} failed with code {code}')
    all_recs = []
    for gpu in (0, 1, 2, 3):
        all_recs.extend(json.loads((OUT / f'GPU{gpu}_done.json').read_text())['results'])
    assert len(all_recs) == 6
    summary = dict(
        status='PASS_ALL6_SETTING_DUAL_TRANSDUCT_ENGINEERING_NO_GT',
        videos=len(all_recs),
        wall_seconds=time.perf_counter() - started,
        GT_open_attempts=sum(r['GT_open_attempts'] for r in all_recs),
        results=all_recs,
    )
    atomic_json(OUT / 'completion.json', summary)
    print(json.dumps(dict(status=summary['status'], videos=6, wall_seconds=round(summary['wall_seconds'], 2))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        main()
