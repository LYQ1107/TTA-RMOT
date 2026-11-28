"""Full 148-unit fixed-evidence primary study for DUAL_TRANSDUCT_COEVOL.

Tests dual-level slow-identity x fast-dynamic hierarchical transduction with
predicate-aware cross-track visual-spatial affinity graphs (OOF LOTO learned,
0 GT) and regularized complementary dual-student co-evolution (S_id, S_dyn).

Fairness & isolation:
- Same target frames, official candidates/Track IDs, 9352D features, 100 AUX,
  and fixed object-balanced UNIFORM paid Qwen answers (0 new Qwen calls).
- Includes retained adequate 6k PARENT_SOFT_KD & PARENT_POOLED (exact immutable
  provider references for raw HOTA + new dual CountMatched), same-regularization
  REG_SOFT_KD / REG_HARD_KD / REG_SOFT_POOLED, and privileged shared-readout
  controls PARENT_SHARED_TRANSDUCT & REG_SHARED_TRANSDUCT.
- All 2960 new outputs seal with GT_open_attempts==0 before isolated TrackEval.
- All data are DEV; fixed seed0; GPUs 0..3 with 0.20 memory fraction.
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
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/dual_transduct_20261127/primary'
PLAN = OUT / 'registration.json'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'dual_transduct_primary_20261127'
PRIMARY = 'DUAL_TRANSDUCT_COEVOL'
ORDINARY_CONTROLS = ('PARENT_SOFT_KD', 'PARENT_POOLED', 'REG_SOFT_KD', 'REG_HARD_KD', 'REG_SOFT_POOLED')
SHARED_CONTROLS = ('PARENT_SHARED_TRANSDUCT', 'REG_SHARED_TRANSDUCT')
COMPARATORS = ORDINARY_CONTROLS + SHARED_CONTROLS
NEW_RAW_METHODS = (PRIMARY, 'REG_SOFT_KD', 'REG_HARD_KD', 'REG_SOFT_POOLED',
                   'PARENT_SHARED_TRANSDUCT', 'REG_SHARED_TRANSDUCT')
PARENT_ALIAS = {'PARENT_SOFT_KD': 'UNIFORM_SOFT_KD288', 'PARENT_POOLED': 'UNIFORM_POOLED'}
SOURCES = (
    'flex_uda/active/dual_transduct_coevol.py',
    'flex_uda/active/dual_transduct_runtime.py',
    'scripts/a100_dual_transduct_contract.py',
    'scripts/a100_dual_transduct_engineering.py',
    'scripts/a100_dual_transduct_primary.py',
    'scripts/a100_dual_transduct_statistics.py',
    'scripts/a100_conditional_state_engineering.py',
    'scripts/a100_factor_rsi_primary_quality.py',
    'flex_uda/presence/predictions.py',
    'flex_uda/a100/evaluation_reuse.py',
)


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    if PLAN.exists():
        raise FileExistsError(PLAN)
    parent = json.loads((BASE / 'registration.json').read_text())
    contract = ROOT / 'a100_runs/visual_classical_20261121/supervisor/dual_transduct_contract_20261127/result.json'
    assert json.loads(contract.read_text())['status'] == 'PASS_DUAL_TRANSDUCT_COEVOL_MATHEMATICAL_CONTRACT'
    eng = OUT.parent / 'engineering/completion.json'
    assert json.loads(eng.read_text())['status'] == 'PASS_ALL6_SETTING_DUAL_TRANSDUCT_ENGINEERING_NO_GT'
    old_provider = PUBLIC / 'fullscope_visual_six_20261126_statistics.json'
    assert old_provider.is_file()

    cases = []
    paths = {BASE / 'registration.json', BASE / 'all4588_sealed_before_GT.json', contract, eng, old_provider}
    for old in parent['cases']:
        case = dict(old)
        folder = BASE / 'units' / case['pair'] / case['evaluated_video']
        case.update(
            parent=str(folder),
            original_scores=str(folder / 'predictions/UNIFORM_SOFT_KD288/scores.npy'),
            original_mask=str(folder / 'predictions/UNIFORM_SOFT_KD288/mask.npy'),
            original_pooled_scores=str(folder / 'predictions/UNIFORM_POOLED/scores.npy'),
            original_pooled_mask=str(folder / 'predictions/UNIFORM_POOLED/mask.npy'),
            uniform_answers=str(folder / 'uniform_answers.jsonl'),
            uniform_draws=str(folder / 'uniform_draws.json'),
        )
        for k in ('original_scores', 'original_mask', 'original_pooled_scores', 'original_pooled_mask',
                  'uniform_answers', 'uniform_draws', 'observations', 'temporal_queries', 'auxiliary_path'):
            paths.add(Path(case[k]))
        paths.update((folder / 'unlabeled_normalizer.npz', folder / 'generation_complete.json'))
        cases.append(case)
    assert len(cases) == 148 and len({c['pair'] for c in cases}) == 6

    source_sha = {name: file_sha(ROOT / name) for name in SOURCES}
    plan = dict(
        status='PREREGISTERED_ALL148_DUAL_TRANSDUCT_COEVOL_PRIMARY_BEFORE_FITS',
        primary=PRIMARY,
        ordinary_controls=ORDINARY_CONTROLS,
        shared_controls=SHARED_CONTROLS,
        comparators=COMPARATORS,
        new_raw_methods=NEW_RAW_METHODS,
        old_statistics_provider=old_provider.name,
        old_statistics_provider_sha256=file_sha(old_provider),
        cases=cases,
        source_sha256=source_sha,
        input_sha256={str(p): file_sha(p) for p in sorted(paths)},
        contract_sha256=file_sha(contract),
        engineering_sha256=file_sha(eng),
        width=256, dropout=0.2, learning_rate=3e-4, weight_decay=2e-2,
        rounds=3, steps_per_round=[500, 300, 300],
        k_nn=5, kappa=4.0, base_decay=0.85,
        seed=0, GPUs=[0, 1, 2, 3], GPU_memory_fraction=0.20,
        existing_GPU2_service_preserved=True,
        new_outputs_per_unit=20, total_new_outputs=2960,
        whole_setting_HOTA_records=132, paired_effects=126,
        hypothesis='Hierarchical E_q x slow persistent track identity U_{q,k} x fast zero-mean dynamic residual + OOF-LOTO learned predicate-aware cross-track visual-spatial graph + regularized complementary dual-student co-evolution (S_id, S_dyn) with M-matrix leverage filtering and ABSENT_NEG beats adequate ordinary and pooled KD across all 6 settings (raw & dual CountMatched).',
        success_ordinary='Primary raw and BOTH CountMatched summary effects positive vs all 5 ordinary controls (including PARENT_POOLED and REG_SOFT_POOLED) in all 6 settings.',
        success_all7='Primary raw and BOTH CountMatched summary effects positive vs all 7 controls (including privileged shared-readout controls) in all 6 settings.',
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
    print(json.dumps(dict(status=plan['status'], units=148, new_outputs=2960, pins=len(source_sha), inputs=len(paths))), flush=True)


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
    from scipy.special import expit
    from flex_uda.a100.mechanism import PAIRS
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.dual_transduct_runtime import (
        FastVideoFeatureAccessor,
        learn_dual_transduct_coevol,
        fit_regularized_ordinary_controls,
        apply_shared_transduct_readout,
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
    context = Context(case, folder, None, gpu)
    accessor = FastVideoFeatureAccessor(context)

    learned = learn_dual_transduct_coevol(context, case, plan, folder / 'primary', device='cuda:0', accessor=accessor)
    ord_started = time.perf_counter()
    reg_scores, reg_receipts = fit_regularized_ordinary_controls(context, case, plan, device='cuda:0', accessor=accessor)
    ord_seconds = time.perf_counter() - ord_started

    pred_started = time.perf_counter()
    parent_soft = np.load(case['original_scores'])
    parent_pool = np.load(case['original_pooled_scores'])
    parent_soft_mask = np.load(case['original_mask'])
    parent_pool_mask = np.load(case['original_pooled_mask'])
    assert np.array_equal(parent_soft > 0, parent_soft_mask)
    assert np.array_equal(parent_pool > 0, parent_pool_mask)

    # Track-mean pooled REG_SOFT_KD (exact same pooling definition as SOFT_POOLED)
    reg_pooled = np.empty_like(reg_scores['REG_SOFT_KD'])
    for track in np.unique(context.video.obs[:, 1]):
        at = np.flatnonzero(context.video.obs[:, 1] == track)
        reg_pooled[:, at] = probability_scores(expit(reg_scores['REG_SOFT_KD'][:, at].astype(float)).mean(1))[:, None]

    parent_shared_prob = apply_shared_transduct_readout(parent_soft, context, learned)
    reg_shared_prob = apply_shared_transduct_readout(reg_scores['REG_SOFT_KD'], context, learned)

    scores = {
        PRIMARY: probability_scores(learned['trans']['frame_prob']),
        'PARENT_SOFT_KD': parent_soft,
        'PARENT_POOLED': parent_pool,
        'REG_SOFT_KD': reg_scores['REG_SOFT_KD'],
        'REG_HARD_KD': reg_scores['REG_HARD_KD'],
        'REG_SOFT_POOLED': reg_pooled,
        'PARENT_SHARED_TRANSDUCT': probability_scores(parent_shared_prob),
        'REG_SHARED_TRANSDUCT': probability_scores(reg_shared_prob),
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
                  extra=dict(seed=0, registration_sha256=plan['registration_sha256'], fixed_teacher_replay=True, **extra))
        tasks.append(dict(pair=case['pair'], video=case['evaluated_video'], method=name, folder=str(dest)))

    for name in NEW_RAW_METHODS:
        assert np.isfinite(scores[name]).all()
        dest = folder / 'predictions' / name
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'scores.npy', scores[name])
        seal(name, masks[name], dict(same_information=True))

    for rank, count in [(PRIMARY, c) for c in COMPARATORS] + [(c, PRIMARY) for c in COMPARATORS]:
        name = rank + '__AT_' + count + '_COUNT'
        mask = projection(scores[rank], masks[count], context.video.obs)
        seal(name, mask, dict(rank_method=rank, count_method=count, counts_from_GT=False, counts_equal_per_query_frame=True))

    assert len(tasks) == 20 and not guard
    receipt = dict(
        status='SEALED_ALL20_DUAL_TRANSDUCT_VIDEO_OUTPUTS',
        pair=case['pair'], video=case['evaluated_video'],
        tasks=tasks, registration_sha256=plan['registration_sha256'],
        GT_open_attempts=0, neural_seed=0, new_teacher_calls=0,
        primary_fit=learned['receipt'], ordinary_fits=reg_receipts,
        ordinary_phase_wall_seconds=ord_seconds,
        prediction_seal_phase_wall_seconds=time.perf_counter() - pred_started,
        total_wall_seconds=time.perf_counter() - started,
        prepaid_inputs_not_fresh_deployment=True,
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
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    plan = read_plan()
    completed = []
    for i, case in enumerate(plan['cases']):
        if i % 4 != gpu:
            continue
        done = run_unit(case, gpu, plan, guard)
        completed.append(dict(pair=case['pair'], video=case['evaluated_video'], outputs=len(done['tasks'])))
        torch.cuda.empty_cache()
    assert not guard
    atomic_json(OUT / 'supervisor' / f'GPU{gpu}_completion.json', dict(
        status='COMPLETE_DUAL_TRANSDUCT_GPU_WORKER', GPU=gpu, execution=execution,
        completed=completed, GT_open_attempts=len(guard), registration_sha256=plan['registration_sha256'],
    ))


def evaluate_and_collect(tasks):
    from flex_uda.a100.evaluation_reuse import evaluate
    from scripts.a100_dual_transduct_statistics import collect, reproduce
    assert len(tasks) == 2960
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda t: evaluate(t['folder'], t['pair']), tasks):
            pass
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
                print(json.dumps(dict(status='STARTED_FULL_DUAL_TRANSDUCT_PRIMARY', GPU=gpu, pid=proc.pid)), flush=True)
            failed = []
            for fut in concurrent.futures.as_completed(futures):
                code = fut.result()
                gpu = futures[fut]
                print(json.dumps(dict(status='DUAL_TRANSDUCT_WORKER_EXIT', GPU=gpu, exit_code=code)), flush=True)
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
            status='ALL2960_DUAL_TRANSDUCT_OUTPUTS_SEALED_BEFORE_GT',
            tasks=tasks, GT_open_attempts=0, registration_sha256=plan['registration_sha256'],
        ))
    sealed = json.loads(barrier.read_text())
    assert sealed['GT_open_attempts'] == 0 and sealed['registration_sha256'] == plan['registration_sha256']
    read_plan()
    print(json.dumps(dict(status='ALL2960_SEALED_START_ISOLATED_OFFICIAL_CPU_EVALUATION', outputs=2960)), flush=True)
    evaluate_and_collect(sealed['tasks'])


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
