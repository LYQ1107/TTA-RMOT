"""Complete 2x2x2 Factorial Ablation (I1 x I2 x I3) across all 148 videos (0 new Qwen calls).

Already sealed & independently reproduced cells of the exact 2x2x2 factorial design:
- (0, 0, 0): UNIFORM_POOLED (fullscope_visual_six_20261126)
- (1, 0, 0): V3 PARENT_SHARED_TRANSDUCT / REG_SHARED_TRANSDUCT (I1 only on UNIFORM)
- (1, 1, 0): V3 DUAL_TRANSDUCT_COEVOL_V3 (I1 + I2 on UNIFORM)
- (0, 0, 1): UNIFIED_V1 ACTIVE_SOFT_POOLED (I3 only on UNIFIED_V1 active answers)
- (1, 0, 1): UNIFIED_V1 ACTIVE_SHARED_TRANSDUCT (I1 + I3 on UNIFIED_V1 active answers)
- (1, 1, 1): UNIFIED_V1 UNIFIED_RSI_RMOT (Full I1 + I2 + I3)

This preregistered fixed-QA replay study generates ONLY the 2 missing factorial cells
plus 1 CHECKLIST-existence ablation across all 148 videos (0 new teacher calls, GT_open_attempts==0):
- (0, 1, 0) I2_ONLY_UNIFORM: Complementary Dual Students (S_id 7482D + S_ctx 9352D,
  probability track pooling, 0.5/0.5 mix) on UNIFORM answers with I1 OFF (no graph,
  no Bayesian anchor fusion, no CHECKLIST existence).
- (0, 1, 1) I2_PLUS_I3_ACTIVE: Exact same Complementary Dual Students on UNIFIED_V1
  active answers with I1 OFF.
- NO_CHECKLIST_EXISTENCE_ACTIVE: Full UNIFIED_V1 (I1+I2+I3) with CHECKLIST existence
  suppression eta_q disabled (eta_q = 1.0) on all 148 videos (exact match on PRESUPPOSED
  C2..C5, isolates CHECKLIST existence effect cleanly on Dance C1/C6!).

Total new outputs: 3 x 148 = 444 raw predictions (plus bidirectional CountMatched vs
UNIFIED_RSI_RMOT_V1 for I2_ONLY_UNIFORM & I2_PLUS_I3_ACTIVE = 4 extra/unit => 7 x 148 = 1,036
outputs), evaluated in ~5 minutes by verified 0.0-error 20-worker fast_evaluation_reuse!
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
V3_OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/dual_transduct_v3_20261127/primary'
U1_OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_rmot_20261128/primary'
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/unified_rsi_factorial_20261128'
PLAN = OUT / 'registration.json'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122'
STEM = 'unified_rsi_factorial_20261128'

NEW_RAW_METHODS = ('I2_ONLY_UNIFORM', 'I2_PLUS_I3_ACTIVE', 'NO_EXISTENCE_ACTIVE')
CM_COMPARATORS = ('I2_ONLY_UNIFORM', 'I2_PLUS_I3_ACTIVE')
SOURCES = (
    'flex_uda/active/dual_transduct_coevol.py',
    'flex_uda/active/dual_transduct_runtime.py',
    'flex_uda/active/dual_transduct_v2.py',
    'flex_uda/active/dual_transduct_v3.py',
    'flex_uda/active/dual_transduct_v3_runtime.py',
    'flex_uda/active/unified_rsi_core.py',
    'flex_uda/active/unified_rsi_runtime.py',
    'scripts/a100_unified_rsi_factorial_ablation.py',
    'scripts/a100_unified_rsi_factorial_statistics.py',
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
    prov_fs = PUBLIC / 'fullscope_visual_six_20261126_statistics.json'
    prov_v3 = PUBLIC / 'dual_transduct_v3_primary_20261127_statistics.json'
    prov_u1 = PUBLIC / 'unified_rsi_rmot_20261128_statistics.json'
    for p in (prov_fs, prov_v3, prov_u1):
        assert p.is_file(), p

    cases = []
    paths = {prov_fs, prov_v3, prov_u1}
    for old in parent['cases']:
        case = dict(old)
        pair, vid = case['pair'], case['evaluated_video']
        f_base = BASE / 'units' / pair / vid
        f_u1 = U1_OUT / 'units' / pair / vid
        case.update(
            parent=str(f_base),
            uniform_answers=str(f_base / 'uniform_answers.jsonl'),
            uniform_draws=str(f_base / 'uniform_draws.json'),
            u1_active_answers=str(f_u1 / 'active_answers.jsonl'),
            u1_primary_scores=str(f_u1 / 'predictions/UNIFIED_RSI_RMOT/scores.npy'),
            u1_primary_mask=str(f_u1 / 'predictions/UNIFIED_RSI_RMOT/mask.npy'),
            u1_learning_receipt=str(f_u1 / 'primary/learning_receipt.json'),
        )
        for k in ('uniform_answers', 'uniform_draws', 'u1_active_answers',
                  'u1_primary_scores', 'u1_primary_mask', 'u1_learning_receipt',
                  'observations', 'temporal_queries', 'auxiliary_path'):
            paths.add(Path(case[k]))
        cases.append(case)
    assert len(cases) == 148

    source_sha = {name: file_sha(ROOT / name) for name in SOURCES}
    plan = dict(
        status='PREREGISTERED_ALL148_2X2X2_FACTORIAL_ABLATION',
        new_raw_methods=NEW_RAW_METHODS,
        cm_comparators=CM_COMPARATORS,
        providers=dict(
            fullscope=dict(file=prov_fs.name, sha256=file_sha(prov_fs)),
            v3=dict(file=prov_v3.name, sha256=file_sha(prov_v3)),
            u1=dict(file=prov_u1.name, sha256=file_sha(prov_u1)),
        ),
        cases=cases,
        source_sha256=source_sha,
        width=256, dropout=0.10, learning_rate=8e-4, weight_decay=1e-3,
        max_steps=4800, target_excess_ce=0.01,
        kappa=6.0, base_decay=0.94,
        seed=0, GPUs=[0, 1, 2, 3], GPU_memory_fraction=0.20,
        new_outputs_per_unit=7, total_new_outputs=1036,
        eval_workers=20,
        all_data_DEV=True, manual_GT_never_used_for_adaptation=True,
        goal_status='ACTIVE',
    )
    assert not guard
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN, plan)
    compact = {k: v for k, v in plan.items() if k != 'cases'}
    compact['video_units'] = 148
    compact['setting_video_counts'] = {p: sum(c['pair'] == p for c in cases) for p in sorted({c['pair'] for c in cases})}
    atomic_json(PUBLIC / (STEM + '_registration.json'), compact)
    print(json.dumps(dict(status=plan['status'], units=148, new_outputs=1036, pins=len(source_sha))), flush=True)


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


def fit_dual_no_i1(context, events, x_id_mat, accessor, id_mean, id_std, aux_id, aux_full, aux_y, aux_w, plan, device):
    """Pure I2 complementary dual students (S_id 7482D + S_ctx 9352D, prob pooling) with I1 OFF."""
    import numpy as np
    from scipy.special import expit
    import torch
    from flex_uda.active.dual_transduct_coevol import RegularizedMLP
    from flex_uda.active.dual_transduct_runtime import predict_full_head_fast, predict_track_id
    from flex_uda.active.dual_transduct_v2 import prob_pooled_track_and_residual
    from flex_uda.active.dual_transduct_v3_runtime import fit_head_adaptive_v3
    from flex_uda.active.unified_rsi_runtime import build_anchor_tensors

    nq, n = context.nq, context.n
    n_tracks = len(context.video.tracks)
    (id_x, id_y, id_w, _), (cx_x, cx_y, cx_w, _), _ = build_anchor_tensors(
        events, x_id_mat, accessor, nq, n)

    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    id_rec = fit_head_adaptive_v3(
        s_id,
        np.concatenate((id_x, aux_id), axis=0),
        np.concatenate((id_y, aux_y), axis=0),
        np.concatenate((id_w, aux_w), axis=0),
        min_steps=600, max_steps=int(plan['max_steps']), target_excess_ce=float(plan['target_excess_ce']),
        device=device, lr=float(plan['learning_rate']), weight_decay=plan['weight_decay'], seed=0)

    torch.manual_seed(0)
    s_ctx = RegularizedMLP(context.mean, context.std, width=plan['width'], dropout=plan['dropout']).to(device)
    cx_rec = fit_head_adaptive_v3(
        s_ctx,
        np.concatenate((cx_x, aux_full), axis=0),
        np.concatenate((cx_y, aux_y), axis=0),
        np.concatenate((cx_w, aux_w), axis=0),
        min_steps=600, max_steps=int(plan['max_steps']), target_excess_ce=float(plan['target_excess_ce']),
        device=device, lr=float(plan['learning_rate']), weight_decay=plan['weight_decay'], seed=100)

    p_id = expit(predict_track_id(s_id, x_id_mat, device))
    f_ctx = predict_full_head_fast(s_ctx, accessor, device)
    f_ctx_tm, _ = prob_pooled_track_and_residual(
        f_ctx, context.video.obs, context.video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_ctx_tm = expit(f_ctx_tm)
    # Pure I2 dual-student track probability (no graph L, no Bayesian anchors, no CHECKLIST eta)
    p_track = 0.5 * p_id + 0.5 * p_ctx_tm
    frame_prob = p_track[:, context.video.columns]
    return probability_scores(frame_prob), dict(s_id=id_rec, s_ctx=cx_rec)


def run_unit(case, gpu, plan, guard):
    import numpy as np
    from scipy.special import expit
    from flex_uda.a100.mechanism import PAIRS
    from flex_uda.active.dual_transduct_coevol import ID_DIM, split_shared_features_np
    from flex_uda.active.dual_transduct_runtime import (
        FastVideoFeatureAccessor, extract_source_events_with_tracks, precompute_graph_distances,
    )
    from flex_uda.active.dual_transduct_v2 import prob_pooled_track_and_residual
    from flex_uda.active.dual_transduct_v3 import TransductiveObserverV3Params
    from flex_uda.active.dual_transduct_v3_runtime import _build_all_gated_laplacians_v3, infer_transductive_v3
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.unified_rsi_runtime import broker_rows_to_events
    from flex_uda.active.acquisition import expression_types
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from scripts.a100_factor_rsi_primary_quality import projection, subset_metadata

    folder = OUT / 'units' / case['pair'] / case['evaluated_video']
    folder.mkdir(parents=True, exist_ok=True)
    finished = folder / 'generation_complete.json'
    if finished.exists():
        done = json.loads(finished.read_text())
        assert done['registration_sha256'] == plan['registration_sha256']
        assert len(done['tasks']) == 7 and done['GT_open_attempts'] == 0
        for task in done['tasks']:
            verify_predictions(task['folder'])
        return done

    started = time.perf_counter()
    shutil.copyfile(Path(case['parent']) / 'unlabeled_normalizer.npz', folder / 'unlabeled_normalizer.npz')
    context = Context(case, folder, None, gpu)
    accessor = FastVideoFeatureAccessor(context)
    x_id_mat = accessor.all_track_id()
    aux_id, _ = split_shared_features_np(context.auxiliary)
    aux_full = context.auxiliary
    aux_y = np.asarray(context.ay, dtype=np.float64)
    aux_w = np.full(len(aux_y), 0.25 / len(aux_y), dtype=np.float64)
    id_mean = x_id_mat.reshape(-1, ID_DIM).mean(axis=0, dtype=np.float64).astype(np.float32)
    id_std = x_id_mat.reshape(-1, ID_DIM).std(axis=0, dtype=np.float64).clip(min=0.1).astype(np.float32)

    # 1. (0, 1, 0) I2_ONLY_UNIFORM
    uni_events, _ = extract_source_events_with_tracks(context, case)
    sc_i2_uni, rec_uni = fit_dual_no_i1(
        context, uni_events, x_id_mat, accessor, id_mean, id_std, aux_id, aux_full, aux_y, aux_w, plan, 'cuda:0')

    # 2. (0, 1, 1) I2_PLUS_I3_ACTIVE (exact UNIFIED_V1 active answers)
    act_rows = [json.loads(line) for line in Path(case['u1_active_answers']).read_text().splitlines() if line.strip()]
    act_events = broker_rows_to_events(context, act_rows)
    sc_i2_act, rec_act = fit_dual_no_i1(
        context, act_events, x_id_mat, accessor, id_mean, id_std, aux_id, aux_full, aux_y, aux_w, plan, 'cuda:0')

    # 3. NO_EXISTENCE_ACTIVE: Exact UNIFIED_V1 scores with CHECKLIST eta_q set to 1.0 (PRESUPPOSED protocol)
    u1_scores = np.load(case['u1_primary_scores'])
    u1_mask = np.load(case['u1_primary_mask'])
    u1_lr = json.loads(Path(case['u1_learning_receipt']).read_text())
    if u1_lr['protocol'] == 'PRESUPPOSED':
        sc_no_exist = u1_scores.copy()
    else:
        # On CHECKLIST (Dance C1/C6), re-apply exact V1 observer with protocol='PRESUPPOSED'
        # to the I2_PLUS_I3_ACTIVE dual-student track prior (or exact p_star without eta_q scaling)!
        obs = u1_lr['observer']
        params = TransductiveObserverV3Params(
            lam_vis_static=obs['lam_vis_static'], lam_vis_dyn=obs['lam_vis_dyn'],
            w_app=obs['w_app'], w_pcd=obs['w_pcd'], w_geo_static=obs['w_geo_static'],
            w_geo_dyn=obs['w_geo_dyn'], dist_gate=obs['dist_gate'], beta_mix=obs['beta_mix'],
            anchor_precision=obs['anchor_precision'], gamma_anc=obs['gamma_anc'],
            gamma_dyn_static=0.0, gamma_dyn_dyn=obs['gamma_dyn_dyn'],
            channel=np.asarray(obs['channel'], dtype=np.float64),
        )
        d_app, d_pcd, d_geo = precompute_graph_distances(context, accessor)
        is_dyn = np.array([bool(set(expression_types(q['query_text'])) & {'MOTION', 'SPATIAL', 'ROLE'})
                           for q in context.video.queries], dtype=bool)
        laps = _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, params, k_nn=3)
        tm_i2, res_i2 = prob_pooled_track_and_residual(
            sc_i2_act, context.video.obs, context.video.columns, context.images, len(context.video.tracks),
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        p_tm = expit(tm_i2)
        out_ne = infer_transductive_v3(
            p_tm, p_tm, res_i2, laps, is_dyn, act_events, context.video.columns, params, protocol='PRESUPPOSED')
        sc_no_exist = probability_scores(out_ne['frame_prob'])

    scores = {
        'UNIFIED_RSI_RMOT': u1_scores,
        'I2_ONLY_UNIFORM': sc_i2_uni,
        'I2_PLUS_I3_ACTIVE': sc_i2_act,
        'NO_EXISTENCE_ACTIVE': sc_no_exist,
    }
    masks = {k: (u1_mask if k == 'UNIFIED_RSI_RMOT' else (v > 0)) for k, v in scores.items()}
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
        dest = folder / 'predictions' / name
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'scores.npy', scores[name])
        seal(name, masks[name], dict(factorial_ablation=True))

    for cmp in CM_COMPARATORS:
        for rank, count in (('UNIFIED_RSI_RMOT', cmp), (cmp, 'UNIFIED_RSI_RMOT')):
            name = rank + '__AT_' + count + '_COUNT'
            mask = projection(scores[rank], masks[count], context.video.obs)
            seal(name, mask, dict(rank_method=rank, count_method=count, counts_from_GT=False))

    assert len(tasks) == 7 and not guard
    receipt = dict(
        status='SEALED_ALL7_FACTORIAL_ABLATION_OUTPUTS',
        pair=case['pair'], video=case['evaluated_video'],
        tasks=tasks, registration_sha256=plan['registration_sha256'],
        GT_open_attempts=0, new_teacher_calls=0, seed=0,
        i2_only_uniform_fit=rec_uni, i2_plus_i3_active_fit=rec_act,
        wall_seconds=time.perf_counter() - started,
    )
    atomic_json(finished, receipt)
    print(json.dumps(dict(status=receipt['status'], pair=case['pair'], video=case['evaluated_video'],
                          GPU=gpu, sec=round(receipt['wall_seconds'], 1))), flush=True)
    return receipt


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    plan = read_plan()
    ordered = sorted(plan['cases'], key=lambda c: (-int(c['logical_budget']), c['pair'], c['evaluated_video']))
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
            run_unit(case, gpu, plan, guard)
            torch.cuda.empty_cache()
    assert not guard


def main():
    plan = read_plan()
    sup = OUT / 'supervisor'
    sup.mkdir(parents=True, exist_ok=True)
    lock = (sup / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    barrier = OUT / 'all1036_sealed_before_GT.json'
    if not barrier.exists():
        def drain(proc, path):
            with path.open('a', buffering=1) as stream:
                for line in proc.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
            return proc.wait()
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
            failed = [futures[f] for f in concurrent.futures.as_completed(futures) if f.result()]
        if failed:
            raise RuntimeError(failed)
        from flex_uda.presence.predictions import verify_predictions
        tasks = []
        for case in plan['cases']:
            done = json.loads((OUT / 'units' / case['pair'] / case['evaluated_video'] / 'generation_complete.json').read_text())
            assert done['registration_sha256'] == plan['registration_sha256'] and done['GT_open_attempts'] == 0
            for t in done['tasks']:
                verify_predictions(t['folder'])
            tasks.extend(done['tasks'])
        assert len(tasks) == 1036
        atomic_json(barrier, dict(status='ALL1036_FACTORIAL_SEALED_BEFORE_GT', tasks=tasks,
                                  GT_open_attempts=0, registration_sha256=plan['registration_sha256']))
    sealed = json.loads(barrier.read_text())
    from flex_uda.a100.fast_evaluation_reuse import evaluate_many
    from scripts.a100_unified_rsi_factorial_statistics import collect, reproduce
    summary = evaluate_many(sealed['tasks'], max_workers=int(plan.get('eval_workers', 20)), use_cache=True)
    print(json.dumps(summary), flush=True)
    collect()
    reproduce()


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
