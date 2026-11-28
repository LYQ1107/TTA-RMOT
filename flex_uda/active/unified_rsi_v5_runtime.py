"""UNIFIED_RSI_RMOT_V5: Strictly Proper Ventral/Dorsal Two-Stream Convex OOF + Exact LOTO-EB Trust-Region RSI.

Refines V4 based strictly on zero-GT mathematical & receipt diagnostics (no GT tuning):
  1. Strictly Proper Two-Stream Convex OOF Gating (I2-A):
       Minimizes the exact strictly proper query-normalized binary cross-entropy (wp=1, wn=1,
       Gneiting & Raftery 2007) separately on static vs dynamic query strata in [0, 1].
       Eliminates V4's class-reweighting distortion (which rewarded upward-biased motion-blind S_id).
  2. Exact Leave-One-Track-Out (LOTO) Empirical-Bayes Trust-Region Co-Evolution (I2-B):
       Replaces V4's 2-fold half-graph split gate (which collapsed small SportsMOT cliques and
       caused false-negative CHECKLIST fold splits on Dance) with exact Sherman-Morrison LOTO
       Empirical-Bayes shrinkage scaling (shrink_q) + sign-balanced mean-neutral trust region on
       PRESUPPOSED queries + full CHECKLIST ABSENT_NEG suppression.
  3. Exact Fixed-QA Replay of Sealed V3 Hybrid Challenger Evidence (I3):
       Reuses the exact 55,019 sealed Qwen3-VL-2B active teacher responses (0 new Qwen forwards, 0 GT).
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch

from flex_uda.active.dual_transduct_coevol import (
    ID_DIM,
    RegularizedMLP,
    split_shared_features_np,
)
from flex_uda.active.dual_transduct_v2 import prob_pooled_track_and_residual
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor,
    precompute_graph_distances,
    predict_full_head_fast,
    predict_track_id,
)
from flex_uda.active.dual_transduct_v3_runtime import (
    _compute_oof_priors_v3,
    fit_head_adaptive_v3,
)
from flex_uda.active.unified_rsi_runtime import (
    broker_rows_to_events,
    build_anchor_tensors,
)
from flex_uda.active.unified_rsi_v4_runtime import (
    apply_shared_transduct_v4_readout,
    compute_balanced_trust_region_weights,
    empirical_bayes_graph_gate_two_stream,
    fit_observer_two_stream_loto,
    infer_transductive_v4,
)
from flex_tta.runtime.pf_receipts import atomic_json


def fit_two_stream_proper_convex_beta(p_id_oof, p_ctx_oof, is_dyn, events):
    """Strictly proper query-normalized convex OOF minimization of beta_static and beta_dyn in [0, 1].

    Unlike class-reweighted CE (whose Bayes optimum shifts upward toward 0.5 when pos_rate < 0.5,
    unfairly rewarding an upward-biased stream), strictly proper binary cross-entropy (wp=1, wn=1)
    is uniquely minimized by the best-calibrated convex posterior mixture on each query stratum.
    """
    nq = p_id_oof.shape[0]
    if not events:
        return 0.5, 0.5, dict(beta_static=0.5, beta_dyn=0.5,
                              static_nll_id=0.0, static_nll_ctx=0.0, static_nll_mix=0.0,
                              dyn_nll_id=0.0, dyn_nll_ctx=0.0, dyn_nll_mix=0.0)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_is_dyn = is_dyn[ev_q]

    def solve_stratum(mask):
        if not np.any(mask):
            return 0.5, 0.0, 0.0, 0.0
        qm = ev_q[mask]
        pm = ev_p[mask]
        pid = np.clip(p_id_oof[qm, ev_k[mask]], 1e-6, 1.0 - 1e-6)
        pcx = np.clip(p_ctx_oof[qm, ev_k[mask]], 1e-6, 1.0 - 1e-6)
        q_cnt = np.bincount(qm, minlength=nq).astype(np.float64)
        w = 1.0 / (max(int((q_cnt > 0).sum()), 1) * np.maximum(q_cnt[qm], 1.0))

        def loss_fn(b):
            p = np.clip((1.0 - b) * pid + b * pcx, 1e-7, 1.0 - 1e-7)
            return float(-np.sum(w * (pm * np.log(p) + (1.0 - pm) * np.log1p(-p))))

        l0 = loss_fn(0.0)
        l1 = loss_fn(1.0)
        res = minimize_scalar(loss_fn, bounds=(0.0, 1.0), method='bounded', options=dict(xatol=1e-6))
        b_opt = float(np.clip(res.x, 0.0, 1.0))
        l_opt = float(res.fun)
        if l0 <= l_opt and l0 <= l1:
            b_opt, l_opt = 0.0, l0
        elif l1 <= l_opt and l1 <= l0:
            b_opt, l_opt = 1.0, l1
        return b_opt, l0, l1, l_opt

    beta_s, s_id, s_cx, s_mix = solve_stratum(~ev_is_dyn)
    beta_d, d_id, d_cx, d_mix = solve_stratum(ev_is_dyn)
    if not np.any(ev_is_dyn):
        beta_d = beta_s
    elif not np.any(~ev_is_dyn):
        beta_s = beta_d

    return beta_s, beta_d, dict(
        beta_static=float(beta_s), beta_dyn=float(beta_d),
        static_nll_id=float(s_id), static_nll_ctx=float(s_cx), static_nll_mix=float(s_mix),
        dyn_nll_id=float(d_id), dyn_nll_ctx=float(d_cx), dyn_nll_mix=float(d_mix),
    )


def learn_unified_rsi_rmot_v5(context, case, plan, folder, device='cuda:0', accessor=None):
    """Run UNIFIED_RSI_RMOT_V5 on sealed V3 active teacher answers (0 new Qwen forwards, 0 GT)."""
    from flex_uda.a100.mechanism import metadata
    from flex_uda.active.acquisition import expression_types
    from flex_uda.active.v8p import classify_protocol

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if accessor is None:
        accessor = FastVideoFeatureAccessor(context)
    video = context.video
    nq, n = context.nq, context.n
    n_tracks = len(video.tracks)
    md = metadata(case['target'])
    sets = {v: [q['query_text'] for q in md['episodes'] if q['video_id'] == v] for v in md['videos']}
    protocol = classify_protocol(sets)['protocol']
    is_dyn = np.array([bool(set(expression_types(q['query_text'])) & {'MOTION', 'SPATIAL', 'ROLE'})
                       for q in video.queries], dtype=bool)

    ans_path = Path(case['v3_active_answers'])
    all_rows = [json.loads(line) for line in ans_path.read_text().splitlines() if line.strip()]
    events = broker_rows_to_events(context, all_rows)
    assert len(events) == sum(context.quotas.values())

    x_id_mat = accessor.all_track_id()
    d_app, d_pcd, d_geo = precompute_graph_distances(context, accessor)
    aux_id, _ = split_shared_features_np(context.auxiliary)
    aux_full = context.auxiliary
    aux_y = np.asarray(context.ay, dtype=np.float64)
    aux_w = np.full(len(aux_y), 0.25 / len(aux_y), dtype=np.float64)

    id_mean = x_id_mat.reshape(-1, ID_DIM).mean(axis=0, dtype=np.float64).astype(np.float32)
    id_std = x_id_mat.reshape(-1, ID_DIM).std(axis=0, dtype=np.float64).clip(min=0.1).astype(np.float32)
    full_mean = context.mean
    full_std = context.std
    track_rows_cache = {k: np.flatnonzero(video.columns == k) for k in range(n_tracks)}

    (id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k), (
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k), q_budget = build_anchor_tensors(
        events, x_id_mat, accessor, nq, n)

    # Step 1: 2-Fold OOF -> Strictly Proper Convex Two-Stream Gating -> Balanced OOF-LOTO -> Exact LOTO-EB Graph Gate
    oof_started = time.perf_counter()
    p_id_oof, f_ctx_oof = _compute_oof_priors_v3(
        x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k,
        aux_id, aux_full, aux_y, aux_w, id_mean, id_std, full_mean, full_std, plan, device)
    f_ctx_tm_oof, f_ctx_res_oof = prob_pooled_track_and_residual(
        f_ctx_oof, video.obs, video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_ctx_tm_oof = expit(f_ctx_tm_oof)

    beta_s, beta_d, two_stream_diag = fit_two_stream_proper_convex_beta(
        p_id_oof, p_ctx_tm_oof, is_dyn, events)
    beta_q = np.where(is_dyn, beta_d, beta_s)

    raw_params, laps, loto_oof = fit_observer_two_stream_loto(
        p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, beta_q, k_nn=plan['k_nn'])
    params, eb_gate = empirical_bayes_graph_gate_two_stream(
        raw_params, laps, p_id_oof, p_ctx_tm_oof, is_dyn, events,
        beta_q, delta_0=float(plan['eb_graph_delta_0']))
    oof_seconds = time.perf_counter() - oof_started

    # Step 2: 3-Round Dual-Student Co-Evolution governed directly by Exact LOTO-EB Trust Region (no 2-fold graph split!)
    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_ctx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    n_rounds = int(plan['rounds'])
    shrink_q = np.where(is_dyn, eb_gate['shrink_dyn'], eb_gate['shrink_static'])

    for r in range(n_rounds):
        r_start = time.perf_counter()
        if r == 0:
            x_id_tr = np.concatenate((id_anchor_x, aux_id), axis=0)
            y_id_tr = np.concatenate((id_anchor_y, aux_y), axis=0)
            w_id_tr = np.concatenate((id_anchor_w, aux_w), axis=0)
            x_cx_tr = np.concatenate((ctx_anchor_x, aux_full), axis=0)
            y_cx_tr = np.concatenate((ctx_anchor_y, aux_y), axis=0)
            w_cx_tr = np.concatenate((ctx_anchor_w, aux_w), axis=0)
            tr_diag = dict(count=0, eff_budget=0.0, up_push_raw=0.0, down_push_raw=0.0, up_damp=1.0)
            absent_neg_used = 0
            min_s, max_s, tgt_ce = 600, int(plan['r0_max_steps']), float(plan['target_excess_ce'])
        else:
            lev = trans['leverage']
            p_tr = trans['transduced_track_prob']
            p_pr = trans['prior_prob']
            eta = trans['existence']
            asked_mask = trans['asked_counts'] > 0

            sup_q, sup_k, raw_w, tr_diag = compute_balanced_trust_region_weights(
                lev, p_tr, p_pr, asked_mask, q_budget, shrink_q,
                lev_thresh=float(plan['leverage_threshold']),
                trans_budget_w=float(plan['transduced_weight']), protocol=protocol)

            id_parts_x = [id_anchor_x, aux_id]
            id_parts_y = [id_anchor_y, aux_y]
            id_parts_w = [id_anchor_w, aux_w]
            if len(sup_q) > 0:
                id_parts_x.append(x_id_mat[sup_q, sup_k])
                id_parts_y.append(p_tr[sup_q, sup_k])
                id_parts_w.append(raw_w)

            absent_neg_used = 0
            abs_q = np.flatnonzero(eta < 0.35) if protocol == 'CHECKLIST' else np.zeros(0, dtype=np.int64)
            if len(abs_q) > 0:
                aq_grid = np.repeat(abs_q, n_tracks)
                ak_grid = np.tile(np.arange(n_tracks, dtype=np.int64), len(abs_q))
                abs_w = 1.0 - eta[aq_grid]
                abs_w = abs_w * (0.35 / max(float(abs_w.sum()), 1e-12))
                id_parts_x.append(x_id_mat[aq_grid, ak_grid])
                id_parts_y.append(np.zeros(len(aq_grid), dtype=np.float64))
                id_parts_w.append(abs_w)
                absent_neg_used = int(len(abs_q))

            if len(sup_q) == 0 and absent_neg_used == 0:
                rounds_log.append(dict(
                    round=r, skipped_exact_convergence=True,
                    trust_region=tr_diag, absent_neg_queries=0,
                    round_wall_seconds=0.0,
                ))
                break

            x_id_tr = np.concatenate(id_parts_x, axis=0)
            y_id_tr = np.concatenate(id_parts_y, axis=0)
            w_id_tr = np.concatenate(id_parts_w, axis=0)

            cx_parts_x = [ctx_anchor_x, aux_full]
            cx_parts_y = [ctx_anchor_y, aux_y]
            cx_parts_w = [ctx_anchor_w, aux_w]
            if len(sup_q) > 0:
                sample_n = min(2048, len(sup_q) * 2)
                pick = rng.integers(len(sup_q), size=sample_n)
                sel_ids, sel_y, sel_w = [], [], []
                for idx_p in pick:
                    q_i, k_i = int(sup_q[idx_p]), int(sup_k[idx_p])
                    rows_k = track_rows_cache[k_i]
                    nd = int(rows_k[rng.integers(len(rows_k))])
                    sel_ids.append(q_i * n + nd)
                    sel_y.append(float(trans['frame_prob'][q_i, nd]))
                    sel_w.append(float(raw_w[idx_p]))
                sel_w = np.asarray(sel_w, dtype=np.float64)
                sel_w = sel_w * (float(raw_w.sum()) / max(float(sel_w.sum()), 1e-12))
                cx_parts_x.append(accessor.full_rows(np.asarray(sel_ids, dtype=np.int64)))
                cx_parts_y.append(np.asarray(sel_y, dtype=np.float64))
                cx_parts_w.append(sel_w)

            if len(abs_q) > 0:
                sample_abs = min(1024, len(abs_q) * 16)
                aq = abs_q[rng.integers(len(abs_q), size=sample_abs)]
                and_ = rng.integers(n, size=sample_abs)
                abs_w_cx = 1.0 - eta[aq]
                abs_w_cx = abs_w_cx * (0.35 / max(float(abs_w_cx.sum()), 1e-12))
                cx_parts_x.append(accessor.full_rows(aq * n + and_))
                cx_parts_y.append(np.zeros(sample_abs, dtype=np.float64))
                cx_parts_w.append(abs_w_cx)

            x_cx_tr = np.concatenate(cx_parts_x, axis=0)
            y_cx_tr = np.concatenate(cx_parts_y, axis=0)
            w_cx_tr = np.concatenate(cx_parts_w, axis=0)
            min_s, max_s, tgt_ce = 300, int(plan['coevol_max_steps']), float(plan['target_excess_ce'])

        lr_r = float(plan['learning_rate']) * (0.5 ** r)
        id_fit = fit_head_adaptive_v3(
            s_id, x_id_tr, y_id_tr, w_id_tr,
            min_steps=min_s, max_steps=max_s, target_excess_ce=tgt_ce,
            device=device, lr=lr_r, weight_decay=plan['weight_decay'], seed=r)
        cx_fit = fit_head_adaptive_v3(
            s_ctx, x_cx_tr, y_cx_tr, w_cx_tr,
            min_steps=min_s, max_steps=max_s, target_excess_ce=tgt_ce,
            device=device, lr=lr_r, weight_decay=plan['weight_decay'], seed=100 + r)

        p_id = expit(predict_track_id(s_id, x_id_mat, device))
        f_ctx_raw = predict_full_head_fast(s_ctx, accessor, device)
        f_ctx_tm, f_ctx_res = prob_pooled_track_and_residual(
            f_ctx_raw, video.obs, video.columns, context.images, n_tracks,
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        p_ctx_tm = expit(f_ctx_tm)
        trans = infer_transductive_v4(
            p_id, p_ctx_tm, f_ctx_res, laps, is_dyn, events,
            video.columns, params, beta_q, protocol=protocol)

        rounds_log.append(dict(
            round=r, lr=lr_r, s_id_fit=id_fit, s_ctx_fit=cx_fit,
            trust_region=tr_diag, absent_neg_queries=absent_neg_used,
            mean_leverage=float(trans['leverage'].mean()),
            mean_existence=float(trans['existence'].mean()),
            mean_frame_prob=float(trans['frame_prob'].mean()),
            round_wall_seconds=time.perf_counter() - r_start,
        ))

    receipt = dict(
        status='COMPLETE_UNIFIED_RSI_RMOT_V5_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        dynamic_queries=int(is_dyn.sum()),
        two_stream_proper_oof=two_stream_diag,
        oof_loto=loto_oof, eb_graph_gate=eb_gate,
        oof_wall_seconds=oof_seconds,
        observer=params.to_dict(), rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        new_teacher_calls=0, GT_open_attempts=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    print(json.dumps(dict(
        status='UNIFIED_RSI_V5_UNIT_LEARNED', pair=case['pair'], video=case['evaluated_video'],
        b_s=round(beta_s, 2), b_d=round(beta_d, 2),
        lam_s=f"{eb_gate['lam_s_raw']:.2f}->{eb_gate['lam_s_gated']:.2f}",
        lam_d=f"{eb_gate['lam_d_raw']:.2f}->{eb_gate['lam_d_gated']:.2f}",
        r1_tr=rounds_log[1]['trust_region']['count'] if len(rounds_log) > 1 else 0,
        p_mean=round(float(trans['frame_prob'].mean()), 4),
        sec=round(receipt['wall_seconds'], 1),
    )), flush=True)
    return dict(
        receipt=receipt, accessor=accessor, s_id=s_id, s_ctx=s_ctx,
        params=params, beta_q=beta_q, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )
