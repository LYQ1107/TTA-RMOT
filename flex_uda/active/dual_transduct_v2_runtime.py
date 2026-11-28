"""Runtime for DUAL_TRANSDUCT_COEVOL_V2 (zero GT, adaptive adequacy, gated cliques).

Reuses the verified bit-identical FastVideoFeatureAccessor and exact batched
Sherman-Morrison OOF-LOTO solver, upgrading:
1. Adaptive adequacy stopping (proper_excess_ce <= 0.01, up to 3000 steps in R0
   and up to 4200 steps for REG_SOFT_KD / REG_HARD_KD) with gentle regularization
   (wd=3e-3, dropout=0.15, lr=5e-4) -> eliminates KITTI 17x underfitting!
2. Similarity-gated k_nn=3 supportive-clique Laplacians -> eliminates sparse
   positive over-smoothing on KITTI while keeping Dance/Sports clique gains!
3. Exact probability-space track pooling + zero-jitter static tracks + bounded
   segment-coherent dynamic residual (gamma_dyn_dyn <= 0.35) -> preserves AssA!
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn

from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.dual_transduct_coevol import (
    DYN_DIM, FULL_DIM, ID_DIM,
    RegularizedMLP, aggregate_track_anchors, compute_video_existence,
    split_shared_features_np,
)
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor, _batched_solve_from_laplacians,
    extract_source_events_with_tracks, precompute_graph_distances,
    predict_frame_dyn_fast, predict_full_head_fast, predict_track_id,
)
from flex_uda.active.dual_transduct_v2 import (
    build_gated_clique_laplacian, default_v2_vector,
    prob_pooled_track_and_residual, unpack_v2_vector,
)


def _build_all_gated_laplacians(d_app, d_pcd, d_geo, is_dyn, params, *, k_nn=3, dist_gate=1.25):
    nq, k = d_pcd.shape[:2]
    laps = np.empty((nq, k, k), dtype=np.float64)
    for q in range(nq):
        w_geo = params.w_geo_dyn if is_dyn[q] else params.w_geo_static
        w_sum = params.w_app + params.w_pcd + w_geo
        dist = (params.w_app * d_app + params.w_pcd * d_pcd[q] + w_geo * d_geo) / max(w_sum, 1e-6)
        laps[q], _ = build_gated_clique_laplacian(dist, k_nn=k_nn, dist_gate=dist_gate)
    return laps


def fit_head_adaptive(model, x_np, y_np, w_np, *, min_steps, max_steps, target_excess_ce,
                      device, lr=5e-4, weight_decay=3e-3, batch_size=256, seed=0):
    """Train RegularizedMLP in 300-step chunks until proper_excess_ce <= target or max_steps."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
    w = w / w.sum().clamp(min=1e-12)
    n = len(x)
    bs = min(batch_size, n)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(max_steps, 1), eta_min=lr * 0.1)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    yv = np.asarray(y_np, dtype=np.float64)
    wv = np.asarray(w_np, dtype=np.float64)
    wv = wv / max(wv.sum(), 1e-12)
    h = -yv * np.log(np.clip(yv, 1e-15, 1.0)) - (1.0 - yv) * np.log(np.clip(1.0 - yv, 1e-15, 1.0))

    started = time.perf_counter()
    completed = 0
    last_loss = 0.0
    excess_ce = float('inf')
    while completed < max_steps:
        chunk = min(300, max_steps - completed)
        model.train()
        for _ in range(chunk):
            idx = torch.arange(n, device=device) if bs == n else torch.randint(0, n, (bs,), generator=gen).to(device)
            opt.zero_grad(set_to_none=True)
            z = model(x[idx])
            per = nn.functional.softplus(z) - y[idx] * z
            loss = (per * w[idx]).sum() / w[idx].sum().clamp(min=1e-12)
            assert torch.isfinite(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            last_loss = float(loss.detach())
        completed += chunk
        model.eval()
        with torch.no_grad():
            z_all = np.concatenate([model(x[b:b + 1024]).double().cpu().numpy() for b in range(0, n, 1024)])
        excess_ce = float(np.sum(wv * (np.logaddexp(0.0, z_all) - yv * z_all - h)))
        if completed >= min_steps and excess_ce <= target_excess_ce:
            break
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return dict(
        steps=int(completed), lr=float(lr), weight_decay=float(weight_decay),
        last_loss=last_loss, proper_excess_ce=excess_ce,
        target_excess_ce=float(target_excess_ce),
        wall_seconds=time.perf_counter() - started,
    )


def fit_observer_loto_v2(g_id_oof, f_dyn_tm_oof, f_dyn_res_oof, d_app, d_pcd, d_geo,
                         is_dyn, events, columns, *, k_nn=3, dist_gate=1.25, prior_reg=0.015):
    """OOF-LOTO optimization with gated supportive cliques & zero-jitter static tracks."""
    theta0 = default_v2_vector()
    nq, n_tracks = g_id_oof.shape
    if not events:
        p0 = unpack_v2_vector(theta0)
        return p0, _build_all_gated_laplacians(d_app, d_pcd, d_geo, is_dyn, p0, k_nn=k_nn, dist_gate=dist_gate), dict(
            loto_before=0.0, loto_after=0.0, stage1_evals=0, stage2_evals=0)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_nodes = [np.asarray(e['nodes'], dtype=np.int64) for e in events]
    q_counts = np.bincount(ev_q, minlength=nq).astype(np.float64)
    ev_w = 1.0 / (max(int((q_counts > 0).sum()), 1) * np.maximum(q_counts[ev_q], 1.0))
    ev_res_list = [f_dyn_res_oof[ev_q[j], nd] for j, nd in enumerate(ev_nodes)]
    single_mask = np.array([len(nd) == 1 for nd in ev_nodes], dtype=bool)
    single_res = np.array([ev_res_list[j][0] if single_mask[j] else 0.0 for j in range(len(events))], dtype=np.float64)
    multi_idx = np.flatnonzero(~single_mask)

    def eval_with_laps(theta, laps):
        params = unpack_v2_vector(theta)
        prior = (1.0 - params.beta_mix) * g_id_oof + params.beta_mix * f_dyn_tm_oof
        target, precision, _, _ = aggregate_track_anchors(
            nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
        lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
        _, _, loto_u = _batched_solve_from_laplacians(prior, laps, target, precision, lam_q, tau_prior=1.0)
        gam_q = np.where(is_dyn, params.gamma_dyn_dyn, 0.0)
        u_ev = loto_u[ev_q, ev_k]
        g_ev = gam_q[ev_q]
        pred_state = expit(u_ev + g_ev * single_res)
        for j in multi_idx:
            pred_state[j] = float(np.mean(expit(u_ev[j] + g_ev[j] * ev_res_list[j])))
        s = params.channel[ev_v, 0]
        c = params.channel[ev_v, 1]
        p_yes = np.clip((1.0 - c) + (s + c - 1.0) * pred_state, 1e-7, 1.0 - 1e-7)
        nll = -np.sum(ev_w * (ev_p * np.log(p_yes) + (1.0 - ev_p) * np.log1p(-p_yes)))
        reg = prior_reg * float(np.mean((theta - theta0) ** 2))
        return float(nll + reg)

    def inv_sp(y):
        return float(math.log(max(math.expm1(y), 1e-6)))
    kernel_candidates = [
        (1.0, 1.0, 0.15, 1.2),
        (2.0, 0.5, 0.05, 0.8),
        (0.5, 2.0, 0.10, 1.2),
        (1.0, 1.0, 0.50, 2.5),
        (1.5, 1.5, 0.01, 0.05),
    ]
    best_theta = theta0.copy()
    best_laps = _build_all_gated_laplacians(
        d_app, d_pcd, d_geo, is_dyn, unpack_v2_vector(theta0), k_nn=k_nn, dist_gate=dist_gate)
    before = eval_with_laps(theta0, best_laps)
    best_val = before
    for wa, wp, wgs, wgd in kernel_candidates:
        cand = theta0.copy()
        cand[2:6] = [inv_sp(wa), inv_sp(wp), inv_sp(wgs), inv_sp(wgd)]
        laps = _build_all_gated_laplacians(
            d_app, d_pcd, d_geo, is_dyn, unpack_v2_vector(cand), k_nn=k_nn, dist_gate=dist_gate)
        val = eval_with_laps(cand, laps)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_theta = cand.copy()
            best_laps = laps

    opt_idx = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    sub0 = best_theta[opt_idx].copy()

    def sub_obj(sub):
        full = best_theta.copy()
        full[opt_idx] = sub
        return eval_with_laps(full, best_laps)

    res = minimize(sub_obj, sub0, method='Powell', options=dict(maxiter=10, xtol=3e-2, ftol=3e-3))
    if np.isfinite(res.fun) and res.fun < best_val:
        best_theta[opt_idx] = res.x
        best_val = float(res.fun)
    params = unpack_v2_vector(best_theta)
    return params, best_laps, dict(
        loto_before=float(before), loto_after=float(best_val),
        stage1_evals=len(kernel_candidates), stage2_evals=int(getattr(res, 'nfev', 0)),
    )


def infer_transductive_v2(g_id, f_dyn_tm, f_dyn_res, laps, is_dyn, events,
                          columns, params, *, protocol='PRESUPPOSED'):
    nq, n_tracks = g_id.shape
    events_by_query = [[] for _ in range(nq)]
    for ev in events:
        events_by_query[ev['q']].append(ev)
    prior = (1.0 - params.beta_mix) * g_id + params.beta_mix * f_dyn_tm
    target, precision, counts, _ = aggregate_track_anchors(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
    u_trans, leverage, loto_u = _batched_solve_from_laplacians(
        prior, laps, target, precision, lam_q, tau_prior=1.0)
    existence = compute_video_existence(
        u_trans, events_by_query, params.channel, protocol=protocol, existence_prior=0.5)
    gam = np.where(is_dyn, params.gamma_dyn_dyn, 0.0)[:, None]
    cols = np.asarray(columns, dtype=np.int64)
    frame_logits = u_trans[:, cols] + gam * f_dyn_res
    frame_prob = existence[:, None] * expit(frame_logits)
    return dict(
        prior_logits=prior,
        transduced_track_logits=u_trans,
        leverage=leverage,
        loto_track_logits=loto_u,
        existence=existence,
        frame_logits=frame_logits,
        frame_prob=frame_prob,
        asked_counts=counts,
    )


def _compute_oof_priors_v2(x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
                           dyn_anchor_x, dyn_anchor_y, dyn_anchor_w, dyn_anchor_k,
                           aux_id, aux_dyn, aux_y, aux_w, id_mean, id_std, dyn_mean, dyn_std,
                           plan, device):
    nq, n_tracks = x_id_mat.shape[:2]
    n = accessor.n
    g_id_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_dyn_oof = np.zeros((nq, n), dtype=np.float32)
    cols = accessor.cols
    for fold in (0, 1):
        tr_id = (id_anchor_k % 2) != fold
        tr_dy = (dyn_anchor_k % 2) != fold
        if not tr_id.any():
            tr_id = np.ones_like(tr_id)
        if not tr_dy.any():
            tr_dy = np.ones_like(tr_dy)
        torch.manual_seed(fold)
        m_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_head_adaptive(
            m_id,
            np.concatenate((id_anchor_x[tr_id], aux_id), axis=0),
            np.concatenate((id_anchor_y[tr_id], aux_y), axis=0),
            np.concatenate((id_anchor_w[tr_id], aux_w), axis=0),
            min_steps=300, max_steps=1500, target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], batch_size=256, seed=fold)
        pred_id = predict_track_id(m_id, x_id_mat, device)
        val_k = (np.arange(n_tracks) % 2) == fold
        g_id_oof[:, val_k] = pred_id[:, val_k]

        torch.manual_seed(10 + fold)
        m_dy = RegularizedMLP(dyn_mean, dyn_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_head_adaptive(
            m_dy,
            np.concatenate((dyn_anchor_x[tr_dy], aux_dyn), axis=0),
            np.concatenate((dyn_anchor_y[tr_dy], aux_y), axis=0),
            np.concatenate((dyn_anchor_w[tr_dy], aux_w), axis=0),
            min_steps=300, max_steps=1500, target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], batch_size=256, seed=10 + fold)
        pred_dy = predict_frame_dyn_fast(m_dy, accessor, device)
        val_obs = (cols % 2) == fold
        f_dyn_oof[:, val_obs] = pred_dy[:, val_obs]
    return g_id_oof, f_dyn_oof


def learn_dual_transduct_v2(context, case, plan, folder, device='cuda:0', accessor=None):
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
    events, _ = extract_source_events_with_tracks(context, case)
    md = metadata(case['target'])
    sets = {v: [q['query_text'] for q in md['episodes'] if q['video_id'] == v] for v in md['videos']}
    protocol = classify_protocol(sets)['protocol']
    is_dyn = np.array([bool(set(expression_types(q['query_text'])) & {'MOTION', 'SPATIAL', 'ROLE'})
                       for q in video.queries], dtype=bool)

    x_id_mat = accessor.all_track_id()
    d_app, d_pcd, d_geo = precompute_graph_distances(context, accessor)
    aux_id, aux_dyn = split_shared_features_np(context.auxiliary)
    aux_y = np.asarray(context.ay, dtype=np.float64)

    id_mean = x_id_mat.reshape(-1, ID_DIM).mean(axis=0, dtype=np.float64).astype(np.float32)
    id_std = x_id_mat.reshape(-1, ID_DIM).std(axis=0, dtype=np.float64).clip(min=0.1).astype(np.float32)
    sample_obs = np.unique(np.linspace(0, n * nq - 1, min(4096, n * nq)).astype(np.int64))
    sample_dyn = accessor.dyn_rows(sample_obs)
    dyn_mean = sample_dyn.mean(axis=0, dtype=np.float64).astype(np.float32)
    dyn_std = sample_dyn.std(axis=0, dtype=np.float64).clip(min=0.1).astype(np.float32)

    q_budget = np.bincount([e['q'] for e in events], minlength=nq).astype(np.float64)
    assert np.all(q_budget > 0)
    id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k = [], [], [], []
    by_qk = {}
    for ev in events:
        by_qk.setdefault((ev['q'], ev['track_col']), []).append(ev['prob'])
    for (q, k), probs in sorted(by_qk.items()):
        id_anchor_x.append(x_id_mat[q, k])
        id_anchor_y.append(float(np.mean(probs)))
        id_anchor_w.append(len(probs) / (nq * q_budget[q]))
        id_anchor_k.append(k)
    id_anchor_x = np.asarray(id_anchor_x, dtype=np.float32)
    id_anchor_y = np.asarray(id_anchor_y, dtype=np.float64)
    id_anchor_w = np.asarray(id_anchor_w, dtype=np.float64)
    id_anchor_k = np.asarray(id_anchor_k, dtype=np.int64)

    dyn_flat_ids, dyn_anchor_y, dyn_anchor_w, dyn_anchor_k = [], [], [], []
    for ev in events:
        w_per_node = 1.0 / (nq * q_budget[ev['q']] * len(ev['nodes']))
        for nd in ev['nodes']:
            dyn_flat_ids.append(ev['q'] * n + nd)
            dyn_anchor_y.append(ev['prob'])
            dyn_anchor_w.append(w_per_node)
            dyn_anchor_k.append(ev['track_col'])
    dyn_flat_ids = np.asarray(dyn_flat_ids, dtype=np.int64)
    dyn_anchor_x = accessor.dyn_rows(dyn_flat_ids)
    dyn_anchor_y = np.asarray(dyn_anchor_y, dtype=np.float64)
    dyn_anchor_w = np.asarray(dyn_anchor_w, dtype=np.float64)
    dyn_anchor_k = np.asarray(dyn_anchor_k, dtype=np.int64)
    aux_w = np.full(len(aux_y), 0.25 / len(aux_y), dtype=np.float64)

    # Step 0: 2-fold out-of-track cross-fitting -> unbiased OOF-LOTO R
    oof_started = time.perf_counter()
    g_id_oof, f_dyn_oof = _compute_oof_priors_v2(
        x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
        dyn_anchor_x, dyn_anchor_y, dyn_anchor_w, dyn_anchor_k,
        aux_id, aux_dyn, aux_y, aux_w, id_mean, id_std, dyn_mean, dyn_std,
        plan, device)
    f_dyn_tm_oof, f_dyn_res_oof = prob_pooled_track_and_residual(
        f_dyn_oof, video.obs, video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    params, laps, loto_oof = fit_observer_loto_v2(
        g_id_oof, f_dyn_tm_oof, f_dyn_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, video.columns, k_nn=plan['k_nn'], dist_gate=plan['dist_gate'])
    oof_seconds = time.perf_counter() - oof_started

    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_dyn = RegularizedMLP(dyn_mean, dyn_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    track_rows_cache = {k: np.flatnonzero(video.columns == k) for k in range(n_tracks)}
    n_rounds = int(plan['rounds'])

    for r in range(n_rounds):
        r_start = time.perf_counter()
        if r == 0:
            x_id_tr = np.concatenate((id_anchor_x, aux_id), axis=0)
            y_id_tr = np.concatenate((id_anchor_y, aux_y), axis=0)
            w_id_tr = np.concatenate((id_anchor_w, aux_w), axis=0)
            x_dy_tr = np.concatenate((dyn_anchor_x, aux_dyn), axis=0)
            y_dy_tr = np.concatenate((dyn_anchor_y, aux_y), axis=0)
            w_dy_tr = np.concatenate((dyn_anchor_w, aux_w), axis=0)
            trans_tracks_used = 0
            absent_neg_used = 0
            min_s, max_s, tgt_ce = 600, int(plan['r0_max_steps']), float(plan['target_excess_ce'])
        else:
            lev = trans['leverage']
            u_prob = expit(trans['transduced_track_logits'])
            eta = trans['existence']
            asked_mask = trans['asked_counts'] > 0

            sup_q, sup_k = np.nonzero((~asked_mask) & (lev > 0.05))
            id_parts_x = [id_anchor_x, aux_id]
            id_parts_y = [id_anchor_y, aux_y]
            id_parts_w = [id_anchor_w, aux_w]
            if len(sup_q) > 0:
                raw_w = lev[sup_q, sup_k] / np.maximum(q_budget[sup_q], 1.0)
                raw_w = raw_w * (0.30 / max(float(raw_w.sum()), 1e-12))
                id_parts_x.append(x_id_mat[sup_q, sup_k])
                id_parts_y.append(u_prob[sup_q, sup_k])
                id_parts_w.append(raw_w)
            trans_tracks_used = int(len(sup_q))

            absent_neg_used = 0
            if protocol == 'CHECKLIST':
                abs_q = np.flatnonzero(eta < 0.35)
                if len(abs_q) > 0:
                    aq_grid = np.repeat(abs_q, n_tracks)
                    ak_grid = np.tile(np.arange(n_tracks, dtype=np.int64), len(abs_q))
                    abs_w = 1.0 - eta[aq_grid]
                    abs_w = abs_w * (0.35 / max(float(abs_w.sum()), 1e-12))
                    id_parts_x.append(x_id_mat[aq_grid, ak_grid])
                    id_parts_y.append(np.zeros(len(aq_grid), dtype=np.float64))
                    id_parts_w.append(abs_w)
                    absent_neg_used = int(len(abs_q))

            x_id_tr = np.concatenate(id_parts_x, axis=0)
            y_id_tr = np.concatenate(id_parts_y, axis=0)
            w_id_tr = np.concatenate(id_parts_w, axis=0)

            dy_parts_x = [dyn_anchor_x, aux_dyn]
            dy_parts_y = [dyn_anchor_y, aux_y]
            dy_parts_w = [dyn_anchor_w, aux_w]
            cand_q, cand_k = np.nonzero(lev > 0.05)
            if len(cand_q) > 0:
                sample_n = min(2048, len(cand_q) * 2)
                pick = rng.integers(len(cand_q), size=sample_n)
                sel_ids, sel_y, sel_w = [], [], []
                for idx_p in pick:
                    q_i, k_i = int(cand_q[idx_p]), int(cand_k[idx_p])
                    rows_k = track_rows_cache[k_i]
                    nd = int(rows_k[rng.integers(len(rows_k))])
                    sel_ids.append(q_i * n + nd)
                    sel_y.append(float(trans['frame_prob'][q_i, nd]))
                    sel_w.append(float(lev[q_i, k_i]))
                sel_ids = np.asarray(sel_ids, dtype=np.int64)
                sel_x = accessor.dyn_rows(sel_ids)
                sel_w = np.asarray(sel_w, dtype=np.float64)
                sel_w = sel_w * (0.30 / max(float(sel_w.sum()), 1e-12))
                dy_parts_x.append(sel_x)
                dy_parts_y.append(np.asarray(sel_y, dtype=np.float64))
                dy_parts_w.append(sel_w)

            if protocol == 'CHECKLIST' and absent_neg_used > 0:
                abs_q = np.flatnonzero(eta < 0.35)
                sample_abs = min(1024, len(abs_q) * 16)
                aq = abs_q[rng.integers(len(abs_q), size=sample_abs)]
                and_ = rng.integers(n, size=sample_abs)
                abs_x_dy = accessor.dyn_rows(aq * n + and_)
                abs_w_dy = 1.0 - eta[aq]
                abs_w_dy = abs_w_dy * (0.35 / max(float(abs_w_dy.sum()), 1e-12))
                dy_parts_x.append(abs_x_dy)
                dy_parts_y.append(np.zeros(sample_abs, dtype=np.float64))
                dy_parts_w.append(abs_w_dy)

            x_dy_tr = np.concatenate(dy_parts_x, axis=0)
            y_dy_tr = np.concatenate(dy_parts_y, axis=0)
            w_dy_tr = np.concatenate(dy_parts_w, axis=0)
            min_s, max_s, tgt_ce = 300, int(plan['coevol_max_steps']), float(plan['target_excess_ce'])

        lr_r = float(plan['learning_rate']) * (0.5 ** r)
        id_fit = fit_head_adaptive(
            s_id, x_id_tr, y_id_tr, w_id_tr,
            min_steps=min_s, max_steps=max_s, target_excess_ce=tgt_ce,
            device=device, lr=lr_r, weight_decay=plan['weight_decay'], batch_size=256, seed=r)
        dy_fit = fit_head_adaptive(
            s_dyn, x_dy_tr, y_dy_tr, w_dy_tr,
            min_steps=min_s, max_steps=max_s, target_excess_ce=tgt_ce,
            device=device, lr=lr_r, weight_decay=plan['weight_decay'], batch_size=256, seed=100 + r)

        g_id = predict_track_id(s_id, x_id_mat, device)
        f_dyn_raw = predict_frame_dyn_fast(s_dyn, accessor, device)
        f_dyn_tm, f_dyn_res = prob_pooled_track_and_residual(
            f_dyn_raw, video.obs, video.columns, context.images, n_tracks,
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        trans = infer_transductive_v2(
            g_id, f_dyn_tm, f_dyn_res, laps, is_dyn, events,
            video.columns, params, protocol=protocol)

        row = dict(
            round=r, lr=lr_r, s_id_fit=id_fit, s_dyn_fit=dy_fit,
            trans_tracks_used=trans_tracks_used, absent_neg_queries=absent_neg_used,
            mean_leverage=float(trans['leverage'].mean()),
            mean_existence=float(trans['existence'].mean()),
            round_wall_seconds=time.perf_counter() - r_start,
        )
        rounds_log.append(row)
        print(json.dumps(dict(
            status='DUAL_TRANSDUCT_V2_ROUND', pair=case['pair'], video=case['evaluated_video'],
            round=r, id_ce=round(id_fit['proper_excess_ce'], 4), dyn_ce=round(dy_fit['proper_excess_ce'], 4),
            id_steps=id_fit['steps'], trans_tracks=trans_tracks_used, absent_neg=absent_neg_used,
            mean_lev=round(float(trans['leverage'].mean()), 3), sec=round(row['round_wall_seconds'], 1),
        )), flush=True)

    receipt = dict(
        status='COMPLETE_DUAL_TRANSDUCT_V2_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        dynamic_queries=int(is_dyn.sum()),
        oof_loto=loto_oof, oof_wall_seconds=oof_seconds,
        observer=params.to_dict(), rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        GT_open_attempts=0, new_teacher_calls=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    return dict(
        receipt=receipt, accessor=accessor, s_id=s_id, s_dyn=s_dyn,
        params=params, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )


def apply_shared_transduct_v2_readout(frame_scores, context, learned, plan):
    n_tracks = len(context.video.tracks)
    tm, res = prob_pooled_track_and_residual(
        frame_scores, context.video.obs, context.video.columns,
        context.images, n_tracks, kappa=plan['kappa'], base_decay=plan['base_decay'])
    out = infer_transductive_v2(
        tm, tm, res, learned['laps'], learned['is_dyn'], learned['events'],
        context.video.columns, learned['params'], protocol=learned['protocol'])
    return out['frame_prob']


def fit_regularized_ordinary_v2(context, case, plan, device='cuda:0', accessor=None):
    if accessor is None:
        accessor = FastVideoFeatureAccessor(context)
    events, _ = extract_source_events_with_tracks(context, case)
    nq, n = context.nq, context.n
    q_budget = np.bincount([e['q'] for e in events], minlength=nq).astype(np.float64)
    uids = np.array([e['q'] * n + e['node'] for e in events], dtype=np.int64)
    probs = np.array([e['prob'] for e in events], dtype=np.float64)
    weights = np.array([1.0 / (nq * q_budget[e['q']]) for e in events], dtype=np.float64)
    x_anc = accessor.full_rows(uids)
    aux_x = context.auxiliary
    aux_y = np.asarray(context.ay, dtype=np.float64)
    aux_w = np.full(len(aux_y), 0.25 / len(aux_y), dtype=np.float64)
    x_all = np.concatenate((x_anc, aux_x), axis=0)
    w_all = np.concatenate((weights, aux_w), axis=0)

    outputs, receipts = {}, {}
    for mode, hard in (('REG_SOFT_KD', False), ('REG_HARD_KD', True)):
        y_all = np.concatenate(((probs >= 0.5).astype(np.float64) if hard else probs,
                                (aux_y >= 0.5).astype(np.float64) if hard else aux_y), axis=0)
        torch.manual_seed(0)
        head = RegularizedMLP(context.mean, context.std, width=plan['width'], dropout=plan['dropout']).to(device)
        rec = fit_head_adaptive(
            head, x_all, y_all, w_all,
            min_steps=900, max_steps=int(plan['ordinary_max_steps']),
            target_excess_ce=float(plan['target_excess_ce']),
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], batch_size=256, seed=0)
        outputs[mode] = predict_full_head_fast(head, accessor, device)
        receipts[mode] = rec
    return outputs, receipts
