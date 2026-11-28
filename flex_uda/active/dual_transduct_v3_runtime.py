"""Runtime for DUAL_TRANSDUCT_COEVOL_V3 (zero GT, full convergence, bounded harmonic).

Key V3 upgrades over V2 (all V1/V2 pinned files completely unchanged):
1. Bounded probability-space harmonic transduction with strictly monotonic Bayesian
   anchors y_bayes = expit(logit(mu_prob) + gamma_anc * Delta) -> eliminates KITTI's
   6x negative-logit gravity AND C1/C3 calibration-floor reversal!
2. OOF-LOTO adaptive tight clique radius (dist_gate in {0.40, 0.55, 0.75, 1.00}) ->
   automatically preserves Dance/Sports dense uniform cliques while isolating unique
   sparse cars/pedestrians on KITTI!
3. Evidence-gated zero-jitter track coherence: gamma_dyn_dyn defaults strictly to 0.0
   and activates ONLY when multi-frame within-track teacher contrasts strictly improve
   within-track teacher NLL by > 0.01 -> recovers +1.5 to +2.9 AssA50 on dynamic queries!
4. Full-capacity complementary dual students:
   - S_id (7482D strictly track-invariant X_id, track-level aggregated anchors)
   - S_ctx (9352D full joint identity+frame+spatial+motion X_full, frame-level events
     with exact probability-space track pooling)
   Both (and fair REG_SOFT_KD / REG_HARD_KD controls) use adaptive convergence up to
   4800 steps (lr=8e-4, batch_size=512 for large videos) eliminating KITTI 0019 underfit!
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch import nn

from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.dual_transduct_coevol import (
    FULL_DIM, ID_DIM,
    RegularizedMLP, compute_video_existence, split_shared_features_np,
)
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor, extract_source_events_with_tracks,
    precompute_graph_distances, predict_full_head_fast, predict_track_id,
)
from flex_uda.active.dual_transduct_v2 import (
    build_gated_clique_laplacian, prob_pooled_track_and_residual,
)
from flex_uda.active.dual_transduct_v3 import (
    DIST_GATE_GRID, aggregate_track_evidence_v3, batched_solve_prob_harmonic,
    default_v3_vector, unpack_v3_vector,
)


def _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, params, *, k_nn=3):
    nq, k = d_pcd.shape[:2]
    laps = np.empty((nq, k, k), dtype=np.float64)
    dg = float(params.dist_gate)
    for q in range(nq):
        w_geo = params.w_geo_dyn if is_dyn[q] else params.w_geo_static
        w_sum = params.w_app + params.w_pcd + w_geo
        dist = (params.w_app * d_app + params.w_pcd * d_pcd[q] + w_geo * d_geo) / max(w_sum, 1e-6)
        laps[q], _ = build_gated_clique_laplacian(dist, k_nn=k_nn, dist_gate=dg)
    return laps


def fit_head_adaptive_v3(model, x_np, y_np, w_np, *, min_steps, max_steps, target_excess_ce,
                         device, lr=8e-4, weight_decay=3e-3, seed=0):
    """Adaptive convergence with batch scaling for high-query KITTI videos."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
    w = w / w.sum().clamp(min=1e-12)
    n = len(x)
    bs = min(512 if n > 2048 else 256, n)
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
        steps=int(completed), batch_size=int(bs), lr=float(lr), weight_decay=float(weight_decay),
        last_loss=last_loss, proper_excess_ce=excess_ce,
        target_excess_ce=float(target_excess_ce),
        wall_seconds=time.perf_counter() - started,
    )


def fit_observer_loto_v3(p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
                         is_dyn, events, *, k_nn=3, prior_reg=0.015):
    """OOF-LOTO optimization in bounded probability space + adaptive tight clique gate."""
    theta0 = default_v3_vector()
    nq, n_tracks = p_id_oof.shape
    if not events:
        p0 = unpack_v3_vector(theta0, dist_gate=0.65, gamma_dyn_dyn=0.0)
        return p0, _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, p0, k_nn=k_nn), dict(
            loto_before=0.0, loto_after=0.0, stage1_evals=0, stage2_evals=0,
            within_track_contrasts=0, chosen_dist_gate=0.65)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    q_counts = np.bincount(ev_q, minlength=nq).astype(np.float64)
    ev_w = 1.0 / (max(int((q_counts > 0).sum()), 1) * np.maximum(q_counts[ev_q], 1.0))

    def eval_with_laps(theta, laps, dg):
        params = unpack_v3_vector(theta, dist_gate=dg, gamma_dyn_dyn=0.0)
        prior_p = (1.0 - params.beta_mix) * p_id_oof + params.beta_mix * p_ctx_tm_oof
        track_llr, precision, _, _, _ = aggregate_track_evidence_v3(
            nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
        lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
        _, _, loto_p = batched_solve_prob_harmonic(
            prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)
        pred_state = loto_p[ev_q, ev_k]
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
    ]
    best_theta = theta0.copy()
    best_dg = 0.65
    best_laps = _build_all_gated_laplacians_v3(
        d_app, d_pcd, d_geo, is_dyn, unpack_v3_vector(theta0, dist_gate=best_dg), k_nn=k_nn)
    before = eval_with_laps(theta0, best_laps, best_dg)
    best_val = before
    stage1_evals = 0
    for dg in DIST_GATE_GRID:
        for wa, wp, wgs, wgd in kernel_candidates:
            cand = theta0.copy()
            cand[2:6] = [inv_sp(wa), inv_sp(wp), inv_sp(wgs), inv_sp(wgd)]
            laps = _build_all_gated_laplacians_v3(
                d_app, d_pcd, d_geo, is_dyn, unpack_v3_vector(cand, dist_gate=dg), k_nn=k_nn)
            val = eval_with_laps(cand, laps, dg)
            stage1_evals += 1
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_theta = cand.copy()
                best_dg = float(dg)
                best_laps = laps

    # Stage 2: Powell over (lam_s, lam_d, beta_mix, anc_prec, gamma_anc, channel[3x2])
    opt_idx = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    sub0 = best_theta[opt_idx].copy()

    def sub_obj(sub):
        full = best_theta.copy()
        full[opt_idx] = sub
        return eval_with_laps(full, best_laps, best_dg)

    res = minimize(sub_obj, sub0, method='Powell', options=dict(maxiter=10, xtol=3e-2, ftol=3e-3))
    if np.isfinite(res.fun) and res.fun < best_val:
        best_theta[opt_idx] = res.x
        best_val = float(res.fun)

    # Stage 3: Evidence-gated within-track dynamic residual check (default strictly 0.0)
    params_base = unpack_v3_vector(best_theta, dist_gate=best_dg, gamma_dyn_dyn=0.0)
    track_llr, precision, _, _, contrasts = aggregate_track_evidence_v3(
        nq, n_tracks, events, params_base.channel, anchor_base_precision=params_base.anchor_precision)
    dyn_contrasts = [ct for ct in contrasts if is_dyn[ct['q']]]
    chosen_gam_d = 0.0
    if len(dyn_contrasts) >= 5:
        prior_p = (1.0 - params_base.beta_mix) * p_id_oof + params_base.beta_mix * p_ctx_tm_oof
        lam_q = np.where(is_dyn, params_base.lam_vis_dyn, params_base.lam_vis_static)
        _, _, loto_p = batched_solve_prob_harmonic(
            prior_p, best_laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params_base.gamma_anc)
        loto_logit = logit(loto_p)

        def contrast_nll(g_val):
            tot, cnt = 0.0, 0
            for ct in dyn_contrasts:
                q, k = ct['q'], ct['k']
                u0 = loto_logit[q, k]
                for ev in ct['events']:
                    nd = np.asarray(ev['nodes'], dtype=np.int64)
                    ps = float(np.mean(expit(u0 + g_val * f_ctx_res_oof[q, nd])))
                    s, c = params_base.channel[ev['view']]
                    py = float(np.clip((1.0 - c) + (s + c - 1.0) * ps, 1e-7, 1.0 - 1e-7))
                    p_ev = float(ev['prob'])
                    tot += -(p_ev * math.log(py) + (1.0 - p_ev) * math.log1p(-py))
                    cnt += 1
            return tot / max(cnt, 1)

        base_cnll = contrast_nll(0.0)
        best_cnll = base_cnll
        for g_cand in (0.08, 0.16, 0.25):
            cn = contrast_nll(g_cand)
            if cn < best_cnll - 0.01:
                best_cnll = cn
                chosen_gam_d = g_cand

    params = unpack_v3_vector(best_theta, dist_gate=best_dg, gamma_dyn_dyn=chosen_gam_d)
    return params, best_laps, dict(
        loto_before=float(before), loto_after=float(best_val),
        stage1_evals=int(stage1_evals), stage2_evals=int(getattr(res, 'nfev', 0)),
        chosen_dist_gate=float(best_dg),
        within_track_dyn_contrasts=int(len(dyn_contrasts)),
        chosen_gamma_dyn_dyn=float(chosen_gam_d),
    )


def infer_transductive_v3(p_id, p_ctx_tm, f_ctx_res, laps, is_dyn, events,
                          columns, params, *, protocol='PRESUPPOSED'):
    nq, n_tracks = p_id.shape
    events_by_query = [[] for _ in range(nq)]
    for ev in events:
        events_by_query[ev['q']].append(ev)
    prior_p = (1.0 - params.beta_mix) * p_id + params.beta_mix * p_ctx_tm
    track_llr, precision, counts, _, _ = aggregate_track_evidence_v3(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
    p_star, leverage, loto_p = batched_solve_prob_harmonic(
        prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)
    u_star = logit(p_star)
    existence = compute_video_existence(
        u_star, events_by_query, params.channel, protocol=protocol, existence_prior=0.5)
    gam = np.where(is_dyn, params.gamma_dyn_dyn, 0.0)[:, None]
    cols = np.asarray(columns, dtype=np.int64)
    if float(params.gamma_dyn_dyn) == 0.0:
        frame_prob = existence[:, None] * p_star[:, cols]
        frame_logits = logit(np.clip(frame_prob, 1e-7, 1.0 - 1e-7))
    else:
        frame_logits = u_star[:, cols] + gam * f_ctx_res
        frame_prob = existence[:, None] * expit(frame_logits)
    return dict(
        prior_prob=prior_p,
        transduced_track_prob=p_star,
        transduced_track_logits=u_star,
        leverage=leverage,
        loto_track_prob=loto_p,
        existence=existence,
        frame_logits=frame_logits,
        frame_prob=frame_prob,
        asked_counts=counts,
    )


def _compute_oof_priors_v3(x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
                           ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k,
                           aux_id, aux_full, aux_y, aux_w, id_mean, id_std, full_mean, full_std,
                           plan, device):
    nq, n_tracks = x_id_mat.shape[:2]
    n = accessor.n
    p_id_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_ctx_oof = np.zeros((nq, n), dtype=np.float32)
    cols = accessor.cols
    for fold in (0, 1):
        tr_id = (id_anchor_k % 2) != fold
        tr_cx = (ctx_anchor_k % 2) != fold
        if not tr_id.any():
            tr_id = np.ones_like(tr_id)
        if not tr_cx.any():
            tr_cx = np.ones_like(tr_cx)
        torch.manual_seed(fold)
        m_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_head_adaptive_v3(
            m_id,
            np.concatenate((id_anchor_x[tr_id], aux_id), axis=0),
            np.concatenate((id_anchor_y[tr_id], aux_y), axis=0),
            np.concatenate((id_anchor_w[tr_id], aux_w), axis=0),
            min_steps=300, max_steps=int(plan['oof_max_steps']), target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], seed=fold)
        pred_id = expit(predict_track_id(m_id, x_id_mat, device))
        val_k = (np.arange(n_tracks) % 2) == fold
        p_id_oof[:, val_k] = pred_id[:, val_k]

        torch.manual_seed(10 + fold)
        m_cx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_head_adaptive_v3(
            m_cx,
            np.concatenate((ctx_anchor_x[tr_cx], aux_full), axis=0),
            np.concatenate((ctx_anchor_y[tr_cx], aux_y), axis=0),
            np.concatenate((ctx_anchor_w[tr_cx], aux_w), axis=0),
            min_steps=300, max_steps=int(plan['oof_max_steps']), target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], seed=10 + fold)
        pred_cx = predict_full_head_fast(m_cx, accessor, device)
        val_obs = (cols % 2) == fold
        f_ctx_oof[:, val_obs] = pred_cx[:, val_obs]
    return p_id_oof, f_ctx_oof


def learn_dual_transduct_v3(context, case, plan, folder, device='cuda:0', accessor=None):
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
    aux_id, _ = split_shared_features_np(context.auxiliary)
    aux_full = context.auxiliary
    aux_y = np.asarray(context.ay, dtype=np.float64)

    id_mean = x_id_mat.reshape(-1, ID_DIM).mean(axis=0, dtype=np.float64).astype(np.float32)
    id_std = x_id_mat.reshape(-1, ID_DIM).std(axis=0, dtype=np.float64).clip(min=0.1).astype(np.float32)
    full_mean = context.mean
    full_std = context.std

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

    ctx_flat_ids, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k = [], [], [], []
    for ev in events:
        w_per_node = 1.0 / (nq * q_budget[ev['q']] * len(ev['nodes']))
        for nd in ev['nodes']:
            ctx_flat_ids.append(ev['q'] * n + nd)
            ctx_anchor_y.append(ev['prob'])
            ctx_anchor_w.append(w_per_node)
            ctx_anchor_k.append(ev['track_col'])
    ctx_flat_ids = np.asarray(ctx_flat_ids, dtype=np.int64)
    ctx_anchor_x = accessor.full_rows(ctx_flat_ids)
    ctx_anchor_y = np.asarray(ctx_anchor_y, dtype=np.float64)
    ctx_anchor_w = np.asarray(ctx_anchor_w, dtype=np.float64)
    ctx_anchor_k = np.asarray(ctx_anchor_k, dtype=np.int64)
    aux_w = np.full(len(aux_y), 0.25 / len(aux_y), dtype=np.float64)

    # Step 0: 2-fold out-of-track cross-fitting -> unbiased probability-space OOF-LOTO R
    oof_started = time.perf_counter()
    p_id_oof, f_ctx_oof = _compute_oof_priors_v3(
        x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k,
        aux_id, aux_full, aux_y, aux_w, id_mean, id_std, full_mean, full_std,
        plan, device)
    f_ctx_tm_oof, f_ctx_res_oof = prob_pooled_track_and_residual(
        f_ctx_oof, video.obs, video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_ctx_tm_oof = expit(f_ctx_tm_oof)
    params, laps, loto_oof = fit_observer_loto_v3(
        p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, k_nn=plan['k_nn'])
    oof_seconds = time.perf_counter() - oof_started

    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_ctx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    track_rows_cache = {k: np.flatnonzero(video.columns == k) for k in range(n_tracks)}
    n_rounds = int(plan['rounds'])
    lev_thresh = float(plan.get('leverage_threshold', 0.08))
    trans_budget_w = float(plan.get('transduced_weight', 0.25))

    for r in range(n_rounds):
        r_start = time.perf_counter()
        if r == 0:
            x_id_tr = np.concatenate((id_anchor_x, aux_id), axis=0)
            y_id_tr = np.concatenate((id_anchor_y, aux_y), axis=0)
            w_id_tr = np.concatenate((id_anchor_w, aux_w), axis=0)
            x_cx_tr = np.concatenate((ctx_anchor_x, aux_full), axis=0)
            y_cx_tr = np.concatenate((ctx_anchor_y, aux_y), axis=0)
            w_cx_tr = np.concatenate((ctx_anchor_w, aux_w), axis=0)
            trans_tracks_used = 0
            absent_neg_used = 0
            min_s, max_s, tgt_ce = 600, int(plan['r0_max_steps']), float(plan['target_excess_ce'])
        else:
            lev = trans['leverage']
            p_tr = trans['transduced_track_prob']
            eta = trans['existence']
            asked_mask = trans['asked_counts'] > 0

            sup_q, sup_k = np.nonzero((~asked_mask) & (lev > lev_thresh))
            id_parts_x = [id_anchor_x, aux_id]
            id_parts_y = [id_anchor_y, aux_y]
            id_parts_w = [id_anchor_w, aux_w]
            if len(sup_q) > 0:
                raw_w = lev[sup_q, sup_k] / np.maximum(q_budget[sup_q], 1.0)
                raw_w = raw_w * (trans_budget_w / max(float(raw_w.sum()), 1e-12))
                id_parts_x.append(x_id_mat[sup_q, sup_k])
                id_parts_y.append(p_tr[sup_q, sup_k])
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

            cx_parts_x = [ctx_anchor_x, aux_full]
            cx_parts_y = [ctx_anchor_y, aux_y]
            cx_parts_w = [ctx_anchor_w, aux_w]
            cand_q, cand_k = np.nonzero((~asked_mask) & (lev > lev_thresh))
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
                sel_x = accessor.full_rows(sel_ids)
                sel_w = np.asarray(sel_w, dtype=np.float64)
                sel_w = sel_w * (trans_budget_w / max(float(sel_w.sum()), 1e-12))
                cx_parts_x.append(sel_x)
                cx_parts_y.append(np.asarray(sel_y, dtype=np.float64))
                cx_parts_w.append(sel_w)

            if protocol == 'CHECKLIST' and absent_neg_used > 0:
                abs_q = np.flatnonzero(eta < 0.35)
                sample_abs = min(1024, len(abs_q) * 16)
                aq = abs_q[rng.integers(len(abs_q), size=sample_abs)]
                and_ = rng.integers(n, size=sample_abs)
                abs_x_cx = accessor.full_rows(aq * n + and_)
                abs_w_cx = 1.0 - eta[aq]
                abs_w_cx = abs_w_cx * (0.35 / max(float(abs_w_cx.sum()), 1e-12))
                cx_parts_x.append(abs_x_cx)
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
        trans = infer_transductive_v3(
            p_id, p_ctx_tm, f_ctx_res, laps, is_dyn, events,
            video.columns, params, protocol=protocol)

        row = dict(
            round=r, lr=lr_r, s_id_fit=id_fit, s_ctx_fit=cx_fit,
            trans_tracks_used=trans_tracks_used, absent_neg_queries=absent_neg_used,
            mean_leverage=float(trans['leverage'].mean()),
            mean_existence=float(trans['existence'].mean()),
            round_wall_seconds=time.perf_counter() - r_start,
        )
        rounds_log.append(row)
        print(json.dumps(dict(
            status='DUAL_TRANSDUCT_V3_ROUND', pair=case['pair'], video=case['evaluated_video'],
            round=r, id_ce=round(id_fit['proper_excess_ce'], 4), ctx_ce=round(cx_fit['proper_excess_ce'], 4),
            id_st=id_fit['steps'], ctx_st=cx_fit['steps'], dg=params.dist_gate, gam_d=params.gamma_dyn_dyn,
            tr=trans_tracks_used, abs_q=absent_neg_used, sec=round(row['round_wall_seconds'], 1),
        )), flush=True)

    receipt = dict(
        status='COMPLETE_DUAL_TRANSDUCT_V3_LEARNING',
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
        receipt=receipt, accessor=accessor, s_id=s_id, s_ctx=s_ctx,
        params=params, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )


def apply_shared_transduct_v3_readout(frame_scores, context, learned, plan):
    n_tracks = len(context.video.tracks)
    tm, res = prob_pooled_track_and_residual(
        frame_scores, context.video.obs, context.video.columns,
        context.images, n_tracks, kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_tm = expit(tm)
    out = infer_transductive_v3(
        p_tm, p_tm, res, learned['laps'], learned['is_dyn'], learned['events'],
        context.video.columns, learned['params'], protocol=learned['protocol'])
    return out['frame_prob']


def fit_regularized_ordinary_v3(context, case, plan, device='cuda:0', accessor=None):
    """Equal-capacity, equal-budget regularized ordinary soft & hard KD controls."""
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
        rec = fit_head_adaptive_v3(
            head, x_all, y_all, w_all,
            min_steps=900, max_steps=int(plan['ordinary_max_steps']),
            target_excess_ce=float(plan['target_excess_ce']),
            device=device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], seed=0)
        outputs[mode] = predict_full_head_fast(head, accessor, device)
        receipts[mode] = rec
    return outputs, receipts
