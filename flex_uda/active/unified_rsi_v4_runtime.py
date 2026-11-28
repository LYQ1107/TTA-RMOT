"""UNIFIED_RSI_RMOT_V4: Ventral/Dorsal Two-Stream Convex OOF Gating + Godel-Machine Verified Trust-Region RSI.

Grounded directly in zero-GT V3 diagnostics across Dance, SportsMOT, and KITTI-V2:
  1. Ventral/Dorsal Two-Stream Strictly Convex OOF Gating (I2-A):
       Minimizes exact balanced 2-fold OOF teacher NLL separately on static vs dynamic query strata:
           p_prior(q, k) = (1 - beta_q) * p_id(q, k) + beta_q * p_ctx(q, k)
       Because binary cross-entropy of a linear probability mixture is strictly convex in beta in [0, 1],
       the unique global optimum guarantees OOF_NLL(two-stream) <= min(OOF_NLL(S_id), OOF_NLL(S_ctx))
       on BOTH static appearance and dynamic spatial/motion queries (eliminating motion-blind S_id bloat).
  2. Sign-Balanced EB-Scaled Trust Region + 2-Fold OOF Godel-Machine Verification Gate (I2-B):
       Removes V3's asymmetric |p_tr - p_pr| > 0.08 upward-bias filter, weights all high-leverage unasked
       tracks proportionally to Empirical-Bayes graph shrinkage with mean-neutral upward-drift damping on
       PRESUPPOSED queries, and accepts R1/R2 self-modification ONLY when 2-fold track-disjoint held-out
       teacher NLL strictly improves (preventing active-sampling confirmation-bias echo chambers).
  3. Exact Fixed-QA Replay of Sealed V3 Hybrid Challenger Evidence (I3):
       Reuses V3's exact 55,019 sealed Qwen3-VL-2B active teacher responses (0 new Qwen forwards, 0 GT).
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
import torch

from flex_uda.active.dual_transduct_coevol import (
    ID_DIM,
    RegularizedMLP,
    compute_video_existence,
    split_shared_features_np,
)
from flex_uda.active.dual_transduct_v2 import prob_pooled_track_and_residual
from flex_uda.active.dual_transduct_v3 import (
    DIST_GATE_GRID,
    TransductiveObserverV3Params,
    aggregate_track_evidence_v3,
    batched_solve_prob_harmonic,
    default_v3_vector,
    unpack_v3_vector,
)
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor,
    precompute_graph_distances,
    predict_full_head_fast,
    predict_track_id,
)
from flex_uda.active.dual_transduct_v3_runtime import (
    _build_all_gated_laplacians_v3,
    fit_head_adaptive_v3,
)
from flex_uda.active.unified_rsi_runtime import (
    broker_rows_to_events,
    build_anchor_tensors,
)
from flex_tta.runtime.pf_receipts import atomic_json


def _balanced_event_weights(ev_q, ev_p, nq, mask=None):
    """Query-normalized balanced weights (70% class-balanced + 30% natural) for OOF/LOTO NLL."""
    if mask is None:
        qm, pm = ev_q, ev_p
    else:
        qm, pm = ev_q[mask], ev_p[mask]
    if len(qm) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 1.0, 1.0
    q_cnt = np.bincount(qm, minlength=nq).astype(np.float64)
    w = 1.0 / (max(int((q_cnt > 0).sum()), 1) * np.maximum(q_cnt[qm], 1.0))
    pos_m = float(np.sum(w * pm))
    neg_m = float(np.sum(w * (1.0 - pm)))
    wp = 0.70 * (0.5 / max(pos_m, 1e-4)) + 0.30
    wn = 0.70 * (0.5 / max(neg_m, 1e-4)) + 0.30
    return w, pm, wp, wn


def fit_two_stream_convex_beta(p_id_oof, p_ctx_oof, is_dyn, events):
    """Exact strictly convex OOF minimization of beta_static and beta_dyn in [0, 1].

    Theorem: binary cross-entropy of (1 - b) * p_id + b * p_ctx is strictly convex in b in [0, 1]
    whenever p_id != p_ctx on any positive-weight event, guaranteeing unique global optimum
    with NLL(b*) <= min(NLL(0), NLL(1)) separately on static and dynamic strata.
    """
    nq = p_id_oof.shape[0]
    if not events:
        return 0.5, 0.5, dict(static_nll_id=0.0, static_nll_ctx=0.0, static_nll_mix=0.0,
                              dyn_nll_id=0.0, dyn_nll_ctx=0.0, dyn_nll_mix=0.0)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_is_dyn = is_dyn[ev_q]

    def solve_stratum(mask):
        if not np.any(mask):
            return 0.5, 0.0, 0.0, 0.0
        pid = np.clip(p_id_oof[ev_q[mask], ev_k[mask]], 1e-6, 1.0 - 1e-6)
        pcx = np.clip(p_ctx_oof[ev_q[mask], ev_k[mask]], 1e-6, 1.0 - 1e-6)
        w, pm, wp, wn = _balanced_event_weights(ev_q, ev_p, nq, mask=mask)

        def loss_fn(b):
            p = np.clip((1.0 - b) * pid + b * pcx, 1e-7, 1.0 - 1e-7)
            return float(-np.sum(w * (wp * pm * np.log(p) + wn * (1.0 - pm) * np.log1p(-p))))

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


def fit_observer_two_stream_loto(p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
                                 is_dyn, events, beta_q, *, k_nn=3, prior_reg=0.015):
    """Balanced OOF-LOTO observer fit with exact two-stream prior (1 - beta_q)*p_id + beta_q*p_ctx."""
    theta0 = default_v3_vector()
    nq, n_tracks = p_id_oof.shape
    prior_p = (1.0 - beta_q[:, None]) * p_id_oof + beta_q[:, None] * p_ctx_tm_oof
    if not events:
        p0 = unpack_v3_vector(theta0, dist_gate=0.65, gamma_dyn_dyn=0.0)
        return p0, _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, p0, k_nn=k_nn), dict(
            loto_before=0.0, loto_after=0.0, stage1_evals=0, stage2_evals=0, chosen_dist_gate=0.65)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_w, _, w_pos, w_neg = _balanced_event_weights(ev_q, ev_p, nq)

    def eval_with_laps(theta, laps, dg):
        params = unpack_v3_vector(theta, dist_gate=dg, gamma_dyn_dyn=0.0)
        track_llr, precision, _, _, _ = aggregate_track_evidence_v3(
            nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
        lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
        _, _, loto_p = batched_solve_prob_harmonic(
            prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)
        pred_state = loto_p[ev_q, ev_k]
        s = params.channel[ev_v, 0]
        c = params.channel[ev_v, 1]
        p_yes = np.clip((1.0 - c) + (s + c - 1.0) * pred_state, 1e-7, 1.0 - 1e-7)
        nll = -np.sum(ev_w * (w_pos * ev_p * np.log(p_yes) + w_neg * (1.0 - ev_p) * np.log1p(-p_yes)))
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
        for wa, wp_k, wgs, wgd in kernel_candidates:
            cand = theta0.copy()
            cand[2:6] = [inv_sp(wa), inv_sp(wp_k), inv_sp(wgs), inv_sp(wgd)]
            laps = _build_all_gated_laplacians_v3(
                d_app, d_pcd, d_geo, is_dyn, unpack_v3_vector(cand, dist_gate=dg), k_nn=k_nn)
            val = eval_with_laps(cand, laps, dg)
            stage1_evals += 1
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_theta = cand.copy()
                best_dg = dg
                best_laps = laps

    # Optimize everything EXCEPT beta_mix (index 6), which is fixed at the exact convex two-stream optimum!
    opt_idx = [0, 1, 7, 8, 9, 10, 11, 12, 13, 14]
    sub0 = best_theta[opt_idx].copy()

    def sub_obj(sub):
        full = best_theta.copy()
        full[opt_idx] = sub
        return eval_with_laps(full, best_laps, best_dg)

    res = minimize(sub_obj, sub0, method='Powell', options=dict(maxiter=10, xtol=3e-2, ftol=3e-3))
    if np.isfinite(res.fun) and res.fun < best_val:
        best_theta[opt_idx] = res.x
        best_val = float(res.fun)

    # Evidence-gated dynamic within-track contrast check (zero-jitter default)
    chosen_gam_d = 0.0
    p_tmp = unpack_v3_vector(best_theta, dist_gate=best_dg, gamma_dyn_dyn=0.0)
    track_llr, precision, _, _, _ = aggregate_track_evidence_v3(
        nq, n_tracks, events, p_tmp.channel, anchor_base_precision=p_tmp.anchor_precision)
    lam_q = np.where(is_dyn, p_tmp.lam_vis_dyn, p_tmp.lam_vis_static)
    _, _, loto_p = batched_solve_prob_harmonic(
        prior_p, best_laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=p_tmp.gamma_anc)
    by_qk = {}
    for ev in events:
        if is_dyn[ev['q']]:
            by_qk.setdefault((ev['q'], ev['track_col']), []).append(ev)
    dyn_contrasts = [evs for evs in by_qk.values() if len(evs) >= 2 and
                     max(e['prob'] for e in evs) - min(e['prob'] for e in evs) >= 0.35]
    if len(dyn_contrasts) >= 5:
        def contrast_nll(g_val):
            tot, cnt = 0.0, 0
            for evs in dyn_contrasts:
                q_i, k_i = evs[0]['q'], evs[0]['track_col']
                base_u = math.log(np.clip(loto_p[q_i, k_i], 1e-6, 1.0 - 1e-6) /
                                  (1.0 - np.clip(loto_p[q_i, k_i], 1e-6, 1.0 - 1e-6)))
                for e in evs:
                    r_val = float(np.mean(f_ctx_res_oof[q_i, e['nodes']]))
                    st_p = float(expit(base_u + g_val * r_val))
                    s_v, c_v = p_tmp.channel[e['view']]
                    py = min(max((1.0 - c_v) + (s_v + c_v - 1.0) * st_p, 1e-7), 1.0 - 1e-7)
                    tot -= e['prob'] * math.log(py) + (1.0 - e['prob']) * math.log1p(-py)
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


def empirical_bayes_graph_gate_two_stream(params, laps, p_id_oof, p_ctx_tm_oof, is_dyn, events,
                                          beta_q, *, delta_0=0.020):
    """Exact OOF-LOTO Empirical-Bayes shrinkage gate using the two-stream prior."""
    nq, n_tracks = p_id_oof.shape
    if not events:
        return params, dict(lam_s_raw=params.lam_vis_static, lam_d_raw=params.lam_vis_dyn,
                            lam_s_gated=params.lam_vis_static, lam_d_gated=params.lam_vis_dyn,
                            gain_static=0.0, gain_dyn=0.0, shrink_static=1.0, shrink_dyn=1.0,
                            delta_0=float(delta_0))

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_is_dyn = is_dyn[ev_q]

    prior_p = (1.0 - beta_q[:, None]) * p_id_oof + beta_q[:, None] * p_ctx_tm_oof
    track_llr, precision, _, _, _ = aggregate_track_evidence_v3(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    s = params.channel[ev_v, 0]
    c = params.channel[ev_v, 1]

    def subset_nll(lam_s, lam_d, mask):
        if not np.any(mask):
            return 0.0
        lam_q = np.where(is_dyn, lam_d, lam_s)
        _, _, loto_p = batched_solve_prob_harmonic(
            prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)
        py = np.clip((1.0 - c[mask]) + (s[mask] + c[mask] - 1.0) * loto_p[ev_q[mask], ev_k[mask]], 1e-7, 1.0 - 1e-7)
        w, pm, wp, wn = _balanced_event_weights(ev_q, ev_p, nq, mask=mask)
        return float(-np.sum(w * (wp * pm * np.log(py) + wn * (1.0 - pm) * np.log1p(-py))))

    st_mask = ~ev_is_dyn
    dy_mask = ev_is_dyn
    gain_s = shrink_s = 0.0
    if np.any(st_mask):
        gain_s = subset_nll(0.0, params.lam_vis_dyn, st_mask) - subset_nll(params.lam_vis_static, params.lam_vis_dyn, st_mask)
        shrink_s = max(gain_s - delta_0, 0.0) / (gain_s + 1e-8) if gain_s > 0 else 0.0

    gain_d = shrink_d = 0.0
    if np.any(dy_mask):
        gain_d = subset_nll(params.lam_vis_static, 0.0, dy_mask) - subset_nll(params.lam_vis_static, params.lam_vis_dyn, dy_mask)
        shrink_d = max(gain_d - delta_0, 0.0) / (gain_d + 1e-8) if gain_d > 0 else 0.0

    gated = TransductiveObserverV3Params(
        lam_vis_static=float(params.lam_vis_static * shrink_s),
        lam_vis_dyn=float(params.lam_vis_dyn * shrink_d),
        w_app=params.w_app, w_pcd=params.w_pcd,
        w_geo_static=params.w_geo_static, w_geo_dyn=params.w_geo_dyn,
        dist_gate=params.dist_gate, beta_mix=params.beta_mix,
        anchor_precision=params.anchor_precision, gamma_anc=params.gamma_anc,
        gamma_dyn_static=0.0, gamma_dyn_dyn=params.gamma_dyn_dyn,
        channel=params.channel.copy(),
    )
    return gated, dict(
        lam_s_raw=float(params.lam_vis_static), lam_d_raw=float(params.lam_vis_dyn),
        lam_s_gated=float(gated.lam_vis_static), lam_d_gated=float(gated.lam_vis_dyn),
        gain_static=float(gain_s), gain_dyn=float(gain_d),
        shrink_static=float(shrink_s), shrink_dyn=float(shrink_d),
        delta_0=float(delta_0),
    )


def infer_transductive_v4(p_id, p_ctx_tm, f_ctx_res, laps, is_dyn, events,
                          columns, params, beta_q, *, protocol='PRESUPPOSED'):
    """Two-stream probability-space harmonic transduction with query-stratum beta_q."""
    nq, n_tracks = p_id.shape
    events_by_query = [[] for _ in range(nq)]
    for ev in events:
        events_by_query[ev['q']].append(ev)

    prior_p = np.clip((1.0 - beta_q[:, None]) * p_id + beta_q[:, None] * p_ctx_tm, 1e-6, 1.0 - 1e-6)
    track_llr, precision, counts, _, _ = aggregate_track_evidence_v3(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
    p_trans, leverage, loto_p = batched_solve_prob_harmonic(
        prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)
    u_trans = logit(np.clip(p_trans, 1e-6, 1.0 - 1e-6))
    existence = compute_video_existence(
        u_trans, events_by_query, params.channel, protocol=protocol, existence_prior=0.5)

    gam = np.where(is_dyn, params.gamma_dyn_dyn, 0.0)[:, None]
    cols = np.asarray(columns, dtype=np.int64)
    p_clipped = np.clip(p_trans[:, cols], 1e-6, 1.0 - 1e-6)
    if np.any(gam > 0):
        base_logits = np.log(p_clipped) - np.log1p(-p_clipped)
        frame_p_raw = expit(base_logits + gam * f_ctx_res)
    else:
        frame_p_raw = p_clipped
    frame_prob = existence[:, None] * frame_p_raw
    return dict(
        prior_prob=prior_p,
        transduced_track_prob=p_trans,
        loto_track_prob=loto_p,
        leverage=leverage,
        asked_counts=counts,
        existence=existence,
        frame_prob=frame_prob,
    )


def compute_balanced_trust_region_weights(lev, p_tr, p_pr, asked_mask, q_budget, shrink_q,
                                          *, lev_thresh=0.15, trans_budget_w=0.18, protocol='PRESUPPOSED'):
    """Sign-balanced, EB-shrinkage-scaled trust-region weights on ALL high-leverage unasked tracks.

    Eliminates V3's asymmetric |p_tr - p_pr| > 0.08 upward-bias filter. On PRESUPPOSED queries,
    attenuates excess net upward probability push so active-selected positive imbalance cannot
    cause runaway base-rate inflation on unasked tracks.
    """
    sel_mask = (~asked_mask) & (lev > lev_thresh) & (shrink_q[:, None] > 1e-6)
    sup_q, sup_k = np.nonzero(sel_mask)
    if len(sup_q) == 0:
        return sup_q, sup_k, np.zeros(0, dtype=np.float64), dict(
            count=0, eff_budget=0.0, up_push_raw=0.0, down_push_raw=0.0, up_damp=1.0)

    base_w = (lev[sup_q, sup_k] * shrink_q[sup_q]) / np.maximum(q_budget[sup_q], 1.0)
    delta = p_tr[sup_q, sup_k] - p_pr[sup_q, sup_k]
    up_mask = delta > 1e-4
    down_mask = delta < -1e-4
    up_push = float(np.sum(base_w[up_mask] * delta[up_mask]))
    down_push = float(np.sum(base_w[down_mask] * (-delta[down_mask])))

    up_damp = 1.0
    if protocol == 'PRESUPPOSED' and up_push > down_push and up_push > 1e-12:
        # Mean-neutral trust region: cap excess net upward push so upward drift cannot snowball
        up_damp = float(min(1.0, (down_push + 0.25 * up_push) / (up_push + 1e-12)))
        base_w = np.where(up_mask, base_w * up_damp, base_w)

    mean_shrink = float(np.mean(shrink_q[sup_q]))
    eff_budget = float(trans_budget_w * mean_shrink)
    w_sum = float(base_w.sum())
    final_w = base_w * (eff_budget / max(w_sum, 1e-12)) if w_sum > 1e-12 else np.zeros_like(base_w)
    return sup_q, sup_k, final_w, dict(
        count=int(len(sup_q)), eff_budget=eff_budget,
        up_push_raw=up_push, down_push_raw=down_push, up_damp=up_damp,
    )


def _compute_oof_priors_with_models_v4(x_id_mat, accessor, id_x, id_y, id_w, id_k,
                                       cx_x, cx_y, cx_w, cx_k,
                                       aux_id, aux_full, aux_y, aux_w,
                                       id_mean, id_std, full_mean, full_std, plan, device):
    """2-fold track-disjoint OOF priors, also returning the fold models & masks for Godel-Machine OOF verification."""
    nq, n_tracks, _ = x_id_mat.shape
    n = accessor.n
    p_id_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_ctx_oof = np.zeros((nq, n), dtype=np.float32)
    cols = accessor.cols
    fold_states = []

    for fold in (0, 1):
        id_tr_mask = (id_k % 2) != fold
        cx_tr_mask = (cx_k % 2) != fold
        torch.manual_seed(fold)
        m_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
        torch.manual_seed(100 + fold)
        m_cx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)
        if np.any(id_tr_mask):
            fit_head_adaptive_v3(
                m_id,
                np.concatenate((id_x[id_tr_mask], aux_id), axis=0),
                np.concatenate((id_y[id_tr_mask], aux_y), axis=0),
                np.concatenate((id_w[id_tr_mask], aux_w), axis=0),
                min_steps=400, max_steps=int(plan['oof_max_steps']), target_excess_ce=0.015,
                device=device, lr=float(plan['learning_rate']), weight_decay=plan['weight_decay'], seed=fold)
        if np.any(cx_tr_mask):
            fit_head_adaptive_v3(
                m_cx,
                np.concatenate((cx_x[cx_tr_mask], aux_full), axis=0),
                np.concatenate((cx_y[cx_tr_mask], aux_y), axis=0),
                np.concatenate((cx_w[cx_tr_mask], aux_w), axis=0),
                min_steps=400, max_steps=int(plan['oof_max_steps']), target_excess_ce=0.015,
                device=device, lr=float(plan['learning_rate']), weight_decay=plan['weight_decay'], seed=100 + fold)
        val_tracks = (np.arange(n_tracks) % 2) == fold
        val_obs = (cols % 2) == fold
        p_id_f = expit(predict_track_id(m_id, x_id_mat, device))
        f_cx_f = predict_full_head_fast(m_cx, accessor, device)
        p_id_oof[:, val_tracks] = p_id_f[:, val_tracks]
        f_ctx_oof[:, val_obs] = f_cx_f[:, val_obs]
        fold_states.append((fold, m_id, m_cx, id_tr_mask, cx_tr_mask, p_id_f, f_cx_f))
    return p_id_oof, f_ctx_oof, fold_states


def evaluate_oof_teacher_nll(p_id_oof, p_ctx_oof, laps, is_dyn, events, params, beta_q, protocol):
    """Exact balanced 2-fold OOF teacher NLL (strictly zero-GT external-evidence objective)."""
    nq, n_tracks = p_id_oof.shape
    if not events:
        return 0.0
    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)

    prior_p = np.clip((1.0 - beta_q[:, None]) * p_id_oof + beta_q[:, None] * p_ctx_oof, 1e-6, 1.0 - 1e-6)
    track_llr, precision, _, _, _ = aggregate_track_evidence_v3(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
    p_trans, _, loto_p = batched_solve_prob_harmonic(
        prior_p, laps, track_llr, precision, lam_q, tau_prior=1.0, gamma_anc=params.gamma_anc)

    events_by_q = [[] for _ in range(nq)]
    for e in events:
        events_by_q[e['q']].append(e)
    u_trans = logit(np.clip(p_trans, 1e-6, 1.0 - 1e-6))
    eta = compute_video_existence(u_trans, events_by_q, params.channel, protocol=protocol, existence_prior=0.5)
    pred_state = np.clip(eta[ev_q] * loto_p[ev_q, ev_k], 1e-7, 1.0 - 1e-7)
    s = params.channel[ev_v, 0]
    c = params.channel[ev_v, 1]
    py = np.clip((1.0 - c) + (s + c - 1.0) * pred_state, 1e-7, 1.0 - 1e-7)
    w, pm, wp, wn = _balanced_event_weights(ev_q, ev_p, nq)
    return float(-np.sum(w * (wp * pm * np.log(py) + wn * (1.0 - pm) * np.log1p(-py))))


def verify_coevol_godel_gate(context, fold_states, x_id_mat, accessor, id_x, id_y, id_w,
                             cx_x, cx_y, cx_w, aux_id, aux_full, aux_y, aux_w,
                             laps, is_dyn, events, params, beta_q, eb_gate, q_budget,
                             track_rows_cache, protocol, plan, device):
    """Godel-Machine 2-fold track-disjoint OOF verification gate for R1/R2 co-evolution (ZERO GT).

    Each fold computes transduced trust-region pseudo-labels strictly from its OWN training-fold
    anchors (zero leakage from held-out fold tracks!), performs one candidate co-evolution update,
    and checks whether held-out fold teacher NLL strictly decreases vs R0.
    """
    video = context.video
    nq, n_tracks, _ = x_id_mat.shape
    n = accessor.n
    cols = video.columns
    shrink_q = np.where(is_dyn, eb_gate['shrink_dyn'], eb_gate['shrink_static'])

    p_id_r0_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_ctx_r0_oof = np.zeros((nq, n), dtype=np.float32)
    p_id_r1_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_ctx_r1_oof = np.zeros((nq, n), dtype=np.float32)
    rng = np.random.default_rng(777)

    for fold, m_id, m_cx, id_tr_mask, cx_tr_mask, p_id_f, f_cx_f in fold_states:
        val_tracks = (np.arange(n_tracks) % 2) == fold
        val_obs = (cols % 2) == fold
        p_id_r0_oof[:, val_tracks] = p_id_f[:, val_tracks]
        f_ctx_r0_oof[:, val_obs] = f_cx_f[:, val_obs]

        # Strictly training-fold events ONLY (zero held-out track leakage!)
        tr_events = [e for e in events if (e['track_col'] % 2) != fold]
        f_cx_tm_f, f_cx_res_f = prob_pooled_track_and_residual(
            f_cx_f, video.obs, cols, context.images, n_tracks,
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        p_cx_tm_f = expit(f_cx_tm_f)
        tr_trans = infer_transductive_v4(
            p_id_f, p_cx_tm_f, f_cx_res_f, laps, is_dyn, tr_events,
            cols, params, beta_q, protocol=protocol)

        sup_q, sup_k, raw_w, _ = compute_balanced_trust_region_weights(
            tr_trans['leverage'], tr_trans['transduced_track_prob'], tr_trans['prior_prob'],
            tr_trans['asked_counts'] > 0, q_budget, shrink_q,
            lev_thresh=float(plan['leverage_threshold']),
            trans_budget_w=float(plan['transduced_weight']), protocol=protocol)
        eta = tr_trans['existence']
        abs_q = np.flatnonzero(eta < 0.35) if protocol == 'CHECKLIST' else np.zeros(0, dtype=np.int64)

        if len(sup_q) == 0 and len(abs_q) == 0:
            p_id_r1_oof[:, val_tracks] = p_id_f[:, val_tracks]
            f_ctx_r1_oof[:, val_obs] = f_cx_f[:, val_obs]
            continue

        id_px = [id_x[id_tr_mask], aux_id]
        id_py = [id_y[id_tr_mask], aux_y]
        id_pw = [id_w[id_tr_mask], aux_w]
        if len(sup_q) > 0:
            id_px.append(x_id_mat[sup_q, sup_k])
            id_py.append(tr_trans['transduced_track_prob'][sup_q, sup_k])
            id_pw.append(raw_w)
        if len(abs_q) > 0:
            aq_grid = np.repeat(abs_q, n_tracks)
            ak_grid = np.tile(np.arange(n_tracks, dtype=np.int64), len(abs_q))
            aw = 1.0 - eta[aq_grid]
            aw = aw * (0.35 / max(float(aw.sum()), 1e-12))
            id_px.append(x_id_mat[aq_grid, ak_grid])
            id_py.append(np.zeros(len(aq_grid), dtype=np.float64))
            id_pw.append(aw)

        cx_px = [cx_x[cx_tr_mask], aux_full]
        cx_py = [cx_y[cx_tr_mask], aux_y]
        cx_pw = [cx_w[cx_tr_mask], aux_w]
        if len(sup_q) > 0:
            sample_n = min(1536, len(sup_q) * 2)
            pick = rng.integers(len(sup_q), size=sample_n)
            sel_ids, sel_y, sel_w = [], [], []
            for idx_p in pick:
                q_i, k_i = int(sup_q[idx_p]), int(sup_k[idx_p])
                rows_k = track_rows_cache[k_i]
                nd = int(rows_k[rng.integers(len(rows_k))])
                sel_ids.append(q_i * n + nd)
                sel_y.append(float(tr_trans['frame_prob'][q_i, nd]))
                sel_w.append(float(raw_w[idx_p]))
            sel_w = np.asarray(sel_w, dtype=np.float64)
            sel_w = sel_w * (float(raw_w.sum()) / max(float(sel_w.sum()), 1e-12))
            cx_px.append(accessor.full_rows(np.asarray(sel_ids, dtype=np.int64)))
            cx_py.append(np.asarray(sel_y, dtype=np.float64))
            cx_pw.append(sel_w)
        if len(abs_q) > 0:
            sample_abs = min(1024, len(abs_q) * 16)
            aq = abs_q[rng.integers(len(abs_q), size=sample_abs)]
            and_ = rng.integers(n, size=sample_abs)
            aw_cx = 1.0 - eta[aq]
            aw_cx = aw_cx * (0.35 / max(float(aw_cx.sum()), 1e-12))
            cx_px.append(accessor.full_rows(aq * n + and_))
            cx_py.append(np.zeros(sample_abs, dtype=np.float64))
            cx_pw.append(aw_cx)

        fit_head_adaptive_v3(
            m_id, np.concatenate(id_px, axis=0), np.concatenate(id_py, axis=0), np.concatenate(id_pw, axis=0),
            min_steps=200, max_steps=600, target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']) * 0.5, weight_decay=plan['weight_decay'], seed=20 + fold)
        fit_head_adaptive_v3(
            m_cx, np.concatenate(cx_px, axis=0), np.concatenate(cx_py, axis=0), np.concatenate(cx_pw, axis=0),
            min_steps=200, max_steps=600, target_excess_ce=0.015,
            device=device, lr=float(plan['learning_rate']) * 0.5, weight_decay=plan['weight_decay'], seed=120 + fold)

        p_id_r1_oof[:, val_tracks] = expit(predict_track_id(m_id, x_id_mat, device))[:, val_tracks]
        f_ctx_r1_oof[:, val_obs] = predict_full_head_fast(m_cx, accessor, device)[:, val_obs]

    tm_r0, _ = prob_pooled_track_and_residual(
        f_ctx_r0_oof, video.obs, cols, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    tm_r1, _ = prob_pooled_track_and_residual(
        f_ctx_r1_oof, video.obs, cols, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    nll_r0 = evaluate_oof_teacher_nll(p_id_r0_oof, expit(tm_r0), laps, is_dyn, events, params, beta_q, protocol)
    nll_r1 = evaluate_oof_teacher_nll(p_id_r1_oof, expit(tm_r1), laps, is_dyn, events, params, beta_q, protocol)
    gain = float(nll_r0 - nll_r1)
    accepted = bool(gain > 1e-4)
    return accepted, dict(
        oof_nll_r0=float(nll_r0), oof_nll_r1=float(nll_r1),
        oof_nll_gain=gain, accepted=accepted,
    )


def learn_unified_rsi_rmot_v4(context, case, plan, folder, device='cuda:0', accessor=None):
    """Run UNIFIED_RSI_RMOT_V4 on sealed V3 active teacher answers (0 new Qwen forwards, 0 GT)."""
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

    # Replay exact sealed V3 active teacher answers (0 new Qwen forwards!)
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

    # Step 1: 2-Fold Track-Disjoint OOF -> Strictly Convex Two-Stream Gating -> Balanced OOF-LOTO -> EB Graph Gate
    oof_started = time.perf_counter()
    p_id_oof, f_ctx_oof, fold_states = _compute_oof_priors_with_models_v4(
        x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k,
        aux_id, aux_full, aux_y, aux_w, id_mean, id_std, full_mean, full_std, plan, device)
    f_ctx_tm_oof, f_ctx_res_oof = prob_pooled_track_and_residual(
        f_ctx_oof, video.obs, video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_ctx_tm_oof = expit(f_ctx_tm_oof)

    beta_s, beta_d, two_stream_diag = fit_two_stream_convex_beta(p_id_oof, p_ctx_tm_oof, is_dyn, events)
    beta_q = np.where(is_dyn, beta_d, beta_s)

    raw_params, laps, loto_oof = fit_observer_two_stream_loto(
        p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, beta_q, k_nn=plan['k_nn'])
    params, eb_gate = empirical_bayes_graph_gate_two_stream(
        raw_params, laps, p_id_oof, p_ctx_tm_oof, is_dyn, events,
        beta_q, delta_0=float(plan['eb_graph_delta_0']))

    # Step 2: Godel-Machine 2-Fold OOF Verification Gate for R1/R2 Co-Evolution (ZERO GT)
    gate_started = time.perf_counter()
    coevol_accepted, godel_diag = verify_coevol_godel_gate(
        context, fold_states, x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w,
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, aux_id, aux_full, aux_y, aux_w,
        laps, is_dyn, events, params, beta_q, eb_gate, q_budget,
        track_rows_cache, protocol, plan, device)
    del fold_states
    gate_seconds = time.perf_counter() - gate_started
    oof_seconds = time.perf_counter() - oof_started

    # Step 3: Final Dual-Student Learning (R0 always; R1/R2 ONLY if Godel-Machine OOF gate accepted OR CHECKLIST ABSENT_NEG!)
    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_ctx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    n_rounds = int(plan['rounds']) if coevol_accepted else (2 if protocol == 'CHECKLIST' else 1)
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

            if coevol_accepted:
                sup_q, sup_k, raw_w, tr_diag = compute_balanced_trust_region_weights(
                    lev, p_tr, p_pr, asked_mask, q_budget, shrink_q,
                    lev_thresh=float(plan['leverage_threshold']),
                    trans_budget_w=float(plan['transduced_weight']), protocol=protocol)
            else:
                sup_q = np.zeros(0, dtype=np.int64)
                sup_k = np.zeros(0, dtype=np.int64)
                raw_w = np.zeros(0, dtype=np.float64)
                tr_diag = dict(count=0, eff_budget=0.0, up_push_raw=0.0, down_push_raw=0.0, up_damp=1.0)

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
        status='COMPLETE_UNIFIED_RSI_RMOT_V4_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        dynamic_queries=int(is_dyn.sum()),
        two_stream_convex_oof=two_stream_diag,
        oof_loto=loto_oof, eb_graph_gate=eb_gate,
        godel_coevol_gate=godel_diag,
        oof_wall_seconds=oof_seconds, gate_wall_seconds=gate_seconds,
        observer=params.to_dict(), rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        new_teacher_calls=0, GT_open_attempts=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    print(json.dumps(dict(
        status='UNIFIED_RSI_V4_UNIT_LEARNED', pair=case['pair'], video=case['evaluated_video'],
        b_s=round(beta_s, 2), b_d=round(beta_d, 2),
        lam_s=f"{eb_gate['lam_s_raw']:.2f}->{eb_gate['lam_s_gated']:.2f}",
        lam_d=f"{eb_gate['lam_d_raw']:.2f}->{eb_gate['lam_d_gated']:.2f}",
        godel=f"{'ACC' if coevol_accepted else 'REJ'}({godel_diag['oof_nll_gain']:+.4f})",
        p_mean=round(float(trans['frame_prob'].mean()), 4),
        sec=round(receipt['wall_seconds'], 1),
    )), flush=True)
    return dict(
        receipt=receipt, accessor=accessor, s_id=s_id, s_ctx=s_ctx,
        params=params, beta_q=beta_q, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )


def apply_shared_transduct_v4_readout(frame_scores, context, learned, plan):
    """Same EB-gated transduction applied to single-stream ACTIVE_SOFT_KD (no S_id, no R1/R2)."""
    n_tracks = len(context.video.tracks)
    tm, res = prob_pooled_track_and_residual(
        frame_scores, context.video.obs, context.video.columns,
        context.images, n_tracks, kappa=plan['kappa'], base_decay=plan['base_decay'])
    p_tm = expit(tm)
    out = infer_transductive_v4(
        p_tm, p_tm, res, learned['laps'], learned['is_dyn'], learned['events'],
        context.video.columns, learned['params'], np.ones(context.nq, dtype=np.float64),
        protocol=learned['protocol'])
    return out['frame_prob']
