"""Runtime for UNIFIED_RSI_RMOT (I1 + I2 + I3 with live frozen Qwen3-VL-2B & zero GT).

Unites all three innovations into one cohesive test-time adaptation loop:
- I3 (Active Challenger-Observer): 4-stage front-loaded dynamic curriculum
  (Stage 0 GLANCE + PRESENCE_COVER, Stages 1-3 Dual-Student Disagreement + Boundary
  Uncertainty + Exact Rank-1 Sherman-Morrison Clique Leverage Gain + 15% Exploration).
  Selection journals are written BEFORE calling frozen Qwen3-VL-2B (TeacherDispatcher).
- I1 (Class-Balanced Bounded Probability-Space Harmonic Transduction):
  2-fold out-of-track OOF-LOTO with class-balanced teacher risk (0.5 NLL_pos + 0.5 NLL_neg)
  selecting dist_gate in {0.40, 0.55, 0.75, 1.00}, kernel weights, lam_vis, beta_mix,
  anchor_precision, gamma_anc, and balanced view channels, plus evidence-gated zero-jitter
  coherence (gamma_dyn_dyn = 0.0 unless >= 5 within-track teacher contrasts prove switch).
- I2 (Complementary Dual-Student Externally-Anchored Co-Evolution):
  S_id (7482D strictly track-invariant) + S_ctx (9352D full contextual + prob pooling),
  gentle regularization (dropout=0.10, wd=1e-3, max_steps=4800), correction-only
  external-leverage feedback ((lev > 0.12) & (|p_star - mu_prob| > 0.05)) + CHECKLIST ABSENT_NEG.
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch

from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.dual_transduct_coevol import (
    ID_DIM, RegularizedMLP, compute_video_existence, split_shared_features_np,
)
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor, precompute_graph_distances,
    predict_full_head_fast, predict_track_id,
)
from flex_uda.active.dual_transduct_v2 import (
    build_gated_clique_laplacian, prob_pooled_track_and_residual,
)
from flex_uda.active.dual_transduct_v3 import (
    DIST_GATE_GRID, aggregate_track_evidence_v3, batched_solve_prob_harmonic,
    default_v3_vector, unpack_v3_vector,
)
from flex_uda.active.dual_transduct_v3_runtime import (
    _build_all_gated_laplacians_v3, _compute_oof_priors_v3,
    fit_head_adaptive_v3, infer_transductive_v3,
)
from flex_uda.active.unified_rsi_core import (
    batched_prob_harmonic_with_inverse, select_challenger_stage_uids, stage_budgets,
)

VIEW_MAP = {'CROP': 0, 'CLIP': 1, 'FRAME_BOX': 2}


def broker_rows_to_events(context, rows):
    """Convert FreshBroker answer rows into deduplicated track-aligned event dicts."""
    n = context.n
    cols = context.video.columns
    frames = context.video.obs[:, 0]
    events = []
    for row in rows:
        uid = int(row['uid'])
        q, oi = divmod(uid, n)
        k = int(cols[oi])
        vname = row.get('effective_view') or row.get('requested_view') or context.queries[q]['view']
        v = VIEW_MAP.get(vname, 0)
        events.append(dict(
            uid=uid, q=int(q), node=int(oi), nodes=[int(oi)],
            track_col=k, frame_id=int(frames[oi]), view=int(v),
            prob=float(row['probability']),
        ))
    return events


def build_anchor_tensors(events, x_id_mat, accessor, nq, n):
    """Build track-level anchors for S_id and frame-level anchors for S_ctx."""
    q_budget = np.bincount([e['q'] for e in events], minlength=nq).astype(np.float64)
    q_pos_n = max(int((q_budget > 0).sum()), 1)
    by_qk = {}
    for ev in events:
        by_qk.setdefault((ev['q'], ev['track_col']), []).append(ev['prob'])
    id_x, id_y, id_w, id_k = [], [], [], []
    for (q, k), probs in sorted(by_qk.items()):
        id_x.append(x_id_mat[q, k])
        id_y.append(float(np.mean(probs)))
        id_w.append(len(probs) / (q_pos_n * max(q_budget[q], 1.0)))
        id_k.append(k)
    id_x = np.asarray(id_x, dtype=np.float32)
    id_y = np.asarray(id_y, dtype=np.float64)
    id_w = np.asarray(id_w, dtype=np.float64)
    id_k = np.asarray(id_k, dtype=np.int64)

    cx_ids, cx_y, cx_w, cx_k = [], [], [], []
    for ev in events:
        w_ev = 1.0 / (q_pos_n * max(q_budget[ev['q']], 1.0))
        cx_ids.append(ev['q'] * n + ev['node'])
        cx_y.append(ev['prob'])
        cx_w.append(w_ev)
        cx_k.append(ev['track_col'])
    cx_ids = np.asarray(cx_ids, dtype=np.int64)
    cx_x = accessor.full_rows(cx_ids)
    cx_y = np.asarray(cx_y, dtype=np.float64)
    cx_w = np.asarray(cx_w, dtype=np.float64)
    cx_k = np.asarray(cx_k, dtype=np.int64)
    return (id_x, id_y, id_w, id_k), (cx_x, cx_y, cx_w, cx_k), q_budget


def fit_observer_balanced_loto(p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
                               is_dyn, events, *, k_nn=3, prior_reg=0.015):
    """Class-balanced OOF-LOTO optimization in bounded probability space.

    Uses soft teacher-probability class balance (0.5 * NLL_pos + 0.5 * NLL_neg, ZERO GT)
    so negative-majority settings (KITTI) never sacrifice rare positive tracks by
    over-smoothing with wide cliques or inflated lam_vis, while Dance/Sports retain
    full supportive-clique gains where visual cliques truly share positive labels.
    """
    theta0 = default_v3_vector()
    nq, n_tracks = p_id_oof.shape
    if not events:
        p0 = unpack_v3_vector(theta0, dist_gate=0.65, gamma_dyn_dyn=0.0)
        return p0, _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, p0, k_nn=k_nn), dict(
            loto_before=0.0, loto_after=0.0, stage1_evals=0, stage2_evals=0,
            chosen_dist_gate=0.65, within_track_dyn_contrasts=0, chosen_gamma_dyn_dyn=0.0)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    q_counts = np.bincount(ev_q, minlength=nq).astype(np.float64)
    ev_w = 1.0 / (max(int((q_counts > 0).sum()), 1) * np.maximum(q_counts[ev_q], 1.0))

    pos_mass = float(np.sum(ev_w * ev_p))
    neg_mass = float(np.sum(ev_w * (1.0 - ev_p)))
    # Blend 70% class-balanced + 30% natural risk so balanced recall AND overall calibration hold
    w_pos = 0.70 * (0.5 / max(pos_mass, 1e-4)) + 0.30
    w_neg = 0.70 * (0.5 / max(neg_mass, 1e-4)) + 0.30

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
    best_dg = 0.55
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

    # Stage 3: Evidence-gated within-track contrast check (default strictly 0.0)
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
        pos_mass=pos_mass, neg_mass=neg_mass,
    )


def learn_unified_rsi_rmot(context, case, plan, folder, device='cuda:0', accessor=None):
    """Run full 4-stage Active Challenger-Observer (I3) + Balanced Harmonic (I1) + Dual-Student Co-Evol (I2)."""
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
    obs_frames = video.obs[:, 0]
    obs_geom = video.geometry[:, :4]

    # Initial bootstrap from prepaid target-collection auxiliary ONLY (0 target video teacher calls)
    active_started = time.perf_counter()
    torch.manual_seed(0)
    s_id_act = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_ctx_act = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)
    aux_w_unit = np.full(len(aux_y), 1.0 / len(aux_y), dtype=np.float64)
    fit_head_adaptive_v3(s_id_act, aux_id, aux_y, aux_w_unit, min_steps=150, max_steps=300,
                         target_excess_ce=0.02, device=device, lr=float(plan['learning_rate']),
                         weight_decay=plan['weight_decay'], seed=0)
    fit_head_adaptive_v3(s_ctx_act, aux_full, aux_y, aux_w_unit, min_steps=150, max_steps=300,
                         target_excess_ce=0.02, device=device, lr=float(plan['learning_rate']),
                         weight_decay=plan['weight_decay'], seed=100)

    # Initial supportive-clique Laplacians for Challenger leverage gain
    p_init = unpack_v3_vector(default_v3_vector(), dist_gate=0.65, gamma_dyn_dyn=0.0)
    laps_init = _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, p_init, k_nn=plan['k_nn'])
    lam_init = np.where(is_dyn, 0.6, 1.0)

    alloc = stage_budgets(context.quotas, stage_fracs=(0.30, 0.25, 0.25, 0.20))
    asked_uids_by_q = {q: set() for q in range(nq)}
    asked_frames_by_qk = {}
    all_rows = []
    stage_logs = []
    journal_path = folder / 'active_selection_journal.jsonl'
    if journal_path.exists():
        journal_path.unlink()

    for st in range(4):
        st_start = time.perf_counter()
        p_id_cur = expit(predict_track_id(s_id_act, x_id_mat, device))
        f_ctx_cur = predict_full_head_fast(s_ctx_act, accessor, device)
        f_ctx_tm_cur, _ = prob_pooled_track_and_residual(
            f_ctx_cur, video.obs, video.columns, context.images, n_tracks,
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        p_ctx_cur = expit(f_ctx_tm_cur)
        prior_cur = 0.5 * p_id_cur + 0.5 * p_ctx_cur

        cur_events = broker_rows_to_events(context, all_rows)
        track_llr, precision, counts, _, _ = aggregate_track_evidence_v3(
            nq, n_tracks, cur_events, p_init.channel, anchor_base_precision=2.0)
        p_star_cur, lev_cur, M_cur = batched_prob_harmonic_with_inverse(
            prior_cur, laps_init, track_llr, precision, lam_init, tau_prior=1.0, gamma_anc=1.0)

        uids, j_rows = select_challenger_stage_uids(
            st, alloc,
            p_id=p_id_cur, p_ctx_tm=p_ctx_cur, p_star=p_star_cur,
            M_inv=M_cur, leverage=lev_cur, asked_counts=counts,
            asked_uids_by_q=asked_uids_by_q, asked_frames_by_qk=asked_frames_by_qk,
            track_rows_cache=track_rows_cache, obs_frames=obs_frames, obs_geom=obs_geom,
            n_obs=n, pi_add=2.0, explore_frac=0.15,
        )
        # Save selection journal BEFORE requesting frozen teacher answers!
        with journal_path.open('a', buffering=1) as stream:
            for jr in j_rows:
                stream.write(json.dumps(jr) + '\n')

        t_qwen0 = time.perf_counter()
        replies = context.primary.request(uids, 'train', f'STAGE{st}')
        t_qwen = time.perf_counter() - t_qwen0
        all_rows.extend(replies)
        for jr in j_rows:
            asked_uids_by_q[jr['query']].add(int(jr['uid']))

        # Incremental dual-student solver update between active stages (st < 3)
        if st < 3:
            ev_now = broker_rows_to_events(context, all_rows)
            (id_x, id_y, id_w, _), (cx_x, cx_y, cx_w, _), _ = build_anchor_tensors(
                ev_now, x_id_mat, accessor, nq, n)
            fit_head_adaptive_v3(
                s_id_act,
                np.concatenate((id_x, aux_id), axis=0),
                np.concatenate((id_y, aux_y), axis=0),
                np.concatenate((id_w, aux_w), axis=0),
                min_steps=300, max_steps=600, target_excess_ce=0.015,
                device=device, lr=float(plan['learning_rate']) * 0.75,
                weight_decay=plan['weight_decay'], seed=10 + st)
            fit_head_adaptive_v3(
                s_ctx_act,
                np.concatenate((cx_x, aux_full), axis=0),
                np.concatenate((cx_y, aux_y), axis=0),
                np.concatenate((cx_w, aux_w), axis=0),
                min_steps=300, max_steps=600, target_excess_ce=0.015,
                device=device, lr=float(plan['learning_rate']) * 0.75,
                weight_decay=plan['weight_decay'], seed=110 + st)

        pos_st = sum(1 for r in replies if float(r['probability']) >= 0.5)
        stage_logs.append(dict(
            stage=st, asked=len(uids), positives=pos_st,
            qwen_wall_seconds=t_qwen, stage_wall_seconds=time.perf_counter() - st_start,
        ))
    del s_id_act, s_ctx_act

    broker_receipt = context.primary.receipt(complete=True)
    active_seconds = time.perf_counter() - active_started
    events = broker_rows_to_events(context, all_rows)
    assert len(events) == sum(context.quotas.values())
    (id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k), (
        ctx_anchor_x, ctx_anchor_y, ctx_anchor_w, ctx_anchor_k), q_budget = build_anchor_tensors(
        events, x_id_mat, accessor, nq, n)

    # Final Learning Phase: Unbiased 2-fold OOF -> Class-Balanced OOF-LOTO Observer R -> 3-Round Co-Evolution
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
    params, laps, loto_oof = fit_observer_balanced_loto(
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
    n_rounds = int(plan['rounds'])
    lev_thresh = float(plan.get('leverage_threshold', 0.12))
    delta_thresh = float(plan.get('correction_delta_threshold', 0.05))
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
            p_pr = trans['prior_prob']
            eta = trans['existence']
            asked_mask = trans['asked_counts'] > 0

            # Correction-only externally-grounded filter: lev > 0.12 AND |p* - mu| > 0.05!
            corr_mask = (~asked_mask) & (lev > lev_thresh) & (np.abs(p_tr - p_pr) > delta_thresh)
            sup_q, sup_k = np.nonzero(corr_mask)
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

    pos_total = sum(1 for e in events if e['prob'] >= 0.5)
    q_with_pos = len({e['q'] for e in events if e['prob'] >= 0.5})
    receipt = dict(
        status='COMPLETE_UNIFIED_RSI_RMOT_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        active_positives=pos_total, active_queries_with_positive=q_with_pos,
        dynamic_queries=int(is_dyn.sum()),
        active_stages=stage_logs, active_wall_seconds=active_seconds,
        broker=broker_receipt,
        oof_loto=loto_oof, oof_wall_seconds=oof_seconds,
        observer=params.to_dict(), rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        GT_open_attempts=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    print(json.dumps(dict(
        status='UNIFIED_RSI_UNIT_LEARNED', pair=case['pair'], video=case['evaluated_video'],
        ev=len(events), pos=pos_total, qpos=f'{q_with_pos}/{nq}',
        new_qwen=broker_receipt['actual_new_teacher_calls'],
        cache_hit=broker_receipt['exact_input_teacher_cache_hits'],
        dg=params.dist_gate, lam_s=round(params.lam_vis_static, 2), lam_d=round(params.lam_vis_dyn, 2),
        sec=round(receipt['wall_seconds'], 1),
    )), flush=True)
    return dict(
        receipt=receipt, accessor=accessor, s_id=s_id, s_ctx=s_ctx,
        params=params, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )


def fit_active_ordinary_controls(context, events, plan, device='cuda:0', accessor=None):
    """Same-information equal-capacity ordinary soft & hard KD on exact active answers."""
    if accessor is None:
        accessor = FastVideoFeatureAccessor(context)
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
    for mode, hard in (('ACTIVE_SOFT_KD', False), ('ACTIVE_HARD_KD', True)):
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
