"""Runtime for UNIFIED_RSI_RMOT_V3: Empirical-Bayes Graph Significance Gate + Hybrid V2/V1 Challenger.

Synthesizes the three exact empirical findings from UNIFIED_V1, UNIFIED_V2, and the
148-video 2x2x2 factorial ablation (all prior pinned sources strictly unchanged):
1. Hybrid Fast-Slow Active Challenger (I3):
   - Stage 0 (35% budget): Pure visual-clique coverage via sequential rank-1 Sherman-Morrison
     DeltaLev + confident PRESENCE_COVER only when max(p0) > 0.65 (proven +2.14 on C6, +0.56 on C1).
   - Stages 1-3 (65% budget): Crisp greedy Challenger (85% argmax uncertainty+disagreement+leverage
     + 15% pure clique-leverage exploration floor, without high-variance 2.5x IWAL negative
     reweighting -> preserves V1's +1.94/+1.63 KITTI positive discovery!).
2. Empirical-Bayes Graph Significance Gate (I1):
   - Tests H0: lam_static = 0 and H0: lam_dyn = 0 separately on exact OOF-LOTO teacher NLL.
   - Applies James-Stein / Empirical-Bayes shrinkage with significance floor delta_0 = 0.020:
       lam_eff = lam_hat * max(gain - delta_0, 0) / (gain + 1e-8)
   - Automatically keeps strong clique diffusion on Dance (gain ~0.15) & Sports (gain ~0.08),
     while shrinking noisy over-smoothing lam -> 0 on KITTI (gain ~0.01) while keeping exact
     monotonic Bayesian teacher anchors & complementary dual students!
3. Graph-Gated Conservative Dual-Student Co-Evolution (I2):
   - Unasked correction-only feedback ((lev > 0.15) & (|p* - mu| > 0.08), weight 0.18)
     + CHECKLIST ABSENT_NEG -> automatically zero positive-drift FP bloat when lam_eff == 0!
"""
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import expit
import torch

from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.dual_transduct_coevol import (
    ID_DIM, RegularizedMLP, split_shared_features_np,
)
from flex_uda.active.dual_transduct_runtime import (
    FastVideoFeatureAccessor, precompute_graph_distances,
    predict_full_head_fast, predict_track_id,
)
from flex_uda.active.dual_transduct_v2 import prob_pooled_track_and_residual
from flex_uda.active.dual_transduct_v3 import (
    TransductiveObserverV3Params, aggregate_track_evidence_v3,
    batched_solve_prob_harmonic, default_v3_vector, unpack_v3_vector,
)
from flex_uda.active.dual_transduct_v3_runtime import (
    _build_all_gated_laplacians_v3, _compute_oof_priors_v3,
    fit_head_adaptive_v3, infer_transductive_v3,
)
from flex_uda.active.unified_rsi_core import (
    batched_prob_harmonic_with_inverse, binary_entropy, pick_best_keyframe,
    sherman_morrison_leverage_gain, sherman_morrison_rank1_anchor_update,
    stage_budgets,
)
from flex_uda.active.unified_rsi_runtime import (
    broker_rows_to_events, build_anchor_tensors, fit_active_ordinary_controls,
    fit_observer_balanced_loto,
)


def select_hybrid_challenger_stage_uids(
    stage_idx, stage_alloc, *,
    p_id, p_ctx_tm, p_star, M_inv, leverage, asked_counts,
    asked_uids_by_q, asked_frames_by_qk,
    track_rows_cache, obs_frames, obs_geom, n_obs,
    pi_add=2.0, explore_frac=0.15,
):
    """Best-of-V2 Stage-0 pure clique coverage + Best-of-V1 crisp greedy Stages 1-3."""
    nq, n_tracks = p_star.shape
    selected_uids = []
    journal_rows = []

    for q in range(nq):
        need = int(stage_alloc[q][stage_idx])
        if need <= 0:
            continue
        M_q = M_inv[q].copy()
        h_q = leverage[q].copy()
        c_q = asked_counts[q].copy()
        asked_q = set(asked_uids_by_q[q])
        avail_count = np.array([
            sum(1 for oi in track_rows_cache[k] if (q * n_obs + int(oi)) not in asked_q)
            for k in range(n_tracks)
        ], dtype=np.int64)
        assert int(avail_count.sum()) >= need, (q, int(avail_count.sum()), need)

        n_explore = int(round(need * explore_frac)) if (stage_idx > 0 and need >= 4) else 0
        n_main = need - n_explore

        for slot in range(need):
            valid = avail_count > 0
            unasked_valid = valid & (c_q == 0)
            pool = unasked_valid if unasked_valid.any() else valid
            lev_gain = sherman_morrison_leverage_gain(M_q, h_q, pi_add=pi_add)
            max_lg = float(lev_gain[pool].max()) if pool.any() else 1.0
            norm_lg = lev_gain / max(max_lg, 1e-8)

            if stage_idx == 0 and slot == 0 and float(p_star[q, pool].max()) > 0.65:
                reason = 'STAGE0_CONFIDENT_PRESENCE_COVER'
                scores = np.where(pool, p_star[q] + 0.05 * norm_lg, -1e18)
            elif stage_idx == 0:
                reason = 'STAGE0_PURE_CLIQUE_COVERAGE'
                scores = np.where(pool, norm_lg / (1.0 + c_q), -1e18)
            elif slot >= n_main:
                reason = 'CHALLENGER_EXPLORE_CLIQUE_COVERAGE'
                scores = np.where(pool, (0.20 + norm_lg) / (1.0 + c_q), -1e18)
            else:
                reason = 'CHALLENGER_DISAGREEMENT_UNCERTAINTY_LEVERAGE'
                ent = binary_entropy(p_star[q])
                disagree = np.abs(p_id[q] - p_ctx_tm[q])
                pos_disc = 0.35 * p_star[q] * (c_q == 0).astype(np.float64)
                util = (ent + disagree + pos_disc) * (0.40 + norm_lg) / (1.0 + c_q)
                scores = np.where(valid, util, -1e18)

            chosen_k = int(np.argmax(scores))
            assert avail_count[chosen_k] > 0
            asked_f = asked_frames_by_qk.setdefault((q, chosen_k), set())
            oi = pick_best_keyframe(
                track_rows_cache[chosen_k], obs_frames, obs_geom,
                asked_q, asked_f, q, n_obs)
            assert oi is not None
            flat_uid = q * n_obs + int(oi)
            asked_q.add(flat_uid)
            asked_f.add(int(obs_frames[oi]))
            avail_count[chosen_k] -= 1
            c_q[chosen_k] += 1
            M_q, h_q = sherman_morrison_rank1_anchor_update(M_q, h_q, chosen_k, pi_add=pi_add)

            uid_str = str(flat_uid)
            selected_uids.append(uid_str)
            journal_rows.append(dict(
                stage=int(stage_idx), query=int(q), track_col=int(chosen_k),
                obs_idx=int(oi), frame_id=int(obs_frames[oi]), uid=uid_str,
                reason=reason, score=float(scores[chosen_k]),
                p_star=float(p_star[q, chosen_k]), p_id=float(p_id[q, chosen_k]),
                p_ctx=float(p_ctx_tm[q, chosen_k]), lev_gain=float(lev_gain[chosen_k]),
            ))
    return selected_uids, journal_rows


def empirical_bayes_graph_gate(params, laps, p_id_oof, p_ctx_tm_oof, is_dyn, events, *,
                               delta_0=0.020):
    """Exact OOF-LOTO Empirical-Bayes shrinkage gate for cross-track graph coupling lam.

    Tests H0: lam_vis = 0 separately on static vs dynamic query teacher events using the
    exact same balanced OOF-LOTO teacher NLL (ZERO GT):
        gain = NLL_LOTO(lam=0) - NLL_LOTO(lam_hat)
        lam_eff = lam_hat * max(gain - delta_0, 0) / (gain + 1e-8)
    """
    nq, n_tracks = p_id_oof.shape
    if not events:
        return params, dict(gain_static=0.0, gain_dyn=0.0, shrink_static=1.0, shrink_dyn=1.0)

    ev_q = np.array([e['q'] for e in events], dtype=np.int64)
    ev_k = np.array([e['track_col'] for e in events], dtype=np.int64)
    ev_v = np.array([e['view'] for e in events], dtype=np.int64)
    ev_p = np.array([e['prob'] for e in events], dtype=np.float64)
    ev_is_dyn = is_dyn[ev_q]

    prior_p = (1.0 - params.beta_mix) * p_id_oof + params.beta_mix * p_ctx_tm_oof
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
        pm = ev_p[mask]
        qm = ev_q[mask]
        q_cnt = np.bincount(qm, minlength=nq).astype(np.float64)
        w = 1.0 / (max(int((q_cnt > 0).sum()), 1) * np.maximum(q_cnt[qm], 1.0))
        pos_m = float(np.sum(w * pm))
        neg_m = float(np.sum(w * (1.0 - pm)))
        wp = 0.70 * (0.5 / max(pos_m, 1e-4)) + 0.30
        wn = 0.70 * (0.5 / max(neg_m, 1e-4)) + 0.30
        return float(-np.sum(w * (wp * pm * np.log(py) + wn * (1.0 - pm) * np.log1p(-py))))

    st_mask = ~ev_is_dyn
    dy_mask = ev_is_dyn

    gain_s = 0.0
    shrink_s = 0.0
    if np.any(st_mask):
        nll_s0 = subset_nll(0.0, params.lam_vis_dyn, st_mask)
        nll_s1 = subset_nll(params.lam_vis_static, params.lam_vis_dyn, st_mask)
        gain_s = nll_s0 - nll_s1
        shrink_s = max(gain_s - delta_0, 0.0) / (gain_s + 1e-8) if gain_s > 0 else 0.0

    gain_d = 0.0
    shrink_d = 0.0
    if np.any(dy_mask):
        nll_d0 = subset_nll(params.lam_vis_static, 0.0, dy_mask)
        nll_d1 = subset_nll(params.lam_vis_static, params.lam_vis_dyn, dy_mask)
        gain_d = nll_d0 - nll_d1
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


def learn_unified_rsi_rmot_v3(context, case, plan, folder, device='cuda:0', accessor=None):
    """Run UNIFIED_RSI_RMOT_V3 (Hybrid V2/V1 Challenger + Empirical-Bayes Graph Gate + Dual-Student Co-Evol)."""
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

    # Bootstrap from prepaid auxiliary only (0 target video teacher calls)
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

    p_init = unpack_v3_vector(default_v3_vector(), dist_gate=0.65, gamma_dyn_dyn=0.0)
    laps_init = _build_all_gated_laplacians_v3(d_app, d_pcd, d_geo, is_dyn, p_init, k_nn=plan['k_nn'])
    lam_init = np.where(is_dyn, 0.6, 1.0)

    alloc = stage_budgets(context.quotas, stage_fracs=(0.35, 0.25, 0.20, 0.20))
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

        uids, j_rows = select_hybrid_challenger_stage_uids(
            st, alloc,
            p_id=p_id_cur, p_ctx_tm=p_ctx_cur, p_star=p_star_cur,
            M_inv=M_cur, leverage=lev_cur, asked_counts=counts,
            asked_uids_by_q=asked_uids_by_q, asked_frames_by_qk=asked_frames_by_qk,
            track_rows_cache=track_rows_cache, obs_frames=obs_frames, obs_geom=obs_geom,
            n_obs=n, pi_add=2.0, explore_frac=0.15,
        )
        with journal_path.open('a', buffering=1) as stream:
            for jr in j_rows:
                stream.write(json.dumps(jr) + '\n')

        t_qwen0 = time.perf_counter()
        replies = context.primary.request(uids, 'train', f'STAGE{st}')
        t_qwen = time.perf_counter() - t_qwen0
        all_rows.extend(replies)
        for jr in j_rows:
            asked_uids_by_q[jr['query']].add(int(jr['uid']))

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

    # Final Learning: 2-fold OOF -> Balanced OOF-LOTO -> Empirical-Bayes Graph Significance Gate -> 3-Round Co-Evol
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
    raw_params, laps, loto_oof = fit_observer_balanced_loto(
        p_id_oof, p_ctx_tm_oof, f_ctx_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, k_nn=plan['k_nn'])
    params, eb_gate = empirical_bayes_graph_gate(
        raw_params, laps, p_id_oof, p_ctx_tm_oof, is_dyn, events,
        delta_0=float(plan['eb_graph_delta_0']))
    oof_seconds = time.perf_counter() - oof_started

    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_ctx = RegularizedMLP(full_mean, full_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    n_rounds = int(plan['rounds'])
    lev_thresh = float(plan['leverage_threshold'])
    delta_thresh = float(plan['correction_delta_threshold'])
    trans_budget_w = float(plan['transduced_weight'])

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

            # Automatically zero unasked feedback when EB graph gate shrinks lam -> 0 (lev == 0 on unasked!)
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

            # If neither unasked graph corrections nor CHECKLIST absent negatives exist, R0 is already exact!
            if trans_tracks_used == 0 and absent_neg_used == 0:
                rounds_log.append(dict(
                    round=r, skipped_exact_convergence=True,
                    trans_tracks_used=0, absent_neg_queries=0,
                    mean_leverage=float(trans['leverage'].mean()),
                    mean_existence=float(trans['existence'].mean()),
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

        rounds_log.append(dict(
            round=r, lr=lr_r, s_id_fit=id_fit, s_ctx_fit=cx_fit,
            trans_tracks_used=trans_tracks_used, absent_neg_queries=absent_neg_used,
            mean_leverage=float(trans['leverage'].mean()),
            mean_existence=float(trans['existence'].mean()),
            round_wall_seconds=time.perf_counter() - r_start,
        ))

    pos_total = sum(1 for e in events if e['prob'] >= 0.5)
    q_with_pos = len({e['q'] for e in events if e['prob'] >= 0.5})
    receipt = dict(
        status='COMPLETE_UNIFIED_RSI_RMOT_V3_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        active_positives=pos_total, active_queries_with_positive=q_with_pos,
        dynamic_queries=int(is_dyn.sum()),
        active_stages=stage_logs, active_wall_seconds=active_seconds,
        broker=broker_receipt,
        oof_loto=loto_oof, eb_graph_gate=eb_gate, oof_wall_seconds=oof_seconds,
        observer=params.to_dict(), rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        GT_open_attempts=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    print(json.dumps(dict(
        status='UNIFIED_RSI_V3_UNIT_LEARNED', pair=case['pair'], video=case['evaluated_video'],
        ev=len(events), pos=pos_total, qpos=f'{q_with_pos}/{nq}',
        new_qwen=broker_receipt['actual_new_teacher_calls'],
        cache_hit=broker_receipt['exact_input_teacher_cache_hits'],
        dg=params.dist_gate,
        lam_s=f"{eb_gate['lam_s_raw']:.2f}->{eb_gate['lam_s_gated']:.2f}(g={eb_gate['gain_static']:+.3f})",
        lam_d=f"{eb_gate['lam_d_raw']:.2f}->{eb_gate['lam_d_gated']:.2f}(g={eb_gate['gain_dyn']:+.3f})",
        sec=round(receipt['wall_seconds'], 1),
    )), flush=True)
    return dict(
        receipt=receipt, accessor=accessor, s_id=s_id, s_ctx=s_ctx,
        params=params, laps=laps, trans=trans, events=events,
        is_dyn=is_dyn, protocol=protocol,
    )
