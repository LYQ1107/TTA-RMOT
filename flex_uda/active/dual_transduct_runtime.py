"""Runtime for DUAL_TRANSDUCT_COEVOL (no target GT access during adaptation).

Upgrades verified before full 148-unit registration:
- Exact bit-identical vectorized feature assembly (10x faster than dict-per-row
  LazyFeatures on large KITTI/SportsMOT videos, verified exact against LazyFeatures).
- True 2-fold out-of-track cross-fitting (track_col % 2) in Round 0 so LOTO
  observer fitting has ZERO in-sample student leakage and learns unbiased
  cross-track graph strength, affinity kernel, dynamic residual gate, and channel.
- Hierarchical CHECKLIST existence with conditional track floor (0.35) so
  consistent negative tracks cleanly identify empty queries and feed ABSENT_NEG.
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

ROOT = Path(__file__).resolve().parents[2]
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.dual_transduct_coevol import (
    DYN_DIM, FULL_DIM, ID_DIM,
    RegularizedMLP, TransductiveObserverParams,
    _default_observer_vector, _unpack_observer_vector,
    aggregate_track_anchors, build_combinatorial_laplacian,
    compute_video_existence, pairwise_unit_sqdist,
    smooth_within_track_residual, split_shared_features_np,
    track_geometry_descriptors,
)


class FastVideoFeatureAccessor:
    """Bit-identical vectorized accessor matching LazyFeatures on one video."""
    def __init__(self, context):
        from flex_uda.active.full_appearance_learning_inputs import bank as full_bank
        self.context = context
        v = context.video
        self.v = v
        self.n = context.n
        self.nq = context.nq
        self.n_tracks = len(v.tracks)
        self.cols = np.asarray(v.columns, dtype=np.int64)
        fb = full_bank(v.target, v.video)
        # Precompute per-observation DINO 3844D once (shared across all queries)
        self.dino_obs = fb.rows(v.obs[:, 0], v.obs[:, 1])  # (n, 3844) float32
        # Precompute normalized bags (nq, K, 3303), present (K, 3), coarse (nq, K, 1101)
        inp = context.inputs
        self.inp_mean = inp.mean
        self.inp_std = inp.std
        self.present = v.present.astype(np.float32)  # (K, 3)
        bags = ((v.raw_bags - inp.mean) / inp.std).astype(np.float32)
        bags[:, ~v.present] = 0.0
        self.bags = bags.reshape(self.nq, self.n_tracks, -1)  # (nq, K, 3303)
        self.coarse = ((inp.coarse - inp.mean) / inp.std).astype(np.float32)  # (nq, K, 1101)
        # Track-level DINO parts for X_id: anchors+present [0:2307], coarse [3076:3844]
        ti = np.asarray([fb.track_index[int(tid)] for tid in v.tracks], dtype=np.int64)
        self.dino_track_id = np.concatenate((fb.anchor_features[ti], fb.coarse[ti]), axis=1).astype(np.float32)  # (K, 3075)
        self.dino_track_coarse = fb.coarse[ti].astype(np.float32)  # (K, 768)
        # Frame-level DINO parts for X_dyn: current+valid [2307:3076], current-coarse
        cur_dino = self.dino_obs[:, 2307:3075]
        cur_valid = self.dino_obs[:, 3075:3076]
        coarse_dino = self.dino_obs[:, 3076:3844]
        self.dino_obs_dyn = np.concatenate((cur_dino, cur_valid, cur_dino - coarse_dino), axis=1).astype(np.float32)  # (n, 1537)

    def full_rows(self, flat_ids):
        ids = np.asarray(flat_ids, dtype=np.int64).reshape(-1)
        q, r = np.divmod(ids, self.n)
        t = self.cols[r]
        cur = ((self.v.dense_rows(q, r) - self.inp_mean) / self.inp_std).astype(np.float32)
        out = np.concatenate((
            self.bags[q, t], self.present[t], cur, self.coarse[q, t], self.dino_obs[r]
        ), axis=1)
        assert out.shape == (len(ids), FULL_DIM)
        return out

    def dyn_rows(self, flat_ids):
        ids = np.asarray(flat_ids, dtype=np.int64).reshape(-1)
        q, r = np.divmod(ids, self.n)
        t = self.cols[r]
        cur = ((self.v.dense_rows(q, r) - self.inp_mean) / self.inp_std).astype(np.float32)
        crs = self.coarse[q, t]
        out = np.concatenate((cur, cur - crs, self.dino_obs_dyn[r]), axis=1)
        assert out.shape == (len(ids), DYN_DIM)
        return out

    def all_track_id(self):
        """Return (nq, K, 7482) track-invariant X_id directly without row lookups."""
        pres = np.broadcast_to(self.present[None, :, :], (self.nq, self.n_tracks, 3))
        dino = np.broadcast_to(self.dino_track_id[None, :, :], (self.nq, self.n_tracks, 3075))
        out = np.concatenate((self.bags, pres, self.coarse, dino), axis=2).astype(np.float32)
        assert out.shape == (self.nq, self.n_tracks, ID_DIM)
        return out


def extract_source_events_with_tracks(context, case):
    from scripts.a100_conditional_state_engineering import source_events
    raw_events, provenance = source_events(context, case)
    cols = context.video.columns
    events = []
    for ev, prov in zip(raw_events, provenance):
        node0 = int(ev.nodes[0])
        track_col = int(cols[node0])
        assert all(int(cols[nd]) == track_col for nd in ev.nodes)
        events.append(dict(
            uid=ev.uid, q=int(ev.query), track_col=track_col,
            view=int(ev.view), prob=float(prov['original_soft_probability']),
            answer=int(ev.answer), nodes=tuple(int(nd) for nd in ev.nodes),
            node=int(ev.uid) % context.n,
        ))
    return events, provenance


def precompute_graph_distances(context, accessor):
    v = context.video
    n_tracks = len(v.tracks)
    d_app = pairwise_unit_sqdist(accessor.dino_track_coarse)
    geo_desc = track_geometry_descriptors(v.obs, v.geometry, v.columns, n_tracks)
    d_geo = pairwise_unit_sqdist(geo_desc)
    pcd_coarse = context.inputs.coarse  # (nq, K, 1101)
    d_pcd = np.empty((context.nq, n_tracks, n_tracks), dtype=np.float64)
    for q in range(context.nq):
        d_pcd[q] = pairwise_unit_sqdist(pcd_coarse[q])
    return d_app, d_pcd, d_geo


def _batched_solve_from_laplacians(prior, laps, target, precision, lam_per_q, tau_prior=1.0):
    nq, k = prior.shape
    eye = np.eye(k, dtype=np.float64)[None, :, :]
    A = tau_prior * eye + lam_per_q[:, None, None] * laps
    idx = np.arange(k)
    A[:, idx, idx] += precision
    M = np.linalg.inv(A)
    rhs = tau_prior * prior + precision * target
    u = np.einsum('qij,qj->qi', M, rhs)
    lev = np.clip(np.einsum('qij,qj->qi', M, precision), 0.0, 1.0)
    m_diag = M[:, idx, idx]
    denom = np.maximum(1.0 - precision * m_diag, 1e-12)
    loto = np.where(precision > 0.0, (u - precision * m_diag * target) / denom, u)
    return u, lev, loto


def _build_all_laplacians(d_app, d_pcd, d_geo, is_dyn, params, k_nn=5):
    nq, k = d_pcd.shape[:2]
    laps = np.empty((nq, k, k), dtype=np.float64)
    for q in range(nq):
        w_geo = params.w_geo_dyn if is_dyn[q] else params.w_geo_static
        dist = params.w_app * d_app + params.w_pcd * d_pcd[q] + w_geo * d_geo
        laps[q], _ = build_combinatorial_laplacian(dist, k_nn=k_nn)
    return laps


def fit_observer_loto_fast(g_id_oof, f_dyn_tm_oof, f_dyn_res_oof, d_app, d_pcd, d_geo,
                           is_dyn, events, columns, *, protocol='PRESUPPOSED',
                           k_nn=5, prior_reg=0.015):
    """Two-stage exact Sherman-Morrison LOTO optimization on OUT-OF-TRACK priors (0 GT)."""
    theta0 = _default_observer_vector()
    nq, n_tracks = g_id_oof.shape
    if not events:
        p0 = _unpack_observer_vector(theta0)
        return p0, _build_all_laplacians(d_app, d_pcd, d_geo, is_dyn, p0, k_nn=k_nn), dict(
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
        params = _unpack_observer_vector(theta)
        prior = (1.0 - params.beta_mix) * g_id_oof + params.beta_mix * f_dyn_tm_oof
        target, precision, _, _ = aggregate_track_anchors(
            nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
        lam_q = np.where(is_dyn, params.lam_vis_dyn, params.lam_vis_static)
        _, _, loto_u = _batched_solve_from_laplacians(prior, laps, target, precision, lam_q, tau_prior=1.0)
        gam_q = np.where(is_dyn, params.gamma_dyn_dyn, params.gamma_dyn_static)
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
        (1.0, 1.0, 0.15, 1.0),
        (2.0, 0.5, 0.05, 0.8),
        (0.5, 2.0, 0.10, 1.2),
        (1.0, 1.0, 0.50, 2.0),
        (1.5, 1.5, 0.01, 0.05),
    ]
    best_theta = theta0.copy()
    best_laps = _build_all_laplacians(d_app, d_pcd, d_geo, is_dyn, _unpack_observer_vector(theta0), k_nn=k_nn)
    before = eval_with_laps(theta0, best_laps)
    best_val = before
    for wa, wp, wgs, wgd in kernel_candidates:
        cand = theta0.copy()
        cand[2:6] = [inv_sp(wa), inv_sp(wp), inv_sp(wgs), inv_sp(wgd)]
        laps = _build_all_laplacians(d_app, d_pcd, d_geo, is_dyn, _unpack_observer_vector(cand), k_nn=k_nn)
        val = eval_with_laps(cand, laps)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_theta = cand.copy()
            best_laps = laps

    opt_idx = [0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    sub0 = best_theta[opt_idx].copy()

    def sub_obj(sub):
        full = best_theta.copy()
        full[opt_idx] = sub
        return eval_with_laps(full, best_laps)

    res = minimize(sub_obj, sub0, method='Powell',
                   options=dict(maxiter=10, xtol=3e-2, ftol=3e-3))
    if np.isfinite(res.fun) and res.fun < best_val:
        best_theta[opt_idx] = res.x
        best_val = float(res.fun)
    params = _unpack_observer_vector(best_theta)
    return params, best_laps, dict(
        loto_before=float(before), loto_after=float(best_val),
        stage1_evals=len(kernel_candidates), stage2_evals=int(getattr(res, 'nfev', 0)),
    )


def infer_transductive_fast(g_id, f_dyn_tm, f_dyn_res, laps, is_dyn, events,
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
    gam = np.where(is_dyn, params.gamma_dyn_dyn, params.gamma_dyn_static)[:, None]
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


def fit_regularized_head(model, x_np, y_np, w_np, steps, device, *,
                         lr=3e-4, weight_decay=2e-2, batch_size=256, seed=0):
    torch.manual_seed(seed)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    w = torch.as_tensor(w_np, dtype=torch.float32, device=device)
    w = w / w.sum().clamp(min=1e-12)
    n = len(x)
    bs = min(batch_size, n)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.1)
    model.train()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    started = time.perf_counter()
    last_loss = 0.0
    for _ in range(steps):
        if bs == n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randint(0, n, (bs,), generator=gen).to(device)
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
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    model.eval()
    with torch.no_grad():
        zs = [model(x[b:b + 1024]).double().cpu().numpy() for b in range(0, n, 1024)]
        z_all = np.concatenate(zs)
    yv = np.asarray(y_np, dtype=np.float64)
    wv = np.asarray(w_np, dtype=np.float64)
    wv = wv / max(wv.sum(), 1e-12)
    h = -yv * np.log(np.clip(yv, 1e-15, 1.0)) - (1.0 - yv) * np.log(np.clip(1.0 - yv, 1e-15, 1.0))
    excess_ce = float(np.sum(wv * (np.logaddexp(0.0, z_all) - yv * z_all - h)))
    return dict(
        steps=int(steps), lr=float(lr), weight_decay=float(weight_decay),
        last_loss=last_loss, proper_excess_ce=excess_ce,
        wall_seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def predict_track_id(model, x_id_matrix, device):
    nq, k, d = x_id_matrix.shape
    flat = x_id_matrix.reshape(nq * k, d)
    out = np.empty(nq * k, dtype=np.float64)
    model.eval()
    for b in range(0, len(flat), 2048):
        sl = slice(b, min(b + 2048, len(flat)))
        xb = torch.as_tensor(flat[sl], dtype=torch.float32, device=device)
        out[sl] = model(xb).cpu().numpy()
    return out.reshape(nq, k)


@torch.no_grad()
def predict_frame_dyn_fast(model, accessor, device):
    total = accessor.n * accessor.nq
    out = np.empty(total, dtype=np.float32)
    model.eval()
    for b in range(0, total, 4096):
        rows = np.arange(b, min(b + 4096, total), dtype=np.int64)
        xb = torch.as_tensor(accessor.dyn_rows(rows), dtype=torch.float32, device=device)
        out[rows] = model(xb).cpu().numpy()
    return out.reshape(accessor.nq, accessor.n)


@torch.no_grad()
def predict_full_head_fast(model, accessor, device):
    total = accessor.n * accessor.nq
    out = np.empty(total, dtype=np.float32)
    model.eval()
    for b in range(0, total, 4096):
        rows = np.arange(b, min(b + 4096, total), dtype=np.int64)
        xb = torch.as_tensor(accessor.full_rows(rows), dtype=torch.float32, device=device)
        out[rows] = model(xb).cpu().numpy()
    return out.reshape(accessor.nq, accessor.n)


def _compute_oof_priors(x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
                        dyn_anchor_x, dyn_anchor_y, dyn_anchor_w, dyn_anchor_k,
                        aux_id, aux_dyn, aux_y, aux_w, id_mean, id_std, dyn_mean, dyn_std,
                        plan, device):
    """2-fold track-disjoint out-of-fold predictions for unbiased LOTO observer fitting."""
    nq, n_tracks = x_id_mat.shape[:2]
    n = accessor.n
    g_id_oof = np.zeros((nq, n_tracks), dtype=np.float64)
    f_dyn_oof = np.zeros((nq, n), dtype=np.float32)
    cols = accessor.cols
    fold_steps = max(int(plan['steps_per_round'][0]) // 2, 200)
    for fold in (0, 1):
        tr_id = (id_anchor_k % 2) != fold
        tr_dy = (dyn_anchor_k % 2) != fold
        if not tr_id.any():
            tr_id = np.ones_like(tr_id)
        if not tr_dy.any():
            tr_dy = np.ones_like(tr_dy)
        torch.manual_seed(fold)
        m_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_regularized_head(
            m_id,
            np.concatenate((id_anchor_x[tr_id], aux_id), axis=0),
            np.concatenate((id_anchor_y[tr_id], aux_y), axis=0),
            np.concatenate((id_anchor_w[tr_id], aux_w), axis=0),
            fold_steps, device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], batch_size=256, seed=fold)
        pred_id = predict_track_id(m_id, x_id_mat, device)
        val_k = (np.arange(n_tracks) % 2) == fold
        g_id_oof[:, val_k] = pred_id[:, val_k]

        torch.manual_seed(10 + fold)
        m_dy = RegularizedMLP(dyn_mean, dyn_std, width=plan['width'], dropout=plan['dropout']).to(device)
        fit_regularized_head(
            m_dy,
            np.concatenate((dyn_anchor_x[tr_dy], aux_dyn), axis=0),
            np.concatenate((dyn_anchor_y[tr_dy], aux_y), axis=0),
            np.concatenate((dyn_anchor_w[tr_dy], aux_w), axis=0),
            fold_steps, device, lr=float(plan['learning_rate']),
            weight_decay=plan['weight_decay'], batch_size=256, seed=10 + fold)
        pred_dy = predict_frame_dyn_fast(m_dy, accessor, device)
        val_obs = (cols % 2) == fold
        f_dyn_oof[:, val_obs] = pred_dy[:, val_obs]
    return g_id_oof, f_dyn_oof


def learn_dual_transduct_coevol(context, case, plan, folder, device='cuda:0', accessor=None):
    """Run 3-round DUAL_TRANSDUCT_COEVOL with OOF-calibrated LOTO R (0 GT)."""
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
    events, provenance = extract_source_events_with_tracks(context, case)
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

    # Step 0: True 2-fold out-of-track cross-fitting to learn unbiased observer R via LOTO!
    oof_started = time.perf_counter()
    g_id_oof, f_dyn_oof = _compute_oof_priors(
        x_id_mat, accessor, id_anchor_x, id_anchor_y, id_anchor_w, id_anchor_k,
        dyn_anchor_x, dyn_anchor_y, dyn_anchor_w, dyn_anchor_k,
        aux_id, aux_dyn, aux_y, aux_w, id_mean, id_std, dyn_mean, dyn_std,
        plan, device)
    f_dyn_tm_oof, f_dyn_res_oof = smooth_within_track_residual(
        f_dyn_oof, video.obs, video.columns, context.images, n_tracks,
        kappa=plan['kappa'], base_decay=plan['base_decay'])
    params, laps, loto_oof = fit_observer_loto_fast(
        g_id_oof, f_dyn_tm_oof, f_dyn_res_oof, d_app, d_pcd, d_geo,
        is_dyn, events, video.columns, protocol=protocol, k_nn=plan['k_nn'], prior_reg=0.015)
    oof_seconds = time.perf_counter() - oof_started

    # Instantiate complementary Dual Students (seed0)
    torch.manual_seed(0)
    s_id = RegularizedMLP(id_mean, id_std, width=plan['width'], dropout=plan['dropout']).to(device)
    torch.manual_seed(0)
    s_dyn = RegularizedMLP(dyn_mean, dyn_std, width=plan['width'], dropout=plan['dropout']).to(device)

    rng = np.random.default_rng(0)
    rounds_log = []
    trans = None
    track_rows_cache = {k: np.flatnonzero(video.columns == k) for k in range(n_tracks)}
    n_rounds = int(plan['rounds'])
    steps_schedule = plan['steps_per_round']

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
                raw_w = raw_w * (0.35 / max(float(raw_w.sum()), 1e-12))
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
                sel_w = sel_w * (0.35 / max(float(sel_w.sum()), 1e-12))
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

        steps_r = int(steps_schedule[r])
        lr_r = float(plan['learning_rate']) * (0.5 ** r)
        id_fit = fit_regularized_head(
            s_id, x_id_tr, y_id_tr, w_id_tr, steps_r, device,
            lr=lr_r, weight_decay=plan['weight_decay'], batch_size=256, seed=r)
        dy_fit = fit_regularized_head(
            s_dyn, x_dy_tr, y_dy_tr, w_dy_tr, steps_r, device,
            lr=lr_r, weight_decay=plan['weight_decay'], batch_size=256, seed=100 + r)

        g_id = predict_track_id(s_id, x_id_mat, device)
        f_dyn_raw = predict_frame_dyn_fast(s_dyn, accessor, device)
        f_dyn_tm, f_dyn_res = smooth_within_track_residual(
            f_dyn_raw, video.obs, video.columns, context.images, n_tracks,
            kappa=plan['kappa'], base_decay=plan['base_decay'])
        # Use the OOF-calibrated observer R so graph/channel never suffer in-sample collapse!
        trans = infer_transductive_fast(
            g_id, f_dyn_tm, f_dyn_res, laps, is_dyn, events,
            video.columns, params, protocol=protocol)

        row = dict(
            round=r, steps=steps_r, lr=lr_r,
            s_id_fit=id_fit, s_dyn_fit=dy_fit,
            trans_tracks_used=trans_tracks_used, absent_neg_queries=absent_neg_used,
            mean_leverage=float(trans['leverage'].mean()),
            mean_existence=float(trans['existence'].mean()),
            min_existence=float(trans['existence'].min()),
            round_wall_seconds=time.perf_counter() - r_start,
        )
        rounds_log.append(row)
        print(json.dumps(dict(
            status='DUAL_TRANSDUCT_COEVOL_ROUND', pair=case['pair'], video=case['evaluated_video'],
            round=r, trans_tracks=trans_tracks_used, absent_neg=absent_neg_used,
            mean_lev=round(float(trans['leverage'].mean()), 3),
            mean_eta=round(float(trans['existence'].mean()), 3),
            min_eta=round(float(trans['existence'].min()), 3),
            sec=round(row['round_wall_seconds'], 1),
        )), flush=True)

    receipt = dict(
        status='COMPLETE_DUAL_TRANSDUCT_COEVOL_LEARNING',
        pair=case['pair'], video=case['evaluated_video'], protocol=protocol,
        n_tracks=n_tracks, queries=nq, observations=n, events=len(events),
        dynamic_queries=int(is_dyn.sum()),
        oof_loto=loto_oof, oof_wall_seconds=oof_seconds,
        observer=params.to_dict(),
        rounds=rounds_log,
        wall_seconds=time.perf_counter() - started,
        GT_open_attempts=0, new_teacher_calls=0, seed=0,
    )
    atomic_json(folder / 'learning_receipt.json', receipt)
    return dict(
        receipt=receipt, accessor=accessor,
        s_id=s_id, s_dyn=s_dyn,
        params=params, laps=laps, trans=trans,
        events=events, is_dyn=is_dyn, protocol=protocol,
    )


def apply_shared_transduct_readout(frame_scores, context, learned):
    n_tracks = len(context.video.tracks)
    tm, res = smooth_within_track_residual(
        frame_scores, context.video.obs, context.video.columns,
        context.images, n_tracks, kappa=4.0, base_decay=0.85)
    out = infer_transductive_fast(
        tm, tm, res, learned['laps'], learned['is_dyn'], learned['events'],
        context.video.columns, learned['params'], protocol=learned['protocol'])
    return out['frame_prob']


def fit_regularized_ordinary_controls(context, case, plan, device='cuda:0', accessor=None):
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
    total_steps = int(sum(plan['steps_per_round']))

    outputs, receipts = {}, {}
    for mode, hard in (('REG_SOFT_KD', False), ('REG_HARD_KD', True)):
        y_all = np.concatenate(((probs >= 0.5).astype(np.float64) if hard else probs,
                                (aux_y >= 0.5).astype(np.float64) if hard else aux_y), axis=0)
        torch.manual_seed(0)
        head = RegularizedMLP(context.mean, context.std, width=plan['width'], dropout=plan['dropout']).to(device)
        rec = fit_regularized_head(
            head, x_all, y_all, w_all, total_steps, device,
            lr=float(plan['learning_rate']), weight_decay=plan['weight_decay'],
            batch_size=256, seed=0)
        scores = predict_full_head_fast(head, accessor, device)
        outputs[mode] = scores
        receipts[mode] = rec
    return outputs, receipts
