"""Dual-level transductive co-evolution for cross-domain RMOT (no GT access).

Unifies three explicit, testable mechanisms grounded in human dual-process
vision, classical transductive inference, and externally anchored co-evolution:
1. Hierarchical Video Existence (E_q) x Slow Persistent Track Identity (U_{q,k})
   x Fast Zero-Mean Dynamic Residual (delta_{q,t}), eliminating the exponential
   long-horizon identity decay of single-level Markov chains.
2. Predicate-aware Cross-Track Visual-Spatial Affinity Graph Laplacian with
   exact M-matrix external-anchor leverage h_q = A_q^{-1} pi_q in [0, 1],
   learned without GT via exact Sherman-Morrison Leave-One-Track-Out (LOTO)
   teacher log-likelihood.
3. Regularized complementary Dual Students (track-level S_id, frame-dynamic
   S_dyn) co-evolving on direct paid teacher anchors, leverage-weighted
   transductive neighbor targets, and CHECKLIST ABSENT_NEG feedback.

Assumptions and boundaries:
- Oriented teacher channels (0.5, 0.99) and correlated-repeat attenuation are
  explicit modeling assumptions, not identified semantic ground truth.
- Exact M-matrix leverage bounds prevent isolated unasked self-training, but
  do not guarantee HOTA improvement over adequate distillation.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch import nn

VIEWS = ('CROP', 'CLIP', 'FRAME_BOX')

# Exact verified slices of the shared 9352D feature vector:
# [0:3306] bags+present, [3306:4407] current PCD, [4407:5508] coarse PCD,
# [5508:7815] DINO anchors+present, [7815:8584] current DINO+valid,
# [8584:9352] coarse DINO.
ID_DIM = 7482
DYN_DIM = 3739
FULL_DIM = 9352


def split_shared_features_np(x):
    """Split shared 9352D features into track-invariant X_id and frame X_dyn."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != FULL_DIM:
        raise ValueError(f'Expected trailing dim {FULL_DIM}, got {x.shape[-1]}')
    x_id = np.concatenate((x[..., 0:3306], x[..., 4407:5508],
                           x[..., 5508:7815], x[..., 8584:9352]), axis=-1)
    cur_pcd, coarse_pcd = x[..., 3306:4407], x[..., 4407:5508]
    cur_dino, coarse_dino = x[..., 7815:8583], x[..., 8584:9352]
    dino_valid = x[..., 8583:8584]
    x_dyn = np.concatenate((cur_pcd, cur_pcd - coarse_pcd,
                            cur_dino, dino_valid, cur_dino - coarse_dino), axis=-1)
    assert x_id.shape[-1] == ID_DIM and x_dyn.shape[-1] == DYN_DIM
    return x_id, x_dyn


def split_shared_features_torch(x):
    """Torch version of shared 9352D split (identical indexing)."""
    if x.shape[-1] != FULL_DIM:
        raise ValueError(f'Expected trailing dim {FULL_DIM}, got {x.shape[-1]}')
    x_id = torch.cat((x[..., 0:3306], x[..., 4407:5508],
                      x[..., 5508:7815], x[..., 8584:9352]), dim=-1)
    cur_pcd, coarse_pcd = x[..., 3306:4407], x[..., 4407:5508]
    cur_dino, coarse_dino = x[..., 7815:8583], x[..., 8584:9352]
    dino_valid = x[..., 8583:8584]
    x_dyn = torch.cat((cur_pcd, cur_pcd - coarse_pcd,
                       cur_dino, dino_valid, cur_dino - coarse_dino), dim=-1)
    return x_id, x_dyn


def bounded_logit(raw, limit=6.0):
    return limit * raw / (limit + raw.abs())


class RegularizedMLP(nn.Module):
    """LayerNorm + Dropout + bounded-logit head resisting sparse-label overfit."""
    def __init__(self, mean, std, width=256, dropout=0.2):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32).clamp(min=0.1))
        self.net = nn.Sequential(
            nn.Linear(len(mean), width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        z = (x - self.mean) / self.std
        return bounded_logit(self.net(z).flatten(), limit=6.0)


def track_geometry_descriptors(observations, geometry, columns, n_tracks):
    """Compute 8D track-level spatial-motion descriptors from official boxes."""
    obs = np.asarray(observations, dtype=np.float64)
    geom = np.asarray(geometry[:, :4], dtype=np.float64)
    cols = np.asarray(columns, dtype=np.int64)
    desc = np.zeros((n_tracks, 8), dtype=np.float64)
    for k in range(n_tracks):
        rows = np.flatnonzero(cols == k)
        if len(rows) == 0:
            continue
        order = rows[np.argsort(obs[rows, 0], kind='stable')]
        g = geom[order]
        t = obs[order, 0]
        desc[k, 0:4] = g.mean(axis=0)
        desc[k, 4:6] = g[:, 0:2].std(axis=0) if len(order) > 1 else 0.0
        dt = max(float(t[-1] - t[0]), 1.0)
        desc[k, 6:8] = (g[-1, 0:2] - g[0, 0:2]) / dt * 30.0
    mean = desc.mean(axis=0, keepdims=True)
    std = np.maximum(desc.std(axis=0, keepdims=True), 1e-4)
    return ((desc - mean) / std).astype(np.float64)


def pairwise_unit_sqdist(vectors):
    """Squared Euclidean distance scaled by median positive off-diagonal entry."""
    v = np.asarray(vectors, dtype=np.float64)
    n = len(v)
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norms > 1e-8):
        v = v / norms
    dot = v @ v.T
    sq = np.sum(v * v, axis=-1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2.0 * dot, 0.0)
    np.fill_diagonal(dist, 0.0)
    off = dist[~np.eye(n, dtype=bool)]
    pos = off[off > 1e-10]
    scale = float(np.median(pos)) if pos.size else 1.0
    return np.minimum(dist / max(scale, 1e-8), 10.0)


def build_combinatorial_laplacian(dist_matrix, k_nn=5):
    """Symmetric normalized affinity -> combinatorial Laplacian (exact L*1 = 0).

    L = diag(W_norm 1) - W_norm is symmetric PSD with L 1 = 0, guaranteeing
    that A = tau I + lambda L + diag(pi) is an exact M-matrix with
    A^{-1}(tau 1 + pi) = 1 and external leverage h = A^{-1} pi in [0, 1].
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    n = len(dist)
    if n < 2:
        return np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)
    k = min(int(k_nn), n - 1)
    masked = dist + np.eye(n) * 1e9
    nbrs = np.argsort(masked, axis=1, kind='stable')[:, :k]
    sigma = np.sqrt(np.maximum(np.take_along_axis(dist, nbrs[:, -1:], axis=1).ravel(), 1e-6))
    raw_w = np.exp(-dist / np.maximum(sigma[:, None] * sigma[None, :], 1e-6))
    connected = np.zeros((n, n), dtype=bool)
    np.put_along_axis(connected, nbrs, True, axis=1)
    w = raw_w * (connected | connected.T)
    np.fill_diagonal(w, 0.0)
    deg = np.maximum(w.sum(axis=1), 1e-8)
    w_norm = w / np.sqrt(deg[:, None] * deg[None, :])
    np.fill_diagonal(w_norm, 0.0)
    lap = np.diag(w_norm.sum(axis=1)) - w_norm
    return lap, w_norm


def channel_event_llr(prob, sensitivity, specificity):
    """Soft-preserving bounded Bayesian channel log-likelihood ratio."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    s = np.clip(float(sensitivity), 0.501, 0.99)
    c = np.clip(float(specificity), 0.501, 0.99)
    return p * math.log(s / (1.0 - c)) + (1.0 - p) * math.log((1.0 - s) / c)


def aggregate_track_anchors(nq, n_tracks, events, reliability, *,
                            anchor_base_precision=2.5, corr_rho=0.5):
    """Aggregate deduplicated teacher events per (query, track) without overcounting.

    Sublinear effective count m / (1 + corr_rho * (m - 1)) prevents repeated
    correlated Qwen observations on the same track from acting as independent votes.
    Returns:
        target_logit: (nq, K) mean channel LLR for asked tracks, 0 elsewhere
        precision: (nq, K) anchor precision pi_{q,k} >= 0 (0 if unasked)
        asked_count: (nq, K) int count of unique events
        mean_prob: (nq, K) mean raw teacher probability on asked tracks (nan if unasked)
    """
    rel = np.asarray(reliability, dtype=np.float64)
    assert rel.shape == (3, 2)
    llr_sum = np.zeros((nq, n_tracks), dtype=np.float64)
    prob_sum = np.zeros((nq, n_tracks), dtype=np.float64)
    counts = np.zeros((nq, n_tracks), dtype=np.int64)
    for ev in events:
        q, k, v, p = int(ev['q']), int(ev['track_col']), int(ev['view']), float(ev['prob'])
        llr_sum[q, k] += channel_event_llr(p, rel[v, 0], rel[v, 1])
        prob_sum[q, k] += p
        counts[q, k] += 1
    asked = counts > 0
    target = np.zeros((nq, n_tracks), dtype=np.float64)
    mean_prob = np.full((nq, n_tracks), np.nan, dtype=np.float64)
    precision = np.zeros((nq, n_tracks), dtype=np.float64)
    if asked.any():
        target[asked] = llr_sum[asked] / counts[asked]
        mean_prob[asked] = prob_sum[asked] / counts[asked]
        m = counts[asked].astype(np.float64)
        eff = m / (1.0 + corr_rho * (m - 1.0))
        conf = 0.5 + np.abs(mean_prob[asked] - 0.5)
        precision[asked] = anchor_base_precision * eff * conf
    return target, precision, counts, mean_prob


def solve_track_transduction(prior_logits, laplacian, target_logits, precision,
                             *, tau_prior=1.0, lam_vis=1.0):
    """Exact M-matrix track graph solve + external leverage + Sherman-Morrison LOTO.

    Solves: (tau I + lam L + diag(pi)) u = tau mu + pi * t
    Returns:
        u: (K,) transduced track logits
        leverage: (K,) external anchor leverage h = A^{-1} pi in [0, 1]
        loto_u: (K,) exact Leave-One-Track-Out logits u^{(-k)}_k for all tracks
    """
    mu = np.asarray(prior_logits, dtype=np.float64)
    L = np.asarray(laplacian, dtype=np.float64)
    t = np.asarray(target_logits, dtype=np.float64)
    pi = np.asarray(precision, dtype=np.float64)
    k = len(mu)
    if k == 0:
        return mu.copy(), np.zeros(0), mu.copy()
    tau = max(float(tau_prior), 1e-6)
    lam = max(float(lam_vis), 0.0)
    A = tau * np.eye(k) + lam * L + np.diag(pi)
    M = np.linalg.inv(A)
    rhs = tau * mu + pi * t
    u = M @ rhs
    leverage = np.clip(M @ pi, 0.0, 1.0)
    # Exact Sherman-Morrison LOTO: u^{(-k)}_k = (u_k - pi_k M_{kk} t_k) / (1 - pi_k M_{kk})
    m_diag = np.diag(M)
    denom = np.maximum(1.0 - pi * m_diag, 1e-12)
    loto_u = np.where(pi > 0.0, (u - pi * m_diag * t) / denom, u)
    return u, leverage, loto_u


def compute_video_existence(prior_logits, events_by_query, reliability, *,
                            protocol='PRESUPPOSED', existence_prior=0.5,
                            neg_cap=-2.5, pos_cap=4.0):
    """Hierarchical video-level existence P(E_q=1 | Y_q) with track-level capping."""
    nq = len(prior_logits)
    if protocol == 'PRESUPPOSED':
        return np.ones(nq, dtype=np.float64)
    if protocol != 'CHECKLIST':
        raise ValueError(f'Unknown protocol: {protocol}')
    rel = np.asarray(reliability, dtype=np.float64)
    eta = np.empty(nq, dtype=np.float64)
    base_logit = float(logit(np.clip(existence_prior, 0.05, 0.95)))
    for q in range(nq):
        by_track = {}
        for ev in events_by_query[q]:
            by_track.setdefault(int(ev['track_col']), []).append(ev)
        total_llr = base_logit
        for k, evs in by_track.items():
            # Correlated-repeat attenuation within a single track: average log-likes scaled by eff count
            m = float(len(evs))
            eff = m / (1.0 + 0.5 * (m - 1.0))
            lp = sum(ev['prob'] * math.log(rel[ev['view'], 0]) +
                     (1.0 - ev['prob']) * math.log(1.0 - rel[ev['view'], 0]) for ev in evs) / m * eff
            ln = sum(ev['prob'] * math.log(1.0 - rel[ev['view'], 1]) +
                     (1.0 - ev['prob']) * math.log(rel[ev['view'], 1]) for ev in evs) / m * eff
            p_k = float(np.clip(max(expit(prior_logits[q, k]), 0.35), 0.35, 0.98))
            m1 = np.logaddexp(math.log(p_k) + lp, math.log(1.0 - p_k) + ln)
            track_llr = float(np.clip(m1 - ln, neg_cap, pos_cap))
            total_llr += track_llr
        eta[q] = float(expit(total_llr))
    return eta


def smooth_within_track_residual(frame_logits, observations, columns, images, n_tracks, *,
                                 kappa=4.0, base_decay=0.85):
    """Appearance-gated zero-mean within-track dynamic residual.

    Preserves instantaneous jumps when DINO appearance shifts (ID switches or
    abrupt state changes) while smoothing frame-to-frame flicker.
    """
    scores = np.asarray(frame_logits, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.int64)
    cols = np.asarray(columns, dtype=np.int64)
    imgs = np.asarray(images, dtype=np.float64)
    nq, n = scores.shape
    track_mean = np.zeros((nq, n_tracks), dtype=np.float64)
    residual = np.zeros((nq, n), dtype=np.float64)
    for k in range(n_tracks):
        rows = np.flatnonzero(cols == k)
        if len(rows) == 0:
            continue
        order = rows[np.argsort(obs[rows, 0], kind='stable')]
        vals = scores[:, order]
        mean_k = vals.mean(axis=1)
        track_mean[:, k] = mean_k
        if len(order) == 1:
            continue
        raw_res = vals - mean_k[:, None]
        # Appearance cosine distance between consecutive observations in track k
        v = imgs[order]
        norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        vn = v / norms
        cos_dist = np.clip(1.0 - np.sum(vn[1:] * vn[:-1], axis=1), 0.0, 2.0)
        alpha = base_decay * np.exp(-kappa * cos_dist)  # high when appearance unchanged
        fwd = np.empty_like(raw_res)
        fwd[:, 0] = raw_res[:, 0]
        for t in range(1, len(order)):
            a = alpha[t - 1]
            fwd[:, t] = a * fwd[:, t - 1] + (1.0 - a) * raw_res[:, t]
        bwd = np.empty_like(raw_res)
        bwd[:, -1] = raw_res[:, -1]
        for t in range(len(order) - 2, -1, -1):
            a = alpha[t]
            bwd[:, t] = a * bwd[:, t + 1] + (1.0 - a) * raw_res[:, t]
        sm = 0.5 * (fwd + bwd)
        # Re-center strictly to zero mean so fast residual never shifts track baseline
        sm -= sm.mean(axis=1, keepdims=True)
        residual[:, order] = sm
    return track_mean, residual


@dataclass
class TransductiveObserverParams:
    """Low-dimensional interpretable parameters learned strictly via LOTO teacher NLL."""
    lam_vis_static: float = 1.5
    lam_vis_dyn: float = 0.8
    w_app: float = 1.0
    w_pcd: float = 1.0
    w_geo_static: float = 0.15
    w_geo_dyn: float = 1.0
    beta_mix: float = 0.4
    anchor_precision: float = 2.5
    gamma_dyn_static: float = 0.15
    gamma_dyn_dyn: float = 0.5
    channel: np.ndarray = None  # (3, 2) in (0.501, 0.99)

    def __post_init__(self):
        if self.channel is None:
            self.channel = np.full((3, 2), 0.82, dtype=np.float64)

    def to_dict(self):
        return dict(
            lam_vis_static=float(self.lam_vis_static),
            lam_vis_dyn=float(self.lam_vis_dyn),
            w_app=float(self.w_app),
            w_pcd=float(self.w_pcd),
            w_geo_static=float(self.w_geo_static),
            w_geo_dyn=float(self.w_geo_dyn),
            beta_mix=float(self.beta_mix),
            anchor_precision=float(self.anchor_precision),
            gamma_dyn_static=float(self.gamma_dyn_static),
            gamma_dyn_dyn=float(self.gamma_dyn_dyn),
            channel=np.asarray(self.channel, dtype=float).tolist(),
        )


def _unpack_observer_vector(theta):
    """Map unconstrained 16D vector to valid interpretable observer parameters."""
    def sp(v):
        return float(np.logaddexp(0.0, v))
    lam_s = min(sp(theta[0]), 12.0)
    lam_d = min(sp(theta[1]), 12.0)
    w_app = sp(theta[2]) + 1e-3
    w_pcd = sp(theta[3]) + 1e-3
    w_gs = sp(theta[4])
    w_gd = sp(theta[5])
    beta_mix = float(expit(theta[6]))
    anc_prec = min(sp(theta[7]) + 0.25, 10.0)
    gam_s = min(sp(theta[8]), 2.0)
    gam_d = min(sp(theta[9]), 2.0)
    ch = 0.60 + 0.37 * expit(theta[10:16].reshape(3, 2))
    return TransductiveObserverParams(
        lam_vis_static=lam_s, lam_vis_dyn=lam_d, w_app=w_app, w_pcd=w_pcd,
        w_geo_static=w_gs, w_geo_dyn=w_gd, beta_mix=beta_mix,
        anchor_precision=anc_prec, gamma_dyn_static=gam_s, gamma_dyn_dyn=gam_d,
        channel=ch,
    )


def _default_observer_vector():
    def inv_sp(y):
        return float(math.log(max(math.expm1(y), 1e-6)))
    return np.array([
        inv_sp(1.5), inv_sp(0.8), inv_sp(1.0), inv_sp(1.0),
        inv_sp(0.15), inv_sp(1.0), float(logit(0.4)), inv_sp(2.25),
        inv_sp(0.15), inv_sp(0.5),
        *[float(logit((0.82 - 0.60) / 0.37))] * 6,
    ], dtype=np.float64)


def run_transductive_inference(g_id, f_dyn_track_mean, f_dyn_residual,
                               d_app, d_pcd, d_geo, is_dyn, events,
                               columns, params, *, protocol='PRESUPPOSED',
                               k_nn=5):
    """Full hierarchical transductive pass for all queries in one video."""
    nq, n_tracks = g_id.shape
    events_by_query = [[] for _ in range(nq)]
    for ev in events:
        events_by_query[int(ev['q'])].append(ev)
    prior = (1.0 - params.beta_mix) * g_id + params.beta_mix * f_dyn_track_mean
    target, precision, counts, _ = aggregate_track_anchors(
        nq, n_tracks, events, params.channel, anchor_base_precision=params.anchor_precision)
    u_trans = np.empty_like(prior)
    leverage = np.empty_like(prior)
    loto_u = np.empty_like(prior)
    for q in range(nq):
        w_geo = params.w_geo_dyn if is_dyn[q] else params.w_geo_static
        dist = params.w_app * d_app + params.w_pcd * d_pcd[q] + w_geo * d_geo
        lap, _ = build_combinatorial_laplacian(dist, k_nn=k_nn)
        lam = params.lam_vis_dyn if is_dyn[q] else params.lam_vis_static
        u_trans[q], leverage[q], loto_u[q] = solve_track_transduction(
            prior[q], lap, target[q], precision[q], tau_prior=1.0, lam_vis=lam)
    existence = compute_video_existence(
        u_trans, events_by_query, params.channel, protocol=protocol, existence_prior=0.5)
    gam = np.where(is_dyn, params.gamma_dyn_dyn, params.gamma_dyn_static)[:, None]
    cols = np.asarray(columns, dtype=np.int64)
    frame_logits = u_trans[:, cols] + gam * f_dyn_residual
    frame_prob = existence[:, None] * expit(frame_logits)
    track_prob = existence[:, None] * expit(u_trans)
    return dict(
        prior_logits=prior,
        transduced_track_logits=u_trans,
        transduced_track_prob=track_prob,
        leverage=leverage,
        loto_track_logits=loto_u,
        existence=existence,
        frame_logits=frame_logits,
        frame_prob=frame_prob,
        asked_counts=counts,
    )


def fit_observer_loto(g_id, f_dyn_track_mean, f_dyn_residual,
                      d_app, d_pcd, d_geo, is_dyn, events, columns, *,
                      protocol='PRESUPPOSED', k_nn=5, maxiter=45):
    """Optimize observer parameters R strictly via LOTO paid-teacher NLL (ZERO GT).

    Evaluates exact Leave-One-Track-Out prediction for every paid event through
    the learned per-view noisy channel, plus gentle prior regularization toward
    the default vector so tiny videos never overfit.
    """
    theta0 = _default_observer_vector()
    nq = g_id.shape[0]
    if not events:
        return _unpack_observer_vector(theta0), dict(loto_before=0.0, loto_after=0.0, nit=0)
    # Pre-group event arrays for fast evaluation
    ev_q = np.array([int(e['q']) for e in events], dtype=np.int64)
    ev_k = np.array([int(e['track_col']) for e in events], dtype=np.int64)
    ev_v = np.array([int(e['view']) for e in events], dtype=np.int64)
    ev_p = np.array([float(e['prob']) for e in events], dtype=np.float64)
    ev_nodes = [np.asarray(e['nodes'], dtype=np.int64) for e in events]
    q_counts = np.bincount(ev_q, minlength=nq).astype(np.float64)
    ev_w = 1.0 / (max(int((q_counts > 0).sum()), 1) * np.maximum(q_counts[ev_q], 1.0))

    def objective(theta):
        params = _unpack_observer_vector(theta)
        out = run_transductive_inference(
            g_id, f_dyn_track_mean, f_dyn_residual, d_app, d_pcd, d_geo,
            is_dyn, events, columns, params, protocol=protocol, k_nn=k_nn)
        loto_u = out['loto_track_logits']
        gam = np.where(is_dyn, params.gamma_dyn_dyn, params.gamma_dyn_static)
        # Predict each held-out event over its exact distinct scope nodes
        pred_state = np.empty(len(events), dtype=np.float64)
        for j, nodes in enumerate(ev_nodes):
            q, k = ev_q[j], ev_k[j]
            z = loto_u[q, k] + gam[q] * f_dyn_residual[q, nodes]
            # For CHECKLIST, use LOTO-free track probability for ranking/channel fit
            # to avoid self-leakage through E_q.
            pred_state[j] = float(np.mean(expit(z)))
        s = params.channel[ev_v, 0]
        c = params.channel[ev_v, 1]
        p_yes = np.clip((1.0 - c) + (s + c - 1.0) * pred_state, 1e-7, 1.0 - 1e-7)
        nll = -np.sum(ev_w * (ev_p * np.log(p_yes) + (1.0 - ev_p) * np.log1p(-p_yes)))
        # Mild Gaussian prior on unconstrained vector prevents degenerate extremes
        reg = 0.005 * float(np.mean((theta - theta0) ** 2))
        return float(nll + reg)

    before = objective(theta0)
    res = minimize(objective, theta0, method='Powell',
                   options=dict(maxiter=maxiter, xtol=1e-2, ftol=1e-3))
    best_theta = res.x if np.isfinite(res.fun) and res.fun <= before else theta0
    after = min(float(res.fun), before) if np.isfinite(res.fun) else before
    return _unpack_observer_vector(best_theta), dict(
        loto_before=float(before), loto_after=float(after),
        nit=int(getattr(res, 'nit', 0)), nfev=int(getattr(res, 'nfev', 0)),
    )
