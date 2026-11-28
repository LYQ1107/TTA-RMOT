"""DUAL_TRANSDUCT_COEVOL_V2 core: similarity-gated cliques + zero-jitter static tracks.

Addresses all three failure modes diagnosed from V1 (148-unit full evaluation):
1. Similarity-gated tight supportive-clique Laplacian (k_nn=3, dist_gate=1.25):
   preserves exact M-matrix properties (L*1=0, h=A^{-1}pi in [0,1]) while
   leaving visually unique/isolated objects (e.g., sparse positive cars on KITTI)
   disconnected from dissimilar negative majorities (zero over-smoothing).
2. Zero-jitter static tracks (gamma_dyn_static = 0.0) + probability-space track
   pooling + high-coherence bounded segment residual (gamma_dyn_dyn in [0, 0.35],
   base_decay=0.94, kappa=6.0) for dynamic queries, preserving whole-track AssA.
3. Unchanged exact Sherman-Morrison LOTO cross-validation & CHECKLIST existence.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import expit, logit
import torch
from torch import nn

from flex_uda.active.dual_transduct_coevol import (
    DYN_DIM, FULL_DIM, ID_DIM, VIEWS,
    RegularizedMLP, aggregate_track_anchors, channel_event_llr,
    compute_video_existence, pairwise_unit_sqdist, solve_track_transduction,
    split_shared_features_np, split_shared_features_torch,
    track_geometry_descriptors,
)


def build_gated_clique_laplacian(dist_matrix, *, k_nn=3, dist_gate=1.25):
    """Similarity-gated tight supportive-clique combinatorial Laplacian (exact L*1=0).

    Edges whose unit-scaled squared distance exceeds dist_gate are zeroed, so
    visually distinct sparse targets are never forcibly averaged with unrelated
    tracks. Remains symmetric PSD with L*1 = 0, preserving exact M-matrix
    external-anchor leverage h = A^{-1} pi in [0, 1].
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    n = len(dist)
    if n < 2:
        return np.zeros((n, n), dtype=np.float64), np.zeros((n, n), dtype=np.float64)
    k = min(int(k_nn), n - 1)
    masked = dist + np.eye(n) * 1e9
    nbrs = np.argsort(masked, axis=1, kind='stable')[:, :k]
    sigma = np.sqrt(np.maximum(np.take_along_axis(dist, nbrs[:, -1:], axis=1).ravel(), 1e-6))
    # Cap sigma so dissimilar neighbors cannot inflate the kernel bandwidth
    sigma = np.minimum(sigma, math.sqrt(dist_gate))
    raw_w = np.exp(-dist / np.maximum(sigma[:, None] * sigma[None, :], 1e-6))
    connected = np.zeros((n, n), dtype=bool)
    np.put_along_axis(connected, nbrs, True, axis=1)
    # Require BOTH top-k neighborhood AND genuine visual-spatial proximity <= dist_gate
    close = dist <= float(dist_gate)
    w = raw_w * (connected | connected.T) * close
    np.fill_diagonal(w, 0.0)
    deg = w.sum(axis=1)
    inv_sqrt = np.where(deg > 1e-8, 1.0 / np.sqrt(np.maximum(deg, 1e-8)), 0.0)
    w_norm = w * (inv_sqrt[:, None] * inv_sqrt[None, :])
    np.fill_diagonal(w_norm, 0.0)
    lap = np.diag(w_norm.sum(axis=1)) - w_norm
    return lap, w_norm


def prob_pooled_track_and_residual(frame_logits, observations, columns, images, n_tracks, *,
                                   kappa=6.0, base_decay=0.94):
    """Exact probability-space track pooling + high-coherence zero-mean residual.

    Track baseline uses logit(mean(expit(z))) matching PARENT_POOLED's robust
    scale (preventing single-frame outlier logits from distorting track rank).
    Residual uses strong forward-backward appearance-gated smoothing so only
    coherent temporal segments (not 1-frame noise) deviate from track baseline.
    """
    scores = np.asarray(frame_logits, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.int64)
    cols = np.asarray(columns, dtype=np.int64)
    imgs = np.asarray(images, dtype=np.float64)
    nq, n = scores.shape
    track_logit = np.zeros((nq, n_tracks), dtype=np.float64)
    residual = np.zeros((nq, n), dtype=np.float64)
    for k in range(n_tracks):
        rows = np.flatnonzero(cols == k)
        if len(rows) == 0:
            continue
        order = rows[np.argsort(obs[rows, 0], kind='stable')]
        vals = scores[:, order]
        mean_p = np.clip(expit(vals).mean(axis=1), 1e-4, 1.0 - 1e-4)
        base_k = logit(mean_p)
        track_logit[:, k] = base_k
        if len(order) == 1:
            continue
        raw_res = np.clip(vals - base_k[:, None], -4.0, 4.0)
        v = imgs[order]
        norms = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        vn = v / norms
        cos_dist = np.clip(1.0 - np.sum(vn[1:] * vn[:-1], axis=1), 0.0, 2.0)
        alpha = base_decay * np.exp(-kappa * cos_dist)
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
        sm -= sm.mean(axis=1, keepdims=True)
        residual[:, order] = sm
    return track_logit, residual


@dataclass
class TransductiveObserverV2Params:
    """Interpretable V2 observer params: zero static jitter + bounded dynamic gate."""
    lam_vis_static: float = 1.2
    lam_vis_dyn: float = 0.6
    w_app: float = 1.0
    w_pcd: float = 1.0
    w_geo_static: float = 0.15
    w_geo_dyn: float = 1.2
    beta_mix: float = 0.5
    anchor_precision: float = 2.0
    gamma_dyn_static: float = 0.0   # Strictly 0.0 -> zero frame-jitter on static tracks!
    gamma_dyn_dyn: float = 0.15     # Bounded in [0.0, 0.35] -> preserves AssA!
    channel: np.ndarray = None

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


def default_v2_vector():
    def inv_sp(y):
        return float(math.log(max(math.expm1(y), 1e-6)))
    return np.array([
        inv_sp(1.2), inv_sp(0.6), inv_sp(1.0), inv_sp(1.0),
        inv_sp(0.15), inv_sp(1.2), float(logit(0.5)), inv_sp(1.75),
        float(logit(0.15 / 0.35)),
        *[float(logit((0.82 - 0.60) / 0.37))] * 6,
    ], dtype=np.float64)


def unpack_v2_vector(theta):
    def sp(v):
        return float(np.logaddexp(0.0, v))
    lam_s = min(sp(theta[0]), 6.0)
    lam_d = min(sp(theta[1]), 6.0)
    w_app = sp(theta[2]) + 1e-3
    w_pcd = sp(theta[3]) + 1e-3
    w_gs = sp(theta[4])
    w_gd = sp(theta[5])
    beta_mix = float(expit(theta[6]))
    anc_prec = min(sp(theta[7]) + 0.25, 8.0)
    gam_d = 0.35 * float(expit(theta[8]))
    ch = 0.60 + 0.37 * expit(theta[9:15].reshape(3, 2))
    return TransductiveObserverV2Params(
        lam_vis_static=lam_s, lam_vis_dyn=lam_d, w_app=w_app, w_pcd=w_pcd,
        w_geo_static=w_gs, w_geo_dyn=w_gd, beta_mix=beta_mix,
        anchor_precision=anc_prec, gamma_dyn_static=0.0, gamma_dyn_dyn=gam_d,
        channel=ch,
    )
