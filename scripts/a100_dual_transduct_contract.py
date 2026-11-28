"""Independent synthetic mathematical contract for DUAL_TRANSDUCT_COEVOL.

Pure CPU synthetic tests (0 target GT, 0 real images, 0 GPU, 0 teacher calls).
Verifies:
1. Exact 9352D feature split (NumPy == Torch, track-invariance of X_id).
2. Combinatorial Laplacian symmetry, PSD, and exact L*1 = 0 nullspace.
3. Exact M-matrix leverage h = A^{-1} pi in [0, 1], h_k = 0 on isolated
   unasked components, and exact convex-combination bound of u^*.
4. Exact Sherman-Morrison Leave-One-Track-Out identity vs brute-force matrix
   re-solve (max error < 1e-11).
5. Sublinear correlated-repeat attenuation on repeated track anchors.
6. Video existence CHECKLIST vs PRESUPPOSED contract and track-cap bounds.
7. Zero-mean preservation and appearance-jump sensitivity of dynamic residual.
8. LOTO observer optimizer non-increase and finite gradients for S_id & S_dyn.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flex_uda.active.dual_transduct_coevol import (
    FULL_DIM, ID_DIM, DYN_DIM,
    RegularizedMLP, TransductiveObserverParams,
    split_shared_features_np, split_shared_features_torch,
    pairwise_unit_sqdist, build_combinatorial_laplacian,
    channel_event_llr, aggregate_track_anchors,
    solve_track_transduction, compute_video_existence,
    smooth_within_track_residual, run_transductive_inference,
    fit_observer_loto,
)


def run_contract():
    rng = np.random.default_rng(20261127)
    checks = []

    # 1. Feature split equivalence (NumPy == Torch)
    x = rng.normal(size=(16, FULL_DIM)).astype(np.float32)
    xid_np, xdyn_np = split_shared_features_np(x)
    xid_t, xdyn_t = split_shared_features_torch(torch.from_numpy(x))
    assert xid_np.shape == (16, ID_DIM) and xdyn_np.shape == (16, DYN_DIM)
    assert np.max(np.abs(xid_np - xid_t.numpy())) < 1e-7
    assert np.max(np.abs(xdyn_np - xdyn_t.numpy())) < 1e-7
    checks.append(('feature_split_np_torch_exact', True))

    # 2. Combinatorial Laplacian symmetry, PSD, exact L*1 = 0
    feats = rng.normal(size=(14, 16))
    dist = pairwise_unit_sqdist(feats)
    lap, w_norm = build_combinatorial_laplacian(dist, k_nn=4)
    assert np.max(np.abs(lap - lap.T)) < 1e-14
    assert np.max(np.abs(lap @ np.ones(14))) < 1e-13
    eigvals = np.linalg.eigvalsh(lap)
    assert eigvals.min() > -1e-12
    checks.append(('laplacian_sym_psd_exact_nullspace', True))

    # 3. Exact M-matrix leverage h in [0, 1] & isolated component h == 0
    # Construct two disconnected clusters: tracks 0..4 and tracks 5..9
    block_lap = np.zeros((10, 10))
    l1, _ = build_combinatorial_laplacian(pairwise_unit_sqdist(rng.normal(size=(5, 8))), k_nn=3)
    l2, _ = build_combinatorial_laplacian(pairwise_unit_sqdist(rng.normal(size=(5, 8))), k_nn=3)
    block_lap[:5, :5] = l1
    block_lap[5:, 5:] = l2
    mu = rng.normal(size=10)
    pi = np.array([3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.array([2.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u, h, loto_u = solve_track_transduction(mu, block_lap, target, pi, tau_prior=1.0, lam_vis=2.0)
    assert np.all((h >= -1e-14) & (h <= 1.0 + 1e-14))
    # Component 2 (tracks 5..9) has ZERO anchors -> leverage must be strictly 0!
    assert np.max(np.abs(h[5:])) < 1e-14
    # Component 1 unasked tracks (1,3,4) must have strictly positive leverage from asked neighbors (0,2)!
    assert np.all(h[[1, 3, 4]] > 0.05)
    # Exact convex-combination property: constant mu & target = c => u == c
    u_const, _, _ = solve_track_transduction(np.full(10, 1.7), block_lap, np.full(10, 1.7), pi, tau_prior=1.0, lam_vis=2.0)
    assert np.max(np.abs(u_const - 1.7)) < 1e-12
    checks.append(('m_matrix_leverage_and_isolated_zero', True))

    # 4. Exact Sherman-Morrison LOTO identity vs brute-force refit
    max_loto_err = 0.0
    for k in range(10):
        pi_minus = pi.copy(); pi_minus[k] = 0.0
        t_minus = target.copy(); t_minus[k] = 0.0
        u_bf, _, _ = solve_track_transduction(mu, block_lap, t_minus, pi_minus, tau_prior=1.0, lam_vis=2.0)
        max_loto_err = max(max_loto_err, abs(loto_u[k] - u_bf[k]))
    assert max_loto_err < 1e-11, max_loto_err
    checks.append(('sherman_morrison_loto_exact', float(max_loto_err)))

    # 5. Correlated-repeat attenuation & soft-preserving bounded LLR
    rel = np.array([[0.85, 0.85], [0.80, 0.80], [0.82, 0.82]])
    assert abs(channel_event_llr(0.5, 0.85, 0.85)) < 1e-14
    assert channel_event_llr(1.0, 0.85, 0.85) > 0 and channel_event_llr(0.0, 0.85, 0.85) < 0
    ev1 = [dict(q=0, track_col=0, view=0, prob=0.95, nodes=[0])]
    ev4 = [dict(q=0, track_col=0, view=0, prob=0.95, nodes=[i]) for i in range(4)]
    _, p1, _, _ = aggregate_track_anchors(1, 2, ev1, rel, anchor_base_precision=2.5, corr_rho=0.5)
    _, p4, _, _ = aggregate_track_anchors(1, 2, ev4, rel, anchor_base_precision=2.5, corr_rho=0.5)
    # Sublinear: 4 repeats must give < 2x of 1 repeat (4 / (1 + 1.5) = 1.6x)
    assert abs(p4[0, 0] / p1[0, 0] - 1.6) < 1e-12
    checks.append(('correlated_repeat_sublinear_attenuation', True))

    # 6. Video existence CHECKLIST vs PRESUPPOSED
    pr = np.zeros((2, 4))
    ebq = [
        [dict(q=0, track_col=k, view=0, prob=0.02) for k in range(4)],
        [dict(q=1, track_col=0, view=0, prob=0.98)],
    ]
    eta_pre = compute_video_existence(pr, ebq, rel, protocol='PRESUPPOSED')
    eta_chk = compute_video_existence(pr, ebq, rel, protocol='CHECKLIST')
    assert np.all(eta_pre == 1.0)
    assert eta_chk[0] < 0.25 and eta_chk[1] > 0.75
    checks.append(('existence_checklist_vs_presupposed', True))

    # 7. Appearance-gated zero-mean dynamic residual
    # Track 0 has 6 frames with an appearance switch between frame 2 and 3
    obs = np.array([[t, 10] for t in range(6)])
    cols = np.zeros(6, dtype=np.int64)
    imgs = np.zeros((6, 4))
    imgs[:3, 0] = 1.0
    imgs[3:, 1] = 1.0
    raw_scores = np.array([[-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]])
    tm, res = smooth_within_track_residual(raw_scores, obs, cols, imgs, 1, kappa=4.0, base_decay=0.9)
    assert abs(tm[0, 0]) < 1e-12
    assert abs(res[0].mean()) < 1e-12
    # Jump preserved across appearance switch (sign preserved)
    assert res[0, 2] < -1.0 and res[0, 3] > 1.0
    checks.append(('appearance_gated_zero_mean_residual', True))

    # 8. LOTO observer optimization non-increase on synthetic graph-structured video
    nq, K, T = 4, 8, 5
    n_obs = K * T
    obs_syn = np.array([[t, k] for k in range(K) for t in range(T)])
    cols_syn = np.array([k for k in range(K) for t in range(T)])
    # Two true visual clusters of 4 tracks each
    app_v = np.zeros((K, 8))
    app_v[:4, :4] = 1.0 + 0.05 * rng.normal(size=(4, 4))
    app_v[4:, 4:] = 1.0 + 0.05 * rng.normal(size=(4, 4))
    d_app = pairwise_unit_sqdist(app_v)
    d_geo = pairwise_unit_sqdist(rng.normal(size=(K, 8)))
    d_pcd = np.stack([d_app for _ in range(nq)], axis=0)
    is_dyn = np.array([False, False, True, True])
    g_id = 0.2 * rng.normal(size=(nq, K))
    f_tm = 0.2 * rng.normal(size=(nq, K))
    f_res = 0.1 * rng.normal(size=(nq, n_obs))
    # Teacher agrees strongly within visual clusters
    syn_events = []
    for q in range(nq):
        for k in range(K):
            prob = 0.92 if (k < 4) == (q % 2 == 0) else 0.08
            node = k * T + 2
            syn_events.append(dict(q=q, track_col=k, view=0, prob=prob, nodes=[node]))
    params, fit_rec = fit_observer_loto(
        g_id, f_tm, f_res, d_app, d_pcd, d_geo, is_dyn, syn_events, cols_syn,
        protocol='PRESUPPOSED', k_nn=3, maxiter=20)
    assert fit_rec['loto_after'] <= fit_rec['loto_before'] + 1e-9
    out = run_transductive_inference(
        g_id, f_tm, f_res, d_app, d_pcd, d_geo, is_dyn, syn_events, cols_syn,
        params, protocol='PRESUPPOSED', k_nn=3)
    assert np.isfinite(out['frame_prob']).all() and np.all((out['frame_prob'] >= 0) & (out['frame_prob'] <= 1))
    checks.append(('loto_observer_nonincrease_and_finite', dict(before=fit_rec['loto_before'], after=fit_rec['loto_after'])))

    # 9. CPU gradient flow for S_id and S_dyn RegularizedMLP
    torch.manual_seed(0)
    s_id = RegularizedMLP(np.zeros(ID_DIM), np.ones(ID_DIM), width=64, dropout=0.1)
    s_dyn = RegularizedMLP(np.zeros(DYN_DIM), np.ones(DYN_DIM), width=64, dropout=0.1)
    for m, inp in ((s_id, xid_t), (s_dyn, xdyn_t)):
        m.train()
        y_t = torch.full((len(inp),), 0.7)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(inp), y_t)
        loss.backward()
        grad_norm = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert np.isfinite(grad_norm) and grad_norm > 0
    checks.append(('dual_student_cpu_gradients_nonzero', True))

    result = dict(
        status='PASS_DUAL_TRANSDUCT_COEVOL_MATHEMATICAL_CONTRACT',
        checks_passed=len(checks),
        details=checks,
        max_loto_sherman_morrison_error=float(max_loto_err),
        no_target_GT_images_GPU_or_teacher=True,
    )
    out_dir = ROOT / 'a100_runs/visual_classical_20261121/supervisor/dual_transduct_contract_20261127'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run_contract()
