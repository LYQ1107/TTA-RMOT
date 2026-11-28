"""Synthetic mathematical contract for DUAL_TRANSDUCT_COEVOL_V2 (0 GT, 0 GPU)."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flex_uda.active.dual_transduct_coevol import solve_track_transduction
from flex_uda.active.dual_transduct_v2 import (
    build_gated_clique_laplacian, default_v2_vector,
    prob_pooled_track_and_residual, unpack_v2_vector,
)
from flex_uda.active.dual_transduct_v2_runtime import infer_transductive_v2


def run():
    rng = np.random.default_rng(20261127)

    # 1. Similarity-gated clique Laplacian: exact L*1=0, PSD, AND dissimilar outlier strictly isolated!
    dist = np.full((6, 6), 5.0)
    np.fill_diagonal(dist, 0.0)
    # Tracks 0,1,2 form a tight clique (dist=0.3); track 3 is an isolated outlier (dist=5.0); 4,5 form a pair (dist=0.4)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            if i != j:
                dist[i, j] = 0.3
    dist[4, 5] = dist[5, 4] = 0.4
    lap, w_norm = build_gated_clique_laplacian(dist, k_nn=3, dist_gate=1.25)
    assert np.max(np.abs(lap - lap.T)) < 1e-14
    assert np.max(np.abs(lap @ np.ones(6))) < 1e-14
    assert np.linalg.eigvalsh(lap).min() > -1e-12
    # Outlier track 3 MUST have zero degree -> ZERO over-smoothing from negative tracks!
    assert np.max(np.abs(w_norm[3])) == 0.0 and lap[3, 3] == 0.0
    # Verify outlier keeps its exact prior + anchor without negative clique contamination
    mu = np.array([-2.0, -2.0, -2.0, 2.0, -2.0, -2.0])
    pi = np.array([3.0, 3.0, 0.0, 0.0, 3.0, 0.0])
    target = np.array([-3.0, -3.0, 0.0, 0.0, -3.0, 0.0])
    u, h, loto_u = solve_track_transduction(mu, lap, target, pi, tau_prior=1.0, lam_vis=3.0)
    assert abs(u[3] - 2.0) < 1e-13 and abs(h[3]) < 1e-14
    # Exact Sherman-Morrison LOTO error < 1e-12
    max_err = 0.0
    for k in range(6):
        pm = pi.copy(); pm[k] = 0.0
        tm = target.copy(); tm[k] = 0.0
        ubf, _, _ = solve_track_transduction(mu, lap, tm, pm, tau_prior=1.0, lam_vis=3.0)
        max_err = max(max_err, abs(loto_u[k] - ubf[k]))
    assert max_err < 1e-12

    # 2. Exact probability-space pooling + zero-jitter static tracks
    obs = np.array([[t, 0] for t in range(8)] + [[t, 1] for t in range(8)])
    cols = np.array([0] * 8 + [1] * 8)
    imgs = np.ones((16, 4))
    raw = rng.normal(size=(2, 16))
    tm, res = prob_pooled_track_and_residual(raw, obs, cols, imgs, 2, kappa=6.0, base_decay=0.94)
    # Track 0 pooled probability matches exact mean(expit(raw))
    expected_t0 = logit(np.clip(expit(raw[0, :8]).mean(), 1e-4, 1 - 1e-4))
    assert abs(tm[0, 0] - expected_t0) < 1e-13
    # Zero-jitter static query (q=0 static, q=1 dynamic)
    params = unpack_v2_vector(default_v2_vector())
    assert params.gamma_dyn_static == 0.0 and 0.0 <= params.gamma_dyn_dyn <= 0.35
    laps = np.stack([np.zeros((2, 2)), np.zeros((2, 2))], axis=0)
    out = infer_transductive_v2(
        tm, tm, res, laps, np.array([False, True]), [], cols, params, protocol='PRESUPPOSED')
    # Static query q=0 MUST be strictly constant within track 0 (zero jitter!)
    assert np.max(np.abs(out['frame_logits'][0, :8] - out['frame_logits'][0, 0])) == 0.0
    # Dynamic query q=1 varies across frames
    assert np.std(out['frame_logits'][1, :8]) > 1e-4

    result = dict(
        status='PASS_DUAL_TRANSDUCT_V2_MATHEMATICAL_CONTRACT',
        checks_passed=6,
        max_loto_sherman_morrison_error=float(max_err),
        outlier_isolation_exact=True,
        static_zero_jitter_exact=True,
        no_target_GT_images_GPU_or_teacher=True,
    )
    dest = ROOT / 'a100_runs/visual_classical_20261121/supervisor/dual_transduct_v2_contract_20261127'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
