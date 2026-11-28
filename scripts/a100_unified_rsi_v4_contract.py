"""Synthetic CPU-only mathematical contract for UNIFIED_RSI_RMOT_V4 (0 GT, 0 teacher calls)."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.unified_rsi_v4_runtime import (
    compute_balanced_trust_region_weights,
    fit_two_stream_convex_beta,
)

OUT = ROOT / 'a100_runs/visual_classical_20261121/supervisor/unified_rsi_v4_contract_20261128'


def run_contract():
    rng = np.random.default_rng(42)

    # 1. Strictly convex Ventral/Dorsal two-stream OOF gating:
    #    Static queries: S_id and S_ctx carry complementary signals -> interior beta_s in (0.15, 0.85)
    #    Dynamic queries: S_id is motion-blind noise around 0.5, S_ctx is accurate -> beta_d -> 1.0
    nq, n_tracks = 20, 12
    is_dyn = np.array([False] * 10 + [True] * 10, dtype=bool)
    truth = rng.binomial(1, 0.4, size=(nq, n_tracks)).astype(np.float64)
    p_id = np.zeros((nq, n_tracks))
    p_ctx = np.zeros((nq, n_tracks))
    # Static: S_id good on even tracks, S_ctx good on odd tracks
    for q in range(10):
        for k in range(n_tracks):
            if k % 2 == 0:
                p_id[q, k] = 0.85 if truth[q, k] > 0.5 else 0.15
                p_ctx[q, k] = 0.52 if truth[q, k] > 0.5 else 0.48
            else:
                p_id[q, k] = 0.52 if truth[q, k] > 0.5 else 0.48
                p_ctx[q, k] = 0.85 if truth[q, k] > 0.5 else 0.15
    # Dynamic: S_id blind (0.55 always), S_ctx accurate
    for q in range(10, 20):
        for k in range(n_tracks):
            p_id[q, k] = 0.55
            p_ctx[q, k] = 0.88 if truth[q, k] > 0.5 else 0.12

    events = []
    for q in range(nq):
        for k in range(n_tracks):
            events.append(dict(q=q, track_col=k, view=0, prob=float(truth[q, k])))

    beta_s, beta_d, diag = fit_two_stream_convex_beta(p_id, p_ctx, is_dyn, events)
    assert 0.25 < beta_s < 0.75, f'Expected interior complementary static beta_s, got {beta_s}'
    assert beta_d > 0.99, f'Expected dorsal beta_d -> 1.0 when S_id blind, got {beta_d}'
    assert diag['static_nll_mix'] < min(diag['static_nll_id'], diag['static_nll_ctx']) - 0.015
    assert diag['dyn_nll_mix'] <= min(diag['dyn_nll_id'], diag['dyn_nll_ctx']) + 1e-12

    # 2. Sign-balanced mean-neutral trust-region weights prevent upward drift on PRESUPPOSED
    lev = np.full((6, 8), 0.5)
    asked = np.zeros((6, 8), dtype=bool)
    asked[:, :2] = True
    p_pr = np.full((6, 8), 0.25)
    p_tr = p_pr.copy()
    # Strong asymmetric upward push (5 tracks up +0.25, 1 track down -0.05)
    p_tr[:, 2:7] = 0.50
    p_tr[:, 7] = 0.20
    q_budget = np.full(6, 4.0)
    shrink_q = np.full(6, 0.5)
    sq, sk, w_tr, tr_diag = compute_balanced_trust_region_weights(
        lev, p_tr, p_pr, asked, q_budget, shrink_q,
        lev_thresh=0.15, trans_budget_w=0.18, protocol='PRESUPPOSED')
    assert len(sq) == 36, f'Expected all 36 unasked tracks selected, got {len(sq)}'
    assert abs(w_tr.sum() - 0.18 * 0.5) < 1e-12, f'EB shrinkage scaling failed: {w_tr.sum()}'
    assert tr_diag['up_damp'] < 0.35, f'Expected strong upward-drift damping, got {tr_diag["up_damp"]}'

    # 3. Zero shrink -> exact zero unasked tracks
    sq0, sk0, w0, d0 = compute_balanced_trust_region_weights(
        lev, p_tr, p_pr, asked, q_budget, np.zeros(6),
        lev_thresh=0.15, trans_budget_w=0.18, protocol='PRESUPPOSED')
    assert len(sq0) == 0 and d0['eff_budget'] == 0.0

    # 4. CHECKLIST does NOT artificially damp downward/upward beyond EB shrink
    sq_c, sk_c, w_c, d_c = compute_balanced_trust_region_weights(
        lev, p_tr, p_pr, asked, q_budget, np.full(6, 0.8),
        lev_thresh=0.15, trans_budget_w=0.18, protocol='CHECKLIST')
    assert d_c['up_damp'] == 1.0 and abs(w_c.sum() - 0.18 * 0.8) < 1e-12

    result = dict(
        status='PASS_UNIFIED_RSI_RMOT_V4_MATHEMATICAL_CONTRACT',
        checks_passed=4,
        two_stream_beta_static=beta_s,
        two_stream_beta_dyn=beta_d,
        static_nll_gain=min(diag['static_nll_id'], diag['static_nll_ctx']) - diag['static_nll_mix'],
        presupposed_up_damp=tr_diag['up_damp'],
        eb_zero_shrink_exact=True,
        GT_open_attempts=0,
        new_teacher_calls=0,
        synthetic_CPU_only=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run_contract()
