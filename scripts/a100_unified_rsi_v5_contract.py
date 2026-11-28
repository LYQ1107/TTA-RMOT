"""Synthetic CPU-only mathematical contract for UNIFIED_RSI_RMOT_V5 (0 GT, 0 teacher calls)."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.active.unified_rsi_v4_runtime import fit_two_stream_convex_beta
from flex_uda.active.unified_rsi_v5_runtime import fit_two_stream_proper_convex_beta

OUT = ROOT / 'a100_runs/visual_classical_20261121/supervisor/unified_rsi_v5_contract_20261128'


def run_contract():
    # Prove strictly proper scoring rule property (Gneiting & Raftery 2007):
    # Under class imbalance (pos_rate = 0.20), suppose S_ctx is EXACTLY calibrated (p_ctx = true_p)
    # while S_id is upward-biased (p_id = min(true_p + 0.25, 0.95)).
    # - Strictly proper CE (V5) MUST pick beta = 1.0 (100% weight on calibrated S_ctx).
    # - Class-reweighted CE (V4) mistakenly gives weight to upward-biased S_id (beta < 0.95)!
    rng = np.random.default_rng(123)
    nq, n_tracks = 40, 20
    is_dyn = np.ones(nq, dtype=bool)
    true_p = np.where(rng.random((nq, n_tracks)) < 0.20, 0.70, 0.08)
    y_exp = true_p  # exact expected probability targets
    p_ctx = true_p.copy()
    p_id = np.clip(true_p + 0.25, 0.01, 0.95)
    events = [dict(q=q, track_col=k, view=0, prob=float(y_exp[q, k]))
              for q in range(nq) for k in range(n_tracks)]

    _, beta_d_proper, d_prop = fit_two_stream_proper_convex_beta(p_id, p_ctx, is_dyn, events)
    _, beta_d_reweighted, _ = fit_two_stream_convex_beta(p_id, p_ctx, is_dyn, events)
    assert beta_d_proper > 0.999, f'Strictly proper CE must select exact calibrated stream, got {beta_d_proper}'
    assert beta_d_reweighted < 0.85, f'Expected reweighted CE to suffer upward-bias distortion, got {beta_d_reweighted}'

    # Also verify complementary static mixing still selects interior optimum
    is_dyn_mix = np.array([False] * 10 + [True] * 10, dtype=bool)
    p_id2 = np.full((20, 12), 0.5)
    p_ctx2 = np.full((20, 12), 0.5)
    tr2 = rng.binomial(1, 0.35, size=(20, 12)).astype(float)
    for q in range(10):
        for k in range(12):
            if k % 2 == 0:
                p_id2[q, k] = 0.85 if tr2[q, k] > 0.5 else 0.15
            else:
                p_ctx2[q, k] = 0.85 if tr2[q, k] > 0.5 else 0.15
    for q in range(10, 20):
        for k in range(12):
            p_id2[q, k] = 0.60
            p_ctx2[q, k] = 0.85 if tr2[q, k] > 0.5 else 0.15
    ev2 = [dict(q=q, track_col=k, view=0, prob=float(tr2[q, k])) for q in range(20) for k in range(12)]
    bs, bd, d2 = fit_two_stream_proper_convex_beta(p_id2, p_ctx2, is_dyn_mix, ev2)
    assert 0.25 < bs < 0.75 and bd > 0.99
    assert d2['static_nll_mix'] < min(d2['static_nll_id'], d2['static_nll_ctx']) - 0.015

    result = dict(
        status='PASS_UNIFIED_RSI_RMOT_V5_MATHEMATICAL_CONTRACT',
        checks_passed=3,
        proper_beta_on_exact_calibrated=beta_d_proper,
        reweighted_beta_distortion_proven=beta_d_reweighted,
        complementary_static_beta=bs,
        blind_dynamic_beta=bd,
        GT_open_attempts=0,
        new_teacher_calls=0,
        synthetic_CPU_only=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run_contract()
