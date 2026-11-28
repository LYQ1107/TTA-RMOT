"""Unified 3-Innovation RSI-RMOT Core (I1 + I2 + I3):

I1: Class-Balanced Bounded Probability-Space Harmonic Transduction over
    OOF-LOTO Gated Visual-Spatial Supportive Cliques + Evidence-Gated Zero-Jitter Coherence.
I2: Complementary Slow-Identity (S_id) / Fast-Context (S_ctx) Dual-Student
    Externally-Anchored Co-Evolution with Correction-Only Leverage Filtering.
I3: Fast-Slow Active Challenger-Observer uniting legacy v8p GLANCE / PRESENCE_COVER /
    keyframe selection with Dual-Student Disagreement and Exact Rank-1 Sherman-Morrison
    External Leverage Gain (submodular-like visual clique covering).
"""
import math
import numpy as np
from scipy.special import expit, logit

from flex_uda.active.dual_transduct_v3 import (
    DIST_GATE_GRID, TransductiveObserverV3Params,
    aggregate_track_evidence_v3, batched_solve_prob_harmonic,
    default_v3_vector, solve_prob_harmonic_transduction, unpack_v3_vector,
)


def binary_entropy(p):
    pc = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return -(pc * np.log(pc) + (1.0 - pc) * np.log1p(-pc))


def sherman_morrison_leverage_gain(M, leverage, pi_add=2.0):
    """Exact total external leverage gain from adding one anchor pi_add at each track k.

    Given A = tau I + lam L + diag(pi), M = A^{-1}, h = M pi in [0, 1]^K:
    Adding pi_add at track k yields A' = A + pi_add e_k e_k^T.
    By Sherman-Morrison, exact new leverage is:
        h' = h + (pi_add * (1 - h_k) / (1 + pi_add * M_kk)) * M[:, k]
    so total leverage gain across all K tracks is:
        DeltaLev(k) = sum_j (h'_j - h_j) = pi_add * (1 - h_k) * sum_j M_{j,k} / (1 + pi_add * M_kk) >= 0.
    Works for both 2D (K, K) and batched 3D (nq, K, K) inverse matrices M.
    """
    M = np.asarray(M, dtype=np.float64)
    h = np.asarray(leverage, dtype=np.float64)
    pi0 = float(pi_add)
    if M.ndim == 2:
        col_sum = M.sum(axis=0)
        m_diag = np.diag(M)
    elif M.ndim == 3:
        col_sum = M.sum(axis=1)
        idx = np.arange(M.shape[1])
        m_diag = M[:, idx, idx]
    else:
        raise ValueError(f'Expected 2D or 3D M, got {M.shape}')
    gain = pi0 * np.maximum(1.0 - h, 0.0) * col_sum / np.maximum(1.0 + pi0 * m_diag, 1e-12)
    return np.maximum(gain, 0.0)


def sherman_morrison_rank1_anchor_update(M, leverage, k, pi_add=2.0):
    """Exact O(K^2) rank-1 update of M = A^{-1} and h = M pi after anchoring track k."""
    pi0 = float(pi_add)
    m_k = M[:, k].copy()
    denom = max(1.0 + pi0 * float(m_k[k]), 1e-12)
    h_new = np.clip(leverage + (pi0 * max(1.0 - float(leverage[k]), 0.0) / denom) * m_k, 0.0, 1.0)
    M_new = M - (pi0 / denom) * np.outer(m_k, m_k)
    return M_new, h_new


def batched_prob_harmonic_with_inverse(prior_prob, laps, track_llr, precision, lam_per_query, *,
                                       tau_prior=1.0, gamma_anc=1.0):
    """Exact probability-space harmonic solve returning (p_star, leverage, M_inv)."""
    mu_p = np.clip(np.asarray(prior_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    nq, k = mu_p.shape
    if k == 0:
        return mu_p.copy(), np.zeros((nq, 0)), np.zeros((nq, 0, 0))
    tau = max(float(tau_prior), 1e-6)
    lam = np.maximum(np.asarray(lam_per_query, dtype=np.float64), 0.0)
    pi = np.asarray(precision, dtype=np.float64)
    llr = np.asarray(track_llr, dtype=np.float64)
    y_bayes = np.where(pi > 0.0, expit(np.clip(logit(mu_p) + float(gamma_anc) * llr, -20.0, 20.0)), 0.0)

    eye = np.eye(k, dtype=np.float64)[None, :, :]
    A = tau * eye + lam[:, None, None] * laps
    diag_idx = np.arange(k)
    A[:, diag_idx, diag_idx] += pi
    M = np.linalg.inv(A)
    rhs = tau * mu_p + pi * y_bayes
    p_star = np.clip(np.einsum('qij,qj->qi', M, rhs), 1e-7, 1.0 - 1e-7)
    leverage = np.clip(np.einsum('qij,qj->qi', M, pi), 0.0, 1.0)
    return p_star, leverage, M


def pick_best_keyframe(track_rows, obs_frames, obs_geom, asked_uids_for_q, asked_frames_for_qk, q_idx, n_obs):
    """Legacy v8p-inspired keyframe selector: temporal diversity + centered + largest area.

    obs_geom[:, :4] contains normalized [cx, cy, w, h] in [0, 1] (or [x, y, w, h]).
    Strictly excludes any already-asked uid for query q_idx.
    """
    avail = [int(oi) for oi in track_rows if (q_idx * n_obs + int(oi)) not in asked_uids_for_q]
    if not avail:
        return None
    best_oi = None
    best_key = None
    for oi in avail:
        f = int(obs_frames[oi])
        dist_t = min((abs(f - af) for af in asked_frames_for_qk), default=1e9)
        g = obs_geom[oi]
        cx, cy, w, h = float(g[0]), float(g[1]), abs(float(g[2])), abs(float(g[3]))
        margin = min(cx - 0.5 * w, 1.0 - (cx + 0.5 * w), cy - 0.5 * h, 1.0 - (cy + 0.5 * h))
        centered = 1 if margin >= 0.02 else 0
        area = w * h
        key = (-dist_t, -centered, -area, f, oi)
        if best_key is None or key < best_key:
            best_key = key
            best_oi = oi
    return best_oi


def stage_budgets(quotas, stage_fracs=(0.30, 0.25, 0.25, 0.20)):
    """Split each query's exact integer budget across 4 front-loaded stages summing exactly to quota."""
    n_stages = len(stage_fracs)
    plan = {q: [0] * n_stages for q in quotas}
    for q, tot in quotas.items():
        tot = int(tot)
        assert tot > 0
        raw = [tot * f for f in stage_fracs]
        counts = [int(math.floor(x)) for x in raw]
        # Ensure stage 0 gets at least 1 if tot >= 1
        if counts[0] == 0:
            counts[0] = 1
        rem = tot - sum(counts)
        if rem > 0:
            order = sorted(range(n_stages), key=lambda i: (-(raw[i] - counts[i]), i))
            for i in range(rem):
                counts[order[i % n_stages]] += 1
        elif rem < 0:
            for i in range(n_stages - 1, -1, -1):
                take = min(counts[i], -rem)
                counts[i] -= take
                rem += take
                if rem == 0:
                    break
        assert sum(counts) == tot and all(c >= 0 for c in counts)
        plan[q] = counts
    return plan


def select_challenger_stage_uids(
    stage_idx, stage_alloc, *,
    p_id, p_ctx_tm, p_star, M_inv, leverage, asked_counts,
    asked_uids_by_q, asked_frames_by_qk,
    track_rows_cache, obs_frames, obs_geom, n_obs,
    pi_add=2.0, explore_frac=0.15,
):
    """Select exact per-query unique uids for one stage via sequential Sherman-Morrison Challenger.

    - Stage 0 (GLANCE):
      Slot 0 verifies the top source-prior candidate (PRESENCE_COVER); remaining slots
      greedily cover visual cliques via sequential rank-1 Sherman-Morrison DeltaLev
      weighted by initial positive/boundary interest.
    - Stages 1..3 (RSI Challenger <-> Dual-Student Solver):
      Combines transduced boundary entropy H(p*), dual-student disagreement |p_id - p_ctx|,
      unverified positive discovery 0.35 * p* * 1[c==0], exact sequential rank-1 clique
      leverage gain DeltaLev, and distinct-track factor 1/(1+c), with a 15% pure-coverage
      exploration floor against blind spots.
    Returns:
      selected_uids: list[str] of unique flat uids
      journal_rows: list[dict] saved BEFORE calling teacher
    """
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
        # Track remaining unasked observations per track
        avail_count = np.array([
            sum(1 for oi in track_rows_cache[k] if (q * n_obs + int(oi)) not in asked_q)
            for k in range(n_tracks)
        ], dtype=np.int64)
        assert int(avail_count.sum()) >= need, (q, int(avail_count.sum()), need)

        n_explore = int(round(need * explore_frac)) if (stage_idx > 0 and need >= 4) else 0
        n_main = need - n_explore

        for slot in range(need):
            valid = avail_count > 0
            # Prefer unasked tracks first if any exist (distinct-object coverage)
            unasked_valid = valid & (c_q == 0)
            lev_gain = sherman_morrison_leverage_gain(M_q, h_q, pi_add=pi_add)
            max_lg = float(lev_gain[valid].max()) if valid.any() else 1.0
            norm_lg = lev_gain / max(max_lg, 1e-8)

            if stage_idx == 0 and slot == 0:
                # PRESENCE_COVER: immediately probe highest-prior positive candidate!
                reason = 'STAGE0_PRESENCE_COVER_TOP_PRIOR'
                pool = unasked_valid if unasked_valid.any() else valid
                scores = np.where(pool, p_star[q] + 0.05 * norm_lg, -1e18)
            elif stage_idx == 0:
                reason = 'STAGE0_GLANCE_CLIQUE_LEVERAGE'
                pool = unasked_valid if unasked_valid.any() else valid
                interest = 0.35 + p_star[q] * (1.0 - p_star[q]) + 0.50 * p_star[q]
                scores = np.where(pool, (0.25 + norm_lg) * interest / (1.0 + c_q), -1e18)
            elif slot >= n_main:
                reason = 'CHALLENGER_EXPLORE_CLIQUE_COVERAGE'
                pool = unasked_valid if unasked_valid.any() else valid
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
            # Exact rank-1 Sherman-Morrison update so the next slot jumps to an uncovered clique!
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
