import numpy as np

from scripts.diag_attrib_v2_common import causal_moments, rank_within_groups, round_budget, top_budget
from scripts.diag_pf_countmatched import topk_ids


def test_vector_frame_ranking_matches_existing_count_control_ties():
    ids = [2, 7, 12, 20]
    values = np.asarray([.2, .8, .8, -.2])
    ranks = rank_within_groups(values, np.zeros(4, dtype=int), 1)
    for k in range(5):
        actual = {i for i, rank in zip(ids, ranks) if rank < k}
        assert actual == topk_ids(ids, dict(zip(ids, values)), k)


def test_historical_restriction_recomputes_rank_without_missing_candidates():
    scores = np.asarray([1., 5., 3., 4., 9.])
    groups = np.asarray([0, 0, 0, 2, 2])
    historical = np.asarray([True, False, True, True, False])
    ranks = rank_within_groups(scores, groups, 3, historical)
    assert np.flatnonzero(historical & (ranks < 1)).tolist() == [2, 3]


def test_global_quantile_tie_is_canonical_and_exact_budget():
    chosen, audit = top_budget([2., 2., 2., -1.], 2)
    assert chosen.tolist() == [True, True, False, False]
    assert audit['boundary_ties'] == 3
    assert audit['selected_boundary_ties'] == 2


def test_causal_window_excludes_current_future_and_older_than_100_frames():
    frames = np.asarray([1, 1, 2, 101, 102])
    scores = np.asarray([1., 3., 100., 9., 20.])
    unique, first, counts, mu, sigma = causal_moments(frames, scores)
    assert counts.tolist() == [0, 2, 3, 2]
    assert mu[1] == 2 and sigma[1] == 1
    assert np.isclose(mu[2], 104/3)
    assert mu[3] == 54.5
    changed = scores.copy(); changed[2:] = [-1000., 999., 888.]
    assert causal_moments(frames, changed)[3][1] == mu[1]


def test_round_half_up_not_bankers_and_clamped():
    assert round_budget(.25, 2) == 1
    assert round_budget(2, 5) == 5
    assert round_budget(-1, 5) == 0


def test_B0_absolute_scales_are_not_clamped_to_one_GT_count():
    from scripts.diag_attrib_v2_common import scale_budget
    assert [scale_budget(a,212541,3588445) for a in (1.25,1.5,2.,3.)] == [265676,318812,425082,637623]
    assert scale_budget(3,212541,500000) == 500000


def verdict_fixture():
    from scripts.diag_attrib_v2_common import BASELINES
    comparisons = {'D1_minus_'+m: {'CI95':[1.,2.]} for m in BASELINES}
    for k in ('OFFICIAL_DANCE_COMPLETE','D1','ORACLE_GT'):
        comparisons['D3_sort_minus_FROZEN_K_'+k] = {'CI95':[-1.,1.]}
    return comparisons


def test_rule_A_contains_zero_is_registered_label_not_equivalence_claim():
    from scripts.report_diag_attrib_v2 import classify
    record = verdict_fixture()
    record['D1_minus_B3_MEAN_GLOBAL']['CI95'] = [-.1,.2]
    assert classify(record, {'D1':345302})['A'] == 'D1 的增益可被无学习校准解释'


def test_rule_B_requires_both_registered_nonoracle_columns():
    from scripts.report_diag_attrib_v2 import classify
    record = verdict_fixture()
    for k in ('OFFICIAL_DANCE_COMPLETE','D1'):
        record['D3_sort_minus_FROZEN_K_'+k]['CI95'] = [.1,1.]
    assert classify(record, {'D1':345302})['B'] == 'D3 的排序增益在合理数量下成立'


def test_rule_B_mixed_case_is_not_silently_called_oracle_only():
    from scripts.report_diag_attrib_v2 import classify
    record = verdict_fixture()
    for k in ('OFFICIAL_DANCE_COMPLETE','ORACLE_GT'):
        record['D3_sort_minus_FROZEN_K_'+k]['CI95'] = [.1,1.]
    assert classify(record, {'D1':345302})['B'].startswith('预注册规则未覆盖')
