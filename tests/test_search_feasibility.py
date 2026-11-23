"""Targeted scientific checks for the fixed CPU simulation, not method training."""
import json
import itertools
import math
import os
import subprocess
import sys
import numpy as np
import pytest
from scripts.search_feasibility_common import (
    ROOT, METHODS, boundary_search, dense_conflict, online_conflict, ranking, sub_seed, suite,
)


def test_exact_bisection_all_monotone_boundaries():
    for n in range(2, 33):
        for boundary in range(n+1):
            answer = tuple(i < boundary for i in range(n))
            b, positions, _ = boundary_search(answer, 'BISECT_EXACT')
            assert b == boundary
            assert len(positions) <= math.ceil(math.log2(n+1))
            assert len(positions) == len(set(positions))


@pytest.mark.parametrize('eps', ['0.1', '0.2', '0.3'])
@pytest.mark.parametrize('presence', [False, True])
def test_integer_pba_matches_independent_posterior(eps, presence):
    answer = (False, True, False, True, False, False, True)
    n = len(answer); p = np.ones(n+1, dtype=float)
    if presence: p[0] = n
    p /= p.sum(); asked = []
    while p.max() < .9 - 1e-12 and len(asked) < n:
        candidates = [i for i in range(1, n+1) if i not in asked]
        deviations = [abs(p[i:].sum() - .5) for i in candidates]
        i = next(i for i, d in zip(candidates, deviations) if np.isclose(d, min(deviations), atol=1e-13, rtol=0))
        asked.append(i); eps_float = float(eps)
        p *= np.where((np.arange(n+1) >= i) == answer[i-1], 1-eps_float, eps_float); p /= p.sum()
    b, trace, maxp = boundary_search(answer, 'PBA', eps, presence)
    assert trace == tuple(asked)
    assert b == next(i for i, x in enumerate(p) if np.isclose(x, p.max(), atol=1e-13, rtol=0))
    assert abs(maxp-p.max()) < 1e-12


def test_tie_and_empty_controls_and_random_budget():
    assert ranking([2., 2., 0., -1.], [5, 2, 3, 4]) == [1, 0, 2, 3]
    rows = suite([False, True, False, True], [2., 1., 0., -1.], sub_seed('list'))
    methods = {r['method']: r for r in rows}
    assert tuple(methods) == METHODS
    assert methods['DENSE']['selected_positions'] == [2, 4]
    assert methods['DENSE']['b_hat'] == 2
    assert methods['NATIVE']['m'] == 0 and methods['NATIVE']['selected_positions'] == [1, 2]
    assert methods['RANDOM_SAME_BUDGET']['m'] == methods['PBA_0.2']['m']
    assert rows == suite([False, True, False, True], [2., 1., 0., -1.], sub_seed('list'))
    for r in rows:
        assert len(r['asked_positions']) == len(set(r['asked_positions']))
        assert r['m'] <= 4 and r['b_hat'] == len(r['selected_positions'])


def test_offline_and_online_conflict_definitions():
    assert dense_conflict([True, True, False, False]) == 0
    assert dense_conflict([False, False, True, True]) == 1
    assert dense_conflict([True, False, False, True]) == .5
    assert dense_conflict([False, False]) is None
    assert online_conflict([1, 2, 3, 4], [False, False, True, True]) == .5
    assert online_conflict([4, 2, 3, 1], [True, False, False, True]) == .5
    assert online_conflict([2, 4, 1, 3], [True, False, True, False]) == 0


def test_uniform_pba_stop_threshold_forces_dense_queries_without_repeats():
    for n in range(2, 9):
        for answers in itertools.product((False, True), repeat=n):
            modes = []
            for eps in ('0.1', '0.2', '0.3'):
                mode, trace, confidence = boundary_search(answers, 'PBA', eps)
                assert len(trace) == n and confidence < .9
                modes.append(mode)
            assert len(set(modes)) == 1


def test_offline_metrics_do_not_confuse_full_gt_count_with_matched_candidates():
    from scripts.evaluate_search_feasibility import selection_metrics
    assert selection_metrics([], [False, False], 0)['F1'] == 1
    assert selection_metrics([0], [False, False], 0)['F1'] == 0
    assert selection_metrics([], [False, False], 2)['F1'] == 0
    out = selection_metrics([0], [True, False], 3)
    assert out['F1'] == 1 and out['count_MAE'] == 2 and out['presence_accuracy'] == 1


def test_search_failure_is_strictly_above_point_one_not_float_roundoff():
    from fractions import Fraction
    from scripts.evaluate_search_feasibility import exact_f1
    dense = exact_f1(dict(selected_count=12, TP=8), 8, 8)  # 0.8
    search = exact_f1(dict(selected_count=12, TP=7), 8, 8)  # 0.7
    assert dense-search == Fraction(1, 10)
    assert not (dense-search > Fraction(1, 10))


def test_video_weighted_rank_statistics_equal_literal_cluster_repetition():
    import pandas as pd
    from scipy.stats import spearmanr
    from scripts.summarize_search_feasibility import spearman_samples, auc_histogram, auc_scores
    d = pd.DataFrame(dict(video=['a','a','b','b','c'],cluster=['a','a','b','b','c'],
        C_dense=[0.,.5,.5,1.,.5],A_GT=[1.,.5,.7,0.,.5],C_online=[0.,.5,.5,1.,0.],search_failure=[0,1,0,1,0]))
    weights = np.asarray([[1,1,1],[2,0,1],[0,3,0]])
    correlations = spearman_samples(d,weights,['a','b','c'])
    aucs = auc_histogram(d,weights,['a','b','c'],sorted(d.C_online.unique()))
    for i,w in enumerate(weights):
        repeated = pd.concat([d[d.video==v] for v,n in zip(['a','b','c'],w) for _ in range(n)])
        rho = spearmanr(repeated.C_dense,repeated.A_GT).statistic
        assert np.isclose(correlations[i],rho,equal_nan=True)
        expected = auc_scores(repeated.C_online,repeated.search_failure)
        assert np.isclose(aucs[i],expected,equal_nan=True)


def test_gt_audit_and_runtime_only_kernel_whitelist(tmp_path):
    code = '''
import json,sys
from pathlib import Path
from scripts.decompose_common import guard,sandbox
attempts=guard()
try:Path('/tmp/search_fixture/gt.txt').read_text()
except PermissionError:pass
else:raise AssertionError('GT access must fail')
isolation=sandbox([], [sys.argv[1]])
try:Path(sys.argv[2]).read_text()
except PermissionError:pass
else:raise AssertionError('GT-derived old outputs are not runtime inputs')
print(json.dumps(dict(blocked=len(attempts),kernel=isolation['kernel_read_whitelist'])))
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='')
    p = subprocess.run([sys.executable, '-c', code, str(tmp_path),
                        str(ROOT / 'results/runs/decompose_20261102/probe/offline_frames.jsonl')],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout) == dict(blocked=1, kernel=True)
