"""Targeted scientific checks for the fixed CPU simulation, not method training."""
import json
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
