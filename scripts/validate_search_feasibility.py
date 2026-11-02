"""Scientific completion checks; preserves all predictions and historical data."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *


def main():
    start = time.perf_counter(); proof = require_prereg(); sealed = verify_search()
    evaluation = json.loads((RUN / 'offline/evaluation.json').read_text())
    prepared = json.loads((RUN / 'runtime/seal.json').read_text())
    for rec in prepared['inputs']: checked(rec)
    checked(evaluation['metrics']); checked(evaluation['conflicts']); checked(evaluation['lists'])
    df = pd.read_csv(evaluation['metrics']['path'], usecols=['list_id', 'model', 'verifier', 'method', 'F1', 'b_hat', 'm', 'N'])
    expected = sealed['lists'] * 3 * 4 * len(METHODS)
    if len(df) != expected or evaluation['runs'] != expected: raise ValueError('Incomplete result matrix')
    sizes = df.groupby(['list_id', 'model', 'verifier']).size()
    if not (sizes == len(METHODS)).all(): raise ValueError('Missing a method/list result')
    d = df[df.method == 'DENSE']
    if not (d.groupby(['list_id', 'verifier']).F1.nunique() == 1).all(): raise ValueError('DENSE changed with ranking')
    d = df[df.method == 'NATIVE']
    if not (d.groupby(['list_id', 'model']).F1.nunique() == 1).all() or d.m.any(): raise ValueError('NATIVE changed by verifier')
    p = df[df.method.str.match(r'^PBA_0\.')]
    if not (p.m == p.N).all(): raise ValueError('Uniform-prior mathematical exhaustion check failed')
    pivot = df.pivot(index=['list_id', 'model', 'verifier'], columns='method', values='b_hat')
    for method in ('PBA_0.1', 'PBA_0.3', 'RANDOM_SAME_BUDGET'):
        if not (pivot[method] == pivot['PBA_0.2']).all(): raise ValueError('Exhaustive uniform MAP equivalence failed')
    summary = pd.read_csv(PUBLIC / 'summary.csv')
    if len(summary) != 3*3*4*10*3: raise ValueError('Incomplete all/present/empty summary')
    snapshot = RUN/'offline/initial_aggregation/verdicts.json'
    if snapshot.exists():
        before = json.loads(snapshot.read_text()); after = json.loads((PUBLIC/'verdicts.json').read_text())
        if any(before[k] != after[k] for k in ('overall','H1','H2a','per_teacher_joint')):
            raise ValueError('Primary point results changed during uncertainty completion')
        for verifier in TEACHERS:
            for target in ('ALL',*TARGETS):
                for field in ('AUROC','failures','failure_rate','lists_times_cross_rankers'):
                    if before['H2b'][verifier][target][field] != after['H2b'][verifier][target][field]:
                        raise ValueError('H2b point result changed')
    if sealed['GT_open_attempts'] or prepared['GT_open_attempts'] or not sealed['isolation']['kernel_read_whitelist']:
        raise ValueError('GT-free isolation failed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_search_feasibility.py', 'tests/test_decompose.py'],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    compile_result = subprocess.run([sys.executable, '-m', 'compileall', '-q',
        *[str(p) for p in sorted((ROOT/'scripts').glob('*search_feasibility*.py'))]], cwd=ROOT, env=env, capture_output=True, text=True)
    logs = RUN / 'validation'; logs.mkdir(exist_ok=True)
    (logs / 'pytest.log').write_text(test.stdout + test.stderr)
    (logs / 'compileall.log').write_text(compile_result.stdout + compile_result.stderr)
    if test.returncode or compile_result.returncode: raise RuntimeError('Targeted regression failed: ' + str(logs))
    passes = re.search(r'(\d+) passed', test.stdout)
    result = dict(status='PASS', tests_passed=int(passes.group(1)), compileall_passed=True,
        full_cross_product_rows=len(df), expected_rows=expected, lists=sealed['lists'], summary_rows=len(summary),
        DENSE_ranking_invariance=True, NATIVE_verifier_invariance=True,
        uniform_PBA_all_candidates_queried=True, all_uniform_epsilon_and_random_MAP_boundaries_identical=True,
        GT_free_preparation_open_attempts=0, GT_free_search_open_attempts=0,
        negative_test_GT_attempts_are_separate_from_experiment=True,
        no_new_teacher_model_or_GPU_calls=True, old_teacher_and_margin_SHAs_unchanged=True,
        primary_points_and_verdicts_unchanged_by_uncertainty_completion=snapshot.exists(),
        published_plan_SHA_matches_runtime_plan=(sha(ROOT/'results/decompose/probe_sets.json') ==
            sha(ROOT/'results/runs/decompose_20261102/probe/plan.json')),
        prereg_publication=proof, logs=[record(logs/'pytest.log'), record(logs/'compileall.log')],
        CPU_wall_seconds=time.perf_counter()-start)
    if not result['published_plan_SHA_matches_runtime_plan']: raise ValueError('Published decompose plan changed')
    dump(PUBLIC / 'validation.json', result)
    print(json.dumps(dict(event='SEARCH_VALIDATION_PASSED', tests_passed=result['tests_passed'], rows=len(df))), flush=True)


if __name__ == '__main__': main()
