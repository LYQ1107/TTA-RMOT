"""Resume the same sealed seed0 official evaluation with eight CPU slots.

Existing isolated evaluators finish under their original PIDs. Only their old
dispatchers are replaced; every new slot receives a disjoint unfinished path.
No prediction, fitting, metric or registered analysis helper is changed.
"""
import concurrent.futures
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_conditional_seed0_scope as scope

OUT = scope.OUT/'evaluation_execution_amendment'
STATE = OUT/'state.json'
READY = OUT/'handoff_ready.json'
PLAN = scope.PUBLIC/'conditional_seed0_evaluation_execution_plan.json'


def live_same(job):
    path = Path('/proc')/str(job['pid'])/'stat'
    if not path.is_file():
        return False
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[0] != 'Z' and fields[19] == job['start_ticks']


def tasks():
    marker = json.loads((scope.OUT/'all_predictions_sealed_before_GT.json').read_text())
    assert marker['methods'] == 876 and marker['GT_open_attempts'] == 0
    rows = [(str(folder), r['pair']) for r in scope.conditions()
        for folder in scope.original.statistical_paths(r).values()]
    assert len(rows) == len(set(rows)) == 876
    return rows


def main():
    plan = json.loads(PLAN.read_text())
    assert plan['max_concurrent_methods'] == 8 and plan['neural_seeds'] == [0]
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    scope.check_registration()
    atomic_json(STATE, dict(status='WAITING_AUTHORIZED_CPU_EVALUATION_HANDOFF', pid=os.getpid(), goal_status='ACTIVE'))
    while not READY.is_file():
        time.sleep(.2)
    ready = json.loads(READY.read_text())
    assert ready['new_controller_pid'] == os.getpid()
    all_tasks = tasks()
    lookup = dict(all_tasks)
    adopted = {r['folder']:r for r in ready['preserved_isolated_evaluators']}
    assert all(folder in lookup and lookup[folder] == j['pair'] for folder, j in adopted.items())
    complete = {folder for folder, _ in all_tasks if (Path(folder)/'official_completion.json').is_file()}
    pending = deque((folder, pair) for folder, pair in all_tasks if folder not in complete and folder not in adopted)
    failure, active = None, {}
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        while pending or active or adopted:
            for folder, job in list(adopted.items()):
                if (Path(folder)/'official_completion.json').is_file():
                    complete.add(folder)
                    del adopted[folder]
                elif not live_same(job):
                    failure = dict(folder=folder, preserved_pid=job['pid'], error='Preserved evaluator ended without completion')
                    del adopted[folder]
            for future, (folder, pair) in list(active.items()):
                if not future.done():
                    continue
                del active[future]
                try:
                    future.result()
                    assert (Path(folder)/'official_completion.json').is_file()
                    complete.add(folder)
                except Exception as error:
                    failure = dict(folder=folder, pair=pair, error=repr(error))
            while pending and len(active)+len(adopted) < 8 and failure is None:
                folder, pair = pending.popleft()
                if (Path(folder)/'official_completion.json').is_file():
                    complete.add(folder)
                else:
                    active[pool.submit(evaluate_one, (folder, pair))] = (folder, pair)
            scope.state('EVALUATING_SAME_SEALED_SEED0_SCOPE_EIGHT_CPU_SLOTS' if failure is None
                else 'PRESERVING_HEALTHY_SEED0_EVALUATORS_AFTER_FAILURE', completed=6, active={},
                official_expected=876, official_completed=len(complete), official_pending=len(pending),
                official_new_active=len(active), preserved_evaluators=list(adopted.values()),
                original_official_reused_at_start=261, max_concurrent_official_methods=8, failure=failure)
            if failure is not None and not active and not adopted:
                raise RuntimeError(json.dumps(failure))
            if pending or active or adopted:
                time.sleep(2)
    assert len(complete) == 876
    results = {folder:json.loads((Path(folder)/'official_completion.json').read_text()) for folder, _ in all_tasks}
    atomic_json(scope.OUT/'official.json', dict(methods=876, own_methods=588,
        fixed_reference_methods=288, original_reused_at_start=261, results=results,
        execution_amendment='Eight CPU slots; original in-flight isolated evaluators preserved; no metric or predictions changed'))
    scope.state('ALL_SEED0_OFFICIAL_COMPLETE_ANALYZING', completed=6, active={})
    scope.analyze()
    subprocess.run([sys.executable, 'scripts/a100_six_setting_statistics_reproduce.py',
        '--stem', scope.STEM], cwd=ROOT, check=True)
    scope.state(scope.SUCCESS, completed=6, active={}, official_expected=876,
        independent_arithmetic_reproduction=True, scientific_goal_complete=False,
        execution_amendment=str(PLAN), max_concurrent_official_methods=8)
    atomic_json(STATE, dict(status='PASS_SAME_SEED0_SCOPE_EIGHT_CPU_EVALUATION_COMPLETE',
        pid=os.getpid(), goal_status='ACTIVE'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        scope.state('FAILED_SINGLE_SEED_SCOPE_REQUIRES_ROOT_CAUSE', error=repr(error))
        atomic_json(STATE, dict(status='FAILED_CPU_EVALUATION_HANDOFF_REQUIRES_ROOT_CAUSE',
            pid=os.getpid(), error=repr(error), goal_status='ACTIVE'))
        raise
