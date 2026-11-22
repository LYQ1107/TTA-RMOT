"""One blocking four-GPU study; all predictions seal before its official GT."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_context_prefix import CONTEXT
from scripts.a100_visual_classical_budget_prefix import BUDGET, METHODS as BASELINES
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts, evaluate_one

STATE = OUT / 'supervisor/context_cohort_state.json'


def conditions():
    return [dict(pair=pair, seed=seed, folder=str(CONTEXT / f'seed{seed}' / pair))
        for pair in ('C4', 'C1') for seed in (0, 17, 42)]


def count_record(record):
    root = Path(record['folder'])
    original = BUDGET / f"rho0.1_seed{record['seed']}" / record['pair']
    sources = {name: str(root/name) for name in
        [r['method'] for r in json.loads((root/'completion.json').read_text())['methods']]}
    sources.update(ORIGINAL_MEAN=str(original/'SOFT_MEAN_400_FINAL'), ORIGINAL_GRAPH=str(original/'MEAN_GRAPH'))
    pairs = [('ORIGINAL_GRAPH', 'ORIGINAL_GRAPH'), ('ORIGINAL_COMPLETE', 'ORIGINAL_GRAPH'),
             ('ORIGINAL_PERMUTED', 'ORIGINAL_GRAPH')]
    if record['pair'] == 'C4':
        pairs += [(name, 'CONTEXT_FULL') for name in ('CONTEXT_FULL', 'CONTEXT_MEAN', 'ORIGINAL_MEAN',
            'CONTEXT_ANCHORS', 'CONTEXT_VISUAL', 'CONTEXT_COMPLETE', 'CONTEXT_PERMUTED')]
        pairs += [(name, donor) for donor in ('CONTEXT_MEAN', 'ORIGINAL_MEAN') for name in ('CONTEXT_FULL', donor)]
    return dict(record, method_sources=sources, projection_pairs=list(dict.fromkeys(pairs)), kind='CONTEXT')


def main():
    prior = json.loads((OUT/'analysis/sparse_observation_analysis.json').read_text())
    assert prior['official_evaluated_methods'] == 404
    prereg = ROOT/'docs/a100_visual_classical_20261121/preregistration_context_cohort_v1.json'
    assert prereg.is_file()
    start = time.perf_counter()
    pending, active, completed, failures = conditions(), {}, [], []
    phase, evaluated, counter_done = 'GENERATING_CONTEXT_PREFIX_NO_NEW_GT', 0, 0
    def state():
        atomic_json(STATE, dict(status=phase, pid=os.getpid(), pending=pending,
            active={str(g): r for g, (_, _, r) in active.items()}, completed=completed, failures=failures,
            official_methods_done=evaluated, count_conditions_done=counter_done,
            preregistration_sha256=file_sha(prereg), wall_seconds=time.perf_counter()-start, goal_status='ACTIVE'))
    state()
    last_progress = 0.
    while pending or active:
        for gpu, (child, log, record) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            log.close()
            record = dict(record, exit_code=code)
            path = Path(record['folder'])/'completion.json'
            if code or not path.is_file():
                failures.append(record)
            else:
                receipt = json.loads(path.read_text())
                assert receipt['GT_open_attempts'] == receipt['original_graph_regression_changed_rows'] == 0
                record['methods'] = len(receipt['methods'])
                completed.append(record)
            del active[gpu]
            print(json.dumps(dict(event='CONTEXT_JOB_DONE', **record)), flush=True)
            state()
        if not failures:
            for gpu in range(4):
                if gpu in active or not pending:
                    continue
                record = pending.pop(0)
                path = Path(record['folder'])/'completion.json'
                if path.is_file():
                    receipt = json.loads(path.read_text())
                    assert receipt['GT_open_attempts'] == receipt['original_graph_regression_changed_rows'] == 0
                    completed.append(dict(record, reused_sealed=True, methods=len(receipt['methods'])))
                    continue
                logfile = OUT/'supervisor'/f"context_{record['pair']}_seed{record['seed']}.log"
                log = logfile.open('a', buffering=1)
                child = subprocess.Popen([sys.executable, 'scripts/a100_visual_classical_context_prefix.py',
                    '--pair', record['pair'], '--seed', str(record['seed'])], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
                record = dict(record, GPU=gpu, pid=child.pid, log=str(logfile))
                active[gpu] = child, log, record
                print(json.dumps(dict(event='CONTEXT_JOB_START', **record)), flush=True)
                state()
        if failures and not active:
            phase = 'FAILED_CONTEXT_PREFIX_PRESERVE_OUTPUTS'
            state()
            for record in failures:
                print(Path(record['log']).read_text()[-7000:], flush=True)
            raise RuntimeError('Context prefix failed; preserve streams and fix first actionable cause')
        if time.perf_counter()-last_progress >= 45:
            print(json.dumps(dict(event='CONTEXT_STUDY_PROGRESS', phase=phase,
                completed_conditions=len(completed), total_conditions=6,
                completed_video_receipts=sum(len(list(Path(r['folder']).glob('videos/*/receipt.json'))) for r in conditions()))), flush=True)
            last_progress = time.perf_counter()
        if pending or active:
            time.sleep(3)
    assert len(completed) == 6
    phase = 'ALL_CONTEXT_RAW_SEALED_GENERATING_QUANTITY_NO_NEW_GT'
    state()
    count_records = [count_record(r) for r in conditions()]
    count_roots = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(generate_counts, r): r for r in count_records}
        for future in concurrent.futures.as_completed(futures):
            folder = future.result()
            counter_done += 1
            count_roots.append((folder, futures[future]['pair']))
            print(json.dumps(dict(event='CONTEXT_COUNTERFACTUAL_DONE', conditions=counter_done, total=6)), flush=True)
            state()
    tasks = []
    for record in conditions():
        folder = Path(record['folder'])
        tasks += [(str(folder/r['method']), record['pair']) for r in json.loads((folder/'completion.json').read_text())['methods']]
    for folder, pair in count_roots:
        tasks += [(str(Path(folder)/r['method']), pair) for r in json.loads((Path(folder)/'completion.json').read_text())['methods']]
    assert len(tasks) == 96, len(tasks)
    phase = 'ALL_CONTEXT_OUTPUTS_SEALED_OFFICIAL_EVALUATING'
    state()
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            folder, row = future.result()
            results[folder] = row
            evaluated += 1
            print(json.dumps(dict(event='CONTEXT_OFFICIAL_PROGRESS', completed=evaluated, total=len(tasks))), flush=True)
            state()
    atomic_json(OUT/'analysis/context_cohort_official.json', dict(status='ALL_CONTEXT_COHORT_OFFICIAL_COMPLETE',
        results=results, methods=len(tasks), goal_status='ACTIVE', all_development_data=True,
        no_adaptation_GT=True, all_new_outputs_sealed_before_this_GT=True))
    phase = 'PASS_CONTEXT_COHORT_STUDY_RESEARCH_GOAL_ACTIVE'
    state()
    print(phase, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        row = json.loads(STATE.read_text()) if STATE.is_file() else {}
        row.update(status='FAILED_CONTEXT_COHORT_REQUIRES_ROOT_CAUSE', error=repr(error), goal_status='ACTIVE')
        atomic_json(STATE, row)
        raise
