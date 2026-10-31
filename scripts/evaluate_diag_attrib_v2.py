#!/usr/bin/env python3
"""Independent post-seal official evaluation; no score calibration here."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_v2_common import *
from scripts.pf_diag_common import official_metrics


def evaluate(method, force_new=False):
    require_prereg()
    verify_seal(method)
    folder = EVAL / method
    done = folder / 'official_completion.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['exit_code'] == 0:
            for name, field in [('pedestrian_summary.txt', 'summary_sha256'), ('pedestrian_detailed.csv', 'detailed_sha256')]:
                if file_sha(folder / 'results' / name) != previous[field]:
                    raise ValueError('completed official result changed: ' + method)
            return previous
    job = RUN / 'official_jobs' / method
    job.mkdir(parents=True, exist_ok=True)
    link = job / method
    if not link.exists():
        link.symlink_to(folder, target_is_directory=True)
    attempt = len(list(job.glob('stdout_*.txt'))) + 1
    log_path = job / f'stdout_{attempt}.txt'
    started = time.perf_counter()
    with log_path.open('w') as log:
        code = subprocess.call([sys.executable, str(ROOT / 'scripts/evaluate_flexhook_dance_tta.py'),
            '--output-root', str(job), '--dataset-root', str(DATA), '--vendor-root', str(ROOT / 'vendor/FlexHook'),
            '--methods', method], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result = dict(status='PASS' if code == 0 else 'FAIL', exit_code=code, method=method,
        new_TrackEval_job=True, worker_wall_seconds=time.perf_counter()-started,
        stdout_path=str(log_path), prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'))
    if code == 0:
        result.update(summary_sha256=file_sha(folder / 'results/pedestrian_summary.txt'),
                      detailed_sha256=file_sha(folder / 'results/pedestrian_detailed.csv'))
    atomic_json(job / f'completion_{attempt}.json', result)
    if code:
        raise RuntimeError('official evaluation failed: ' + method + '; see ' + str(log_path))
    verify_seal(method)
    atomic_json(done, result)
    print(json.dumps(dict(event='ATTRIB_V2_OFFICIAL_PASS', method=method,
                         metrics=official_metrics(folder / 'results'), seconds=result['worker_wall_seconds'])), flush=True)
    return result


def references():
    report = json.loads((RUN / 'reference_reproduction.json').read_text())
    if report['status'] != 'PASS':
        raise ValueError('reference row reproduction must pass first')
    complete = 'OFFICIAL_DANCE_COMPLETE'
    historical = historical_name(complete)
    evaluate(complete)
    evaluate(historical, force_new=True)
    actual = official_metrics(EVAL / complete / 'results')
    old = official_metrics(EVAL / historical / 'results')
    if actual['Dets'] != 254910 or old['Dets'] != 254118 or old['HOTA'] != 32.134:
        raise ValueError('v2 reference/historical official evaluation gate failed')
    gate = dict(status='PASS', complete_metrics=actual, historical_metrics=old,
        complete_HOTA_expected_for_matrix_diagonal=actual['HOTA'],
        complete_prediction_seal_sha256=file_sha(EVAL / complete / 'predictions_seal.json'),
        historical_native_rows_exactly_equal=True, historical_HOTA_reproduced=True,
        new_full_reference_HOTA_not_assumed_to_equal_32_134=True)
    atomic_json(RUN / 'reference_validation.json', gate)
    print(json.dumps(dict(event='ATTRIB_V2_REFERENCE_GATE_PASS', **gate)), flush=True)


def batch(methods):
    # Four independent jobs; each unchanged official evaluator uses two CPU
    # workers. Whole episodes remain independent and all425 are evaluated.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluate, m) for m in methods]):
            future.result()


def prepared():
    record = json.loads((RUN / 'GT_free_predictions.json').read_text())
    if record['status'] != 'ALL_GT_FREE_PREDICTIONS_SEALED' or record['GT_free_target_open_attempts']:
        raise ValueError('seal GT-free predictions before evaluator')
    for item in record['equivalent_old_results']:
        reuse_metrics(item['method'], item['reference'])
    expected = {matrix_name(m, m): v for m, v in HISTORICAL_HOTA.items()}
    for m, h1, h2 in [('D1',29.4,7.8978),('D2',22.854,7.838),('D3',27.103,8.1589)]:
        expected[matrix_name('FROZEN',m)] = h1
        expected[matrix_name(m,'FROZEN')] = h2
    expected[matrix_name('OFFICIAL_DANCE_COMPLETE','OFFICIAL_DANCE_COMPLETE')] = json.loads((RUN / 'reference_validation.json').read_text())['complete_HOTA_expected_for_matrix_diagonal']
    for method, hota in expected.items():
        if abs(official_metrics(EVAL / method / 'results')['HOTA']-hota) > 1e-7:
            raise ValueError('registered old control/diagonal HOTA mismatch: ' + method)
    atomic_json(RUN / 'matrix_self_checks.json',dict(status='PASS', all425_row_equivalences=record['equivalent_old_results'], expected_HOTA=expected))
    batch(record['methods'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--methods', nargs='*')
    args = parser.parse_args()
    if args.references:
        references()
    elif args.prepared:
        prepared()
    else:
        batch(args.methods or [])
