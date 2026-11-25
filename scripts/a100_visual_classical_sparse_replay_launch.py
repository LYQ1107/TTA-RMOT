"""Isolated prediction/evaluation processes for fixed-QA sparse-source control."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def evaluate_one(folder):
    from flex_uda.a100.evaluation_reuse import evaluate
    return Path(folder).name, evaluate(folder, 'C4')


def main():
    supervisor = OUT / 'supervisor';state_path = supervisor / 'sparse_replay_state.json'
    root = OUT / 'cold_full_c4/C4_sparse_fixedQA_seed0';begin = time.perf_counter()
    log_path = supervisor / 'recovery_strong_field_cold_full_sparse_replay_worker.log'
    with log_path.open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, 'scripts/a100_visual_classical_sparse_replay.py'], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        atomic_json(supervisor / 'sparse_replay_worker.json', dict(pid=child.pid, physical_GPU=2, log=str(log_path)))
        code = child.wait()
    if code:
        atomic_json(state_path, dict(status='FAILED_SPARSE_REPLAY_REQUIRES_ROOT_CAUSE', exit_code=code, pid=os.getpid()))
        raise RuntimeError('Fixed-QA sparse worker failed: ' + str(code))
    sealed = json.loads((root / 'completion.json').read_text())
    assert sealed['GT_open_attempts'] == sealed['forbidden_read_attempts'] == sealed['new_teacher_calls'] == 0
    atomic_json(state_path, dict(status='EVALUATING_SPARSE_FIXED_QA', pid=os.getpid()))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap()
    folders = [root / 'methods' / item['method'] for item in sealed['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        evaluations = dict(pool.map(evaluate_one, folders))
    references = {label: OUT / 'cold_full_c4' / name for label, name in [
        ('FULL', 'C4_fresh_full_stream_seed0'), ('SPARSE', 'C4_fresh_track_three_seed0')]}
    atomic_json(state_path, dict(status='WAITING_FOR_FRESH_REFERENCE_EVALUATIONS', pid=os.getpid()))
    while not all((p / 'official_evaluation.json').is_file() for p in references.values()):
        for file in ('cold_full_state.json', 'cold_sparse_state.json'):
            status = json.loads((supervisor / file).read_text()).get('status', '')
            if status.startswith('FAILED'):
                raise RuntimeError('Actual source stream failed: ' + file)
        time.sleep(15)
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_mechanism_analyze import compare_official
    data = {folder.name: read_stats(folder) for folder in folders};comparisons = {}
    for method in data:
        comparisons[method + '_sparse_source_vs_full_FIXED_QA'] = compare_official(
            data[method], read_stats(references['FULL'] / 'methods' / method))
        comparisons[method + '_actual_sparse_vs_fixed_QA_sparse'] = compare_official(
            read_stats(references['SPARSE'] / 'methods' / method), data[method])
    for label in ('RAW', 'CLIP'):
        comparisons[label + '_same_QA_COHORT_vs_KD'] = compare_official(data[label + '_COHORT'], data[label + '_KD'])
    result = dict(status='PASS_SPARSE_SOURCE_FIXED_QA_CONTROL', HOTA={k: v['HOTA_unrounded'] for k, v in evaluations.items()},
                  comparisons=comparisons, logical_teacher_examples=sealed['logical_teacher_examples'], new_teacher_calls=0,
                  same_reference_features_and_teacher_answers=True, only_source_prior_and_presence_sparsified=True,
                  actual_full_vs_sparse_PCD_max_delta=max(r['full_vs_sparse_actual_PCD_max_delta'] for r in sealed['receipts']),
                  actual_full_vs_sparse_RGB_max_delta=max(r['RGB_max_delta'] for r in sealed['receipts']),
                  quality_only_not_a_cold_runtime=True, generation_GT_open_attempts=0,
                  all_development_data=True, supervisor_wall_includes_wait_for_actual_streams=time.perf_counter()-begin,
                  actual_sparse_vs_fixed_control_also_changes_questions_and_actual_numeric_execution=True)
    atomic_json(root / 'official_evaluation.json', result)
    atomic_json(OUT / 'analysis/fresh_sparse_fixedQA.json', result)
    atomic_json(state_path, dict(status='PASS_SPARSE_SOURCE_FIXED_QA_CONTROL', pid=os.getpid()))
    print(json.dumps(dict(event='SPARSE_FIXED_QA_COMPLETE', HOTA=result['HOTA'], comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic_json(OUT / 'supervisor/sparse_replay_state.json', dict(status='FAILED_SPARSE_REPLAY_REQUIRES_ROOT_CAUSE', pid=os.getpid()))
        raise
