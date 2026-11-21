"""Actually new10% C4/C1 mean-KD/graph stream, physical worker receipt."""
import argparse
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

NAME = 'C4_key_rho0p1_mean_pair_20261122'
DESTINATION = OUT/'cold_full_c4'/NAME
STATE = OUT/'supervisor/budget_cold_state.json'


def main(pair='C4'):
    prereg = ROOT/'docs/a100_visual_classical_20261121'/('preregistration_budget_cold_v1.json' if pair=='C4' else 'preregistration_budget_cold_C1_v1.json')
    plan = json.loads(prereg.read_text())
    destination=OUT/'cold_full_c4'/plan['run_name']
    state=STATE if pair=='C4' else OUT/'supervisor/budget_cold_C1_state.json'
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination/'worker_receipt.json'
    if (destination/'completion.json').is_file():
        assert receipt_path.is_file(), 'Do not replace a missing original physical receipt with a no-op timing'
    else:
        if pair=='C1':
            atomic_json(state,dict(status='WAITING_FOR_CONTEXT_GPU_RELEASE',pid=os.getpid(),run_root=str(destination),goal_status='ACTIVE'))
            while True:
                context=json.loads((OUT/'supervisor/context_cohort_state.json').read_text())
                if context['status'].startswith('FAILED_'):
                    raise RuntimeError('Context generation failed: resolve it before using its physical GPU0')
                if context['status'].startswith(('ALL_CONTEXT_', 'PASS_CONTEXT_')):
                    break
                time.sleep(5)
        log_path = destination/'worker.log'
        start = time.perf_counter()
        command = [sys.executable,'scripts/a100_visual_classical_cold_full_c4.py','--name',plan['run_name'],'--pair',pair,
            '--perception-mode','track-three','--teacher-fraction','.1','--methods','CLIP_MEAN,CLIP_MEAN_COHORT']
        with log_path.open('a', buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['physical_GPU']), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            atomic_json(state, dict(status='RUNNING_FRESH10_PERCENT_'+pair,pid=os.getpid(),worker_pid=child.pid,
                worker_log=str(log_path), run_root=str(destination), goal_status='ACTIVE',
                preregistration_sha256=file_sha(prereg)))
            code = child.wait()
        worker = dict(exit_code=code,actual_worker_process_wall_seconds=time.perf_counter()-start,
            physical_GPU=plan['physical_GPU'],methods=plan['methods'],actual_two_branch_process=True,
            independent_single_arm_processes=False,all_startup_and_inference_included=True,
            official_GT_evaluation_excluded=True,OS_NAS_shared_background_jobs_disclosed=True)
        attempt = len(list(destination.glob('worker_attempt_*.json')))
        atomic_json(destination/f'worker_attempt_{attempt}.json', worker)
        attempts = [json.loads(p.read_text()) for p in sorted(destination.glob('worker_attempt_*.json'))]
        atomic_json(receipt_path, dict(worker, attempts=attempts,
            actual_worker_process_wall_seconds=sum(r['actual_worker_process_wall_seconds'] for r in attempts),
            failed_attempt_wall_seconds=sum(r['actual_worker_process_wall_seconds'] for r in attempts if r['exit_code'])))
        if code:
            print(log_path.read_text()[-7000:],flush=True)
            raise RuntimeError('Fresh budget worker failed; keep receipt and stream')
    completion = json.loads((destination/'completion.json').read_text())
    assert completion['GT_open_attempts'] == completion['prepaid_artifact_reads'] == 0
    assert completion['logical_teacher_calls'] == plan['expected_questions']
    assert completion['teacher']['new_calls'] == plan['expected_questions']
    assert completion['teacher']['cache_hits'] == 0
    assert {r['method'] for r in completion['methods']} == set(plan['methods'])
    atomic_json(state,dict(status='FRESH10_PERCENT_SEALED_EVALUATING',pid=os.getpid(),run_root=str(destination),goal_status='ACTIVE'))
    from flex_uda.a100.evaluation_reuse import evaluate
    results = {name:evaluate(destination/'methods'/name,pair) for name in plan['methods']}
    atomic_json(destination/'official_evaluation.json',dict(status='COMPLETE_FRESH_BUDGET_PAIR',methods=results,
        actual_worker_receipt=json.loads(receipt_path.read_text()),no_target_GT_in_adaptation=True,
        original_predictions_sealed_before_GT=True,goal_status='ACTIVE'))
    atomic_json(state,dict(status='PASS_FRESH_BUDGET_PAIR_RESEARCH_GOAL_ACTIVE',pid=os.getpid(),run_root=str(destination),goal_status='ACTIVE'))
    print('PASS_FRESH_BUDGET_PAIR_RESEARCH_GOAL_ACTIVE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',choices=['C4','C1'],default='C4')
    args=parser.parse_args()
    try:
        main(args.pair)
    except Exception as error:
        state=STATE if args.pair=='C4' else OUT/'supervisor/budget_cold_C1_state.json'
        row=json.loads(state.read_text()) if state.is_file() else {}
        row.update(status='FAILED_FRESH_BUDGET_PAIR_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(state,row)
        raise
