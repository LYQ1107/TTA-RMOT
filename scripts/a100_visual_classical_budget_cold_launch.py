"""Actually new10% C4/C1 mean-KD/graph stream, physical worker receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_visual_classical_prepare import OUT

NAME = 'C4_key_rho0p1_mean_pair_20261122'
DESTINATION = OUT/'cold_full_c4'/NAME
STATE = OUT/'supervisor/budget_cold_state.json'


def custom_plan(path,job_key=None):
    path=Path(path).absolute();plan=json.loads(path.read_text())
    if job_key is not None:plan=dict(plan,**plan['jobs'][job_key])
    return path,plan


def state_path(pair,graph_scope,single_method=None,plan_file=None,job_key=None):
    if plan_file:
        _,plan=custom_plan(plan_file,job_key)
        return OUT/'supervisor'/plan.get('state_filename',plan['run_name']+'_state.json')
    if single_method:return OUT/'supervisor'/('final_single_'+single_method.lower()+'_state.json')
    return OUT/'supervisor/minimal_final_cold_state.json' if graph_scope=='extended' else STATE if pair=='C4' else OUT/'supervisor/budget_cold_C1_state.json'


def main(pair='C4',graph_scope='clothing',single_method=None,plan_file=None,job_key=None):
    if plan_file:
        prereg,plan=custom_plan(plan_file,job_key)
        graph_scope=plan.get('query_graph_scope',graph_scope)
    else:
        prereg = ROOT/'docs/a100_visual_classical_20261121'/('preregistration_minimal_final_single_v2.json' if single_method else 'preregistration_minimal_final_cold_v1.json' if graph_scope=='extended' else 'preregistration_budget_cold_v1.json' if pair=='C4' else 'preregistration_budget_cold_C1_v1.json')
        plan = json.loads(prereg.read_text())
        if single_method:
            assert graph_scope=='extended'
            plan=dict(plan,**plan['jobs'][single_method])
    assert plan['pair']==pair
    destination=OUT/'cold_full_c4'/plan['run_name']
    state=state_path(pair,graph_scope,single_method,plan_file,job_key)
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination/'worker_receipt.json'
    if (destination/'completion.json').is_file():
        assert receipt_path.is_file(), 'Do not replace a missing original physical receipt with a no-op timing'
    else:
        if pair=='C1' and plan_file is None:
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
        command = [sys.executable,'scripts/a100_visual_classical_cold_full_c4.py','--name',plan['run_name'],'--pair',pair,'--query-graph-scope',graph_scope,
            '--perception-mode','track-three','--teacher-fraction',str(plan.get('teacher_fraction',.1)),'--methods',','.join(plan['methods']),
            '--head-query-scope',plan.get('head_query_scope','clothing'),'--set-decoder',plan.get('set_decoder','original')]
        command+=['--graph-metric',plan.get('graph_metric','combined'),'--graph-prior',plan.get('graph_prior','constant')]
        if plan.get('task_protocol'):command+=['--task-protocol',plan['task_protocol']]
        with log_path.open('a', buffering=1) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['physical_GPU']), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            running=dict(status='RUNNING_FRESH10_PERCENT_'+pair,pid=os.getpid(),worker_pid=child.pid,
                worker_log=str(log_path), run_root=str(destination), goal_status='ACTIVE',
                query_graph_scope=graph_scope,preregistration_sha256=file_sha(prereg))
            atomic_json(state,running)
            stop_progress=threading.Event()
            def progress():
                while not stop_progress.wait(45):
                    count=len(list((destination/'source').glob('*/receipt.json')))
                    atomic_json(state,dict(running,completed=count))
                    print(json.dumps(dict(event='PHYSICAL_PAIR_PROGRESS',pair=pair,scope=graph_scope,
                        completed_videos=count,total_videos=45 if pair=='C4' else 25)),flush=True)
            monitor=threading.Thread(target=progress,daemon=True);monitor.start()
            code = child.wait()
            observed=time.perf_counter()-start
            stop_progress.set();monitor.join()
        worker = dict(exit_code=code,actual_worker_process_wall_seconds=observed,
            physical_GPU=plan['physical_GPU'],methods=plan['methods'],actual_two_branch_process=len(plan['methods'])==2,
            independent_single_arm_processes=len(plan['methods'])==1,all_startup_and_inference_included=True,
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
    protocol=completion['protocol']
    assert protocol.get('head_query_scope','clothing')==plan.get('head_query_scope','clothing')
    assert protocol.get('set_decoder','original')==plan.get('set_decoder','original')
    assert protocol.get('task_protocol')==plan.get('task_protocol')
    assert protocol.get('graph_metric','combined')==plan.get('graph_metric','combined')
    assert protocol.get('graph_prior','constant')==plan.get('graph_prior','constant')
    assert completion['GT_open_attempts'] == completion['prepaid_artifact_reads'] == 0
    assert completion['logical_teacher_calls'] == plan['expected_questions']
    assert completion['teacher']['new_calls'] == plan['expected_questions']
    assert completion['teacher']['cache_hits'] == 0
    assert {r['method'] for r in completion['methods']} == set(plan['methods'])
    atomic_json(state,dict(status='FRESH10_PERCENT_SEALED_EVALUATING',pid=os.getpid(),run_root=str(destination),goal_status='ACTIVE'))
    from flex_uda.a100.evaluation_reuse import evaluate
    results = {name:evaluate(destination/'methods'/name,pair) for name in plan['methods']}
    atomic_json(destination/'official_evaluation.json',dict(status='COMPLETE_FRESH_BUDGET_PAIR' if len(plan['methods'])==2 else 'COMPLETE_FRESH_BUDGET_SINGLE',methods=results,
        actual_worker_receipt=json.loads(receipt_path.read_text()),no_target_GT_in_adaptation=True,
        original_predictions_sealed_before_GT=True,goal_status='ACTIVE'))
    atomic_json(state,dict(status='PASS_FRESH_BUDGET_PAIR_RESEARCH_GOAL_ACTIVE',pid=os.getpid(),run_root=str(destination),goal_status='ACTIVE'))
    print('PASS_FRESH_BUDGET_PAIR_RESEARCH_GOAL_ACTIVE',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',choices=['C4','C1'],default='C4')
    parser.add_argument('--scope',choices=['clothing','extended'],default='clothing')
    parser.add_argument('--single-method',choices=['CLIP_MEAN','CLIP_MEAN_COHORT'])
    parser.add_argument('--plan-file',type=Path,help='Explicit registered deployment plan; keeps default historical plans unchanged')
    parser.add_argument('--job-key',help='Named job within the explicit plan')
    args=parser.parse_args()
    try:
        main(args.pair,args.scope,args.single_method,args.plan_file,args.job_key)
    except Exception as error:
        state=state_path(args.pair,args.scope,args.single_method,args.plan_file,args.job_key)
        row=json.loads(state.read_text()) if state.is_file() else {}
        row.update(status='FAILED_FRESH_BUDGET_PAIR_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(state,row)
        raise
