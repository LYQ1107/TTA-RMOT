"""Four independent actual workers; a shared seal barrier precedes any GT."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one

PLAN=ROOT/'docs/a100_visual_classical_20261121/preregistration_appearance_joint_cold_v1.json'
STATE=OUT/'supervisor/appearance_cold_final_state.json'
SUCCESS='PASS_APPEARANCE_COLD_FINAL_RESEARCH_REVIEW_REQUIRED'


def main():
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];assert plan['defer_official_evaluation']
    for name,path in dict(worker='scripts/a100_visual_classical_cold_full_c4.py',runtime='flex_uda/active/appearance_runtime.py',
            calibration='flex_uda/active/semantic_colour.py',graph='flex_uda/active/grounded_graph.py').items():
        assert file_sha(ROOT/path)==plan['code_sha256'][name]
    started=time.perf_counter();active={};done={};last=0;failure=None
    def state(status):
        arms={}
        for key,job in jobs.items():
            folder=OUT/'cold_full_c4'/job['run_name'];status_file=OUT/'supervisor'/(job['run_name']+'_state.json')
            child=json.loads(status_file.read_text()) if status_file.is_file() else {}
            arms[key]=dict(physical_GPU=job['physical_GPU'],completed_videos=len(list((folder/'source').glob('*/receipt.json'))),
                total_videos=45 if job['pair']=='C4' else 25,status=child.get('status','INITIALIZING'))
            phase=folder/'phase_ledger.json'
            if phase.is_file():
                rows=json.loads(phase.read_text()).get('common_phases',[])
                arms[key]['last_phase']=rows[-1]['phase'] if rows else None
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),arms=arms,failure=failure,
            worker_exit_codes=done,wall_seconds=time.perf_counter()-started,preregistration_sha256=file_sha(PLAN),goal_status='ACTIVE'))
    for key,job in jobs.items():
        log=OUT/'supervisor'/('appearance_cold_'+key+'.log');handle=log.open('a',buffering=1)
        process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_budget_cold_launch.py','--pair',job['pair'],
            '--scope','extended','--plan-file',str(PLAN),'--job-key',key],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        active[key]=process,handle,log
    state('RUNNING_FOUR_INDEPENDENT_APPEARANCE_COLD_WORKERS_NO_GT')
    while active:
        changed=False
        for key,(process,handle,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();done[key]=rc;del active[key];changed=True
            if rc:failure=dict(job=key,exit_code=rc,log=str(log),last_log=log.read_text()[-7000:])
        if changed or time.perf_counter()-last>45:
            state('RUNNING_FOUR_INDEPENDENT_APPEARANCE_COLD_WORKERS_NO_GT' if not failure else 'PRESERVING_REMAINING_APPEARANCE_COLD_WORKERS_AFTER_FAILURE');last=time.perf_counter()
        if active:time.sleep(3)
    if failure:raise RuntimeError(json.dumps(failure))
    tasks=[]
    for key,job in jobs.items():
        folder=OUT/'cold_full_c4'/job['run_name'];receipt=json.loads((folder/'completion.json').read_text())
        assert receipt['GT_open_attempts']==receipt['prepaid_artifact_reads']==0
        assert receipt['single_method_execution'] and receipt['single_uninterrupted_attempt']
        assert receipt['teacher']['new_calls']==job['expected_questions'] and receipt['teacher']['cache_hits']==0
        assert receipt['logical_teacher_calls']==job['expected_questions']
        assert {r['method'] for r in receipt['methods']}==set(job['methods'])
        tasks.append((str(folder/'methods'/job['methods'][0]),job['pair']))
    state('ALL_FOUR_ACTUAL_COLD_WORKERS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'analysis/appearance_cold_official.json',dict(status='ALL_FOUR_APPEARANCE_COLD_OFFICIAL_COMPLETE',
        results=results,all_four_predictions_sealed_before_GT=True,goal_status='ACTIVE'))
    state('APPEARANCE_COLD_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_visual_classical_appearance_cold_analyze.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','appearance_cold'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_APPEARANCE_COLD_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
