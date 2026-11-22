"""Conditional four fresh deployments; one supplied-protocol ordinary baseline."""
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
from scripts.a100_visual_classical_protocol_calibration_review import quality_checks
from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PLAN=PUBLIC/'preregistration_protocol_cold_v1.json'
STATE=OUT/'supervisor/protocol_cold_state.json'
SUCCESS='PASS_PROTOCOL_FINAL_ACTUAL_COST_RESEARCH_REVIEW_REQUIRED'


def main():
    assert json.loads((PUBLIC/'protocol_calibration_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    quality=json.loads((PUBLIC/'protocol_calibration_analysis.json').read_text());checks,_=quality_checks(quality)
    assert all(checks.values()), ('Quality criteria unmet; no physical worker',checks)
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];begin=time.perf_counter();active={};done={};failure=None;last=0
    for path,sha in plan['implementation_sha256'].items():
        assert file_sha(ROOT/path)==sha, path
    for job in jobs.values():
        assert file_sha(ROOT/job['worker_script'])==job['worker_sha256'] and job['physical_GPU'] in (0,1,2,3)
    pending=list(jobs)
    def state(status):
        arms={}
        for key,job in jobs.items():
            root=OUT/'cold_full_c4'/job['run_name'];file=root/'phase_ledger.json'
            rows=json.loads(file.read_text()).get('common_phases',[]) if file.is_file() else []
            arms[key]=dict(physical_GPU=job['physical_GPU'],completed_videos=len(list((root/'source').glob('*/receipt.json'))),
                total_videos=45 if job['pair']=='C4' else 25,sealed=key in done,last_phase=rows[-1]['phase'] if rows else None)
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),arms=arms,failure=failure,
            quality_checks=checks,wall_seconds=time.perf_counter()-begin,preregistration_sha256=file_sha(PLAN),goal_status='ACTIVE'))
    state('WAITING_GPU0_3_RELEASE_FOR_FOUR_FRESH_FINAL_PROTOCOL_ARMS')
    while pending or active:
        changed=False
        for key,(process,handle,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();done[key]=rc;del active[key];changed=True
            if rc:failure=dict(job=key,exit_code=rc,last_log=log.read_text()[-7000:])
        if pending and not failure:
            ids=','.join(str(jobs[k]['physical_GPU']) for k in pending)
            rows=subprocess.check_output(['nvidia-smi','-i',ids,'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.strip().splitlines()}
            allocation=dict(requested_GPUs=[jobs[k]['physical_GPU'] for k in pending],used_MiB=memory,
                only_start_below_MiB=512,no_other_process_terminated=True)
            atomic_json(OUT/'supervisor/protocol_cold_last_allocation.json',allocation)
            for key in list(pending):
                job=jobs[key]
                if memory[job['physical_GPU']]>=512:continue
                log=OUT/'supervisor'/('protocol_cold_'+key+'.log');handle=log.open('a',buffering=1)
                child=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_budget_cold_launch.py',
                    '--pair',job['pair'],'--scope','extended','--plan-file',str(PLAN),'--job-key',key],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'))
                active[key]=child,handle,log;pending.remove(key);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_FOUR_FRESH_FINAL_PROTOCOL_ARMS_NO_GT' if not failure else 'PRESERVING_REMAINING_PROTOCOL_ARMS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    tasks=[]
    for key,job in jobs.items():
        root=OUT/'cold_full_c4'/job['run_name'];r=json.loads((root/'completion.json').read_text())
        assert r['GT_open_attempts']==r['prepaid_artifact_reads']==r['teacher']['cache_hits']==0 and r['single_uninterrupted_attempt']
        assert r['teacher']['new_calls']==r['logical_teacher_calls']==job['expected_questions']
        assert r['protocol']['confidence_weighted_teacher_head'] and r['protocol']['shared_head_readout']=='CONVERGED_FINAL'
        assert not r['protocol']['source_motion_presence_gate_applied'] and r['single_method_execution']
        tasks.append((str(root/'methods'/job['methods'][0]),job['pair']))
    state('ALL4_FINAL_PROTOCOL_PREDICTIONS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'analysis/protocol_cold_official.json',dict(status='ALL4_PROTOCOL_FINAL_COLD_OFFICIAL_COMPLETE',
        results=results,all_four_fresh_predictions_sealed_before_followup_GT=True,goal_status='ACTIVE'))
    state('PROTOCOL_FINAL_COLD_OFFICIAL_COMPLETE_ANALYZING')
    for command in (['scripts/a100_visual_classical_protocol_cold_analyze.py'],['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','protocol_cold']):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_PROTOCOL_FINAL_COLD_REQUIRES_ROOT_CAUSE_OR_UNMET_QUALITY',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'))
        raise
