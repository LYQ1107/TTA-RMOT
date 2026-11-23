"""True fresh optimized KD, adopting the two uninterrupted fresh J workers."""
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

PLAN=ROOT/'docs/a100_visual_classical_20261121/preregistration_native_optimized_cold_v2.json'
STATE=OUT/'supervisor/native_optimized_cold_state.json'
SUCCESS='PASS_NATIVE_OPTIMIZED_COLD_FINAL_RESEARCH_REVIEW_REQUIRED'


def main():
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];started=time.perf_counter();active={};done={};failure=None;last=0
    pending=[key for key,job in jobs.items() if not job['reuse_live_v1_process']]
    external=[key for key,job in jobs.items() if job['reuse_live_v1_process']]
    for job in jobs.values():assert file_sha(ROOT/job['worker_script'])==job['worker_sha256']
    for key,path in dict(runtime='flex_uda/active/appearance_runtime.py',calibration='flex_uda/active/semantic_colour.py',graph='flex_uda/active/grounded_graph.py').items():
        assert file_sha(ROOT/path)==plan['code_sha256'][key]
    def state(status):
        arms={}
        for key,job in jobs.items():
            folder=OUT/'cold_full_c4'/job['run_name'];status_file=OUT/'supervisor'/(job['run_name']+'_state.json')
            child=json.loads(status_file.read_text()) if status_file.is_file() else {}
            arms[key]=dict(physical_GPU=job['physical_GPU'],completed_videos=len(list((folder/'source').glob('*/receipt.json'))),
                total_videos=45 if job['pair']=='C4' else 25,status=child.get('status','INITIALIZING'),
                handcrafted_RGB_statistics_computed=job['compute_RGB_statistics'],adopted_running_J_from_v1=job['reuse_live_v1_process'])
            file=folder/'phase_ledger.json'
            if file.is_file():
                rows=json.loads(file.read_text()).get('common_phases',[]);arms[key]['last_phase']=rows[-1]['phase'] if rows else None
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),arms=arms,failure=failure,
            worker_exit_codes=done,wall_seconds=time.perf_counter()-started,preregistration_sha256=file_sha(PLAN),goal_status='ACTIVE'))
    state('WAITING_GPU0_2_RELEASE_FOR_OPTIMIZED_NATIVE_KD')
    while pending or active or external:
        changed=False
        for key in list(external):
            folder=OUT/'cold_full_c4'/jobs[key]['run_name'];status_file=OUT/'supervisor'/(jobs[key]['run_name']+'_state.json')
            child=json.loads(status_file.read_text()) if status_file.is_file() else {}
            if child.get('status')=='PASS_FRESH_SINGLE_SEALED_WAITING_GROUP_GT_BARRIER':
                receipt=json.loads((folder/'worker_receipt.json').read_text());assert receipt['exit_code']==0
                done[key]=0;external.remove(key);changed=True
            elif child.get('status','').startswith('FAILED'):
                failure=dict(job=key,adopted_J_failure=child);external.remove(key);changed=True
        for key,(process,handle,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();done[key]=rc;del active[key];changed=True
            if rc:failure=dict(job=key,exit_code=rc,log=str(log),last_log=log.read_text()[-7000:])
        if not failure and pending:
            requested=','.join(str(jobs[key]['physical_GPU']) for key in pending)
            rows=subprocess.check_output(['nvidia-smi','-i',requested,'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            free={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.strip().splitlines()}
            for key in list(pending):
                job=jobs[key]
                if free[job['physical_GPU']]>=512:continue
                log=OUT/'supervisor'/('native_optimized_cold_'+key+'.log');handle=log.open('a',buffering=1)
                child=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_budget_cold_launch.py','--pair',job['pair'],
                    '--scope','extended','--plan-file',str(PLAN),'--job-key',key],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[key]=child,handle,log;pending.remove(key);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_OPTIMIZED_NATIVE_KD_AND_UNINTERRUPTED_J_NO_GT' if not failure else 'PRESERVING_REMAINING_NATIVE_ACTUAL_ARMS_AFTER_FAILURE');last=time.perf_counter()
        if pending or active or external:time.sleep(3)
        if failure and not active and not external:raise RuntimeError(json.dumps(failure))
    if failure:raise RuntimeError(json.dumps(failure))
    tasks=[]
    for key,job in jobs.items():
        folder=OUT/'cold_full_c4'/job['run_name'];r=json.loads((folder/'completion.json').read_text())
        assert r['GT_open_attempts']==r['prepaid_artifact_reads']==0 and r['single_method_execution'] and r['single_uninterrupted_attempt']
        assert r['teacher']['new_calls']==r['logical_teacher_calls']==job['expected_questions'] and r['teacher']['cache_hits']==0
        assert r['protocol']['head_source_logit']=='zero' and r['protocol']['head_feature_dimension']==768
        assert r['protocol'].get('RGB_statistics_computed',True)==job['compute_RGB_statistics']
        tasks.append((str(folder/'methods'/job['methods'][0]),job['pair']))
    state('ALL4_OPTIMIZED_NATIVE_ACTUAL_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    correction=json.loads((ROOT/'docs/a100_visual_classical_20261121/native_cost_protocol_correction.json').read_text())
    abandoned={}
    for pair in ('C4','C1'):
        folder=Path(correction['cancelled_KD'][pair]['run_root']);receipt=folder/'worker_receipt.json'
        if receipt.is_file():abandoned[pair]=json.loads(receipt.read_text())
    atomic_json(OUT/'analysis/native_cold_official.json',dict(status='ALL4_NATIVE_OPTIMIZED_OFFICIAL_COMPLETE',results=results,
        all_four_predictions_sealed_before_GT=True,abandoned_unoptimized_KD_research_attempts=abandoned,
        unoptimized_research_attempts_not_charged_as_final_deployment=True,goal_status='ACTIVE'))
    state('NATIVE_OPTIMIZED_ACTUAL_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_visual_classical_native_cold_analyze.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','native_cold'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_NATIVE_OPTIMIZED_COLD_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
