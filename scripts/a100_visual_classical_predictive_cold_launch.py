"""Conditional real inference follow-up, preserving optimized physical KD."""
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

PUBLIC=ROOT/'docs/a100_visual_classical_20261121'
PLAN=PUBLIC/'preregistration_predictive_topology_cold_v1.json'
STATE=OUT/'supervisor/predictive_cold_state.json'
SUCCESS='PASS_PREDICTIVE_TOPOLOGY_ACTUAL_COST_RESEARCH_REVIEW_REQUIRED'


def promotion_checks(quality):
    checks={}
    for pair in ('C4','C1'):
        row=quality['repeated']['PREDICTIVE_'+pair]['comparisons'];e=row['SELECTED_JOINT_vs_KD']
        checks[pair+'_beats_native_KD']=all(v>0 for v in e['per_neural_seed_delta']) and e['parent_and_neural_seed_sensitivity_CI95'][0]>0
        local=row['SELECTED_JOINT_vs_LOCAL_JOINT']['parent_and_neural_seed_sensitivity_CI95']
        global_=row['SELECTED_JOINT_vs_GLOBAL_JOINT']['parent_and_neural_seed_sensitivity_CI95']
        checks[pair+'_local_control']=local[0]>0 if pair=='C1' else local[1]>=0
        checks[pair+'_global_control']=global_[0]>0 if pair=='C4' else global_[1]>=0
    return checks


def main():
    for stem in ('predictive_topology','native_cold'):
        assert json.loads((PUBLIC/(stem+'_statistical_reproduction.json')).read_text())['status'].startswith('PASS_')
    quality=json.loads((PUBLIC/'predictive_topology_analysis.json').read_text());checks=promotion_checks(quality)
    assert all(checks.values()),('Registered promotion criteria not met; no new GPU worker',checks)
    plan=json.loads(PLAN.read_text());jobs=plan['jobs'];begin=time.perf_counter();active={};done={};failure=None;last=0
    for job in jobs.values():assert file_sha(ROOT/job['worker_script'])==job['worker_sha256']
    for name,path in dict(runtime='flex_uda/active/appearance_runtime.py',calibration='flex_uda/active/semantic_colour.py',graph='flex_uda/active/grounded_graph.py',selector='flex_uda/active/topology_selection.py').items():
        assert file_sha(ROOT/path)==plan['code_sha256'][name]
    for key,job in jobs.items():
        if job['reuse_previous_fresh_KD']:
            root=OUT/'cold_full_c4'/job['run_name'];r=json.loads((root/'completion.json').read_text());w=json.loads((root/'worker_receipt.json').read_text())
            assert w['exit_code']==r['GT_open_attempts']==r['prepaid_artifact_reads']==0 and r['single_uninterrupted_attempt']
            assert r['teacher']['new_calls']==job['expected_questions'] and r['teacher']['cache_hits']==0
            done[key]=dict(reused_fresh_physical_baseline=True,exit_code=0)
    def state(status):
        arms={}
        for key,job in jobs.items():
            root=OUT/'cold_full_c4'/job['run_name'];ledger=root/'phase_ledger.json'
            phases=json.loads(ledger.read_text()).get('common_phases',[]) if ledger.is_file() else []
            arms[key]=dict(physical_GPU=job['physical_GPU'],completed_videos=len(list((root/'source').glob('*/receipt.json'))),total_videos=45 if job['pair']=='C4' else 25,
                reused_physical_baseline=job['reuse_previous_fresh_KD'],sealed=key in done,last_phase=phases[-1]['phase'] if phases else None)
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),arms=arms,failure=failure,quality_promotion_checks=checks,
            wall_seconds=time.perf_counter()-begin,preregistration_sha256=file_sha(PLAN),goal_status='ACTIVE'))
    pending=[key for key,job in jobs.items() if not job['reuse_previous_fresh_KD']]
    state('WAITING_RELEASED_GPU1_3_FOR_FRESH_SELECTED_TOPOLOGY')
    while pending or active:
        changed=False
        for key,(process,handle,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();done[key]=dict(exit_code=rc,reused_fresh_physical_baseline=False);del active[key];changed=True
            if rc:failure=dict(job=key,exit_code=rc,last_log=log.read_text()[-7000:])
        if pending and not failure:
            ids=','.join(str(jobs[k]['physical_GPU']) for k in pending)
            rows=subprocess.check_output(['nvidia-smi','-i',ids,'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.strip().splitlines()}
            for key in list(pending):
                job=jobs[key]
                if memory[job['physical_GPU']]>=512:continue
                log=OUT/'supervisor'/('predictive_cold_'+key+'.log');handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_budget_cold_launch.py','--pair',job['pair'],'--scope','extended',
                    '--plan-file',str(PLAN),'--job-key',key],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[key]=process,handle,log;pending.remove(key);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_TWO_FRESH_SELECTED_ARMS_PLUS_TWO_TRUE_OPTIMIZED_KD_RECORDS' if not failure else 'PRESERVING_OTHER_SELECTED_ARM_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    tasks=[]
    for key,job in jobs.items():
        root=OUT/'cold_full_c4'/job['run_name'];r=json.loads((root/'completion.json').read_text())
        assert r['GT_open_attempts']==r['prepaid_artifact_reads']==r['teacher']['cache_hits']==0 and r['single_uninterrupted_attempt']
        assert r['teacher']['new_calls']==r['logical_teacher_calls']==job['expected_questions']
        if not job['reuse_previous_fresh_KD']:assert r['protocol']['graph_topology_policy']=='teacher_TrackID_crossfit_predictive_LOCAL_GLOBAL'
        tasks.append((str(root/'methods'/job['methods'][0]),job['pair']))
    state('BOTH_NEW_SELECTED_PREDICTIONS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'analysis/predictive_cold_official.json',dict(status='PREDICTIVE_COLD_OFFICIAL_COMPLETE',results=results,
        both_new_selected_sealed_before_followup_GT=True,previous_KD_already_sealed_and_evaluated_separately=True,goal_status='ACTIVE'))
    state('PREDICTIVE_ACTUAL_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_visual_classical_predictive_cold_analyze.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','predictive_cold'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_PREDICTIVE_COLD_REQUIRES_ROOT_CAUSE_OR_UNMET_PROMOTION',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
