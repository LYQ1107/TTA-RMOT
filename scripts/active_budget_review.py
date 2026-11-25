"""Post-stage, GT-free cost review; never changes the registered cost/gate."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_plan import jobs,dimensions
from scripts.active_run import PAIRS


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--durable-dir',type=Path,required=True);a=p.parse_args()
    execution,_=authorize_active();public=execution.public
    end=json.loads((a.durable_dir/'completion.json').read_text())
    if end['exit_code']:raise ValueError('Cost review requires the completed tier')
    plan=json.loads((public/'registered_jobs.json').read_text())
    with (public/'costs.csv').open() as stream:costs=list(csv.DictReader(stream))
    allowed={(r['pair'],r['method'],r['rho']) for r in plan['jobs']}
    cost_rows=[r for r in costs if r['rho'] and (r['pair'],r['method'],float(r['rho'])) in allowed]
    sources=[];units=[]
    for pair in ('C1','C3'):
        folder=execution.active/pair/'FULL/rho_0.3';seal=verify_predictions(folder)
        receipts=[json.loads(path.read_text()) for path in (folder/'videos').glob('*/receipt.json')]
        updates=sum(r['slow']['optimizer_updates'] for v in receipts for r in v['rounds'])
        work=sum(v['rescore_candidate_query_rows'] for v in receipts)
        units.append(dict(seconds_per_update=seal['LoRA_GPU_seconds']/updates,
            seconds_per_rescore_candidate_query=seal['rescore_GPU_seconds']/work,
            seconds_per_teacher_call=sum(v['teacher_delta']['equivalent_forward_seconds'] for v in receipts)/seal['teacher']['logical_calls']))
        sources.append(dict(path=str(folder/'predictions_seal.json'),sha256=file_sha(folder/'predictions_seal.json'),
            successful_optimizer_updates=updates,rescore_candidate_query_rows=work))
    unit={key:max(row[key] for row in units) for key in units[0]}
    unit['seconds_per_CROP_call']=max(float(r['committed_teacher_equivalent_forward_GPU_hours'])*3600/int(r['teacher_calls'])
        for r in cost_rows if r['method'].startswith('YESTRACK_BUDGET'))
    estimates=[]
    for job in jobs():
        ds=dimensions(PAIRS[job['pair']][1]);method=job['method'];rho=job['rho']
        calls=(sum(d['yestrack_full_d10_calls'] for d in ds) if method=='YESTRACK_FULL_D10' else
            sum(int(__import__('numpy').floor(rho*d['eligible_tracks']*d['queries'])) for d in ds))
        no_slow=method.startswith('YESTRACK') or method in ('UNIFORM_K','MINUS_SLOW')
        teacher=calls*unit['seconds_per_CROP_call' if method.startswith('YESTRACK') else 'seconds_per_teacher_call']/3600
        slow=0. if no_slow else 4*50*len(ds)*unit['seconds_per_update']/3600
        rescore=0. if no_slow else 5*sum(d['rescore_rows'] for d in ds)*unit['seconds_per_rescore_candidate_query']/3600
        estimates.append(dict(job,teacher_GPU_hours=teacher,LoRA_GPU_hours=slow,rescore_GPU_hours=rescore,
            total_GPU_hours_estimate=teacher+slow+rescore,completed_in_current_tier=(job['pair'],method,rho) in allowed))
    critical=[r for r in estimates if r['rho']==.3 and r['method'] in
        ('FULL','MINUS_SLOW','UNIFORM_K','YESTRACK_BUDGET_RANDOM','YESTRACK_BUDGET_PRIOR')]
    initial=json.loads((public/'execution_started.json').read_text())
    start_ns=int(Path(initial['coordinator']).name.split('_')[-1])
    original=json.loads((public/'BC_cost_estimate.json').read_text())
    result=dict(status='COMPLETED_TIER_REVIEW_NOT_NEW_RUN_AUTHORIZATION',full108_goal_not_complete=True,
        registered_original_full_matrix_GPU_hours_estimate=original['total_GPU_hours_estimate'],
        updated_full108_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in estimates),
        updated_remaining98_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in estimates if not r['completed_in_current_tier']),
        original_matrix_main_critical_subset_jobs=len(critical),
        main_critical_subset_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in critical),
        additional_main_critical_subset_jobs=sum(not r['completed_in_current_tier'] for r in critical),
        additional_main_critical_subset_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in critical if not r['completed_in_current_tier']),
        observed_tier_worker_wall_hours=sum(float(r['assigned_worker_wall_hours']) for r in cost_rows),
        observed_tier_worker_wall_is_upper_bound=any(r['worker_wall_is_upper_bound']=='True' for r in cost_rows),
        observed_tier_new_teacher_forward_hours=sum(float(r['teacher_forward_GPU_hours']) for r in cost_rows),
        observed_tier_successful_lineage_LoRA_hours=sum(float(r['LoRA_GPU_hours']) for r in cost_rows),
        observed_tier_successful_lineage_rescore_hours=sum(float(r['rescore_GPU_hours']) for r in cost_rows),
        continuation_start_to_final_manager_wall_hours=(end['finished_time_ns']-start_ns)/3.6e12,
        wall_note='includes downtime/CPU reporting; worker ledger also contains earlier reused first-video pilots, so their ratio is not a measured parallel speedup',
        unit_costs=unit,sources=sources,costs_csv_sha256=file_sha(public/'costs.csv'),
        original_estimate_sha256=file_sha(public/'BC_cost_estimate.json'),
        limits=['no KITTI unit benchmark; maximum measured C1/C3 units substituted',
            'all4 rounds execute50 steps and5 rescores per video; no skip/cache discounts',
            'teacher equivalence is origin-batch average allocation, not isolated-call timing; mixed views/text lengths/shared GPU affect it',
            'future idle failures/loading overhead are not predicted; actual interrupted/idle work remains in the worker ledger',
            '28-job subset contains only already registered jobs; no new C2/C5 MINUS_SLOW is added',
            'this review does not replace or unlock the original40h gate or ten-job manifest'],
        newly_authorized_GPU_jobs=0,model_teacher_or_GT_calls_by_estimator=0,estimates=estimates)
    atomic_json(public/'measured_budget_review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='estimates'},indent=2))


if __name__=='__main__':main()
