"""Only the user-registered B1/B2/B3, conditional on the complete A gate."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.runtime import target_metadata
from scripts.active_run import PAIRS


def tier_jobs(tier):
    if tier=='B1':
        return [dict(pair=p,method=m,rho=.3) for m in ('RANDOM_ACQ','MINUS_DS','MINUS_ROUTE') for p in PAIRS]+[
            dict(pair=p,method=m,rho=.3) for m in ('MINUS_GLANCE','MINUS_EB','MINUS_PRES') for p in ('C1','C6')]
    if tier=='B2':
        return [dict(pair=p,method=m,rho=r) for m in ('MINUS_SLOW','YESTRACK_BUDGET_PRIOR','UNIFORM_K')
                for r in (.1,1.) for p in PAIRS]
    if tier=='B3':
        return [dict(pair='C4',method=m,rho=.3) for m in ('FULL','SLOW_REPLAY')]
    raise ValueError(tier)


def prepare_tier(execution,tier):
    gatepath=execution.public/'stage_A_gate.json';gate=json.loads(gatepath.read_text())
    assert gate['result']=='PASS' and all(gate['checks'].values())
    jobs=tier_jobs(tier)
    units=json.loads((ROOT/'results/presence_active/redesign_v5/active_map_v1_critical28/remaining_budget_review.json').read_text())['target_unit_costs']
    dimensions=json.loads((execution.public/'BC_cost_estimate.json').read_text())['dimensions']
    costs=[]
    for job in jobs:
        pair=job['pair'];target=PAIRS[pair][1];meta=target_metadata(target);calls=0
        for video,bank in meta['videos'].items():
            _,counts=np.unique(np.asarray(bank['observations'])[:,1],return_counts=True)
            nq=sum(q['video_id']==video for q in meta['episodes'])
            calls+=int(np.floor(job['rho']*nq*int((counts>=3).sum())))
        seconds=calls*units[target]['seconds_per_teacher_call']
        if tier=='B3':
            bounds=json.loads((ROOT/'results/presence_active/redesign_v6/budget_preflight.json').read_text())['estimates']
            b=next(b for b in bounds if b['pair']==pair and b['method']=='FULL')
            slow=b['LoRA_GPU_hours']+b['full_frame_score_GPU_hours']+b['sample32_score_GPU_hours']
        else:slow=0
        costs.append(dict(job,logical_teacher_budget_upper_bound=calls,
            total_GPU_hours_estimate=slow+(0 if job['method']=='SLOW_REPLAY' else seconds/3600)))
    costpath=execution.public/('BC_cost_'+tier+'.json')
    atomic_json(costpath,dict(tier=tier,jobs=costs,
        total_GPU_hours_estimate=sum(c['total_GPU_hours_estimate'] for c in costs),
        conservative_no_cache_discount=True,maximum_total_project_GPUs=2))
    manifest=dict(execution.manifest,allowed_registered_jobs=jobs,registered_tier='V8P_'+tier,
        stage_A_gate_path=str(gatepath),stage_A_gate_sha256=file_sha(gatepath))
    if tier=='B3':
        # New caches only; historical raw images/checkpoints are never moved.
        cache=Path('/data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v8p')
        required=2*dimensions['C4']['candidate_query_rows']*4*769+(4<<30)
        if shutil.disk_usage(cache.parent).free<required:
            atomic_json(execution.public/'B3_deferred.json',dict(status='DEFERRED_STORAGE',
                required_bytes=required,available_bytes=shutil.disk_usage(cache.parent).free,
                no_GPU_or_teacher_calls=True))
            return None,None
        manifest['frame_cache_root']=str(cache)
    manifestpath=execution.public/('execution_'+tier+'.json')
    atomic_json(manifestpath,manifest)
    planpath=execution.public/('registered_jobs_'+tier+'.json')
    atomic_json(planpath,dict(tier='V8P_'+tier,jobs=jobs,requires_idle_GPU=tier=='B3',
        prereg_path=manifest['prereg_path'],prereg_sha256=manifest['prereg_sha256'],
        cost_estimate_path=str(costpath),cost_estimate_sha256=file_sha(costpath),
        stage_A_gate_sha256=file_sha(gatepath),maximum_GPUs=2))
    return manifestpath,planpath


def run_conditional(execution):
    gate=json.loads((execution.public/'stage_A_gate.json').read_text())
    assert gate['result']=='PASS'
    for tier in ('B1','B2','B3'):
        if tier=='B3':
            from scripts.select_gpu import query
            waiting=False
            while True:
                audit=query()
                original=Path(execution.manifest['prior_v8_durable_completion'])
                if not original.exists():
                    indices={g['index'] for g in json.loads(Path(execution.manifest['prior_v8_admission_path']).read_text())['selected']}
                    eligible=[g for g in audit if g['index'] in indices]
                else:eligible=audit
                atomic_json(execution.public/'B3_idle_admission.json',dict(audit=audit,
                    only_genuinely_idle_GPU=True,foreign_processes_stopped=0,
                    original_v8_still_running=not original.exists()))
                if any(g['compute_processes']==0 and g['free_mib']>=43008 for g in eligible):break
                if not waiting:
                    print(json.dumps(dict(event='V8P_B3_WAITING_GENUINELY_IDLE_GPU',
                        maximum_total_project_GPUs=2,original_queue_unmodified=True)),flush=True)
                    waiting=True
                # Internal resource wait in the same durable command, not an
                # agent polling loop or a silent decision to drop B3.
                time.sleep(30)
        manifest,plan=prepare_tier(execution,tier)
        if manifest is None:break
        env=dict(os.environ,PRESENCE_EXECUTION_MANIFEST=str(manifest),CUDA_VISIBLE_DEVICES='')
        env.pop('PRESENCE_REUSE_EVALUATIONS_FROM',None)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-tier',str(plan)],
                       cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/active_v8p_stage_b_report.py'),'--tier',tier],
                       cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
    atomic_json(execution.public/'stage_B_completion.json',dict(status='REGISTERED_AVAILABLE_B_TIERS_CLOSED',
        B1=(execution.public/'B1_summary.csv').exists(),B2=(execution.public/'B2_summary.csv').exists(),
        B3=(execution.public/'B3_summary.csv').exists(),
        B3_deferred=json.loads((execution.public/'B3_deferred.json').read_text()) if (execution.public/'B3_deferred.json').exists() else None,
        old80_launched=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-tier',type=Path,required=True);a=p.parse_args()
    from flex_uda.active.execution import authorize_active
    from scripts.active_v8p_queue import run
    execution,_=authorize_active()
    assert file_sha(execution.manifest['stage_A_gate_path'])==execution.manifest['stage_A_gate_sha256']
    assert json.loads(Path(execution.manifest['stage_A_gate_path']).read_text())['result']=='PASS'
    run(execution,a.run_tier)
    from scripts.active_v8p_validate import validate
    validate(execution,a.run_tier)
