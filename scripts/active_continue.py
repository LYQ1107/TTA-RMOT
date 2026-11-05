"""One blocking main/critical queue with independent post-seal CPU evaluation."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def folder_for(execution,job):
    return execution.active / job['pair'] / job['method'] / ('rho_'+str(job['rho']))


def prepare(execution):
    cost_path=execution.public / 'BC_cost_estimate.json'
    cost=json.loads(cost_path.read_text())
    jobs=execution.manifest['allowed_registered_jobs']
    selected=[row for row in cost['jobs'] if {k:row[k] for k in ('pair','method','rho')} in jobs]
    if len(selected)!=len(jobs):raise ValueError('Cost estimate does not cover the fixed tier')
    pilots=[]
    for pair in ('C1','C3'):
        folder=folder_for(execution,dict(pair=pair,method='FULL',rho=.3))
        receipt=folder / 'pilot_receipts/first_001.json'
        record=json.loads(receipt.read_text())
        if record['GT_open_attempts'] or len(record['videos'])!=1:
            raise ValueError('Resume requires the sealed GT-free first video')
        checkpoint=folder / 'stream_000.pt'
        pilots.append(dict(pair=pair,pilot_receipt_path=str(receipt),pilot_receipt_sha256=file_sha(receipt),
            checkpoint_path=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),next_video=1))
    plan=dict(tier='MAIN_AND_CRITICAL_ONLY',jobs=jobs,
        cost_estimate_sha256=file_sha(cost_path),
        selected_total_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in selected),
        full_matrix_GPU_hours_estimate=cost['total_GPU_hours_estimate'],
        estimate_is_conservative_no_cache_and_does_not_deduct_finished_first_videos=True,
        resumed_first_videos=pilots,prereg_path=execution.manifest['prereg_path'],
        prereg_sha256=execution.manifest['prereg_sha256'],maximum_GPUs=2,
        remaining_matrix_not_authorized=True)
    path=execution.public / 'registered_jobs.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Do not overwrite a fixed continuation plan')
    else:atomic_json(path,plan)
    return path,plan


def run(execution,plan_path):
    plan=json.loads(plan_path.read_text())
    if file_sha(plan['prereg_path'])!=plan['prereg_sha256']:
        raise ValueError('Changed continuation preregistration')
    run_root=execution.launchers / ('coordinator_'+str(time.time_ns()))
    run_root.mkdir(parents=True)
    atomic_json(run_root / 'plan.json',plan)
    launcher=execution.manifest.get('launcher_script','scripts/active_launch.py') if execution.manifest else 'scripts/active_launch.py'
    launch=subprocess.Popen([sys.executable,str(ROOT / launcher),
        '--phase','registered','--jobs-json',str(plan_path)],cwd=ROOT)
    print(json.dumps(dict(event='CONTINUATION_STARTED',coordinator=str(run_root),
        jobs=len(plan['jobs']),launch_pid=launch.pid)),flush=True)
    jobs=plan['jobs'];evaluated={};evaluation=None;errors=[]
    from scripts.active_run import PAIRS
    while True:
        if evaluation is not None and evaluation['process'].poll() is not None:
            code=evaluation['process'].wait();evaluation['stream'].close()
            job=evaluation['job'];key=(job['pair'],job['method'],job['rho'])
            row=dict(job,exit_code=code,log=str(evaluation['log']))
            evaluated[key]=row
            if code:errors.append(row)
            else:
                row['metrics']=json.loads((folder_for(execution,job) / 'offline_summary.json').read_text())
            atomic_json(run_root / ('evaluation_'+job['pair']+'_'+job['method']+'.json'),row)
            print(json.dumps(dict(event='POST_SEAL_EVALUATION_FINISHED',**row)),flush=True)
            evaluation=None
        ready=[job for job in jobs if (job['pair'],job['method'],job['rho']) not in evaluated
            and (folder_for(execution,job) / 'predictions_seal.json').exists()]
        if evaluation is None and ready:
            job=ready[0];target=PAIRS[job['pair']][1]
            command=[sys.executable,str(ROOT / 'scripts/active_eval.py'),'--folder',str(folder_for(execution,job)),
                '--target',target]
            if target=='SportsMOT':command.append('--sports418')
            log=run_root / ('evaluation_'+job['pair']+'_'+job['method']+'.log')
            stream=log.open('x')
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                stdout=stream,stderr=subprocess.STDOUT)
            evaluation=dict(job=job,process=process,stream=stream,log=log)
            print(json.dumps(dict(event='POST_SEAL_EVALUATION_STARTED',job=job,log=str(log))),flush=True)
        if launch.poll() is not None and evaluation is None and not ready:break
        # Internal waiting only. The agent drains this single command; it does
        # not independently poll GPU processes or experimental files.
        time.sleep(2)
    launch_code=launch.wait()
    completion=dict(status='COMPLETE' if not launch_code and not errors and len(evaluated)==len(jobs) else 'FAILED',
        launcher_exit_code=launch_code,evaluated_jobs=list(evaluated.values()),evaluation_errors=errors,
        expected_jobs=len(jobs),full108_matrix_complete=False,
        no_other_processes_stopped=True)
    atomic_json(run_root / 'completion.json',completion)
    if completion['status']!='COMPLETE':
        raise RuntimeError('Preserve completed streams; inspect '+str(run_root / 'completion.json'))
    report_script=execution.manifest.get('report_script','scripts/report_active.py') if execution.manifest else 'scripts/report_active.py'
    subprocess.run([sys.executable,str(ROOT / report_script),'--partial'],cwd=ROOT,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    print(json.dumps(dict(event='MAIN_CRITICAL_COMPLETE',coordinator=str(run_root),
        report=str(execution.public / 'REPORT.md'),jobs=len(jobs))),flush=True)


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--jobs-json',type=Path);a=p.parse_args()
    execution,_=authorize_active()
    if a.prepare:
        path,plan=prepare(execution)
        print(json.dumps(dict(path=str(path),sha256=file_sha(path),jobs=len(plan['jobs']),
            selected_GPU_hours_estimate=plan['selected_total_GPU_hours_estimate']),indent=2),flush=True)
    else:
        if not a.jobs_json:p.error('Supply the published --jobs-json plan')
        run(execution,a.jobs_json)


if __name__=='__main__':main()
