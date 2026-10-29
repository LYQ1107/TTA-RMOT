#!/usr/bin/env python3
"""One blocking CPU completion pipeline after every registered GPU lane exits.

Internal receipt/process waiting is intentional. Do not replace it with agent
polling; unfinished/healthy training is never stopped by this process.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.run_psr_matrix import DEFAULT


def finish(run):
    run=Path(run).resolve();evaluation=run/'evaluation/merged';public=ROOT/'results/psr/full'
    output=run/'cpu_pipeline';output.mkdir(exist_ok=True);last=None
    while True:
        complete=[]
        for method in DEFAULT:
            p=evaluation/method/'full_audit.json'
            if p.exists():
                audit=json.loads(p.read_text())
                if audit['status']=='PASS' and audit['valid_episodes']==425 and audit['unique_videos']==25:complete.append(method)
        launchers=list((run/'matrix_launchers').glob('*/gpu_admission.json'))
        active=[p.parent.name for p in launchers if not (p.parent/'launcher_status.json').exists()]
        current=(tuple(complete),tuple(active))
        if current!=last:
            print(json.dumps(dict(event='PSR_CPU_PIPELINE_WAIT',sealed_methods=complete,required_methods=len(DEFAULT),active_launchers=len(active))),flush=True);last=current
        if len(complete)==len(DEFAULT) and not active:break
        time.sleep(20)
    stages=[
        ('controls',['prepare_psr_controls.py','--evaluation-root',str(evaluation),'--training-root',str(run/'students')]),
        ('official',['evaluate_mllm_trackeval.py','--output-root',str(evaluation),'--max-jobs','4']),
        ('analysis',['analyze_psr_matrix.py','--evaluation-root',str(evaluation),'--training-root',str(run/'students'),'--output',str(public)]),
        ('tables',['render_psr_tables.py','--public-root',str(public)]),
        ('export',['export_psr_results.py','--run-root',str(run),'--output',str(public)])]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    records=[]
    for name,args in stages:
        folder=output/name;folder.mkdir(exist_ok=True);receipt=folder/'completion.json'
        if receipt.exists() and json.loads(receipt.read_text())['status']=='COMPLETE':
            records.append(json.loads(receipt.read_text()));print(json.dumps(dict(event='PSR_CPU_PHASE_PRESERVED',phase=name)),flush=True);continue
        start=time.perf_counter();print(json.dumps(dict(event='PSR_CPU_PHASE_STARTED',phase=name)),flush=True)
        with (folder/'stdout.txt').open('a') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:
                log.write(line);log.flush()
                if line.startswith('{"event":'):print(line.rstrip(),flush=True)
            code=child.wait()
        result=dict(phase=name,status='COMPLETE' if code==0 else 'FAILED_PRESERVED',exit_code=code,seconds=time.perf_counter()-start,
            GPU_calls=0,stdout=str(folder/'stdout.txt'))
        atomic_json(receipt,result);records.append(result);print(json.dumps(dict(event='PSR_CPU_PHASE_FINISHED',**result)),flush=True)
        if code:raise RuntimeError('fix only failed CPU phase and resume this same pipeline; no GPU reruns: '+name)
    launch_history={p.parent.name:json.loads((p.parent/'launcher_status.json').read_text()) for p in launchers}
    result=dict(status='COMPLETE_PRIMARY_AND_MATCHED_CONTROLS',student_methods=DEFAULT,episodes_each=425,videos=25,
        protocol='USER_AUTHORIZED25VAL_SAME_SET_NOT_TRAIN_TO_VAL',stages=records,launcher_history=launch_history,
        recovered_resource_failures_retained=True,target_GT_used_only_in_independent_CPU_evaluation=True,
        scientific_success='NOT_IMPLIED_BY_PIPELINE_COMPLETION',conditional_multiseed_confirmation='decide only from registered main comparison; no threshold changes')
    atomic_json(output/'completion.json',result);atomic_json(public/'pipeline_completion.json',result)
    print(json.dumps(dict(event='PSR_COMPLETE_CPU_PIPELINE',student_methods=len(DEFAULT))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();finish(a.run_root)
