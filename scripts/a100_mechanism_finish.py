"""One blocking post-seal pipeline; waits happen inside this supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.a100.mechanism import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def wait_for(relative):
    while not (OUT/relative).is_file():
        paths=[OUT/'evidence/supervisor_receipt.json',*(OUT/'cpu_supervisor').glob('*.json')]
        for path in paths:
            if path.is_file() and json.loads(path.read_text()).get('status')=='FAILED':
                raise RuntimeError('Producer failed; inspect '+str(path))
        time.sleep(5)


def run(script,args,name):
    folder=OUT/'post_supervisor';folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/(name+'.json')
    if receipt.is_file() and json.loads(receipt.read_text()).get('status')=='COMPLETE':return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts'/script),*args]
    begin=time.perf_counter();logpath=folder/f'{name}_attempt{len(list(folder.glob(name+"_attempt*.log"))):02d}.log'
    print(json.dumps(dict(event='POST_START',job=name)),flush=True)
    with logpath.open('w') as log:code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(status='COMPLETE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.perf_counter()-begin,command=command,log=str(logpath))
    atomic_json(receipt,result);print(json.dumps(dict(event='POST_END',job=name,**result)),flush=True)
    if code:raise RuntimeError('Post-seal child failed; inspect '+str(logpath))


if __name__=='__main__':
    wait_for('mechanism/bank_decisions.json')
    run('a100_mechanism_analyze.py',['--phase','bank'],'bank_analysis')
    wait_for('cpu_supervisor/completion.json')
    run('a100_mechanism_analyze.py',['--phase','official'],'official_analysis')
    wait_for('profile/supervisor_receipt.json')
    run('a100_mechanism_analyze.py',['--phase','costs'],'cost_analysis')
    run('a100_mechanism_report.py',[],'report')
    atomic_json(OUT/'post_supervisor/completion.json',dict(status='ALL_ANALYSES_AND_REPORT_COMPLETE'))
