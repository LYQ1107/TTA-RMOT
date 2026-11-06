"""One CPU blocking queue; all GT diagnostics are separate post-seal children."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.active_v9_inputs import PUBLIC,PAIRS,DATA
from flex_tta.runtime.pf_receipts import atomic_json


def run_one(method):
    log=DATA/(method+'_queue.log');log.parent.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    with log.open('a') as handle:
        child=subprocess.Popen([sys.executable,str(ROOT/'scripts/active_v9_replay.py'),'--estimator',method],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(json.dumps(dict(event='V9_CPU_ESTIMATOR_STARTED',estimator=method,pid=child.pid,log=str(log))),flush=True)
        for line in child.stdout:
            handle.write(line);handle.flush();print(line,end='',flush=True)
        code=child.wait()
    receipt=dict(estimator=method,exit_code=code,wall_seconds=time.perf_counter()-begin,log=str(log))
    atomic_json(DATA/(method+'_queue_completion.json'),receipt)
    if code:raise RuntimeError('CPU estimator failed; preserve sealed units: '+str(log))
    return receipt


def main():
    begin=time.perf_counter()
    assert all((DATA/'E0'/p/'estimates_seal.json').exists() for p in PAIRS)
    with ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(run_one,['E1','E3']))
    run_one('E2')  # fixed D3 NOT_APPLICABLE; no fitting/teacher invocation
    for script in ('active_v9_diagnostics.py','active_v9_report.py'):
        subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
    result=dict(status='V9_OFFLINE_COMPLETE',CPU_receipts=receipts,wall_seconds=time.perf_counter()-begin,
        GPU_calls=0,teacher_calls=0,stage3_untouched=True,stage4_launched=False)
    atomic_json(PUBLIC/'queue_completion.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
