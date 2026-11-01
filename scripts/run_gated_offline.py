"""One blocking post-teacher CPU pipeline, streamed logs, no agent polling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *

def run():
    setup();require_plan();start=time.perf_counter();folder=RUN/'CPU_launchers'/str(time.time_ns());folder.mkdir(parents=True);jobs=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    try:
        # This is the single blocking waiter, not repeated model shell probes.
        while not (RUN/'teacher_launch_completion.json').exists():
            launches=sorted((RUN/'teacher_launchers').iterdir())
            if launches:
                for name in ('smoke.json','full.json'):
                    p=launches[-1]/name
                    if p.exists() and json.loads(p.read_text())['exit_code']!=0:raise RuntimeError('teacher failed; preserve cache and inspect '+str(p))
            time.sleep(5)
        for stage,script in [('generate','gated_decode.py'),('official','evaluate_gated_decode.py'),('report','report_gated_decode.py')]:
            begin=time.perf_counter();log=folder/(stage+'.txt')
            with log.open('w') as h:
                p=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in p.stdout:h.write(line);h.flush();print(stage+': '+line,end='',flush=True)
                code=p.wait()
            r=dict(stage=stage,pid=p.pid,exit_code=code,wall_seconds=time.perf_counter()-begin,log=str(log));jobs.append(r);atomic_json(folder/(stage+'.json'),r)
            if code:raise RuntimeError('CPU phase failed; preserve completed work; see '+str(log))
    finally:atomic_json(folder/'launcher_status.json',dict(status='FAILED' if sys.exc_info()[0] else 'COMPLETE',jobs=jobs,wall_seconds=time.perf_counter()-start))
    atomic_json(PUBLIC/'CPU_pipeline_costs.json',dict(status='COMPLETE',jobs=jobs,wall_seconds=time.perf_counter()-start,GPUs=0))

if __name__=='__main__':run()
