"""One blocking CPU pipeline after the single source-GPU job completes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *

def run():
    setup();require_plan()
    if not (RUN/'source/scores_seal.json').exists():raise ValueError('wait for complete source scoring, not partial arrays')
    folder=RUN/'CPU_launchers'/str(time.time_ns());folder.mkdir(parents=True);start=time.perf_counter();records=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
    def child(name,args):
        begin=time.perf_counter();log=folder/(name+'.txt')
        with log.open('w') as h:
            p=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in p.stdout:h.write(line);h.flush();print(name+': '+line,end='',flush=True)
            code=p.wait()
        result=dict(stage=name,pid=p.pid,exit_code=code,wall_seconds=time.perf_counter()-begin,stdout=str(log))
        records.append(result);atomic_json(folder/(name+'.json'),result)
        if code:raise RuntimeError('CPU phase failed; preserve complete methods: '+str(log))
        return result
    try:
        child('thresholds',[ROOT/'scripts/xquery_source.py','thresholds'])
        child('target_predictions',[ROOT/'scripts/xquery_decode.py'])
        child('source_predictions',[ROOT/'scripts/xquery_source_boundary.py'])
        # All target predictions and teacher combos are now sealed. GT is
        # accessed only in these independent subprocesses, never in decoders.
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs=[pool.submit(child,'target_official',[ROOT/'scripts/evaluate_xquery_decode.py']),
                  pool.submit(child,'source_official',[ROOT/'scripts/xquery_source_boundary.py','--evaluate']),
                  pool.submit(child,'teacher_offline',[ROOT/'scripts/xquery_teacher_combo.py','--evaluate'])]
            for future in as_completed(jobs):future.result()
        child('report',[ROOT/'scripts/report_xquery_decode.py'])
    finally:
        atomic_json(folder/'launcher_status.json',dict(status='FAILED' if sys.exc_info()[0] else 'COMPLETE',jobs=records,
            actual_wall_seconds=time.perf_counter()-start,GPUs_used=0))
    atomic_json(PUBLIC/'CPU_pipeline_costs.json',dict(status='COMPLETE',jobs=records,actual_wall_seconds=time.perf_counter()-start,
        GPU_calls=0,new_teacher_calls=0,student_updates=0))

if __name__=='__main__':run()
