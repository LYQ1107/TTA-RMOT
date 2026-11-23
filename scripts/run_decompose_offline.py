"""Blocking CPU parent: independent oracle children, then official evaluation."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import RUN,require_plan,atomic_json


def main(combos):
    require_plan();out=RUN/'offline_launchers'/str(time.time_ns());out.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def child(tag,args):
        start=time.perf_counter()
        with (out/(tag+'.log')).open('w') as log:
            p=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in p.stdout:
                log.write(line);log.flush()
                if line.startswith('{"event":') or 'Traceback' in line or 'Error:' in line:print(line.rstrip(),flush=True)
            code=p.wait()
        r=dict(tag=tag,pid=p.pid,exit_code=code,wall_seconds=time.perf_counter()-start,log=str(out/(tag+'.log')))
        atomic_json(out/(tag+'.json'),r)
        if code:raise RuntimeError('Offline stage stopped; preserve completed units and inspect '+r['log'])
        return r
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(child,'oracle_'+c,[ROOT/'scripts/decompose_oracle.py','--combo',c]) for c in combos]):records.append(f.result())
    records.append(child('official',[ROOT/'scripts/evaluate_decompose.py','--combos',*combos]))
    atomic_json(out/'completion.json',dict(status='COMPLETE',workers=records))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--combos',nargs='+',required=True);a=p.parse_args();main(a.combos)
