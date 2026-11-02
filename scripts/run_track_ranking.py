"""One blocking CPU parent, complete-video recovery; never polls GPUs/processes."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *


def main(phase):
    require_plan();start=time.perf_counter()
    dest=RUN/'launchers'/(phase+'_'+str(time.time_ns()));dest.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
    results=[]
    def child(tag,args):
        begin=time.perf_counter();log=dest/(tag+'.log')
        with log.open('w') as stream:
            p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in p.stdout:
                stream.write(line);stream.flush();print(line.rstrip(),flush=True)
            code=p.wait()
        r=dict(stage=tag,exit_code=code,wall_seconds=time.perf_counter()-begin,log=str(log))
        if code:
            dump(dest/(tag+'_failed.json'),r)
            raise RuntimeError('Preserve completed videos and fix first actionable failure: '+str(log))
        return r
    def save(r):
        results.append(r);dump(dest/'status.json',dict(phase=phase,stages=results))
    if phase=='metrics':
        if (RUN/'plan.json').exists():
            plan=json.loads((RUN/'plan.json').read_text())
            for r in plan['inputs']:checked(r)
            checked(plan['clarifications'])
        else:save(child('preflight',[ROOT/'scripts/prepare_track_ranking.py']))
        if not (RUN/'noise_rates.json').exists():save(child('noise_rates',[ROOT/'scripts/track_ranking_noise.py']))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(child,'target_'+t,[ROOT/'scripts/evaluate_track_ranking.py','--target',t]) for t in TARGETS]
            for f in as_completed(futures):save(f.result())
    else:
        names=(('summarize_track_ranking',) if phase=='aggregate' else
               ('validate_track_ranking','report_track_ranking') if phase=='report' else
               ('summarize_track_ranking','validate_track_ranking','report_track_ranking'))
        for name in names:
            save(child(name,[ROOT/'scripts'/(name+'.py')]))
    dump(dest/'completion.json',dict(status='COMPLETE',phase=phase,stages=results,wall_seconds=time.perf_counter()-start,
                                    new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--phase',choices=('metrics','summary','aggregate','report'),required=True);main(p.parse_args().phase)
