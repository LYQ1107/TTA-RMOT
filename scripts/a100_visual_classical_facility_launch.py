"""Persistent real-teacher collection, causal inference and isolated evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_prepare import OUT
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    jobs={};logs=[];state=OUT/'supervisor/facility_state.json';begin=time.perf_counter()
    def start(name,script,args,gpu=None):
        path=OUT/'supervisor'/('recovery_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        p=subprocess.Popen([sys.executable,'scripts/'+script,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[name]=dict(process=p,pid=p.pid,log=str(path),gpu=gpu)
    def snapshot(status):
        data=dict(status=status,supervisor_pid=os.getpid(),wall_seconds=time.perf_counter()-begin,
            jobs={k:{a:b for a,b in v.items() if a!='process'}|dict(exit_code=v['process'].poll()) for k,v in jobs.items()})
        atomic_json(state,data);return data
    for pair,tgpu,sgpu in [('C4',2,0),('C1',3,1)]:
        start('facility_collect_'+pair,'a100_visual_classical_facility_collect.py',['--pair',pair],tgpu)
        start('facility_prefix_'+pair,'a100_visual_classical_color_prefix.py',['--pair',pair,'--policies','FACILITY','--group','facility_prefix'],sgpu)
        start('facility_counts_'+pair,'a100_visual_classical_prefix_countmatched.py',['--pair',pair,'--family','facility'])
    evaluated=set()
    while True:
        data=snapshot('RUNNING_REAL_TEACHER_FACILITY_EXPERIMENT')
        failed=[k for k,v in data['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:
            snapshot('FAILED_REQUIRES_ROOT_CAUSE_FIX');raise RuntimeError(str(failed))
        for kind,group,primary in [('prefix','facility_prefix','FACILITY__COHORT'),('counts','facility_countmatched','FULL__AT_FULL_COUNT')]:
            if kind not in evaluated and all(jobs['facility_'+kind+'_'+p]['process'].poll()==0 for p in ('C4','C1')):
                start('facility_'+kind+'_evaluate','a100_visual_classical_evaluate_group.py',['--group',group,'--primary',primary,'--workers','4']);evaluated.add(kind)
        if len(evaluated)==2 and all(v['process'].poll()==0 for v in jobs.values()):break
        time.sleep(15)
    snapshot('PASS_FACILITY_FULL_PROTOCOL')
    for log in logs:log.close()


if __name__=='__main__':main()
