"""Four-GPU causal strong-baseline check; groups have independent frozen protocols."""
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
    jobs={};logs=[];evaluated=set();begin=time.perf_counter()
    def launch(name,args,gpu=None):
        path=OUT/'supervisor'/('recovery_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        p=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[name]=dict(process=p,pid=p.pid,gpu=gpu,log=str(path))
    for policy,pair,gpu in [('COVER','C4',0),('COVER','C1',1),('UNIFORM','C4',2),('UNIFORM','C1',3)]:
        launch('strong_'+policy+'_'+pair,['scripts/a100_visual_classical_strong_prefix.py','--pair',pair,'--policy',policy],gpu)
    while True:
        state=dict(status='RUNNING_STRONG_PREFIX_CONTROL',supervisor_pid=os.getpid(),wall_seconds=time.perf_counter()-begin,
            jobs={k:{a:b for a,b in v.items() if a!='process'}|dict(exit_code=v['process'].poll()) for k,v in jobs.items()})
        failed=[k for k,v in state['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:state['status']='FAILED_REQUIRES_ROOT_CAUSE_FIX'
        atomic_json(OUT/'supervisor/strong_state.json',state)
        if failed:raise RuntimeError(str(failed))
        for policy in ('COVER','UNIFORM'):
            if policy not in evaluated and all(jobs['strong_'+policy+'_'+p]['process'].poll()==0 for p in ('C4','C1')):
                launch('strong_'+policy+'_evaluate',['scripts/a100_visual_classical_evaluate_group.py','--group',
                    'strong_'+policy.lower()+'_seed0','--primary','COHORT','--workers','4']);evaluated.add(policy)
        if len(evaluated)==2 and all(v['process'].poll()==0 for v in jobs.values()):break
        time.sleep(15)
    state['status']='PASS_ALL_STRONG_PREFIX_GROUPS';atomic_json(OUT/'supervisor/strong_state.json',state)
    for log in logs:log.close()


if __name__=='__main__':main()
