"""Fixed-recipe acquisition + optimization repeats, durable across chat turns."""
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
    jobs={};logs=[];begin=time.perf_counter();state_path=OUT/'supervisor/repeats_state.json'
    def launch(name,args,gpu=None):
        path=OUT/'supervisor'/('recovery_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        process=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[name]=dict(process=process,pid=process.pid,gpu=gpu,log=str(path))
    for pair,gpu in [('C4',2),('C1',3)]:launch('repeat_collect_'+pair,['scripts/a100_visual_classical_repeat_collect.py','--pair',pair],gpu)
    launched=set();evaluated=set()
    def done(name):return name in jobs and jobs[name]['process'].poll()==0
    while True:
        state=dict(status='RUNNING_FIXED_RECIPE_REPEATS',pid=os.getpid(),wall_seconds=time.perf_counter()-begin,
            jobs={k:{a:b for a,b in v.items() if a!='process'}|dict(exit_code=v['process'].poll()) for k,v in jobs.items()})
        failed=[k for k,v in state['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:state['status']='FAILED_REPEATS_REQUIRE_ROOT_CAUSE'
        atomic_json(state_path,state)
        if failed:raise RuntimeError(str(failed))
        for pair,seed,gpu in [('C4',17,0),('C4',42,2),('C1',17,3),('C1',42,0)]:
            name='repeat_fit_'+pair+'_'+str(seed)
            if name in launched or not done('repeat_collect_'+pair):continue
            if pair=='C1' and seed==42 and not done('repeat_fit_C4_17'):continue
            launch(name,['scripts/a100_visual_classical_strong_prefix.py','--pair',pair,'--policy','COVER',
                '--seed',str(seed),'--acquisition-seed',str(seed)],gpu);launched.add(name)
        for seed in (17,42):
            group='strong_cover_seed'+str(seed)
            for pair in ('C4','C1'):
                name='repeat_counts_'+pair+'_'+str(seed)
                if name not in launched and done('repeat_fit_'+pair+'_'+str(seed)):
                    launch(name,['scripts/a100_visual_classical_strong_countmatched.py','--pair',pair,'--group',group]);launched.add(name)
            if seed not in evaluated and all(done('repeat_counts_'+pair+'_'+str(seed)) for pair in ('C4','C1')):
                for suffix,primary in [('', 'COHORT'),('_counts','COHORT__AT_COHORT_COUNT')]:
                    launch('repeat_eval_'+str(seed)+suffix,['scripts/a100_visual_classical_evaluate_group.py','--group',group+suffix,
                        '--primary',primary,'--workers','4'])
                launch('repeat_analysis_'+str(seed),['scripts/a100_visual_classical_strong_analyze.py','--seed',str(seed),'--policies','cover'])
                evaluated.add(seed)
        if len(evaluated)==2 and all(v['process'].poll()==0 for v in jobs.values()):break
        time.sleep(15)
    state['status']='PASS_FIXED_RECIPE_REPEATS';atomic_json(state_path,state)
    for log in logs:log.close()


if __name__=='__main__':main()
