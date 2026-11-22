"""Durable graph-search experiment; never preempts another running job."""
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
    jobs={};logs=[];launched=set();evaluated=set();begin=time.perf_counter();state_path=OUT/'supervisor/field_state.json'
    def launch(name,args,gpu=None):
        path=OUT/'supervisor'/('recovery_strong_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        child=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[name]=dict(process=child,pid=child.pid,gpu=gpu,log=str(path));launched.add(name)
    def done(name):return name in jobs and jobs[name]['process'].poll()==0
    def previous_done(name):
        p=OUT/'supervisor/repeats_state.json'
        return p.is_file() and json.loads(p.read_text()).get('jobs',{}).get(name,{}).get('exit_code')==0
    while True:
        for pair,gpu,dependency in [('C4',2,'repeat_fit_C4_42'),('C1',3,'repeat_fit_C1_17')]:
            name='field_collect_'+pair
            if name not in launched and previous_done(dependency):launch(name,['scripts/a100_visual_classical_field_collect.py','--pair',pair],gpu)
        clipped_ready=all((OUT/'strong_cover_seed0_clipped'/p/'completion.json').is_file() for p in ('C4','C1'))
        for pair,policy,gpu in [('C4','FIELD',2),('C4','FIELD_STATIC',0),('C1','FIELD',3),('C1','FIELD_STATIC',1)]:
            name='field_fit_'+pair+'_'+policy
            if name in launched or not done('field_collect_'+pair):continue
            if gpu==0 and not previous_done('repeat_fit_C1_42'):continue
            if gpu==1 and not clipped_ready:continue
            launch(name,['scripts/a100_visual_classical_strong_prefix.py','--pair',pair,'--policy',policy],gpu)
        for policy in ('FIELD','FIELD_STATIC'):
            group='strong_'+policy.lower()+'_seed0'
            for pair in ('C4','C1'):
                name='field_counts_'+pair+'_'+policy
                if name not in launched and done('field_fit_'+pair+'_'+policy):
                    launch(name,['scripts/a100_visual_classical_strong_countmatched.py','--pair',pair,'--group',group,
                        '--exclusive-projection'])
            if policy not in evaluated and all(done('field_counts_'+pair+'_'+policy) for pair in ('C4','C1')):
                for suffix,primary in [('', 'COHORT'),('_counts_exclusive','COHORT__AT_COHORT_COUNT')]:
                    launch('field_evaluate_'+policy+suffix,['scripts/a100_visual_classical_evaluate_group.py','--group',group+suffix,
                        '--primary',primary,'--workers','4'])
                evaluated.add(policy)
        state=dict(status='RUNNING_GRAPH_FIELD_SEARCH',pid=os.getpid(),wall_seconds=time.perf_counter()-begin,
            jobs={k:{a:b for a,b in v.items() if a!='process'}|dict(exit_code=v['process'].poll()) for k,v in jobs.items()})
        failed=[k for k,v in state['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:state['status']='FAILED_GRAPH_SEARCH_REQUIRES_ROOT_CAUSE'
        atomic_json(state_path,state)
        if failed:raise RuntimeError(str(failed))
        if len(evaluated)==2 and all(v['process'].poll()==0 for v in jobs.values()):break
        time.sleep(15)
    with (OUT/'supervisor/recovery_strong_field_analysis.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_field_analyze.py'],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
    assert code==0
    state['status']='PASS_GRAPH_FIELD_SEARCH';atomic_json(state_path,state)
    for log in logs:log.close()


if __name__=='__main__':main()
