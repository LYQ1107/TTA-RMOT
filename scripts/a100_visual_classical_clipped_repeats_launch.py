"""Three-seed matched bounded-label controls, unchanged .02 from original graph."""
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
    jobs={};logs=[];done_eval=set();begin=time.perf_counter();state_path=OUT/'supervisor/clipped_repeats_state.json'
    def launch(name,args,gpu=None):
        path=OUT/'supervisor'/('recovery_strong_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        p=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        jobs[name]=dict(process=p,pid=p.pid,gpu=gpu,log=str(path))
    def done(name):return name in jobs and jobs[name]['process'].poll()==0
    while True:
        previous=json.loads((OUT/'supervisor/repeats_state.json').read_text())
        raw_gpu0_free=previous['jobs'].get('repeat_fit_C1_42',{}).get('exit_code')==0
        for seed,gpu in [(17,0),(42,1)]:
            c4='clip_repeat_C4_'+str(seed);c1='clip_repeat_C1_'+str(seed)
            if c4 not in jobs and (gpu!=0 or raw_gpu0_free):
                launch(c4,['scripts/a100_visual_classical_strong_prefix.py','--pair','C4','--policy','COVER',
                    '--seed',str(seed),'--acquisition-seed',str(seed),'--teacher-target-clip','.02'],gpu)
            if c1 not in jobs and done(c4):
                launch(c1,['scripts/a100_visual_classical_strong_prefix.py','--pair','C1','--policy','COVER',
                    '--seed',str(seed),'--acquisition-seed',str(seed),'--teacher-target-clip','.02'],gpu)
        for seed in (0,17,42):
            group='strong_cover_seed'+str(seed)+'_clipped'
            for pair in ('C4','C1'):
                ready=(OUT/group/pair/'completion.json').is_file();name='clip_repeat_counts_'+pair+'_'+str(seed)
                if ready and name not in jobs:launch(name,['scripts/a100_visual_classical_strong_countmatched.py','--pair',pair,'--group',group,'--exclusive-projection'])
            if seed not in done_eval and all(done('clip_repeat_counts_'+pair+'_'+str(seed)) for pair in ('C4','C1')):
                for suffix,primary in [('', 'COHORT'),('_counts_exclusive','COHORT__AT_COHORT_COUNT')]:
                    launch('clip_repeat_eval_'+str(seed)+suffix,['scripts/a100_visual_classical_evaluate_group.py','--group',group+suffix,'--primary',primary,'--workers','4'])
                done_eval.add(seed)
        state=dict(status='RUNNING_MATCHED_CLIPPED_REPEATS',pid=os.getpid(),wall_seconds=time.perf_counter()-begin,
            jobs={k:{a:b for a,b in v.items() if a!='process'}|dict(exit_code=v['process'].poll()) for k,v in jobs.items()})
        failed=[k for k,v in state['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:state['status']='FAILED_CLIPPED_REPEATS_REQUIRE_ROOT_CAUSE'
        atomic_json(state_path,state)
        if failed:raise RuntimeError(str(failed))
        if len(done_eval)==3 and all(v['process'].poll()==0 for v in jobs.values()):break
        time.sleep(15)
    with (OUT/'supervisor/recovery_strong_clipped_three_seed_analysis.log').open('a',buffering=1) as log:
        code=subprocess.call([sys.executable,'scripts/a100_visual_classical_repeats_analyze.py','--group-suffix','_clipped',
            '--counts-suffix','_counts_exclusive','--output-name','three_seed_clipped_results'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert code==0
    state['status']='PASS_MATCHED_CLIPPED_THREE_SEEDS';atomic_json(state_path,state)
    for log in logs:log.close()


if __name__=='__main__':main()
