"""Resume interrupted, checkpointed jobs under one independent supervisor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def main():
    record_path=OUT/'supervisor/recovery_state.json';jobs={};logs=[];begin=time.perf_counter()
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def start(name,arguments):
        path=OUT/'supervisor'/('recovery_'+name+'.log');log=path.open('a',buffering=1);logs.append(log)
        p=subprocess.Popen([sys.executable,*arguments],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        jobs[name]=dict(process=p,pid=p.pid,log=str(path),arguments=arguments)
    def snapshot(status):
        row=dict(status=status,supervisor_pid=os.getpid(),jobs={name:{k:v for k,v in job.items() if k!='process'}|
            dict(exit_code=job['process'].poll()) for name,job in jobs.items()},wall_seconds=time.perf_counter()-begin)
        atomic_json(record_path,row)
        return row
    start('first_round',['scripts/a100_visual_classical_finish.py'])
    start('strict_prefix',['scripts/a100_visual_classical_strict_launch.py'])
    for pair in ('C4','C1'):start('exemplar_'+pair,['scripts/a100_visual_classical_exemplar_prefix.py','--pair',pair])
    evaluation_started=False
    while True:
        row=snapshot('RUNNING_CHECKPOINT_RECOVERY');failed=[k for k,v in row['jobs'].items() if v['exit_code'] not in (None,0)]
        if failed:
            snapshot('FAILED_JOB_REQUIRES_ROOT_CAUSE_FIX');raise RuntimeError(str(failed))
        if not evaluation_started and all(jobs['exemplar_'+p]['process'].poll()==0 for p in ('C4','C1')):
            start('exemplar_evaluate',['scripts/a100_visual_classical_evaluate_group.py','--group','exemplar_prefix','--primary','COVER__SUPPORT_MIX','--workers','4']);evaluation_started=True
        if evaluation_started and all(j['process'].poll()==0 for j in jobs.values()):break
        time.sleep(15)
    snapshot('ALL_RECOVERED_JOBS_FINISHED')
    for log in logs:log.close()


if __name__=='__main__':main()
