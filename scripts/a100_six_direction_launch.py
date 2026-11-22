"""New C1..C6 transfer matrix; one controller and one blocking observer.

Internal GPU allocation/progress checks are confined to this controller.
Every raw prediction and quantity control is sealed before this study's GT.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT,PUBLIC,PAIRS,SEEDS,source

STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_SIX_DIRECTION_TRANSFER_MATRIX_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def counts(record):
    from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts
    return generate_counts(dict(record,coarse_scores=True,fresh_prior_root=str(source(record['pair'])/'source'),
        projection_pairs=[('JOINT','JOINT'),('BASE_PROTOCOL','JOINT'),('JOINT','BASE_PROTOCOL'),('BASE_PROTOCOL','BASE_PROTOCOL')]))


def main():
    plan=json.loads((PUBLIC/'preregistration_transfer_v1.json').read_text())
    pending=[dict(kind='EVIDENCE',pair=p) for p in ('C2','C3','C5','C6')]+[dict(kind='QUALITY',**r) for r in conditions()]
    active={};done=[];failure=None;start=time.perf_counter();last=0
    # A source-registry error must not hold up healthy GPUs or restart their
    # streams. A replacement controller adopts only the recorded exact PIDs.
    previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
    def alive(pid):
        path=Path('/proc')/str(pid)/'stat'
        return path.is_file() and path.read_text().rsplit(')',1)[1].split()[0]!='Z'
    class Adopted:
        def __init__(self,job):self.job=job
        def poll(self):
            if alive(self.job['pid']):return None
            receipt=(source(self.job['pair']) if self.job['kind']=='EVIDENCE' else Path(self.job['folder']))/'completion.json'
            return 0 if receipt.is_file() else 1
    if previous.get('active'):
        assert not alive(previous['pid']),'Do not start a second active group controller'
        adopted=[]
        for gpu,job in previous['active'].items():
            if not alive(job['pid']):continue
            command=(Path('/proc')/str(job['pid'])/'cmdline').read_bytes()
            assert b'a100_six_direction.py' in command and job['pair'].encode() in command
            active[int(gpu)]=Adopted(job),Path(job['log']).open('a'),job
            pending=[j for j in pending if not (j['kind']==job['kind'] and j['pair']==job['pair'] and j.get('seed')==job.get('seed'))]
            adopted.append(job)
        atomic_json(OUT/'supervisor/adopted_healthy_workers.json',dict(previous_controller_pid=previous['pid'],
            previous_failure=previous.get('failure'),adopted=adopted,workers_restarted=False,GT_reads=0))
    def state(status):
        progress={}
        for gpu,(_,_,job) in active.items():
            folder=source(job['pair'])/'source' if job['kind']=='EVIDENCE' else Path(job['folder'])/'videos'
            progress[str(gpu)]=dict(job,completed_videos=len(list(folder.glob('*/receipt.json'))))
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,active=progress,
            pending=[{k:v for k,v in j.items() if k!='folder'} for j in pending],failure=failure,
            wall_seconds=time.perf_counter()-start,goal_status='ACTIVE',six_direction_goal_complete=False))
    def ready(job):
        if job['kind']!='EVIDENCE':return (source(job['pair'])/'completion.json').is_file()
        if (source(job['pair'])/'completion.json').is_file():return True
        # The independent LaMOT repair may still own this evidence stream.
        # Keep fitting other ready conditions instead of occupying a free GPU
        # with a second process waiting on the same collection lock.
        import fcntl
        folder=source(job['pair']);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'collection.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
            return True
    state('INITIALIZING_NEW_SIX_DIRECTION_STUDY')
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();del active[gpu]
            receipt=(source(job['pair']) if job['kind']=='EVIDENCE' else Path(job['folder']))/'completion.json'
            if rc or not receipt.is_file():
                failure=dict(job=job,exit_code=rc,last_log=Path(job['log']).read_text()[-7000:])
            else:
                completion=json.loads(receipt.read_text());assert completion['GT_open_attempts']==0
                done.append(dict(job,exit_code=0))
            changed=True
        runnable=[j for j in pending if ready(j)]
        candidates=[g for g in (0,1,2,3) if g not in active]
        if runnable and candidates and not failure:
            rows=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in rows.strip().splitlines()}
            allocation=dict(candidate_GPU_memory_MiB=memory,only_start_below_MiB=512,no_other_process_terminated=True)
            atomic_json(OUT/'supervisor/last_GPU_allocation.json',allocation)
            for gpu in candidates:
                if memory[gpu]>=512 or not runnable:continue
                job=runnable.pop(0);pending.remove(job)
                receipt=(source(job['pair']) if job['kind']=='EVIDENCE' else Path(job['folder']))/'completion.json'
                if receipt.is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
                slug=job['kind']+'_'+job['pair']+('_seed'+str(job['seed']) if 'seed' in job else '')
                log=OUT/'supervisor'/(slug+'.log');handle=log.open('a',buffering=1)
                command=[sys.executable,'-u','scripts/a100_six_direction.py','--pair',job['pair']]
                command+=['--collect'] if job['kind']=='EVIDENCE' else ['--seed',str(job['seed'])]
                proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_NEW_SIX_DIRECTION_EVIDENCE_AND_QUALITY_NO_GT' if not failure else 'PRESERVING_ACTIVE_SIX_DIRECTION_JOBS_AFTER_FAILURE')
            last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==4+len(conditions())
    state('ALL18_RAW_CONDITIONS_SEALED_GENERATING_QUANTITY_CONTROLS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        count_roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),count_roots):
        for folder in (Path(record['folder']),Path(count_root)):
            result=json.loads((folder/'completion.json').read_text());assert result['GT_open_attempts']==0
            assert not any(result.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in result['methods'])
    assert len(tasks)==234
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(conditions=18,methods=len(tasks),
        all_raw_and_countmatched_sealed=True,adaptation_GT_open_attempts=0,all_data_development=True))
    state('ALL234_SIX_DIRECTION_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(status='SIX_DIRECTION_OFFICIAL_COMPLETE',results=results,methods=len(tasks),goal_status='ACTIVE'))
    state('SIX_DIRECTION_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_six_direction_analyze.py'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
        previous.update(status='FAILED_SIX_DIRECTION_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,previous);raise
