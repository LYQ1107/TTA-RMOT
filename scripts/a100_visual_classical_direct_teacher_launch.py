"""Queue direct-KD fairness audit only after four physical workers exit."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_direct_teacher import DIRECT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/direct_teacher_state.json'
SUCCESS='PASS_DIRECT_TEACHER_NORES_STUDY_RESEARCH_REVIEW_REQUIRED'


def conditions():
    return [dict(pair=p,seed=s,folder=str(DIRECT/p/f'seed{s}')) for s in (0,17,42) for p in ('C4','C1')]


def count_record(record):
    return dict(record,kind='DIRECT_NORES',coarse_scores=True,
        projection_pairs=[('JOINT','JOINT'),('KD_MEAN','JOINT'),('JOINT','KD_MEAN'),('KD_MEAN','KD_MEAN')],
        fresh_prior_root=str(source_spec(record['pair'])[0]/'source'))


def main():
    begin=time.perf_counter();pending=conditions();active={};done=[];completed=0;failure=None;last=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_direct_teacher_nores_v1.json'
    def state(status):
        progress={r['pair']+f"_seed{r['seed']}":dict(videos=len(list((Path(r['folder'])/'videos').glob('*/receipt.json'))),total=45 if r['pair']=='C4' else 25)
            for _,_,r in active.values()}
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=dict(conditions=completed,active_videos=progress) if active else completed,
            finished_conditions=done,pending=pending,active={str(g):r for g,(_,_,r) in active.items()},failure=failure,
            preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    physical=json.loads((ROOT/'docs/a100_visual_classical_20261121/preregistration_appearance_joint_cold_v1.json').read_text())
    while True:
        ready=True
        for job in physical['jobs'].values():
            folder=OUT/'cold_full_c4'/job['run_name'];receipt=folder/'worker_receipt.json'
            if not (folder/'completion.json').is_file() or not receipt.is_file():ready=False;continue
            actual=json.loads(receipt.read_text());assert actual['exit_code']==0,('Resolve physical-worker failure first',job)
        if ready:break
        if time.perf_counter()-last>45:state('WAITING_ALL_FOUR_PHYSICAL_WORKERS_EXIT_NO_GPU_CONTENTION');last=time.perf_counter()
        time.sleep(5)
    while True:
        rows=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.strip().splitlines()}
        if all(memory[g]<512 for g in (0,1,2,3)):break
        state('WAITING_FREE_GPU0_TO3_AFTER_PHYSICAL_EXIT');time.sleep(15)
    atomic_json(OUT/'supervisor/direct_teacher_GPU_release_preflight.json',dict(physical_workers_all_exited=True,GPU_memory_MiB=memory,goal_status='ACTIVE'))
    from tools.deduplicate_immutable_predictions import run as deduplicate
    while pending or active:
        changed=False
        for gpu,(process,handle,record) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();receipt=Path(record['folder'])/'completion.json';record=dict(record,exit_code=rc)
            if rc or not receipt.is_file():failure=dict(record,last_log=Path(record['log']).read_text()[-7000:])
            else:
                result=json.loads(receipt.read_text());assert result['GT_open_attempts']==result['new_teacher_calls']==0
                assert result['source_logit_residual_disabled'];deduplicate(DIRECT,True)
            done.append(record);del active[gpu];completed+=1;changed=True
        if not failure:
            for gpu in (0,1,2,3):
                if gpu in active or not pending:continue
                record=pending.pop(0)
                if (Path(record['folder'])/'completion.json').is_file():done.append(dict(record,reused_sealed=True));completed+=1;changed=True;continue
                log=OUT/'supervisor'/f"direct_teacher_{record['pair']}_seed{record['seed']}.log";handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_direct_teacher.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=process,handle,dict(record,pid=process.pid,log=str(log),GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_DIRECT_TEACHER_NORES_HEADS' if not failure else 'PRESERVING_RUNNING_DIRECT_HEADS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert completed==6
    state('ALL_DIRECT_RAW_SEALED_GENERATING_QUANTITIES_NO_GT');records=conditions();roots=[]
    for record in records:roots.append(generate_counts(count_record(record)));deduplicate(DIRECT/record['pair'],True)
    tasks=[]
    for record,counts in zip(records,roots):
        for folder in (Path(record['folder']),Path(counts)):
            receipt=json.loads((folder/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/m['method']),record['pair']) for m in receipt['methods'])
    assert len(tasks)==72,len(tasks)
    completed=0;state('ALL72_DIRECT_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;completed+=1;state('ALL72_DIRECT_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/direct_teacher_official.json',dict(status='ALL_DIRECT_TEACHER_OFFICIAL_COMPLETE',methods=72,results=results,goal_status='ACTIVE'))
    state('DIRECT_TEACHER_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_visual_classical_direct_teacher_analyze.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','direct_teacher'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_DIRECT_TEACHER_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
