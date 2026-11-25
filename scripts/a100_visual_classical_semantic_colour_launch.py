"""Six equal-information fits on GPUs0..3; all174seal before study GT."""
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
from scripts.a100_visual_classical_semantic_colour import SEMANTIC
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/semantic_colour_state.json'
SUCCESS='PASS_SEMANTIC_COLOUR_STUDY_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair=p,seed=s,folder=str(SEMANTIC/p/f'seed{s}')) for s in (0,17,42) for p in ('C4','C1')]


def count_record(record):
    full='MONO';others=['SEM_MEAN','ZERO_MEAN','BIN','NAIVE_MEAN']
    pairs=[(full,full)]+[(other,full) for other in others]+[(rank,other) for other in others for rank in (full,other)]
    pairs += [('SEM_MEAN','ZERO_MEAN'),('ZERO_MEAN','SEM_MEAN')]
    return dict(record,kind='SEMANTIC_COLOUR',projection_pairs=pairs,fresh_prior_root=str(source_spec(record['pair'])[0]/'source'))


def main():
    sanity=json.loads((ROOT/'docs/a100_visual_classical_20261121/semantic_colour_generation_sanity.json').read_text())
    assert sanity['status']=='PASS_REAL_SEMANTIC_CUES_AND_INDEPENDENT_PAVA'
    assert sanity['semantic_implementation_sha256']==file_sha(ROOT/'flex_uda/active/semantic_colour.py')
    begin=time.perf_counter();pending=conditions();active={};done=[];failure=None;completed=0;last=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_semantic_colour_v1.json'
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=completed,finished_conditions=done,pending=pending,
            active={str(g):r for g,(_,_,r) in active.items()},failure=failure,
            preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    from tools.deduplicate_immutable_predictions import run as deduplicate
    while pending or active:
        changed=False
        for gpu,(process,handle,record) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();completion=Path(record['folder'])/'completion.json';record=dict(record,exit_code=rc)
            if rc or not completion.is_file():failure=dict(record,last_log=Path(record['log']).read_text()[-7000:])
            else:
                receipt=json.loads(completion.read_text());assert receipt['GT_open_attempts']==receipt['new_teacher_calls']==0
                record['sealed_methods']=len(receipt['methods']);deduplicate(SEMANTIC,True)
            done.append(record);completed+=1;del active[gpu];changed=True
        if not failure:
            for gpu in (0,1,2,3):
                if gpu in active or not pending:continue
                record=pending.pop(0);completion=Path(record['folder'])/'completion.json'
                if completion.is_file():done.append(dict(record,exit_code=0,reused_sealed_condition=True));completed+=1;changed=True;continue
                log=OUT/'supervisor'/f"semantic_colour_{record['pair']}_seed{record['seed']}.log";handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_semantic_colour.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                record=dict(record,GPU=gpu,pid=process.pid,log=str(log));active[gpu]=process,handle,record;changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_SEMANTIC_COLOUR_HEADS' if not failure else 'STOPPING_PRESERVING_SEMANTIC_RUNNING_JOBS');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert completed==6
    records=conditions();roots=[];state('ALL_SEMANTIC_RAW_SEALED_GENERATING_QUANTITIES_NO_NEW_GT')
    for record in records:
        roots.append(generate_counts(count_record(record)));deduplicate(SEMANTIC/record['pair'],True)
    tasks=[]
    for record,counts in zip(records,roots):
        for root in (Path(record['folder']),Path(counts)):
            receipt=json.loads((root/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(root/m['method']),record['pair']) for m in receipt['methods'])
    assert len(tasks)==174,len(tasks)
    completed=0;state('ALL174_SEMANTIC_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;completed+=1;state('ALL174_SEMANTIC_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/semantic_colour_official.json',dict(status='ALL_SEMANTIC_COLOUR_OFFICIAL_COMPLETE',methods=174,results=results,goal_status='ACTIVE'))
    state('SEMANTIC_OFFICIAL_COMPLETE_ANALYZING')
    for cmd in (['scripts/a100_visual_classical_semantic_colour_analyze.py'],['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','semantic_colour']):
        subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_SEMANTIC_COLOUR_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
