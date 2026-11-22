"""Fit equal-RGB heads on GPUs0/3, then seal all114 evaluations before GT."""
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
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/strict_rgb_state.json'
SUCCESS='PASS_STRICT_RGB_PARITY_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair=p,seed=s,folder=str(PARITY/p/f'seed{s}')) for p in ('C1','C4') for s in (0,17,42)]


def count_record(record):
    full='RGB_GRAPH_MEAN';others=['RGB_KD_MEAN']
    if record['pair']=='C4':others.append('ORIG_GRAPH_MEAN')
    pairs=[(full,full)]+[(other,full) for other in others]+[(rank,other) for other in others for rank in (full,other)]
    return dict(record,projection_pairs=pairs,kind='STRICT_RGB',fresh_prior_root=str(source_spec(record['pair'])[0]/'source'))


def main():
    begin=time.perf_counter();pending=conditions();completed=[];active={};failure=None;done=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_strict_rgb_parity_v1.json'
    assert prereg.is_file()
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=done,total_conditions=6,
            finished_conditions=completed,active={str(g):r for g,(_,_,r) in active.items()},pending=pending,
            failure=failure,preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    state('RUNNING_EQUAL_RGB_FRESH_SOURCE_QUALITY_REPLAY')
    from tools.deduplicate_immutable_predictions import run as deduplicate
    last_heartbeat=0
    while pending or active:
        changed=False
        for gpu,(process,handle,record) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();completion=Path(record['folder'])/'completion.json';record=dict(record,exit_code=rc)
            if rc or not completion.is_file():
                failure=dict(record,last_log=Path(record['log']).read_text()[-8000:])
            else:
                receipt=json.loads(completion.read_text());assert receipt['GT_open_attempts']==receipt['new_teacher_calls']==0
                record['sealed_methods']=len(receipt['methods']);deduplicate(PARITY,True)
            completed.append(record);del active[gpu];done+=1;changed=True
            print(json.dumps(dict(event='STRICT_RGB_CONDITION_END',**record)),flush=True)
        if not failure:
            for gpu in (0,3):
                if gpu in active or not pending:continue
                record=pending.pop(0);completion=Path(record['folder'])/'completion.json'
                if completion.is_file():
                    receipt=json.loads(completion.read_text());assert receipt['GT_open_attempts']==0
                    completed.append(dict(record,exit_code=0,reused_sealed_condition=True,sealed_methods=len(receipt['methods'])));done+=1;changed=True
                    continue
                logpath=OUT/'supervisor'/f"strict_rgb_{record['pair']}_seed{record['seed']}.log"
                handle=logpath.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'scripts/a100_visual_classical_rgb_parity.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                record=dict(record,GPU=gpu,pid=process.pid,log=str(logpath));active[gpu]=process,handle,record;changed=True
                print(json.dumps(dict(event='STRICT_RGB_CONDITION_START',**record)),flush=True)
        if changed or time.perf_counter()-last_heartbeat>45:
            state('STOPPING_PRESERVING_RUNNING_RGB_JOBS' if failure else 'RUNNING_EQUAL_RGB_FRESH_SOURCE_QUALITY_REPLAY')
            last_heartbeat=time.perf_counter()
        if failure and not active:raise RuntimeError('Strict RGB condition failed: '+json.dumps(failure))
        if active or pending:time.sleep(3)
    assert done==6 and all(r['exit_code']==0 for r in completed)
    state('ALL_RGB_RAW_SEALED_GENERATING_QUANTITY_CONTROLS_NO_NEW_GT')
    records=conditions();roots=[]
    for record in records:
        roots.append(generate_counts(count_record(record)))
        deduplicate(PARITY/record['pair'],True)
    tasks=[]
    for record,counts in zip(records,roots):
        for root in (Path(record['folder']),Path(counts)):
            receipt=json.loads((root/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(root/r['method']),record['pair']) for r in receipt['methods'])
    assert len(tasks)==114,len(tasks)
    done=0;state('ALL114_RGB_PREDICTIONS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;done+=1
            state('ALL114_RGB_PREDICTIONS_SEALED_OFFICIAL_EVALUATING')
            print(json.dumps(dict(event='STRICT_RGB_OFFICIAL',completed=done,total=114)),flush=True)
    atomic_json(OUT/'analysis/strict_rgb_official.json',dict(status='ALL_STRICT_RGB_OFFICIAL_COMPLETE',methods=114,results=results,goal_status='ACTIVE'))
    state('RGB_OFFICIAL_COMPLETE_ANALYZING')
    for command in (['scripts/a100_visual_classical_rgb_parity_analyze.py'],
        ['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','strict_rgb']):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        row=json.loads(STATE.read_text()) if STATE.is_file() else {}
        row.update(status='FAILED_STRICT_RGB_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,row);raise
