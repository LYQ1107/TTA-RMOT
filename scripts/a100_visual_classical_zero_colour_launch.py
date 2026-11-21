"""Six matched feature controls, then42sealed original-vendor evaluations."""
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
from scripts.a100_visual_classical_zero_colour import ZERO
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/zero_colour_state.json'
SUCCESS='PASS_ZERO_COLOUR_STUDY_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair=p,seed=s,folder=str(ZERO/p/f'seed{s}')) for p in ('C1','C4') for s in (0,17,42)]


def count_record(record):
    full='RGB_MEAN';base='ZERO_MEAN'
    return dict(record,projection_pairs=[(full,full),(base,full),(full,base),(base,base)],kind='ZERO_COLOUR',
        method_sources={full:str(PARITY/record['pair']/f"seed{record['seed']}"/'RGB_KD_MEAN')},
        fresh_prior_root=str(source_spec(record['pair'])[0]/'source'))


def main():
    begin=time.perf_counter();pending=conditions();active={};done=[];failure=None;completed=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_zero_colour_v1.json'
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=completed,finished_conditions=done,pending=pending,
            active={str(g):r for g,(_,_,r) in active.items()},failure=failure,
            preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    from tools.deduplicate_immutable_predictions import run as deduplicate
    last=0
    while pending or active:
        changed=False
        for gpu,(process,handle,record) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();receipt=Path(record['folder'])/'completion.json';record=dict(record,exit_code=rc)
            if rc or not receipt.is_file():failure=dict(record,last_log=Path(record['log']).read_text()[-6000:])
            else:
                assert json.loads(receipt.read_text())['GT_open_attempts']==0;deduplicate(ZERO,True)
            done.append(record);completed+=1;del active[gpu];changed=True
        release=OUT/'supervisor/strict_rgb_release_state.json'
        released=release.is_file() and json.loads(release.read_text())['status']=='PASS_RGB_RELEASE_HELPER_RESEARCH_GOAL_ACTIVE'
        if not failure:
            for gpu in ([0,3,1,2] if released else [0,3]):
                if gpu in active or not pending:continue
                record=pending.pop(0);receipt=Path(record['folder'])/'completion.json'
                if receipt.is_file():done.append(dict(record,exit_code=0,reused_sealed_condition=True));completed+=1;changed=True;continue
                log=OUT/'supervisor'/f"zero_colour_{record['pair']}_seed{record['seed']}.log";handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_zero_colour.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                record=dict(record,GPU=gpu,pid=process.pid,log=str(log));active[gpu]=process,handle,record;changed=True
        if changed or time.perf_counter()-last>45:state('RUNNING_ZERO_COLOUR_FEATURE_ABLATION' if not failure else 'STOPPING_PRESERVING_RUNNING_ZERO_JOBS');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert completed==6
    # Reference heads must also be sealed; never consult their metrics.
    for record in conditions():assert (PARITY/record['pair']/f"seed{record['seed']}"/'completion.json').is_file()
    roots=[];state('ALL_ZERO_RAW_SEALED_GENERATING_QUANTITIES_NO_NEW_GT')
    for record in conditions():
        roots.append(generate_counts(count_record(record)));deduplicate(ZERO/record['pair'],True)
    tasks=[]
    for record,counts in zip(conditions(),roots):
        for root in (Path(record['folder']),Path(counts)):
            r=json.loads((root/'completion.json').read_text());assert r['GT_open_attempts']==0;assert not any(r.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(root/m['method']),record['pair']) for m in r['methods'])
    assert len(tasks)==42,len(tasks)
    completed=0;state('ALL42_ZERO_COLOUR_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;completed+=1;state('ALL42_ZERO_COLOUR_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/zero_colour_official.json',dict(status='ALL_ZERO_COLOUR_OFFICIAL_COMPLETE',methods=42,results=results,goal_status='ACTIVE'))
    state('ZERO_COLOUR_OFFICIAL_COMPLETE_ANALYZING')
    for cmd in (['scripts/a100_visual_classical_zero_colour_analyze.py'],['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','zero_colour']):subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_ZERO_COLOUR_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
