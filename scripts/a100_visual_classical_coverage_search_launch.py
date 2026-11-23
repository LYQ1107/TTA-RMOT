"""Same-budget teacher-feedback search; sealed strong heads and classical ablations."""
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
from scripts.a100_visual_classical_coverage_search import DIRECT
from scripts.a100_visual_classical_search_collect import EVIDENCE
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/coverage_search_state.json'
SUCCESS='PASS_COVERAGE_SEARCH_STUDY_RESEARCH_REVIEW_REQUIRED'


def conditions():
    return [dict(pair=p,seed=s,folder=str(DIRECT/p/f'seed{s}')) for s in (0,17,42) for p in ('C4',)]


def count_record(record):
    primary='JOINT';others=('KD_MEAN','GRAPH','CAL','COVER_KD','COVER_JOINT','COMPLETE')
    pairs=[(primary,primary)]
    for other in others:pairs.extend([(other,primary),(primary,other),(other,other)])
    return dict(record,kind='COVERAGE_SEARCH',coarse_scores=True,projection_pairs=pairs,
        fresh_prior_root=str(source_spec(record['pair'])[0]/'source'),method_sources=dict(
            COVER_KD=str(OUT/'confidence_teacher'/record['pair']/f"seed{record['seed']}"/'KD_MEAN'),
            COVER_JOINT=str(OUT/'weighted_classical'/record['pair']/f"seed{record['seed']}"/'JOINT')))


def main():
    begin=time.perf_counter();pending=conditions();active={};done=[];completed=0;failure=None;last=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_coverage_search_v1.json'
    def state(status):
        progress={r['pair']+f"_seed{r['seed']}":dict(videos=len(list((Path(r['folder'])/'videos').glob('*/receipt.json'))),total=45 if r['pair']=='C4' else 25)
            for _,_,r in active.values()}
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=dict(conditions=completed,active_videos=progress) if active else completed,
            finished_conditions=done,pending=pending,active={str(g):r for g,(_,_,r) in active.items()},failure=failure,
            preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    assert json.loads((ROOT/'docs/a100_visual_classical_20261121/native_cold_statistical_reproduction.json').read_text())['status'].startswith('PASS_')
    allocation_receipts=[]
    evidence=EVIDENCE/'C4'/'completion.json'
    if not evidence.is_file():
        worker=None;handle=None
        while worker is None:
            memory=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            free=[int(line.split(',')[0]) for line in memory.strip().splitlines() if int(line.split(',')[1])<512]
            if free:
                gpu=free[0];log=OUT/'supervisor/coverage_search_collect.log';handle=log.open('a',buffering=1)
                allocation_receipts.append(dict(phase='teacher_evidence',GPU=gpu,preflight=memory))
                worker=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_search_collect.py'],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            else:state('WAITING_ALLOWED_GPU_FOR_SEARCH_EVIDENCE');time.sleep(15)
        while worker.poll() is None:
            completed=len(list((EVIDENCE/'C4/source').glob('*/receipt.json')))
            state('COLLECTING_REAL_MISSING_SEARCH_TEACHER_ANSWERS');time.sleep(30)
        handle.close()
        if worker.returncode or not evidence.is_file():raise RuntimeError('Search evidence failed: '+log.read_text()[-7000:])
    completed=0
    assert json.loads(evidence.read_text())['GT_open_attempts']==0
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
            candidates=[g for g in (0,1,2,3) if g not in active];available=set()
            if candidates and pending:
                rows=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                    '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in rows.strip().splitlines()}
                available={g for g in candidates if memory[g]<512}
                if available:
                    allocation_receipts.append(dict(candidate_GPU_memory_MiB=memory,available=sorted(available)))
                    atomic_json(OUT/'supervisor/coverage_search_GPU_preflight.json',dict(allocation_receipts=allocation_receipts,goal_status='ACTIVE'))
            for gpu in (0,1,2,3):
                if gpu not in available or gpu in active or not pending:continue
                record=pending.pop(0)
                if (Path(record['folder'])/'completion.json').is_file():done.append(dict(record,reused_sealed=True));completed+=1;changed=True;continue
                log=OUT/'supervisor'/f"coverage_search_{record['pair']}_seed{record['seed']}.log";handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_coverage_search.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=process,handle,dict(record,pid=process.pid,log=str(log),GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state(('RUNNING_CONFIDENCE_WEIGHTED_DIRECT_HEADS_ON_RELEASED_GPUS' if active else 'QUEUED_CONFIDENCE_WEIGHTED_WAIT_PHYSICAL_PREVIOUS_RELEASED_GPU') if not failure else 'PRESERVING_RUNNING_DIRECT_HEADS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert completed==3
    state('ALL_DIRECT_RAW_SEALED_GENERATING_QUANTITIES_NO_GT');records=conditions();roots=[]
    for record in records:roots.append(generate_counts(count_record(record)));deduplicate(DIRECT/record['pair'],True)
    tasks=[]
    for record,counts in zip(records,roots):
        for folder in (Path(record['folder']),Path(counts)):
            receipt=json.loads((folder/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/m['method']),record['pair']) for m in receipt['methods'])
    assert len(tasks)==84,len(tasks)
    completed=0;state('ALL84_DIRECT_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;completed+=1;state('ALL84_DIRECT_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/coverage_search_official.json',dict(status='ALL_DIRECT_TEACHER_OFFICIAL_COMPLETE',methods=84,results=results,goal_status='ACTIVE'))
    state('DIRECT_TEACHER_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_visual_classical_coverage_search_analyze.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','coverage_search'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_DIRECT_TEACHER_PCD_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
