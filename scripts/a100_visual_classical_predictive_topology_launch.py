"""CPU-only sealed-head study; every156 output precedes new study GT."""
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
from scripts.a100_visual_classical_predictive_topology import PREDICTIVE
from scripts.a100_visual_classical_native_factorial import FACTORIAL
from scripts.a100_visual_classical_direct_teacher_pcd import DIRECT
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/predictive_topology_state.json'
SUCCESS='PASS_PREDICTIVE_TOPOLOGY_STUDY_RESEARCH_REVIEW_REQUIRED'


def conditions():
    return [dict(pair=p,seed=s,folder=str(PREDICTIVE/p/f'seed{s}')) for s in (0,17,42) for p in ('C4','C1')]


def count_record(record):
    primary='SELECTED_JOINT';others=('KD','CAL','SELECTED_GRAPH','LOCAL_JOINT','GLOBAL_JOINT','MEAN_JOINT')
    pairs=[(primary,primary)]
    for other in others:pairs.extend([(other,primary),(primary,other),(other,other)])
    previous=DIRECT/record['pair']/f"seed{record['seed']}"
    global_reference=FACTORIAL/record['pair']/f"seed{record['seed']}"/'COMPLETE'
    return dict(record,kind='PREDICTIVE_TOPOLOGY',projection_pairs=pairs,coarse_scores=True,
        fresh_prior_root=str(source_spec(record['pair'])[0]/'source'),
        method_sources=dict(KD=str(previous/'KD_MEAN'),CAL=str(previous/'CAL'),LOCAL_JOINT=str(previous/'JOINT'),GLOBAL_JOINT=str(global_reference)))


def main():
    previous=json.loads((ROOT/'docs/a100_visual_classical_20261121/native_factorial_statistical_reproduction.json').read_text())
    assert previous['status'].startswith('PASS_')
    begin=time.perf_counter();pending=conditions();active={};done=[];failure=None;completed=0;last=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_predictive_topology_v1.json'
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=(dict(conditions=completed,active_videos={r['pair']+f"_seed{r['seed']}":len(list((Path(r['folder'])/'videos').glob('*/receipt.json'))) for _,_,r in active.values()}) if active else completed),finished_conditions=done,pending=pending,
            active={str(slot):r for slot,(_,_,r) in active.items()},failure=failure,wall_seconds=time.perf_counter()-begin,
            preregistration_sha256=file_sha(prereg),goal_status='ACTIVE'))
    from tools.deduplicate_immutable_predictions import run as deduplicate
    while pending or active:
        changed=False
        for slot,(process,handle,record) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            handle.close();completion=Path(record['folder'])/'completion.json';record=dict(record,exit_code=rc)
            if rc or not completion.is_file():failure=dict(record,last_log=Path(record['log']).read_text()[-7000:])
            else:
                receipt=json.loads(completion.read_text());assert receipt['GT_open_attempts']==receipt['new_teacher_calls']==receipt['new_head_fits']==0
                deduplicate(PREDICTIVE,True)
            done.append(record);completed+=1;del active[slot];changed=True
        if not failure:
            for slot in (0,1):
                if slot in active or not pending:continue
                record=pending.pop(0)
                if (Path(record['folder'])/'completion.json').is_file():done.append(dict(record,reused_sealed=True));completed+=1;changed=True;continue
                log=OUT/'supervisor'/f"predictive_topology_{record['pair']}_seed{record['seed']}.log";handle=log.open('a',buffering=1)
                process=subprocess.Popen([sys.executable,'-u','scripts/a100_visual_classical_predictive_topology.py','--pair',record['pair'],'--seed',str(record['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                active[slot]=process,handle,dict(record,pid=process.pid,log=str(log));changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_PREDICTIVE_TOPOLOGY_GENERATION' if not failure else 'STOPPING_PRESERVING_PREDICTIVE_TOPOLOGY_RUNNING_JOBS');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert completed==6
    state('ALL_PREDICTIVE_TOPOLOGY_RAW_SEALED_GENERATING_QUANTITIES_NO_NEW_GT');records=conditions();roots=[]
    for record in records:
        roots.append(generate_counts(count_record(record)));deduplicate(PREDICTIVE/record['pair'],True)
    tasks=[]
    for record,counts in zip(records,roots):
        for folder in (Path(record['folder']),Path(counts)):
            receipt=json.loads((folder/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/m['method']),record['pair']) for m in receipt['methods'])
    assert len(tasks)==156,len(tasks)
    completed=0;state('ALL156_PREDICTIVE_TOPOLOGY_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;completed+=1;state('ALL156_PREDICTIVE_TOPOLOGY_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/predictive_topology_official.json',dict(status='ALL_PREDICTIVE_TOPOLOGY_OFFICIAL_COMPLETE',methods=156,results=results,goal_status='ACTIVE'))
    state('PREDICTIVE_TOPOLOGY_OFFICIAL_COMPLETE_ANALYZING')
    for command in (['scripts/a100_visual_classical_predictive_topology_analyze.py'],['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','predictive_topology']):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_PREDICTIVE_TOPOLOGY_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
