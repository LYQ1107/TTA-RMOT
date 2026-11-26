"""A prespecified CPU follow-up; all66 new outputs seal before its GT."""
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
from scripts.a100_visual_classical_null_set import NULL
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/null_set_state.json'
SUCCESS='PASS_NULL_SET_SAFEGUARD_STUDY_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair='C4',seed=s,folder=str(NULL/f'seed{s}'/'C4')) for s in (0,17,42)]


def count_record(record):
    full='SAFE_MEAN';others=['KD_MEAN_PRESENCE','GRAPH_MEAN_PRESENCE']
    pairs=[(full,full)]+[(other,full) for other in others]+[(rank,other) for other in others for rank in (full,other)]
    return dict(record,projection_pairs=pairs,kind='NULL_SET',fresh_prior_root=str(source_spec('C4')[0]/'source'))


def main():
    begin=time.perf_counter();done=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_null_set_v1.json'
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=done,preregistration_sha256=file_sha(prereg),wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    while True:
        previous=json.loads((OUT/'supervisor/strict_rgb_state.json').read_text())
        if previous['status'].startswith('FAILED_'):raise RuntimeError('Strict RGB source study failed; preserve unfinished follow-up.')
        if previous['status']=='PASS_STRICT_RGB_PARITY_RESEARCH_GOAL_ACTIVE':break
        state('WAITING_FOR_REGISTERED_RGB_SOURCES_NO_METRICS_READ');time.sleep(15)
    from tools.deduplicate_immutable_predictions import run as deduplicate
    records=conditions()
    for record in records:
        state('GENERATING_NULL_SET_ALL_ARMS_NO_NEW_GT')
        log=OUT/'supervisor'/f"null_set_seed{record['seed']}.log"
        with log.open('a') as handle:
            result=subprocess.run([sys.executable,'-u','scripts/a100_visual_classical_null_set.py','--seed',str(record['seed'])],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        if result.returncode:raise RuntimeError(str(log)+'\n'+log.read_text()[-6000:])
        done+=1;deduplicate(NULL,True)
    state('ALL_NULL_SET_RAW_SEALED_GENERATING_QUANTITIES_NO_NEW_GT');roots=[]
    for record in records:
        roots.append(generate_counts(count_record(record)));deduplicate(NULL/f"seed{record['seed']}",True)
    tasks=[]
    for record,counts in zip(records,roots):
        for root in (Path(record['folder']),Path(counts)):
            receipt=json.loads((root/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(root/r['method']),'C4') for r in receipt['methods'])
    assert len(tasks)==66,len(tasks)
    done=0;state('ALL66_NULL_SET_OUTPUTS_SEALED_OFFICIAL_EVALUATING');results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;done+=1;state('ALL66_NULL_SET_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/null_set_official.json',dict(status='ALL_NULL_SET_OFFICIAL_COMPLETE',methods=66,results=results,goal_status='ACTIVE'))
    state('NULL_SET_OFFICIAL_COMPLETE_ANALYZING')
    for command in (['scripts/a100_visual_classical_null_set_analyze.py'],['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','null_set']):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_NULL_SET_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
