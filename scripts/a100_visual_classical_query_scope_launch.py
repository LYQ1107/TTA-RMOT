"""CPU same-head minimality; all raw/count outputs seal before105 officials."""
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
from scripts.a100_visual_classical_query_scope import SCOPE,head_sources
from scripts.a100_visual_classical_sparse_observation_evaluate import generate_counts,evaluate_one

STATE=OUT/'supervisor/query_scope_state.json'


def conditions():
    return [dict(pair=pair,seed=seed,folder=str(SCOPE/f'seed{seed}'/pair)) for pair in ('C4','C1') for seed in (0,17,42)]


def generate(record):
    folder=Path(record['folder'])
    logpath=OUT/'supervisor'/f"query_scope_{record['pair']}_seed{record['seed']}.log"
    with logpath.open('a') as log:
        result=subprocess.run([sys.executable,'scripts/a100_visual_classical_query_scope.py',
            '--pair',record['pair'],'--seed',str(record['seed'])],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:
        raise RuntimeError('Query scope generation failed: '+str(logpath)+'\n'+logpath.read_text()[-6000:])
    receipt=json.loads((folder/'completion.json').read_text())
    assert receipt['GT_open_attempts']==receipt['head_updates']==receipt['original_graph_changed_rows']==0
    return dict(record,methods=len(receipt['methods']))


def counts_record(record):
    folder=Path(record['folder'])
    sources={r['method']:str(folder/r['method']) for r in json.loads((folder/'completion.json').read_text())['methods']}
    sources.update({'KD_'+head:str(path) for head,path in head_sources(record['pair'],record['seed']).items()})
    full='ORIGINAL_MEAN_EXTENDED';others=['ORIGINAL_MEAN_CLOTHING','KD_ORIGINAL_MEAN']
    pairs=[(full,full)]+[(other,full) for other in others]+[(rank,other) for other in others for rank in (full,other)]
    return dict(record,method_sources=sources,projection_pairs=pairs,kind='QUERY_SCOPE')


def main():
    begin=time.perf_counter();records=conditions();done=0
    prereg=ROOT/'docs/a100_visual_classical_20261121/preregistration_query_scope_v1.json'
    assert prereg.is_file()
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=done,preregistration_sha256=file_sha(prereg),
            wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    state('GENERATING_ALL_SAME_HEAD_SCOPES_NO_NEW_GT')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(generate,records):
            done+=1
            print(json.dumps(dict(event='QUERY_SCOPE_GENERATED',completed=done,total=6,**row)),flush=True)
            state('GENERATING_ALL_SAME_HEAD_SCOPES_NO_NEW_GT')
    state('DEDUPLICATING_SEALED_SCOPE_OUTPUTS_NO_NEW_GT')
    from tools.deduplicate_immutable_predictions import run as reuse_immutable
    reuse_immutable(SCOPE,True)
    state('GENERATING_ALL_SCOPE_QUANTITIES_NO_NEW_GT');count_roots=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for folder in pool.map(generate_counts,[counts_record(r) for r in records]): count_roots.append(folder)
    tasks=[]
    for record,counts in zip(records,count_roots):
        for folder in (Path(record['folder']),Path(counts)):
            receipt=json.loads((folder/'completion.json').read_text())
            assert receipt['GT_open_attempts']==0
            if 'CountSelf_changed_query_files' in receipt: assert not any(receipt['CountSelf_changed_query_files'].values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in receipt['methods'])
    assert len(tasks)==105,len(tasks)
    state('ALL_SCOPE_PREDICTIONS_SEALED_OFFICIAL_EVALUATING');results={};done=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(evaluate_one,task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            name,row=future.result();results[name]=row;done+=1
            print(json.dumps(dict(event='QUERY_SCOPE_OFFICIAL_PROGRESS',completed=done,total=105)),flush=True)
            state('ALL_SCOPE_PREDICTIONS_SEALED_OFFICIAL_EVALUATING')
    atomic_json(OUT/'analysis/query_scope_official.json',dict(status='ALL_QUERY_SCOPE_OFFICIAL_COMPLETE',methods=105,results=results,goal_status='ACTIVE'))
    state('OFFICIAL_COMPLETE_ANALYZING_SCOPE')
    for command in (['scripts/a100_visual_classical_query_scope_analyze.py'],
        ['scripts/a100_visual_classical_sparse_statistics_reproduce.py','--stem','query_scope']):
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    state('PASS_QUERY_SCOPE_STUDY_RESEARCH_GOAL_ACTIVE')
    print('PASS_QUERY_SCOPE_STUDY_RESEARCH_GOAL_ACTIVE',flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        row=json.loads(STATE.read_text()) if STATE.is_file() else {}
        row.update(status='FAILED_QUERY_SCOPE_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,row)
        raise
