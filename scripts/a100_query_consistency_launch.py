"""Two CPU slots, no GPU fits, no new teacher; await each sealed input."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT as TRANSFER,PAIRS,SEEDS,source
from scripts.a100_query_consistency import OUT

STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_QUERY_CONSISTENCY_SIX_DIRECTION_RESEARCH_GOAL_ACTIVE'


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def counts(record):
    from flex_uda.a100.mechanism import metadata,generation_guard
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();pair=record['pair'];model,target=PAIRS[pair];m=metadata(target)
    root=Path(record['folder']);destination=root/'countmatched'
    if (destination/'completion.json').is_file():return str(destination)
    pairs=[('LATTICE_JOINT','LATTICE_JOINT'),('BASE_PROTOCOL','LATTICE_JOINT'),
           ('LATTICE_JOINT','BASE_PROTOCOL'),('BASE_PROTOCOL','BASE_PROTOCOL')]
    masks={key:np.zeros(m['candidate_query_rows'],bool) for key in pairs}
    for video,bank in sorted(m['videos'].items()):
        queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        with np.load(root/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];scores={name:a[name] for name in ('LATTICE_JOINT','BASE_PROTOCOL')}
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:prior=a['prior']
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        bits={name:np.load(root/name/'videos'/video/'mask.npy') for name in scores}
        for frame in np.unique(obs[:,0]):
            ids=np.flatnonzero(obs[:,0]==frame);ti=columns[ids]
            for qi,q in enumerate(queries):
                order={name:ids[np.lexsort((obs[ids,1],-prior[qi,ti],-value[qi,ti]))] for name,value in scores.items()}
                for (rank,donor),mask in masks.items():
                    n=int(bits[donor][qi,ids].sum());mask[q['start']+order[rank][:n]]=True
    methods=[]
    for (rank,donor),mask in masks.items():
        name=rank+'__AT_'+donor+'_COUNT'
        seal=seal_mask(destination/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(rank_method=rank,count_method=donor,counts_from_GT=False,counts_equal_per_query_frame=True,
                independent_decoder_every_arm=True,no_assumed_query_mutual_exclusivity=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    changed={}
    for name in ('LATTICE_JOINT','BASE_PROTOCOL'):
        old=json.loads((root/name/'predictions_seal.json').read_text());new=json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        before={(r['video'],r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name]=sum(before[r['video'],r['expression']]!=r['prediction_sha256'] for r in new['episodes'])
    assert not any(changed.values()),changed
    assert not guard
    atomic_json(destination/'completion.json',dict(status='SEALED_QUERY_LOGIC_COUNTS',methods=methods,
        CountSelf_changed_query_files=changed,GT_open_attempts=0,independent_decoder_every_arm=True))
    return str(destination)


def main():
    start=time.perf_counter();pending=[dict(kind='DEPENDENCY',pair=p,seed=s) for s in SEEDS for p in ('C1','C4')]
    pending += [dict(kind='CONSISTENCY',**r) for r in conditions()]
    active={};done=[];failure=None;last=0
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),active={str(k):j for k,(_,_,j) in active.items()},
            pending=[{k:v for k,v in j.items() if k!='folder'} for j in pending],failure=failure,goal_status='ACTIVE',
            wall_seconds=time.perf_counter()-start))
    def folder(job):return TRANSFER/'quality'/job['pair']/f"seed{job['seed']}" if job['kind']=='DEPENDENCY' else Path(job['folder'])
    while pending or active:
        changed=False
        for slot,(proc,handle,job) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();del active[slot]
            if rc or not (folder(job)/'completion.json').is_file():failure=dict(job=job,exit_code=rc,last_log=Path(job['log']).read_text()[-7000:])
            else:done.append(job)
            changed=True
        runnable=[j for j in pending if j['kind']=='DEPENDENCY' or (TRANSFER/'quality'/j['pair']/f"seed{j['seed']}"/'completion.json').is_file()]
        for slot in range(2):
            if failure or slot in active or not runnable:continue
            job=runnable.pop(0);pending.remove(job)
            if (folder(job)/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
            log=OUT/'supervisor'/(job['kind']+'_'+job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
            script='scripts/a100_six_direction.py' if job['kind']=='DEPENDENCY' else 'scripts/a100_query_consistency.py'
            proc=subprocess.Popen([sys.executable,'-u',script,'--pair',job['pair'],'--seed',str(job['seed'])],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            active[slot]=proc,handle,dict(job,pid=proc.pid,log=str(log));changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_OR_AWAITING_SIX_DIRECTION_QUERY_CONSISTENCY_NO_GT' if not failure else 'PRESERVING_ACTIVE_QUERY_CONSISTENCY_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    state('ALL18_QUERY_CONSISTENCY_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,counts_root in zip(conditions(),roots):
        for root in (Path(record['folder']),Path(counts_root)):
            receipt=json.loads((root/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(root/r['method']),record['pair']) for r in receipt['methods'])
    assert len(tasks)==252
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=252,conditions=18,GT_open_attempts=0))
    state('ALL252_QUERY_CONSISTENCY_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=252,results=results,goal_status='ACTIVE'))
    state('QUERY_CONSISTENCY_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_query_consistency_analyze.py'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
        previous.update(status='FAILED_QUERY_CONSISTENCY_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE');atomic_json(STATE,previous);raise
