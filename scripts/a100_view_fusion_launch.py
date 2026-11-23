"""Fixed requested-view attribution after the live transfer GPU stage.

This controller shares the existing registry observer. It never starts a
second source/teacher stream, modifies a running numeric helper, or scores
GT before every arm and quantity control in this study has been sealed.
"""
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
from scripts.a100_six_direction import PAIRS,SEEDS,source,OUT as TRANSFER,PUBLIC
from scripts.a100_view_fusion import OUT,METHODS

STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_FIXED_VIEW_FUSION_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
PRIMARY='DS_FULL'
BASELINES=('RAW_FINAL','RAW_MEAN','HARD_FINAL','RAW_SRC_FINAL')


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def projection_pairs():
    pairs=[(PRIMARY,PRIMARY)]
    for baseline in BASELINES:
        pairs.extend(((baseline,PRIMARY),(PRIMARY,baseline),(baseline,baseline)))
    return pairs


def counts(record):
    from flex_uda.a100.mechanism import metadata,generation_guard
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();pair=record['pair'];model,target=PAIRS[pair];m=metadata(target)
    root=Path(record['folder']);destination=root/'countmatched'
    if (destination/'completion.json').is_file():return str(destination)
    names=(PRIMARY,)+BASELINES;pairs=projection_pairs()
    masks={key:np.zeros(m['candidate_query_rows'],bool) for key in pairs}
    for video,bank in sorted(m['videos'].items()):
        queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        with np.load(root/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];scores={name:a[name] for name in names}
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:prior=a['prior']
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        bits={name:np.load(root/name/'videos'/video/'mask.npy') for name in names}
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
                independent_per_query_decoder_every_arm=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    changed={}
    for name in names:
        old=json.loads((root/name/'predictions_seal.json').read_text())
        new=json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        before={(r['video'],r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name]=sum(before[r['video'],r['expression']]!=r['prediction_sha256'] for r in new['episodes'])
    assert not any(changed.values()),changed
    assert not guard
    atomic_json(destination/'completion.json',dict(status='SEALED_FIXED_VIEW_FUSION_COUNTS',methods=methods,
        CountSelf_changed_query_files=changed,GT_open_attempts=0,independent_decoder_every_arm=True))
    return str(destination)


def transfer_GPU_stage_finished():
    state_path=TRANSFER/'supervisor/state.json'
    if not state_path.is_file():return False
    status=json.loads(state_path.read_text())['status']
    return status in ('ALL18_RAW_CONDITIONS_SEALED_GENERATING_QUANTITY_CONTROLS_NO_GT',
        'ALL234_SIX_DIRECTION_OUTPUTS_SEALED_OFFICIAL_EVALUATING','SIX_DIRECTION_OFFICIAL_COMPLETE_ANALYZING',
        'PASS_SIX_DIRECTION_TRANSFER_MATRIX_RESEARCH_GOAL_ACTIVE')


def main():
    plan=json.loads((PUBLIC/'view_fusion_registration_v1.json').read_text())
    assert plan['total_outputs']==468 and plan['GT_used_for_design_or_adaptation'] is False
    start=time.perf_counter();pending=conditions();active={};done=[];failure=None;last=0
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
            active={str(g):j for g,(_,_,j) in active.items()},pending=pending,failure=failure,
            wall_seconds=time.perf_counter()-start,goal_status='ACTIVE',research_goal_complete=False))
    state('AWAITING_LIVE_TRANSFER_GPU_STAGE_RELEASE_NO_GT')
    while not transfer_GPU_stage_finished():time.sleep(5)
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();del active[gpu]
            completion=Path(job['folder'])/'completion.json'
            if rc or not completion.is_file():
                failure=dict(job=job,exit_code=rc,last_log=Path(job['log']).read_text()[-7000:])
            else:
                receipt=json.loads(completion.read_text());assert receipt['GT_open_attempts']==0
                done.append(job)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in output.strip().splitlines()}
            atomic_json(OUT/'supervisor/last_GPU_allocation.json',dict(candidate_GPU_memory_MiB=memory,
                only_start_below_MiB=512,no_other_process_terminated=True))
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():
                    done.append(dict(job,reused_sealed=True));changed=True;continue
                log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u','scripts/a100_view_fusion.py','--pair',job['pair'],
                    '--seed',str(job['seed'])],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_FIXED_VIEW_FUSION_SIX_SETTINGS_NO_GT' if not failure else 'PRESERVING_ACTIVE_FIXED_VIEW_FUSION_AFTER_FAILURE')
            last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==18
    state('ALL18_FIXED_VIEW_FUSION_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            receipt=json.loads((folder/'completion.json').read_text());assert receipt['GT_open_attempts']==0
            assert not any(receipt.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in receipt['methods'])
    assert len(tasks)==468
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=468,conditions=18,
        GT_open_attempts=0,all_raw_and_quantity_controls_sealed=True))
    state('ALL468_FIXED_VIEW_FUSION_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=468,results=results,goal_status='ACTIVE'))
    state('FIXED_VIEW_FUSION_OFFICIAL_COMPLETE_ANALYZING')
    subprocess.run([sys.executable,'scripts/a100_view_fusion_analyze.py'],cwd=ROOT,check=True)
    state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
        previous.update(status='FAILED_FIXED_VIEW_FUSION_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
        atomic_json(STATE,previous);raise
