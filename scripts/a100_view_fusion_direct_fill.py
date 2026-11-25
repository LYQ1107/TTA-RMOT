"""CPU-only direct-teacher baseline for the preregistered multiview study.

Core fits and numerics stay immutable. This supplement has its own raw/count
seal and waits for the core analysis before reading official statistics.
"""
import argparse
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
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.presence.predictions import seal_mask
from scripts.a100_six_direction import SEEDS,PUBLIC,source
from scripts.a100_view_fusion import OUT as CORE,labels

OUT=CORE/'direct_fill_supplement'
STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_MULTIVIEW_DIRECT_FILL_SUPPLEMENT_RESEARCH_GOAL_ACTIVE'
BASE='RAW_TEACHER_FILL';FULL='DS_FULL'


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def generate(pair,seed):
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    guard=generation_guard();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/'quality'/pair/f'seed{seed}';core=CORE/'quality'/pair/f'seed{seed}'
    assert (core/'completion.json').is_file()
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    (root/FULL).symlink_to(core/FULL,target_is_directory=True) if not (root/FULL).exists() else None
    mask=np.zeros(m['candidate_query_rows'],bool)
    for video,bank in sorted(m['videos'].items()):
        queries=[q for q in m['episodes'] if q['video_id']==video]
        with np.load(core/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];unary=a['RAW_FINAL'];full=a[FULL]
        cells,probability,_=labels(pair,video,queries,tracks)
        direct=unary.copy();direct[cells[:,0],cells[:,1]]=np.nansum(probability,1)/np.isfinite(probability).sum(1)
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([lookup[int(t)] for _,t in obs]);bits=plain(direct[:,columns],False)
        destination=root/BASE/'videos'/video;destination.mkdir(parents=True,exist_ok=True)
        np.save(destination/'mask.npy',bits)
        immediate_predictions(root/BASE,type('Context',(),dict(bank=bank,queries=queries))(),bits)
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,DS_FULL=full,RAW_TEACHER_FILL=direct)
        for qi,q in enumerate(queries):mask[q['start']:q['stop']]=bits[qi]
    seal=seal_mask(root/BASE,m,mask,target=target,model=model,method='MULTIVIEW_DIRECT_TEACHER_FILL',GT_open_attempts=len(guard),
        extra=dict(same_all_requested_views_as_core=True,teacher_raw_view_mean_at_known_cells=True,
            otherwise_same_converged_RAW_head=True,new_teacher_encoder_or_head_calls=0,independent_decoder=True))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_MULTIVIEW_DIRECT_TEACHER_FILL',pair=pair,seed=seed,
        methods=[dict(method=BASE,rows=seal['prediction_rows'])],GT_open_attempts=0,new_teacher_encoder_head_calls=0,
        wall_seconds=time.perf_counter()-begin))


def counts(record):
    guard=generation_guard();pair=record['pair'];model,target=PAIRS[pair];m=metadata(target)
    root=Path(record['folder']);destination=root/'countmatched'
    if (destination/'completion.json').is_file():return str(destination)
    pairs=[(FULL,FULL),(BASE,FULL),(FULL,BASE),(BASE,BASE)]
    masks={key:np.zeros(m['candidate_query_rows'],bool) for key in pairs}
    for video,bank in sorted(m['videos'].items()):
        queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        with np.load(root/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];scores={name:a[name] for name in (FULL,BASE)}
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
                independent_per_query_decoder=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    changed={}
    for name in (FULL,BASE):
        old=json.loads((root/name/'predictions_seal.json').read_text())
        new=json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        before={(r['video'],r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name]=sum(before[r['video'],r['expression']]!=r['prediction_sha256'] for r in new['episodes'])
    assert not any(changed.values()),changed
    assert not guard
    atomic_json(destination/'completion.json',dict(status='SEALED_DIRECT_TEACHER_COUNT_CONTROLS',methods=methods,
        CountSelf_changed_query_files=changed,GT_open_attempts=0))
    return str(destination)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==90
    data={};sufficient={};max_error=0.
    names=[FULL,BASE,FULL+'__AT_'+FULL+'_COUNT',BASE+'__AT_'+FULL+'_COUNT',
        FULL+'__AT_'+BASE+'_COUNT',BASE+'__AT_'+BASE+'_COUNT']
    for r in conditions():
        root=Path(r['folder']);data[r['pair'],r['seed']]={n:read_stats(root/n if n in (FULL,BASE) else root/'countmatched'/n) for n in names}
        for name,row in data[r['pair'],r['seed']].items():
            num,den=row['parent_stats'];max_error=max(max_error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            sufficient[r['pair']+f"_seed{r['seed']}/"+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
    assert max_error<1e-8,max_error
    repeated={}
    for pair in PAIRS:
        rows=[data[pair,s] for s in SEEDS]
        repeated[pair]=dict(source=PAIRS[pair][0],target=PAIRS[pair][1],
            mean_HOTA={n:float(np.mean([r[n]['HOTA'] for r in rows])) for n in (FULL,BASE)},
            comparisons={name:comparison([r[a] for r in rows],[r[b] for r in rows]) for name,a,b in (
                ('FULL_vs_DIRECT_TEACHER_FILL',FULL,BASE),
                ('RANK_AT_FULL_COUNT',FULL+'__AT_'+FULL+'_COUNT',BASE+'__AT_'+FULL+'_COUNT'),
                ('RANK_AT_DIRECT_FILL_COUNT',FULL+'__AT_'+BASE+'_COUNT',BASE+'__AT_'+BASE+'_COUNT'))})
    result=dict(status='MULTIVIEW_DIRECT_FILL_SUPPLEMENT_ANALYZED',repeated=repeated,
        all90_supplement_outputs_sealed_before_supplement_GT=True,core468_separate_preregistered_GT_group=True,
        primary_and_parameters_unchanged=True,all_data_development=True,new_teacher_encoder_head_calls=0,
        HOTA_parent_sum_max_error=max_error,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'view_fusion_direct_fill_analysis.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    (PUBLIC/'view_fusion_direct_fill_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='MULTIVIEW_DIRECT_TEACHER_FILL_COMPARISON',repeated=repeated,goal_status='ACTIVE')),flush=True)


def main():
    start=time.perf_counter();pending=conditions();active={};done=[];failure=None;last=0
    def state(status):
        atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),active={str(s):j for s,(_,_,j) in active.items()},
            pending=pending,failure=failure,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    while pending or active:
        changed=False
        for slot,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[slot]
            if code or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-5000:])
            else:done.append(job)
            changed=True
        runnable=[j for j in pending if (CORE/'quality'/j['pair']/f"seed{j['seed']}"/'completion.json').is_file()]
        for slot in range(2):
            if failure or slot in active or not runnable:continue
            job=runnable.pop(0);pending.remove(job)
            if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
            log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            active[slot]=proc,handle,dict(job,pid=proc.pid,log=str(log));changed=True
        if changed or time.perf_counter()-last>45:
            state('AWAITING_OR_GENERATING_DIRECT_TEACHER_FILL_NO_GT' if not failure else 'PRESERVING_ACTIVE_DIRECT_FILL_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    state('ALL18_DIRECT_FILL_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:count_roots=list(pool.map(counts,conditions()))
    tasks=[]
    for r,count_root in zip(conditions(),count_roots):
        for folder in (Path(r['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/e['method']),r['pair']) for e in row['methods'])
    assert len(tasks)==90
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=90,conditions=18,GT_open_attempts=0))
    state('ALL90_DIRECT_FILL_OUTPUTS_SEALED_AWAITING_CORE_ANALYSIS')
    while not (CORE/'analysis.json').is_file():time.sleep(5)
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    state('ALL90_DIRECT_FILL_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=90,results=results,goal_status='ACTIVE'))
    analyze();state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS));p.add_argument('--seed',type=int);args=p.parse_args()
    if args.pair:generate(args.pair,args.seed)
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_DIRECT_FILL_SUPPLEMENT_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
