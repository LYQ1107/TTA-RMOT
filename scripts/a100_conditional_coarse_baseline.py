"""Preserve strong track-mean ordinary KD for the new shared-input study.

This registered supplement does not modify the live conditional recipe.
Every target is the raw own-policy mean; all image fields are deterministic
summaries of the same three observations given to the primary learner.
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
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.observation_learning_inputs import ObservationVideo,prefix_data,DIMENSION,ANSWERS
from scripts.a100_conditional_six_setting import OUT as PARENT,STATE as PARENT_STATE,PRIMARY
from scripts.a100_six_direction import OUT as SIX,PUBLIC,SEEDS

OUT=SIX/'conditional_coarse_baseline_20261123';STATE=OUT/'supervisor/state.json'
REGISTRATION=PUBLIC/'conditional_coarse_registration_v1.json';STEM='conditional_coarse'
SUCCESS='PASS_CONDITIONAL_COARSE_ORDINARY_BASELINE_SIX_SETTINGS_GOAL_ACTIVE'
METHODS=('COARSE_SOFT_FINAL','COARSE_SOFT_MEAN','COARSE_HARD_FINAL','COARSE_HARD_MEAN','COARSE_TEACHER_FILL')
CONTRASTS={'FULL_vs_'+b:(PRIMARY,b) for b in METHODS}
for b in METHODS:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b]=(PRIMARY+'__AT_'+b+'_COUNT',b+'__AT_'+b+'_COUNT')


def conditions():return [dict(pair=p,seed=s,folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def coarse_inputs(video):
    raw=video.raw_bags;present=video.present;count=present.sum(1)[None,:,None]
    values=raw.sum(2)/count
    # Same raw histograms: sqrt(mean h) rather than mean(sqrt h).
    colour=raw[...,768:1080];flags=raw[...,1080:1083]
    mass=np.square(colour).sum(2).reshape(raw.shape[:2]+(3,104))
    denom=flags.sum(2)
    values[...,768:1080]=np.sqrt(mass/np.maximum(denom[...,None],1)).reshape(raw.shape[:2]+(312,))
    return values.astype('float32')


def supervision(video,values):
    rows=video.supervision('DESIGN');groups=rows['bag_group'];n=int(groups.max())+1
    # Every observed frame/view-mixture is counted once in the raw bag mean.
    raw_y=np.bincount(groups,weights=rows['raw_target'])/np.bincount(groups)
    first=np.asarray([np.flatnonzero(groups==g)[0] for g in range(n)])
    # np.unique(axis=0) in the shared input helper uses this same tuple
    # order. Read actual indices rather than trying to infer an ID from
    # feature hashes, which would fail on genuinely identical feature bags.
    doc=json.loads((ANSWERS/'policy_answers'/video.pair/video.video/'DESIGN_answers.json').read_text())
    keys=np.asarray(sorted({(int(r['local_query']),int(r['local_track'])) for r in doc['answers']}),int)
    assert len(keys)==n and np.array_equal(keys[:,0],rows['query_group'][first])
    current=values[keys[:,0],keys[:,1]]
    # Only the local current vector is consumed by the ordinary model.
    # Use a one-present-slot bag to share the already audited optimizer API.
    bags=np.zeros((len(current),3,DIMENSION),dtype='float32');bags[:,0]=current
    masks=np.zeros((len(current),3),bool);masks[:,0]=True
    return dict(bags=bags,present=masks,current=current,raw_target=raw_y.astype('float32'),
        query_group=keys[:,0],bag_group=np.arange(len(current)),keys=keys,
        report=dict(rows['report'],query_track_targets=len(current),raw_frame_mean_target=True))


def run(pair,seed):
    import torch
    from flex_uda.active.observation_learning_fit import fit,forward
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_device(0)
    started=time.perf_counter();model,target=PAIRS[pair];md=metadata(target);root=OUT/pair/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').is_file():return
    stream=root/'stream.pt';previous={};receipts=[];next_video=0
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False);previous,receipts,next_video=[saved[k] for k in ('previous','receipts','next_video')]
    total=np.zeros(DIMENSION);square=total.copy();count=0;records=[]
    for vi,video in enumerate(sorted(md['videos'])):
        stamp=time.perf_counter();inputs=ObservationVideo(pair,video);values=coarse_inputs(inputs);r=supervision(inputs,values);records.append(r)
        flat=values.reshape(-1,DIMENSION).astype(float);total+=flat.sum(0);square+=np.square(flat).sum(0);count+=len(flat)
        mean=total/count;std=np.maximum(np.sqrt(np.maximum(square/count-mean**2,0)),1e-3)
        if vi<next_video:continue
        data=prefix_data(records,mean,std);current=((values.reshape(-1,DIMENSION)-mean)/std).astype('float32')
        bags=np.zeros((len(current),3,DIMENSION),dtype='float32');bags[:,0]=current
        masks=np.zeros((len(current),3),bool);masks[:,0]=True
        bb=torch.as_tensor(bags,device='cuda:0');mm=torch.as_tensor(masks,device='cuda:0');cc=torch.as_tensor(current,device='cuda:0')
        scores={};fits={};new_previous={}
        for label,recipe in (('SOFT','FRAME_SOFT'),('HARD','FRAME_HARD')):
            head,early,checkpoint,receipt=fit(data,recipe,seed=seed,mean=mean,std=std,previous=previous.get(label))
            with torch.no_grad():
                final=torch.sigmoid(forward(head,'local',bb,mm,cc)).cpu().numpy().reshape(values.shape[:2])
                first=torch.sigmoid(forward(early,'local',bb,mm,cc)).cpu().numpy().reshape(values.shape[:2])
            scores['COARSE_'+label+'_FINAL']=final;scores['COARSE_'+label+'_MEAN']=.5*(first+final)
            fits[label]=receipt;new_previous[label]=checkpoint
        fill=scores['COARSE_SOFT_FINAL'].copy();keys=r['keys'];fill[keys[:,0],keys[:,1]]=r['raw_target'];scores['COARSE_TEACHER_FILL']=fill
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=inputs.tracks,**scores)
        for name,score in scores.items():
            bits=score[:,inputs.columns]>.5;destination=root/name/'videos'/video;destination.mkdir(parents=True,exist_ok=True)
            np.save(destination/'mask.npy',bits);immediate_predictions(root/name,type('Context',(),dict(bank=inputs.bank,queries=inputs.queries))(),bits)
        receipt=dict(video=video,index=vi,fits=fits,teacher_evidence=r['report'],GT_open_attempts=len(guard),actual_head_fits=2,
            same_three_state_information=True,no_source_residual=True,wall_seconds=time.perf_counter()-stamp)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt);next_video=vi+1;previous=new_previous
        temp=stream.with_suffix('.tmp');torch.save(dict(previous=previous,receipts=receipts,next_video=next_video),temp);temp.replace(stream)
    methods=[]
    for name in METHODS:
        mask=np.zeros(md['candidate_query_rows'],bool)
        for video in sorted(md['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in md['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        row=seal_mask(root/name,md,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,no_source_logit_residual=True,current_video_transductive=True,no_future_answers=True))
        methods.append(dict(method=name,rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_CONDITIONAL_COARSE_ORDINARY_NO_GT',methods=methods,receipts=receipts,
        GT_open_attempts=0,target_fit_failures=[dict(video=r['video'],arm=a,metrics=f['metrics']) for r in receipts for a,f in r['fits'].items() if not f['target_fit_gate_pass']],
        wall_seconds=time.perf_counter()-started,goal_status='ACTIVE'))


def counts(record):
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();pair=record['pair'];model,target=PAIRS[pair];md=metadata(target)
    root=Path(record['folder']);parent=PARENT/pair/f"seed{record['seed']}";destination=root/'countmatched'
    if (destination/'completion.json').is_file():return str(destination)
    pairs=[(PRIMARY,PRIMARY)]+[p for b in METHODS for p in ((b,PRIMARY),(PRIMARY,b),(b,b))]
    masks={p:np.zeros(md['candidate_query_rows'],bool) for p in pairs}
    for video,bank in sorted(md['videos'].items()):
        obs=np.asarray(bank['observations']);queries=[q for q in md['episodes'] if q['video_id']==video]
        with np.load(root/'videos'/video/'coarse_scores.npz') as a:
            tracks=a['tracks'];lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
            scores={name:a[name][:,columns] for name in METHODS}
        scores[PRIMARY]=np.load(parent/'videos'/video/(PRIMARY+'_scores.npy'),mmap_mode='r')
        bits={name:np.load((parent if name==PRIMARY else root)/name/'videos'/video/'mask.npy') for name in (*METHODS,PRIMARY)}
        for frame in np.unique(obs[:,0]):
            ids=np.flatnonzero(obs[:,0]==frame)
            for qi,q in enumerate(queries):
                order={name:ids[np.lexsort((obs[ids,1],-np.asarray(score[qi,ids])))] for name,score in scores.items()}
                for (rank,donor),mask in masks.items():mask[q['start']+order[rank][:int(bits[donor][qi,ids].sum())]]=True
    methods=[]
    for (rank,donor),mask in masks.items():
        name=rank+'__AT_'+donor+'_COUNT';row=seal_mask(destination/name,md,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(counts_from_GT=False,counts_equal_per_query_frame=True,rank_method=rank,count_method=donor))
        methods.append(dict(method=name,rows=row['prediction_rows']))
    changed={}
    for name in (*METHODS,PRIMARY):
        old=json.loads(((parent if name==PRIMARY else root)/name/'predictions_seal.json').read_text())
        new=json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text());lookup={(r['video'],r['expression']):r['prediction_sha256'] for r in old['episodes']}
        changed[name]=sum(lookup[r['video'],r['expression']]!=r['prediction_sha256'] for r in new['episodes'])
    assert not guard and not any(changed.values()),changed
    atomic_json(destination/'completion.json',dict(status='SEALED_CONDITIONAL_COARSE_COUNTS_NO_GT',methods=methods,GT_open_attempts=0,CountSelf_changed_query_files=changed))
    return str(destination)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    data={};statistics={};summary={};error=0.
    for record in conditions():
        root=Path(record['folder']);slug=record['pair']+f"_seed{record['seed']}"
        raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        paths={r['method']:root/r['method'] for r in raw['methods']};paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        paths[PRIMARY]=PARENT/record['pair']/f"seed{record['seed']}"/PRIMARY
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in (*METHODS,PRIMARY)},target_fit_failures=raw['target_fit_failures'])
    repeated={p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in (*METHODS,PRIMARY)},
        comparisons={label:comparison([data[p+f'_seed{s}'][a] for s in SEEDS],[data[p+f'_seed{s}'][b] for s in SEEDS]) for label,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    assert error<1e-8 and len(statistics)==396
    result=dict(status='CONDITIONAL_COARSE_ORDINARY_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        HOTA_parent_sum_max_error=error,shared_primary_from_original_sealed_study=True,all_own_and_parent_outputs_sealed_before_this_GT=True,
        all_data_development=True,no_independent_blind_validation=True,quality_replay_not_cold_cost=True,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics,separators=(',',':'))+'\n')


def main():
    reg=json.loads(REGISTRATION.read_text())
    for name,sha in reg['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    STATE.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(STATE,dict(status='AWAITING_MAIN_CONDITIONAL_GPU_STAGE_RELEASE',pid=os.getpid(),completed=0))
    while True:
        previous=json.loads(PARENT_STATE.read_text()) if PARENT_STATE.is_file() else {}
        if previous.get('status','').startswith('FAILED_'):raise RuntimeError('Resolve primary study first')
        if previous.get('status') in ('ALL18_CONDITIONAL_RAW_SEALED_GENERATING_COUNTS_NO_GT','ALL1692_CONDITIONAL_OUTPUTS_SEALED_OFFICIAL_EVALUATING','PASS_CONDITIONAL_OBSERVATION_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'):break
        time.sleep(15)
    pending=conditions();done=[];active={};failure=None
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        pending=pending,active={str(g):j for g,(_,_,j) in active.items()},failure=failure,goal_status='ACTIVE'))
    while pending or active:
        for gpu,(proc,handle,job) in list(active.items()):
            if proc.poll() is None:continue
            handle.close();del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=proc.returncode,last_log=Path(job['log']).read_text()[-8000:])
            else:done.append(job)
        free=[g for g in (0,1,2,3) if g not in active]
        if pending and free and not failure:
            text=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,free)),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()}
            for gpu in free:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0);log=STATE.parent/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu)
        state('RUNNING_CONDITIONAL_COARSE_ORDINARY_NO_GT' if not failure else 'PRESERVING_HEALTHY_COARSE_WORKERS_AFTER_FAILURE')
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(5)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    # Primary study can finish its own independent evaluation earlier. This
    # supplementary recipe is fixed without reading those quality numbers.
    while not (PARENT/'all_predictions_sealed_before_GT.json').is_file():time.sleep(15)
    tasks=[]
    for record,counts_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(counts_root)):
            doc=json.loads((folder/'completion.json').read_text());assert doc['GT_open_attempts']==0
            tasks.extend((str(folder/r['method']),record['pair']) for r in doc['methods'])
    assert len(tasks)==378
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=378,parent_methods=1692,GT_open_attempts=0))
    state('ALL378_COARSE_OWN_OUTPUTS_AND1692_PARENT_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    # Wait for parent aggregate instead of guessing an individual result
    # path. This cannot refit any learner or change its fixed predictions.
    while not (PARENT/'official.json').is_file():time.sleep(15)
    atomic_json(OUT/'official.json',dict(methods=378,results=results));analyze();state(SUCCESS)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS));p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.pair:
        import fcntl
        root=OUT/a.pair/f'seed{a.seed}';root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);run(a.pair,a.seed)
    else:
        try:main()
        except Exception as error:
            old=json.loads(STATE.read_text()) if STATE.is_file() else {};old.update(status='FAILED_CONDITIONAL_COARSE_BASELINE',error=repr(error));atomic_json(STATE,old);raise
