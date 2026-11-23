"""Six-setting conditional-observation learning against adequate ordinary KD.

This is a new matched study, not the archived v8P experiment. Shared dense
source features are prepaid research inputs. This run cannot establish a
single-method cold deployment time or an independent holdout result.
"""
import argparse
import concurrent.futures
import hashlib
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
from flex_uda.active.observation_learning_inputs import ObservationVideo,prefix_data,DIMENSION
from scripts.a100_dense_rgb_collect import OUT as DENSE_RGB
from scripts.a100_six_direction import OUT as SIX,PUBLIC,SEEDS

OUT=SIX/'conditional_observation_quality_20261123';STATE=OUT/'supervisor/state.json'
REGISTRATION=PUBLIC/'conditional_observation_registration_v1.json'
SUCCESS='PASS_CONDITIONAL_OBSERVATION_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
STEM='conditional_observation'
ARMS=('FRAME_SOFT','FRAME_HARD','FLAT_SOFT','FLAT_HARD','DIRECT_POINT','DIRECT_BALANCED',
      'DIRECT_PROFILE','CONDITIONAL_BALANCED','FULL','FULL_PRIOR','FULL_UNIFORM','FULL_COVERAGE')
METHODS=tuple(a+'_'+d for a in ARMS for d in ('FINAL','MEAN'))
PRIMARY='FULL_FINAL';BASELINES=tuple(m for m in METHODS if m!=PRIMARY)
CONTRASTS={'FULL_vs_'+b:(PRIMARY,b) for b in BASELINES}
CONTRASTS.update(CONDITIONAL_STRUCTURE=(PRIMARY,'DIRECT_PROFILE_FINAL'),
    QUERY_PROFILE=(PRIMARY,'CONDITIONAL_BALANCED_FINAL'),
    EXPOSURE_BALANCE=('DIRECT_BALANCED_FINAL','DIRECT_POINT_FINAL'),
    PROFILE_ON_DIRECT=('DIRECT_PROFILE_FINAL','DIRECT_BALANCED_FINAL'))
for b in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b]=(PRIMARY+'__AT_'+b+'_COUNT',b+'__AT_'+b+'_COUNT')


def conditions():
    return [dict(pair=p,seed=s,folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def run(pair,seed):
    import torch
    from flex_uda.active.observation_learning_fit import fit,forward
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);start=time.perf_counter();source_model,target=PAIRS[pair]
    root=OUT/pair/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').is_file():return
    metadata_bank=metadata(target);stream=root/'stream.pt';next_video=0;previous={};receipts=[]
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False)
        previous,receipts,next_video=[saved[k] for k in ('previous','receipts','next_video')]
    total=np.zeros(DIMENSION);square=total.copy();observations=0
    policies=('DESIGN','PRIOR','UNIFORM','COVERAGE_ONLY');records={p:[] for p in policies}
    for vi,video in enumerate(sorted(metadata_bank['videos'])):
        stamp=time.perf_counter();video_inputs=ObservationVideo(pair,video)
        values=video_inputs.raw_bags[:,video_inputs.present].reshape(-1,DIMENSION).astype(float)
        total+=values.sum(0);square+=np.square(values).sum(0);observations+=len(values)
        mean=total/observations;std=np.maximum(np.sqrt(np.maximum(square/observations-mean**2,0)),1e-3)
        evidence={}
        for policy in policies:
            current=video_inputs.supervision(policy);records[policy].append(current);evidence[policy]=current['report']
        if vi<next_video:continue
        atomic_json(root/'progress.json',dict(pair=pair,seed=seed,completed=vi,total=len(metadata_bank['videos']),video=video,stage='FITTING'))
        prepared={};models={};fits={};cache={};new_previous={};actual_fits=0
        for arm in ARMS:
            policy={'FULL_PRIOR':'PRIOR','FULL_UNIFORM':'UNIFORM','FULL_COVERAGE':'COVERAGE_ONLY'}.get(arm,'DESIGN')
            recipe='FULL' if arm.startswith('FULL_') else arm
            if policy not in prepared:prepared[policy]=prefix_data(records[policy],mean,std)
            data=prepared[policy];old=previous.get(arm);h=hashlib.sha256(recipe.encode())
            for field in ('bags','present','current','raw_target','group','balanced_weight','confidence'):
                h.update(np.ascontiguousarray(data[field]).tobytes())
            if old is not None:
                for k,v in sorted(old['model'].items()):h.update(k.encode());h.update(v.numpy().tobytes())
                h.update(np.asarray(old['mean']).tobytes());h.update(np.asarray(old['std']).tobytes())
            key=h.hexdigest()
            if key in cache:
                owner,model,early,checkpoint,receipt=cache[key]
                receipt=dict(receipt,identical_own_prefix_fit_reused_from=owner)
            else:
                model,early,checkpoint,receipt=fit(data,recipe,seed=seed,mean=mean,std=std,previous=old)
                actual_fits+=1;cache[key]=arm,model,early,checkpoint,receipt
            models[arm]=(model,early,checkpoint['kind']);new_previous[arm]=checkpoint;fits[arm]=receipt
            atomic_json(root/'progress.json',dict(pair=pair,seed=seed,completed=vi,total=len(metadata_bank['videos']),
                video=video,stage='FITTING',last_arm=arm,completed_arms=len(models),fit_gate_pass=receipt['target_fit_gate_pass']))
        # Dense image descriptors must have been completed, checked against
        # the fixed anchors, and offered to every learner before inference.
        wait=time.perf_counter();rgb_receipt=DENSE_RGB/target/video/'receipt.json'
        while not rgb_receipt.is_file():
            status_path=DENSE_RGB/'supervisor/state.json'
            state=json.loads(status_path.read_text()) if status_path.is_file() else {}
            if state.get('status','').startswith('FAILED_'):raise RuntimeError('Resolve dense RGB preparation first')
            atomic_json(root/'progress.json',dict(pair=pair,seed=seed,completed=vi,total=len(metadata_bank['videos']),
                video=video,stage='WAITING_DENSE_RGB_SHARED_INPUTS',wait_seconds=time.perf_counter()-wait))
            time.sleep(15)
        waiting=time.perf_counter()-wait
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        nq,no=len(video_inputs.queries),len(video_inputs.obs);prediction_begin=time.perf_counter()
        scores={m:np.lib.format.open_memmap(local/(m+'_scores.npy'),mode='w+',dtype='float32',shape=(nq,no)) for m in METHODS}
        normalized=((video_inputs.raw_bags-mean)/std).astype('float32');normalized[:,~video_inputs.present]=0
        with torch.no_grad():
            for a in range(0,nq*no,2048):
                flat=np.arange(a,min(a+2048,nq*no));qi,rows=flat//no,flat%no;ti=video_inputs.columns[rows]
                current=video_inputs.dense_rows(qi,rows);current=((current-mean)/std).astype('float32')
                b=torch.as_tensor(normalized[qi,ti],device='cuda:0')
                m=torch.as_tensor(video_inputs.present[ti],device='cuda:0');c=torch.as_tensor(current,device='cuda:0')
                for arm,(model,early,kind) in models.items():
                    final=torch.sigmoid(forward(model,kind,b,m,c)).cpu().numpy()
                    first=torch.sigmoid(forward(early,kind,b,m,c)).cpu().numpy()
                    scores[arm+'_FINAL'].reshape(-1)[flat]=final
                    scores[arm+'_MEAN'].reshape(-1)[flat]=.5*(first+final)
        for name,score in scores.items():
            score.flush();bits=np.asarray(score)>.5;dest=root/name/'videos'/video;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'mask.npy',bits)
            immediate_predictions(root/name,type('Context',(),dict(bank=video_inputs.bank,queries=video_inputs.queries))(),bits)
        torch.cuda.synchronize();prediction_seconds=time.perf_counter()-prediction_begin
        receipt=dict(video=video,index=vi,fits=fits,teacher_evidence=evidence,actual_head_fits=actual_fits,
            exact_identical_fit_reuse=len(ARMS)-actual_fits,GT_open_attempts=len(guard),
            normalization_rows=observations,normalization_current_and_past_anchors_only=True,
            dense_shared_feature_wait_seconds=waiting,prediction_and_output_wall_seconds=prediction_seconds,
            wall_seconds=time.perf_counter()-stamp,quality_replay_not_deployment_cost=True,
            historical_full_PCD_version_identical_for_all_heads=True)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt);previous=new_previous;next_video=vi+1
        temporary=stream.with_suffix('.tmp');torch.save(dict(previous=previous,receipts=receipts,next_video=next_video),temporary);temporary.replace(stream)
        del prepared,models,cache,scores;torch.cuda.empty_cache()
    sealed=[]
    for name in METHODS:
        mask=np.zeros(metadata_bank['candidate_query_rows'],bool)
        for video in sorted(metadata_bank['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in metadata_bank['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        row=seal_mask(root/name,metadata_bank,mask,target=target,model=source_model,method=name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,current_video_transductive=True,no_future_answers_or_normalization=True,
                no_source_logit_residual=True,all_heads_same_dense_input_version=True,quality_replay_not_cold_deployment=True))
        sealed.append(dict(method=name,rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_CONDITIONAL_OBSERVATION_RAW_NO_GT',pair=pair,seed=seed,
        methods=sealed,receipts=receipts,GT_open_attempts=0,actual_head_fits=sum(r['actual_head_fits'] for r in receipts),
        target_fit_failures=[dict(video=r['video'],arm=a,metrics=f['metrics']) for r in receipts for a,f in r['fits'].items() if not f['target_fit_gate_pass']],
        new_teacher_or_student_calls=0,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))


def projection_pairs():
    return [(PRIMARY,PRIMARY)]+[p for b in BASELINES for p in ((b,PRIMARY),(PRIMARY,b),(b,b))]


def counts(record):
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();pair=record['pair'];model,target=PAIRS[pair];md=metadata(target)
    root=Path(record['folder']);destination=root/'countmatched'
    if (destination/'completion.json').is_file():return str(destination)
    pairs=projection_pairs();masks={p:np.zeros(md['candidate_query_rows'],bool) for p in pairs}
    for video,bank in sorted(md['videos'].items()):
        queries=[q for q in md['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        scores={name:np.load(root/'videos'/video/(name+'_scores.npy'),mmap_mode='r') for name in METHODS}
        bits={name:np.load(root/name/'videos'/video/'mask.npy') for name in METHODS}
        for frame in np.unique(obs[:,0]):
            ids=np.flatnonzero(obs[:,0]==frame)
            for qi,q in enumerate(queries):
                order={name:ids[np.lexsort((obs[ids,1],-np.asarray(score[qi,ids])))] for name,score in scores.items()}
                for (rank,donor),mask in masks.items():
                    n=int(bits[donor][qi,ids].sum());mask[q['start']+order[rank][:n]]=True
    methods=[]
    for (rank,donor),mask in masks.items():
        name=rank+'__AT_'+donor+'_COUNT'
        row=seal_mask(destination/name,md,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(rank_method=rank,count_method=donor,counts_from_GT=False,counts_equal_per_query_frame=True))
        methods.append(dict(method=name,rows=row['prediction_rows']))
    changed={}
    for name in METHODS:
        before=json.loads((root/name/'predictions_seal.json').read_text())
        after=json.loads((destination/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        lookup={(r['video'],r['expression']):r['prediction_sha256'] for r in before['episodes']}
        changed[name]=sum(lookup[r['video'],r['expression']]!=r['prediction_sha256'] for r in after['episodes'])
    assert not guard and not any(changed.values()),changed
    atomic_json(destination/'completion.json',dict(status='SEALED_CONDITIONAL_OBSERVATION_COUNTS_NO_GT',methods=methods,
        CountSelf_changed_query_files=changed,GT_open_attempts=0))
    return str(destination)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==1692
    data={};statistics={};summary={};error=0.
    for record in conditions():
        root=Path(record['folder']);slug=record['pair']+f"_seed{record['seed']}"
        raw=json.loads((root/'completion.json').read_text());cnt=json.loads((root/'countmatched/completion.json').read_text())
        paths={r['method']:root/r['method'] for r in raw['methods']}
        paths.update({r['method']:root/'countmatched'/r['method'] for r in cnt['methods']})
        data[slug]={name:read_stats(path) for name,path in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},target_fit_failures=raw['target_fit_failures'],
            actual_head_fits=raw['actual_head_fits'],CountSelf_changed_query_files=cnt['CountSelf_changed_query_files'])
    repeated={}
    for pair in PAIRS:
        keys=[pair+f'_seed{s}' for s in SEEDS]
        interaction=[data[k]['FULL_FINAL']['HOTA']-data[k]['CONDITIONAL_BALANCED_FINAL']['HOTA']-
            data[k]['DIRECT_PROFILE_FINAL']['HOTA']+data[k]['DIRECT_BALANCED_FINAL']['HOTA'] for k in keys]
        repeated[pair]=dict(mean_HOTA={n:float(np.mean([data[k][n]['HOTA'] for k in keys])) for n in METHODS},
            comparisons={label:comparison([data[k][a] for k in keys],[data[k][b] for k in keys]) for label,(a,b) in CONTRASTS.items()},
            factorization_profile_interaction=dict(per_seed=interaction,mean=float(np.mean(interaction)),CI95=None,
                scope='Descriptive four-arm interaction; no population or independent holdout claim.'))
    assert error<1e-8 and len(statistics)==1692,(error,len(statistics))
    result=dict(status='CONDITIONAL_OBSERVATION_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        HOTA_parent_sum_max_error=error,all_outputs_sealed_before_this_GT=True,manual_target_GT_in_adaptation=False,
        all_data_development=True,independent_blind_validation=False,quality_replay_not_cold_deployment=True,
        latent_bag_support_not_semantically_identified=True,raw_view_projection_and_CLIP_anchor_approximation_shared=True,
        no_claim_classical_identity_is_novel_theorem=True,goal_status='ACTIVE',research_goal_complete=False)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/(STEM+'_analysis.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/(STEM+'_parent_statistics.json')).write_text(json.dumps(statistics,separators=(',',':'))+'\n')


def main():
    plan=json.loads(REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    pending=conditions();done=[];active={};failure=None;start=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        active={str(g):j for g,(_,_,j) in active.items()},pending=pending,failure=failure,
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    STATE.parent.mkdir(parents=True,exist_ok=True);state('QUEUED_CONDITIONAL_LEARNING_SIX_SETTINGS_NO_GT')
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            if proc.poll() is None:continue
            handle.close();del active[gpu]
            if proc.returncode or not (Path(job['folder'])/'completion.json').is_file():
                failure=dict(job=job,exit_code=proc.returncode,last_log=Path(job['log']).read_text()[-8000:])
            else:done.append(job)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
                log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_CONDITIONAL_LEARNING_SIX_SETTINGS_NO_GT' if not failure else 'PRESERVING_HEALTHY_CONDITIONAL_WORKERS_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(5)
    state('ALL18_CONDITIONAL_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks)==1692
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=1692,GT_open_attempts=0))
    state('ALL1692_CONDITIONAL_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=1692,results=results));analyze();state(SUCCESS)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS));p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.pair:
        import fcntl
        root=OUT/a.pair/f'seed{a.seed}';root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);run(a.pair,a.seed)
    else:
        try:main()
        except Exception as error:
            old=json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_CONDITIONAL_SIX_SETTING_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE');atomic_json(STATE,old);raise
