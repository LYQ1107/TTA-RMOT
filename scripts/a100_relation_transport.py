"""Preregistered frame-dependent relation transport versus matched ordinary KD.

The source network stays frozen. Classical within-object nuisance elimination
motivates a task-specific relation field, not a generic score-calibration claim.
All 450 raw/count outputs precede this study's isolated official GT evaluation.
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
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.relational_transport import transport,horizontal_direction
from flex_uda.active.relational_features import geometry,extra_features,moments,frame_supervision,references
from scripts.a100_six_direction import OUT as SIX,PUBLIC,SEEDS,source
from scripts.a100_view_fusion import OUT as CORE
from scripts.a100_view_fusion_flat_rgb import OUT as RGB

OUT=SIX/'relational_transport/quality_study';PREP=SIX/'relational_transport/teacher_prepare'
STATE=OUT/'supervisor/state.json';SUCCESS='PASS_RELATION_TRANSPORT_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE'
METHODS=('COARSE_DS','COARSE_RGB','FRAME_KD_FINAL','FRAME_TEACHER_FILL','RELATION_FULL',
    'RELATION_TIME_AVERAGE','RELATION_SHUFFLED','RGB_RELATION','UNCENTERED_RELATION')
PRIMARY='RELATION_FULL'
BASELINES=('COARSE_DS','COARSE_RGB','FRAME_KD_FINAL','FRAME_TEACHER_FILL','RELATION_TIME_AVERAGE')
CONTRASTS={
    **{'FULL_vs_'+b:(PRIMARY,b) for b in BASELINES},
    'TIME_ALIGNMENT_vs_SHUFFLE':(PRIMARY,'RELATION_SHUFFLED'),
    'OBJECT_REFERENCE_CONDITIONAL':(PRIMARY,'UNCENTERED_RELATION'),
    'RELATION_ON_RAW_RGB':('RGB_RELATION','COARSE_RGB'),
    'OLD_STRUCTURE_GIVEN_RELATION':(PRIMARY,'RGB_RELATION')}
for b in BASELINES:
    CONTRASTS['RANK_AT_FULL_COUNT_vs_'+b]=(PRIMARY+'__AT_'+PRIMARY+'_COUNT',b+'__AT_'+PRIMARY+'_COUNT')
    CONTRASTS['RANK_AT_BASE_COUNT_vs_'+b]=(PRIMARY+'__AT_'+b+'_COUNT',b+'__AT_'+b+'_COUNT')


def conditions():return [dict(pair=p,seed=s,folder=str(OUT/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def coarse_features(pcd,colours,prior,tracks,obs,texts):
    from flex_uda.active.context_cohort import distillation_features
    from flex_uda.active.semantic_colour import appearance_cues
    lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
    cue,available,_,_=appearance_cues(colours,texts,lengths)
    raw=np.concatenate((distillation_features(pcd,colours,texts,'extended'),
        np.where(available,2*cue-1,0).astype('float32')[:,:,None]),axis=2)
    assert raw.shape==prior.shape+(1081,)
    return raw


def predict_frame_head(checkpoint,raw,coords,columns,directions,coarse):
    import torch
    from torch import nn
    device=torch.device('cuda:0');model=nn.Sequential(nn.Linear(1087,64),nn.GELU(),nn.Linear(64,1)).to(device)
    model.load_state_dict(checkpoint['model']);model.eval();score=np.empty((len(raw),len(coords)),float)
    mean,std=checkpoint['mean'],checkpoint['std']
    with torch.no_grad():
        for qi,direction in enumerate(directions):
            for start in range(0,len(coords),4096):
                stop=min(start+4096,len(coords));ids=columns[start:stop]
                x=np.concatenate((raw[qi,ids],extra_features(coords[start:stop],direction)),axis=1)
                residual=model(torch.as_tensor((x-mean)/std,dtype=torch.float32,device=device)).squeeze(-1)
                z=torch.as_tensor(logit(np.clip(coarse[qi,ids],1e-7,1-1e-7)),dtype=torch.float32,device=device)
                score[qi,start:stop]=torch.sigmoid(z+residual).cpu().numpy()
    return score


def relation_fields(ds,rgb,coords,columns,directions,reference,beta,obs,tracks,video):
    expanded=ds[:,columns];ordinary=rgb[:,columns]
    position=np.where(directions[:,None]>0,coords[None,:,0],1-coords[None,:,0])
    eligible=directions[:,None]!=0
    full=transport(expanded,position,reference[:,columns],beta,eligible)
    values=dict(COARSE_DS=expanded,COARSE_RGB=ordinary,RELATION_FULL=full,
        RGB_RELATION=transport(ordinary,position,reference[:,columns],beta,eligible),
        UNCENTERED_RELATION=transport(expanded,position,.5,beta,eligible))
    average=full.copy();shuffled=full.copy()
    for track in tracks:
        ids=np.flatnonzero(obs[:,1]==track)
        for qi in np.flatnonzero(directions):average[qi,ids]=float(full[qi,ids].mean())
        if len(ids)>1:
            key=int(hashlib.sha256((video+':'+str(int(track))+':relation-state-permutation').encode()).hexdigest()[:8],16)
            source_ids=np.roll(ids,1+key%(len(ids)-1));shuffled[:,ids]=full[:,source_ids]
    values.update(RELATION_TIME_AVERAGE=average,RELATION_SHUFFLED=shuffled)
    return values


def run(pair,seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    assert (CORE/'quality'/pair/f'seed{seed}'/'completion.json').is_file()
    assert (RGB/'quality'/pair/f'seed{seed}'/'completion.json').is_file()
    assert json.loads((PREP/'completion.json').read_text())['target_GT_open_attempts']==0
    root.mkdir(parents=True,exist_ok=True);model,target=PAIRS[pair];m=metadata(target);begin=time.perf_counter()
    total=np.zeros(1087);square=np.zeros(1087);rows=0;bx=[];bz=[];by=[];receipts=[];next_video=0;state=root/'stream.pt'
    if state.is_file():
        saved=torch.load(state,map_location='cpu',weights_only=False)
        total,square,rows,bx,bz,by,receipts,next_video=[saved[k] for k in ('total','square','rows','bx','bz','by','receipts','next_video')]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        if vi<next_video:continue
        start=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        obs=np.asarray(bank['observations']);coords,width,height=geometry(bank,target)
        directions=np.asarray([horizontal_direction(t) for t in texts],int)
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=[a[k] for k in ('pcd','colours','prior','tracks')]
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        with np.load(CORE/'quality'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks']);ds=a['DS_FULL']
        with np.load(RGB/'quality'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks']);rgb=a['RAW_RGB_FINAL']
        raw=coarse_features(pcd,colours,prior,tracks,obs,texts)
        records=json.loads((history(pair)/'videos'/video/'questions.json').read_text())
        cells,teacher,teacher_coords,evidence=frame_supervision(records,queries,tracks,width,height)
        qidx,tidx,frames=cells.T
        features=np.concatenate((raw[qidx,tidx],extra_features(teacher_coords,directions[qidx])),axis=1).astype('float32')
        bx.append(features);by.append(teacher);bz.append(logit(np.clip(rgb[qidx,tidx],1e-7,1-1e-7)))
        t,s,n=moments(raw,coords,columns,directions);total+=t;square+=s;rows+=n
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        x=np.concatenate(bx);raw_y=np.concatenate(by);y=np.clip(raw_y,.02,.98);z=np.concatenate(bz)
        # The current prediction probe is tiny. Full frame inference is batched
        # separately so millions of Q x observation x feature rows never reside
        # on the GPU together. Ordinary training uses all eligible paid answers.
        _,fit,checkpoint=fit_head(x,z,y,x[:1],z[:1],mean,std,
            sample_weight=1+3*np.abs(2*raw_y-1),soft=True,seed=seed)
        fit.update(architecture='1087->64 GELU->1 residual on causal ordinary RAW_RGB_FINAL',
            all_queries_and_all_paid_views_used=True,not_restricted_to_our_relation_query_gate=True)
        torch.save(checkpoint,root/'FRAME_KD_latest_head.pt')
        prepare=json.loads((PREP/pair/video/'teacher_rows.json').read_text());relation=json.loads((PREP/pair/video/'prefix_fit.json').read_text())
        assert prepare['query_directions']==directions.tolist()
        if directions.any():assert prepare['width']==width and prepare['height']==height
        query_index={q['expression_id']:i for i,q in enumerate(queries)}
        teacher_rows=[dict(r,local_query=query_index[r['query']]) for r in prepare['rows']]
        reference=references(teacher_rows,obs,tracks,directions,coords)
        scores=relation_fields(ds,rgb,coords,columns,directions,reference,relation['beta'],obs,tracks,video)
        scores['FRAME_KD_FINAL']=predict_frame_head(checkpoint,raw,coords,columns,directions,rgb)
        direct=rgb[:,columns].copy();observation_index={(int(f),int(t)):i for i,(f,t) in enumerate(obs)};filled=0
        for (q,t,f),p in zip(cells,teacher):
            index=observation_index.get((int(f),int(tracks[t])))
            if index is not None:direct[q,index]=p;filled+=1
        scores['FRAME_TEACHER_FILL']=direct
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        for name in METHODS:
            assert scores[name].shape==(len(queries),len(obs)) and np.isfinite(scores[name]).all()
            mask=plain(scores[name],False);destination=root/name/'videos'/video;destination.mkdir(parents=True,exist_ok=True)
            np.save(destination/'mask.npy',mask);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        assert np.array_equal(scores['RELATION_FULL'][directions==0],scores['COARSE_DS'][directions==0])
        np.savez_compressed(local/'frame_scores.npz',tracks=tracks,**scores)
        receipt=dict(video=video,index=vi,fits={'FRAME_KD':fit},teacher_evidence=evidence,
            relation_prefix_fit=relation,frame_teacher_cells_on_candidate_observations=filled,
            raw_teacher_cells_outside_current_observation_bank=len(cells)-filled,
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-start,
            current_video_transductive=True,no_future_teacher_or_heads=True,
            geometry_information_given_to_ordinary_KD=True,new_teacher_encoder_calls=0)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,bz=bz,by=by,receipts=receipts,next_video=vi+1)
        temporary=state.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(state)
        print(json.dumps(dict(event='RELATION_TRANSPORT_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,frame_dependent_predictions=True,same_fixed_paid_QA=True,
                independent_per_query_decoder=True,no_target_manual_labels=True,quality_replay_not_deployment_cost=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_RELATION_TRANSPORT_RAW_NO_GT',pair=pair,seed=seed,
        methods=methods,receipts=receipts,GT_open_attempts=0,new_teacher_encoder_calls=0,actual_head_fits=len(receipts),
        wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


def projection_pairs():
    return [(PRIMARY,PRIMARY)]+[pair for b in BASELINES for pair in ((b,PRIMARY),(PRIMARY,b),(b,b))]


def counts(record):
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();root=Path(record['folder']);dest=root/'countmatched';pair=record['pair'];model,target=PAIRS[pair];m=metadata(target)
    if (dest/'completion.json').is_file():return str(dest)
    pairs=projection_pairs();names=(PRIMARY,)+BASELINES;masks={p:np.zeros(m['candidate_query_rows'],bool) for p in pairs}
    for video,bank in sorted(m['videos'].items()):
        obs=np.asarray(bank['observations']);queries=[q for q in m['episodes'] if q['video_id']==video]
        with np.load(root/'videos'/video/'frame_scores.npz') as a:tracks=a['tracks'];scores={n:a[n] for n in names}
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:prior=a['prior']
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        bits={n:np.load(root/n/'videos'/video/'mask.npy') for n in names}
        for frame in np.unique(obs[:,0]):
            ids=np.flatnonzero(obs[:,0]==frame)
            for qi,q in enumerate(queries):
                order={n:ids[np.lexsort((obs[ids,1],-prior[qi,columns[ids]],-scores[n][qi,ids]))] for n in names}
                for (rank,donor),mask in masks.items():mask[q['start']+order[rank][:int(bits[donor][qi,ids].sum())]]=True
    methods=[]
    for (rank,donor),mask in masks.items():
        name=rank+'__AT_'+donor+'_COUNT';seal=seal_mask(dest/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),
            extra=dict(rank_method=rank,count_method=donor,counts_equal_per_query_frame=True,counts_from_GT=False,frame_dependent_rank=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    changed={}
    for name in names:
        before=json.loads((root/name/'predictions_seal.json').read_text());after=json.loads((dest/(name+'__AT_'+name+'_COUNT')/'predictions_seal.json').read_text())
        original={(r['video'],r['expression']):r['prediction_sha256'] for r in before['episodes']}
        changed[name]=sum(original[r['video'],r['expression']]!=r['prediction_sha256'] for r in after['episodes'])
    assert not any(changed.values()),changed
    assert not guard
    atomic_json(dest/'completion.json',dict(status='SEALED_FRAME_RELATION_COUNTS_NO_GT',methods=methods,CountSelf_changed_query_files=changed,GT_open_attempts=0))
    return str(dest)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==450
    data={};summary={};statistics={};error=0.;head_fits=0
    for record in conditions():
        root=Path(record['folder']);slug=record['pair']+f"_seed{record['seed']}";raw=json.loads((root/'completion.json').read_text());count=json.loads((root/'countmatched/completion.json').read_text())
        paths={r['method']:root/r['method'] for r in raw['methods']};paths.update({r['method']:root/'countmatched'/r['method'] for r in count['methods']})
        data[slug]={n:read_stats(p) for n,p in paths.items()}
        for name,row in data[slug].items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            statistics[slug+'/'+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
        head_fits+=raw['actual_head_fits'];summary[slug]=dict(raw_HOTA={n:data[slug][n]['HOTA'] for n in METHODS},
            teacher_fit_audit=fit_audit(raw['receipts']),CountSelf_changed_query_files=count['CountSelf_changed_query_files'])
    assert error<1e-8 and head_fits==444,(error,head_fits)
    repeated={p:dict(mean_HOTA={n:float(np.mean([data[p+f'_seed{s}'][n]['HOTA'] for s in SEEDS])) for n in METHODS},
        comparisons={n:comparison([data[p+f'_seed{s}'][a] for s in SEEDS],[data[p+f'_seed{s}'][b] for s in SEEDS]) for n,(a,b) in CONTRASTS.items()}) for p in PAIRS}
    for row in repeated.values():
        h=row['mean_HOTA'];row['relation_structure_interaction_point']=(h[PRIMARY]-h['COARSE_DS'])-(h['RGB_RELATION']-h['COARSE_RGB'])
    result=dict(status='RELATION_TRANSPORT_SIX_SETTINGS_ANALYZED',primary=PRIMARY,conditions=summary,repeated=repeated,
        HOTA_parent_sum_max_error=error,actual_head_fits_total=head_fits,new_teacher_encoder_calls=0,
        all450_outputs_sealed_before_this_GT=True,manual_target_labels_in_adaptation=False,
        all_data_development=True,independent_blind_validation=False,goal_status='ACTIVE',research_goal_complete=False,
        scope='Fixed paid QA and prepaid 3-observation features. Scalar relation has a literal left/right query gate, not six independently tuned recipes; other queries exactly retain DS_FULL.',
        ordinary_comparator='Adequate raw-QA residual frame head receives 1081D PCD/RGB/cue plus normalized xywh, oriented x, and the same semantic gate. It trains on all queries, not only our eligible subgroup.',
        theory_scope='Within-group nuisance cancellation and bounded convex scalar fitting under stated assumptions; no theorem of correct semantics or HOTA improvement.',
        quality_replay_is_not_real_deployment_cost=True,no_posthoc_control_promoted=True)
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'relation_transport_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'relation_transport_parent_statistics.json').write_text(json.dumps(statistics,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='RELATION_TRANSPORT_ANALYSIS',repeated=repeated,goal_status='ACTIVE')),flush=True)


def ordinary_GPU_stage_finished():
    state=RGB/'supervisor/state.json'
    if not state.is_file():return False
    d=json.loads(state.read_text())
    return d.get('completed')==18 and not d.get('active') and not d.get('pending') and not d.get('failure')


def main():
    plan=json.loads((PUBLIC/'relation_transport_registration_v1.json').read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    assert plan['total_outputs']==450 and plan['actual_neural_fits']==444
    pending=conditions();done=[];active={};failure=None;start=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),finished=done,
        active={str(g):j for g,(_,_,j) in active.items()},pending=pending,failure=failure,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    state('AWAITING_FLAT_RGB_GPU_STAGE_RELEASE_NO_GT')
    while not ordinary_GPU_stage_finished():time.sleep(5)
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-6000:])
            else:done.append(job)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
                log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_FRAME_RELATION_SIX_SETTINGS_NO_GT' if not failure else 'PRESERVING_ACTIVE_RELATION_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==18
    state('ALL18_RELATION_RAW_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for record,count_root in zip(conditions(),roots):
        for folder in (Path(record['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/r['method']),record['pair']) for r in row['methods'])
    assert len(tasks)==450
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=450,conditions=18,GT_open_attempts=0))
    state('ALL450_RELATION_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=450,results=results,goal_status='ACTIVE'))
    analyze();state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS));p.add_argument('--seed',type=int);args=p.parse_args()
    if args.pair:
        import fcntl
        root=OUT/args.pair/f'seed{args.seed}';root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);run(args.pair,args.seed)
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {};previous.update(status='FAILED_RELATION_TRANSPORT_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
