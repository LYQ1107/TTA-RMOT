"""New fixed-evidence six-way attribution of view fusion and classic inference.

Old v8P predictions/beliefs/identity votes are not reused. Only the actually
requested frozen-teacher view answers enter this study. The new source PCD
stream and the adequate ordinary-head recipe are shared by all new arms.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.view_evidence import VIEWS,targets
from scripts.a100_six_direction import OUT as TRANSFER,PROTOCOL,source

OUT=TRANSFER/'view_fusion'
METHODS=('RAW_FINAL','RAW_MEAN','HARD_FINAL','RAW_SRC_FINAL','FIXED_FINAL','DS_FINAL',
         'DS_CONTEXT','DS_VISUAL','DS_JOINT','DS_FULL','FIXED_FULL','RAW_FULL','DS_SHUFFLED')


def labels(pair,video,queries,tracks):
    old=history(pair)/'videos'/video/'questions.json'
    rows=json.loads(old.read_text());qi={q['expression_id']:i for i,q in enumerate(queries)};ti={int(t):i for i,t in enumerate(tracks)}
    values=defaultdict(list);sha=set();kinds=defaultdict(int)
    for row in rows:
        q,t=qi[row['query_id']],ti[int(row['track_id'])]
        view=row['effective_view'];assert view in VIEWS
        values[q,t,VIEWS.index(view)].append(float(row['yes_probability']))
        sha.add(row['frozen_weight_sha256']);kinds[view]+=1
    assert len(sha)==1
    cells=sorted({(q,t) for q,t,v in values});probability=np.full((len(cells),len(VIEWS)),np.nan)
    for i,(q,t) in enumerate(cells):
        for vi in range(len(VIEWS)):
            if (q,t,vi) in values:probability[i,vi]=np.mean(values[q,t,vi])
    return np.asarray(cells,int),probability,dict(source=str(old),logical_questions=len(rows),distinct_cells=len(cells),
        effective_view_calls=dict(kinds),within_track_view_calls_aggregated=len(rows)-len(values),teacher_weight_sha256=next(iter(sha)),
        grouped_calls_may_use_different_frames_and_are_not_exact_cache_duplicates=True,
        legacy_beliefs_and_identity_votes_ignored=True)


def refine(unary,teacher,pcd,colours,texts,lengths,tracks,video,appearance_history,protocol):
    from flex_uda.active.protocol_calibration import global_pool,contract_graph
    from flex_uda.active.topology_selection import graph_matrices
    from flex_uda.active.appearance_runtime import regularize_appearance
    if protocol=='CHECKLIST':
        _,_,glob,valid,supported=graph_matrices(pcd,colours,texts)
        context=global_pool(unary,teacher,glob,valid,supported)
    else:context=unary.copy()
    context=contract_graph(unary,context,protocol)
    visual,inputs,receipt=regularize_appearance(unary,colours,texts,lengths,teacher,tracks,video,appearance_history)
    joint,_,_=regularize_appearance(context,colours,texts,lengths,teacher,tracks,video,appearance_history)
    return context,visual,joint,inputs,receipt


def run(pair,seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.active.appearance_runtime import empty_appearance_history,commit_appearance_history
    from flex_uda.active.query_consistency import relations,project
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'quality'/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);model,target=PAIRS[pair];m=metadata(target);start=time.perf_counter()
    total=np.zeros(768);square=np.zeros(768);rows=0;bx=[];bz=[];bv=[];receipts=[]
    histories={key:empty_appearance_history() for key in ('RAW','FIXED','DS')}
    state_path=root/'stream.pt';next_video=0
    if state_path.is_file():
        saved=torch.load(state_path,map_location='cpu',weights_only=False)
        total,square,rows,bx,bz,bv,receipts,histories,next_video=[saved[k] for k in ('total','square','rows','bx','bz','bv','receipts','histories','next_video')]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        if vi<next_video:continue
        stamp=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=[a[k] for k in ('pcd','colours','prior','tracks')]
        cells,view_probability,evidence_receipt=labels(pair,video,queries,tracks)
        first_answer=json.loads((source(pair)/'source'/video/'answers.json').read_text())['answers'][0]
        assert first_answer['frozen_weight_sha256']==evidence_receipt['teacher_weight_sha256'],'Do not mix teacher weights'
        qidx,tidx=cells.T;bx.append(pcd[qidx,tidx]);z=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);bz.append(z[qidx,tidx]);bv.append(view_probability)
        current=pcd.reshape(-1,768).astype('float32');total+=current.sum(0,dtype=float)
        square+=np.square(current.astype(float)).sum(0);rows+=len(current)
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        raw,fixed,adaptive,fusion=targets(np.concatenate(bv));x=np.concatenate(bx);source_z=np.concatenate(bz)
        weight=1.+3.*np.abs(2.*raw-1.);heads={};fits={}
        for name,y,soft,residual in (('RAW',raw,True,False),('HARD',raw,False,False),
                ('RAW_SRC',raw,True,True),('FIXED',fixed,True,False),('DS',adaptive,True,False)):
            result,fit,checkpoint=fit_head(x,source_z if residual else np.zeros(len(x)),np.clip(y,.02,.98),
                current,z.ravel() if residual else np.zeros(prior.size),mean,std,sample_weight=weight,soft=soft,seed=seed)
            fits[name]=dict(fit,common_raw_teacher_confidence_weights=True,source_residual=residual)
            torch.save(checkpoint,root/(name+'_latest_head.pt'))
            heads[name]={k:v.reshape(prior.shape) for k,v in result.items()}
        scores={name+'_FINAL':heads[name]['CONVERGED'] for name in heads}
        scores['RAW_MEAN']=.5*(heads['RAW']['400']+heads['RAW']['CONVERGED'])
        obs=np.asarray(bank['observations']);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks]);_,edges,_=relations(texts)
        matrices={};new_inputs={};post={}
        offset=len(raw)-len(cells)
        for label,all_y in (('RAW',raw),('FIXED',fixed),('DS',adaptive)):
            teacher=np.full(prior.shape,np.nan);teacher[qidx,tidx]=all_y[offset:]
            matrices[label]=teacher
            context,visual,joint,inputs,receipt=refine(scores[label+'_FINAL'],teacher,pcd,colours,texts,lengths,tracks,video,histories[label],PROTOCOL[pair])
            scores[label+'_FULL'],projection=project(joint,edges)
            new_inputs[label]=inputs;post[label]=dict(appearance=receipt,projection=projection)
            if label=='DS':scores.update(DS_CONTEXT=context,DS_VISUAL=visual,DS_JOINT=joint)
        seed_hash=int(__import__('hashlib').sha256((video+':query-implication-placebo').encode()).hexdigest()[:8],16)
        permutation=np.random.default_rng(seed_hash).permutation(len(queries))
        scores['DS_SHUFFLED'],_=project(scores['DS_JOINT'],permutation[edges])
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        for name in METHODS:
            mask=plain(scores[name][:,columns],False);destination=root/name/'videos'/video
            destination.mkdir(parents=True,exist_ok=True);np.save(destination/'mask.npy',mask)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**scores)
        receipt=dict(video=video,index=vi,fits=fits,teacher_evidence=evidence_receipt,fusion=fusion,postprocess=post,
            current_teacher_cells=len(cells),prefix_teacher_cells=len(x),GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        for label in histories:commit_appearance_history(histories[label],new_inputs[label],matrices[label])
        saved=dict(total=total,square=square,rows=rows,bx=bx,bz=bz,bv=bv,receipts=receipts,histories=histories,next_video=vi+1)
        temp=state_path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(state_path)
        print(json.dumps(dict(event='VIEW_FUSION_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    records=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            bits=np.load(root/name/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='VIEW_FUSION_'+name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,fixed_legacy_rho03_requested_QA=True,no_legacy_prediction_or_belief_reuse=True,
                independent_per_query_decoder_every_arm=True,current_video_transductive=True,no_future_teacher=True,
                new_teacher_encoder_calls=0,quality_replay_not_deployment_cost=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_VIEW_FUSION_SIX_DIRECTION',pair=pair,seed=seed,methods=records,
        receipts=receipts,GT_open_attempts=0,new_teacher_encoder_calls=0,actual_head_fits=5*len(receipts),
        logical_teacher_questions=sum(r['teacher_evidence']['logical_questions'] for r in receipts),
        original_teacher_budget_fraction=.3,source_features_from_new_three_observation_stream=True,
        old_active_selection_policy_not_revalidated=True,all_data_development=True,
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    import fcntl
    folder=OUT/'quality'/a.pair/f'seed{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'condition.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(a.pair,a.seed)
