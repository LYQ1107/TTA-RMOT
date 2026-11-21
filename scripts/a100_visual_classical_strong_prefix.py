"""Fully fitted, channel-normalized causal KD and same-head classic controls."""
import argparse
import json
import re
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.active.strong_prefix_teacher import fit_head
from flex_uda.active.color_cohort import augmented_features,part_mask
from flex_uda.active.peer_evidence import audit_weights
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.acquisition import exclusive_group,expression_types
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import read_guard,plain,immediate_predictions

METHODS=('SOFT_KD_400','SOFT_KD_4000','SOFT_KD','HARD_KD_400','HARD_KD_4000','HARD_KD',
    'NO_GRAPH','COHORT','WHITENED_COHORT','STRONG_PRIOR','PEER','UNIFORM_PRECISION')


def run(pair,policy,seed=0,acquisition_seed=0,teacher_target_clip=0.):
    import torch
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model,target=PAIRS[pair];m=metadata(target);group='strong_'+policy.lower()+'_seed'+str(seed)+('_clipped' if teacher_target_clip else '');root=OUT/group/pair
    if (root/'completion.json').is_file():
        assert json.loads((root/'completion.json').read_text()).get('teacher_target_clip',0.)==teacher_target_clip
        return
    begin=time.perf_counter();dimension=1080;moment_sum=np.zeros(dimension);moment_square=np.zeros(dimension);image_rows=0
    bx=[];bz=[];by=[];receipts=[];start=0;state_path=root/'stream.pt'
    if state_path.is_file():
        saved=torch.load(state_path,map_location='cpu',weights_only=False);start=saved['next_video'];receipts=saved['receipts']
        assert saved.get('teacher_target_clip',0.)==teacher_target_clip
        moment_sum=saved['sum'];moment_square=saved['square'];image_rows=saved['image_rows'];bx=saved['x'];bz=saved['z'];by=saved['y']
    videos=sorted(m['videos']);visible.update(videos[:start]);root.mkdir(parents=True,exist_ok=True)
    for vi,video in enumerate(videos):
        if vi<start:continue
        began=time.perf_counter();visible.add(video);fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:original=a['features'];source=np.clip(logit(np.clip(a['prior'],.05,.95)),-2,2)
        with np.load(OUT/'color_features'/pair/video/'features.npz') as a:colour=a['colours']
        texts=[q['query_text'] for q in fc.queries];geometry,_,selected_colour=augmented_features(original,colour,texts)
        raw=np.concatenate([original,selected_colour],axis=2).astype('float32');current=raw.reshape(-1,dimension)
        moment_sum+=current.sum(0,dtype=np.float64);moment_square+=np.square(current.astype(float)).sum(0);image_rows+=len(current)
        mean=moment_sum/image_rows;std=np.maximum(np.sqrt(np.maximum(moment_square/image_rows-mean**2,0)),1e-3)
        shard_policy=policy if acquisition_seed==0 else policy+'_S'+str(acquisition_seed)
        shard_path=OUT/'causal_teacher'/pair/shard_policy/(video+'.json')
        while not shard_path.is_file():time.sleep(5)
        shard=json.loads(shard_path.read_text());teacher=np.full(source.shape,np.nan)
        for row in shard['answers']:teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
        known=np.isfinite(teacher);bx.append(raw[known]);bz.append(source[known]);by.append(teacher[known])
        x=np.concatenate(bx);z=np.concatenate(bz);y=np.concatenate(by);predictions={};fits={}
        for label,soft in [('SOFT',True),('HARD',False)]:
            targets=np.clip(y,teacher_target_clip,1-teacher_target_clip) if soft and teacher_target_clip else y
            scores,receipt,head=fit_head(x,z,targets,current,source.ravel(),mean,std,soft=soft,seed=seed)
            receipt['teacher_target_clip']=teacher_target_clip if soft else 0.
            fits[label]=receipt;torch.save(head,root/(label+'_latest_head.pt'))
            for point,value in scores.items():predictions[label+'_KD'+('' if point=='CONVERGED' else '_'+point)]=value.reshape(source.shape)
        kd=predictions['SOFT_KD'];whitened=((current-mean)/std).reshape(raw.shape)
        for name in ('NO_GRAPH','COHORT','WHITENED_COHORT','STRONG_PRIOR','PEER','UNIFORM_PRECISION'):predictions[name]=kd.copy()
        audit=[]
        for qi,text in enumerate(texts):
            required=part_mask(text)>0
            if not required.any():continue
            valid=np.all(colour[:,required].sum(2)>0,axis=1)
            factor,record=audit_weights(geometry[qi],teacher[qi],valid)
            if re.search(r'\b(referee|goalkeeper|libero|setter|coach)\b',text.lower()):
                factor[:]=1;record=dict(record,downweighted=0,skip_reason='named role clause')
            seen=np.isfinite(teacher[qi]);base=np.zeros(len(seen));base[seen]=1+3*np.abs(2*teacher[qi,seen]-1)
            uniform=np.full(len(seen),(base*factor).sum()/base.sum() if base.sum() else 1.)
            audit.append(dict(query=qi,**record))
            variants=[('NO_GRAPH',geometry[qi],False,.1,None),('COHORT',geometry[qi],True,.1,None),
                ('WHITENED_COHORT',whitened[qi],True,.1,None),('STRONG_PRIOR',geometry[qi],True,1.,None),
                ('PEER',geometry[qi],True,.1,factor),('UNIFORM_PRECISION',geometry[qi],True,.1,uniform)]
            for name,feature,graph,weight,factors in variants:predictions[name][qi]=refine_distilled_prior(kd[qi],feature,teacher[qi],
                graph=graph,source_weight=weight,node_mask=valid,teacher_weight_factor=factors)
        exclusive=exclusive_group(texts,target);lookup={t:i for i,t in enumerate(fc.tracks)};columns=np.asarray([lookup[int(t)] for _,t in fc.observations]);output={}
        for name,score in predictions.items():
            expanded=score[:,columns];mask=plain(expanded,exclusive)
            for qi,q in enumerate(fc.queries):
                if 'MOTION' in expression_types(q['query_text']):
                    p=expit(fc.raw_presence[qi]);mask[qi]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
            folder=root/name;local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',mask);np.save(local/'scores.npy',expanded.astype('float32'))
            output[name]=immediate_predictions(folder,fc,mask)
        receipt=dict(video=video,index=vi,visible_teacher_videos=sorted(visible),normalization_image_rows=image_rows,
            teacher_examples=len(y),fits=fits,output_hashes=output,audits=audit,GT_open_attempts=len(gt),wall_seconds=time.perf_counter()-began)
        atomic_json(root/'videos'/video/'receipt.json',receipt);receipts.append(receipt)
        state=dict(next_video=vi+1,receipts=receipts,sum=moment_sum,square=moment_square,image_rows=image_rows,x=bx,z=bz,y=by,teacher_target_clip=teacher_target_clip)
        tmp=state_path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(state_path)
        print(json.dumps(dict(event='STRONG_NORMALIZED_PREFIX_VIDEO',pair=pair,policy=policy,seed=seed,video=video,prefix=vi+1,total=len(videos),
            steps={k:v['steps'] for k,v in fits.items()},teacher_KL={k:v['teacher_KL'] for k,v in fits.items()})),flush=True)
    methods=[]
    for name in METHODS:
        folder=root/name;mask=np.zeros(m['candidate_query_rows'],dtype=bool)
        for video in videos:
            local=np.load(folder/'videos'/video/'mask.npy')
            for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
        seal=seal_mask(folder,m,mask,target=target,model=model,method=group+'__'+name,GT_open_attempts=len(gt),extra=dict(
            seed=seed,acquisition_seed=acquisition_seed,teacher_target_clip=teacher_target_clip,teacher_acquisition=policy,logical_teacher_calls=sum(map(len,by)),same_head_in_SOFT_KD_and_classic_arms=True,
            current_prefix_channel_normalization=True,no_future_teacher_feedback=True,video_stream_causal=True,
            current_video_transductive=True,all_prefix_images_not_only_queried_used_for_normalization=True,
            actual_hard_and_soft_KD=True,teacher_fit_min_steps=4000,teacher_fit_max_steps=16000,
            checkpoints400_and4000_are_strong_baseline_controls=True,not_live_end_to_end_wall=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not gt and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_STRONG_NORMALIZED_PREFIX',pair=pair,policy=policy,seed=seed,acquisition_seed=acquisition_seed,teacher_target_clip=teacher_target_clip,
        methods=methods,GT_open_attempts=0,forbidden_read_attempts=0,receipts=receipts,teacher_reads=reads,
        wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);p.add_argument('--policy',choices=['COVER','UNIFORM','PRIOR','FACILITY','FIELD','FIELD_STATIC'],default='COVER');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--acquisition-seed',type=int,default=0)
    p.add_argument('--teacher-target-clip',type=float,default=0.)
    a=p.parse_args();run(a.pair,a.policy,a.seed,a.acquisition_seed,a.teacher_target_clip)
