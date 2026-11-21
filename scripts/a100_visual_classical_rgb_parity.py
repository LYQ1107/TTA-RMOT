"""Same-dimension RGB exposure on one actual source/QA stream, no source GT."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.color_cohort import augmented_features
from flex_uda.active.context_cohort import context_part_mask
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

PARITY=OUT/'strict_rgb_parity'


def source_spec(pair):
    name='C4_minimal_single_KD_rho0p1_20261122' if pair=='C4' else 'C1_key_rho0p1_mean_pair_20261122'
    state='final_single_clip_mean_state.json' if pair=='C4' else 'budget_cold_C1_state.json'
    return OUT/'cold_full_c4'/name,OUT/'supervisor'/state


def run(pair,seed):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,target=PAIRS[pair];m=metadata(target);videos=sorted(m['videos'])
    source,source_state=source_spec(pair);root=PARITY/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);start_time=time.perf_counter()
    labels=('ORIG','RGB') if pair=='C4' else ('RGB',)
    moments={k:dict(total=np.zeros(1080),square=np.zeros(1080),rows=0) for k in labels}
    bx={k:[] for k in labels};bz=[];by=[];receipts=[];start=0;methods=None
    stream=root/'stream.pt'
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False)
        moments,bx,bz,by=saved['moments'],saved['bx'],saved['bz'],saved['by']
        receipts,start,methods=saved['receipts'],saved['next_video'],saved['methods']
    for vi,video in enumerate(videos):
        if vi<start:continue
        stamp=time.perf_counter();folder=source/'source'/video;wait_start=time.perf_counter()
        while not (folder/'receipt.json').is_file():
            state=json.loads(source_state.read_text()) if source_state.is_file() else {}
            if state.get('status','').startswith(('FAILED_','PASS_')):
                raise RuntimeError('Source video unavailable under terminal status: '+video+' '+str(state))
            time.sleep(3)
        waited=time.perf_counter()-wait_start
        with np.load(folder/'sparse_inputs.npz') as data:
            pcd,colours,prior,tracks=data['pcd'],data['colours'],data['prior'],data['tracks']
        bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        geometry,normalized,old_colour=augmented_features(pcd,colours,texts)
        masks=np.stack([context_part_mask(text) for text in texts])
        rgb=np.stack([(colours*mask[None,:,None]).reshape(len(colours),-1) for mask in masks])
        rgb/=np.maximum(np.linalg.norm(rgb,axis=2,keepdims=True),1e-8)
        extended=np.concatenate([normalized,rgb],axis=2).astype('float32')
        arrays=dict(ORIG=np.concatenate([pcd,old_colour],axis=2).astype('float32'),
            RGB=np.concatenate([pcd,rgb],axis=2).astype('float32'))
        if pair=='C1':assert np.array_equal(arrays['ORIG'],arrays['RGB'])
        z=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.)
        teacher=np.full(z.shape,np.nan)
        for row in json.loads((folder/'answers.json').read_text())['answers']:
            teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
        known=np.isfinite(teacher);bz.append(z[known]);by.append(teacher[known]);statistics={}
        for label in labels:
            current=arrays[label].reshape(-1,1080);moment=moments[label]
            moment['total']+=current.sum(0,dtype=np.float64);moment['square']+=np.square(current.astype(float)).sum(0);moment['rows']+=len(current)
            mean=moment['total']/moment['rows'];std=np.maximum(np.sqrt(np.maximum(moment['square']/moment['rows']-mean**2,0)),1e-3)
            statistics[label]=(mean,std);bx[label].append(arrays[label][known])
        y,s=np.concatenate(by),np.concatenate(bz);heads={};fits={}
        for name,label,soft in ([('ORIG','ORIG',True)] if pair=='C4' else [])+[('RGB','RGB',True),('HARD','RGB',False)]:
            mean,std=statistics[label]
            scores,fit,head=fit_head(np.concatenate(bx[label]),s,np.clip(y,.02,.98),arrays[label].reshape(-1,1080),z.ravel(),mean,std,soft=soft,seed=seed)
            heads[name]={point:value.reshape(z.shape) for point,value in scores.items()};fits[name]=fit
            torch.save(head,root/(name+'_latest_head.pt'))
        if pair=='C1':heads['ORIG']=heads['RGB']
        predictions={}
        for label in (('ORIG','RGB') if pair=='C4' else ('RGB',)):
            scores=heads[label];predictions[label+'_KD400']=scores['400'];predictions[label+'_KD_FINAL']=scores['CONVERGED']
            predictions[label+'_KD_MEAN']=.5*(scores['400']+scores['CONVERGED'])
        predictions['RGB_HARD400']=heads['HARD']['400'];predictions['RGB_HARD_FINAL']=heads['HARD']['CONVERGED']
        configurations=[('RGB_GRAPH400','RGB_KD400'),('RGB_GRAPH_FINAL','RGB_KD_FINAL'),('RGB_GRAPH_MEAN','RGB_KD_MEAN'),
            ('RGB_GRAPH_HARD400','RGB_HARD400'),('RGB_GRAPH_HARD_FINAL','RGB_HARD_FINAL'),('RGB_ANCHORS_MEAN','RGB_KD_MEAN')]
        if pair=='C4':configurations+=[('ORIG_GRAPH400','ORIG_KD400'),('ORIG_GRAPH_MEAN','ORIG_KD_MEAN')]
        for name,base in configurations:
            value=predictions[base].copy()
            for q,mask in enumerate(masks):
                if mask.any():value[q]=refine_distilled_prior(value[q],extended[q],teacher[q],graph=name!='RGB_ANCHORS_MEAN',
                    source_weight=.1,node_mask=np.all(colours[:,mask>0].sum(2)>0,axis=1))
            predictions[name]=value
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        exclusive=exclusive_group(texts,target);decoded={name:plain(value[:,columns],exclusive) for name,value in predictions.items()}
        baseline_name='ORIG_KD_MEAN' if pair=='C4' else 'RGB_KD_MEAN';changed=0;graph_changed=0
        if seed==0:
            expected=np.load(source/'methods/CLIP_MEAN/videos'/video/'mask.npy')
            changed=int(np.count_nonzero(expected!=decoded[baseline_name]));assert changed==0,('Physical seed0 KD replay',pair,video,changed)
            if pair=='C1':
                expected=np.load(source/'methods/CLIP_MEAN_COHORT/videos'/video/'mask.npy')
                graph_changed=int(np.count_nonzero(expected!=decoded['RGB_GRAPH_MEAN']))
                assert graph_changed==0,('Physical seed0 graph replay',video,graph_changed)
        methods=list(predictions)
        for name,value in predictions.items():
            destination=root/name;local=destination/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',decoded[name])
            if name in ('RGB_KD_MEAN','RGB_GRAPH_MEAN','ORIG_GRAPH_MEAN'):np.save(local/'scores.npy',value[:,columns])
            immediate_predictions(destination,type('Context',(),dict(bank=bank,queries=queries))(),decoded[name])
        receipt=dict(video=video,index=vi,fits=fits,teacher_examples=len(y),current_teacher_examples=int(known.sum()),
            physical_seed0_KD_changed_rows=changed,physical_seed0_graph_changed_rows=graph_changed,
            source_wait_seconds=waited,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        receipts.append(receipt);atomic_json(root/'videos'/video/'receipt.json',receipt)
        saved=dict(next_video=vi+1,moments=moments,bx=bx,bz=bz,by=by,receipts=receipts,methods=methods)
        temporary=stream.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(stream)
        print(json.dumps(dict(event='STRICT_RGB_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(videos))),flush=True)
    records=[]
    for name in methods:
        destination=root/name;mask=np.zeros(m['candidate_query_rows'],bool)
        for video in videos:
            local=np.load(destination/'videos'/video/'mask.npy')
            for q,episode in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[episode['start']:episode['stop']]=local[q]
        seal=seal_mask(destination,m,mask,target=target,model=model,method='STRICT_RGB_'+name,GT_open_attempts=len(guard),
            extra=dict(seed=seed,source_pool=str(source),same_full_RGB_information=True,neural_dimension=1080,
                new_teacher_calls=0,no_future_feedback=True,no_position_features=True,quality_replay_not_cold_timing=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_STRICT_RGB_PARITY',pair=pair,seed=seed,methods=records,
        receipts=receipts,GT_open_attempts=0,new_teacher_calls=0,source_pool=str(source),
        logical_teacher_calls=sum(r['current_teacher_examples'] for r in receipts),wall_seconds=time.perf_counter()-start_time,
        implementation_sha256=file_sha(Path(__file__)),goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a.pair,a.seed)
