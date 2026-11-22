"""Equal-information KD and teacher-only semantic monotone correction."""
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
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.context_cohort import distillation_features
from flex_uda.active.semantic_colour import appearance_cues,shuffled_cues,crossfit_predictions,correct_belief,append_history
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

SEMANTIC=OUT/'semantic_colour'
METHODS=('ZERO400','ZERO_FINAL','ZERO_MEAN','SEM400','SEM_FINAL','SEM_MEAN','SEM_HARD400','SEM_HARD_FINAL',
    'MONO','BIN','SHUFFLED','NAIVE_MEAN','MONO_ONLY','MONO_ZERO')
EXPANDED={'MONO','SEM_MEAN','ZERO_MEAN','BIN','NAIVE_MEAN'}


def empty_history():return {f:dict(x=np.empty(0),y=np.empty(0)) for f in (0,1)}


def run(pair,seed):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard=generation_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=SEMANTIC/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    implementation=dict(worker=file_sha(Path(__file__)),semantic=file_sha(ROOT/'flex_uda/active/semantic_colour.py'))
    model,target=PAIRS[pair];m=metadata(target);videos=sorted(m['videos']);source=source_spec(pair)[0]
    total={k:np.zeros(1081) for k in ('ZERO','SEM')};square={k:np.zeros(1081) for k in total};rows=0
    bx={k:[] for k in total};bz=[];by=[];history=empty_history();permuted_history=empty_history();receipts=[];start=0;stream=root/'stream.pt'
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False);assert saved['implementation']==implementation
        total,square,rows=saved['total'],saved['square'],saved['rows'];bx,bz,by=saved['bx'],saved['bz'],saved['by']
        history,permuted_history=saved['history'],saved['permuted_history'];receipts,start=saved['receipts'],saved['next_video']
    for vi,video in enumerate(videos):
        if vi<start:continue
        stamp=time.perf_counter();folder=source/'source'/video;assert (folder/'receipt.json').is_file()
        with np.load(folder/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        bank=m['videos'][video];obs=np.asarray(bank['observations']);queries=[e for e in m['episodes'] if e['video_id']==video];texts=[e['query_text'] for e in queries]
        lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks]);cue,available,families,specs=appearance_cues(colours,texts,lengths)
        assert np.isfinite(cue).all() and np.all((cue>=0)&(cue<=1))
        rgb=distillation_features(pcd,colours,texts,'extended');feature=np.where(available,2*cue-1,0).astype('float32')
        raw=dict(ZERO=np.concatenate([rgb,np.zeros(feature.shape+(1,),dtype='float32')],axis=2),
            SEM=np.concatenate([rgb,feature[:,:,None]],axis=2))
        z=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);teacher=np.full(z.shape,np.nan)
        for answer in json.loads((folder/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        known=np.isfinite(teacher);bz.append(z[known]);by.append(teacher[known]);rows+=len(rgb.reshape(-1,1080))
        statistics={}
        for name,array in raw.items():
            current=array.reshape(-1,1081);total[name]+=current.sum(0,dtype=np.float64);square[name]+=np.square(current.astype(float)).sum(0)
            mean=total[name]/rows;std=np.maximum(np.sqrt(np.maximum(square[name]/rows-mean**2,0)),1e-3)
            statistics[name]=(mean,std);bx[name].append(array[known])
        heads={};fits={}
        for label,scope,soft in [('ZERO','ZERO',True),('SEM','SEM',True),('HARD','SEM',False)]:
            mean,std=statistics[scope]
            predictions,fit,checkpoint=fit_head(np.concatenate(bx[scope]),np.concatenate(bz),np.clip(np.concatenate(by),.02,.98),
                raw[scope].reshape(-1,1081),z.ravel(),mean,std,soft=soft,seed=seed)
            fits[label]=fit;heads[label]={k:v.reshape(z.shape) for k,v in predictions.items()}
            torch.save(checkpoint,root/(label+'_latest_head.pt'));del checkpoint
        output=dict(ZERO400=heads['ZERO']['400'],ZERO_FINAL=heads['ZERO']['CONVERGED'],
            ZERO_MEAN=.5*(heads['ZERO']['400']+heads['ZERO']['CONVERGED']),
            SEM400=heads['SEM']['400'],SEM_FINAL=heads['SEM']['CONVERGED'],
            SEM_MEAN=.5*(heads['SEM']['400']+heads['SEM']['CONVERGED']),
            SEM_HARD400=heads['HARD']['400'],SEM_HARD_FINAL=heads['HARD']['CONVERGED'])
        post_begin=time.perf_counter()
        probability,base,active,calibration=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
        binned,bprior,bactive,bin_receipts=crossfit_predictions(cue,available,families,teacher,tracks,video,history,monotone=False)
        permuted=shuffled_cues(cue,available,video)
        shuffled,sprior,sactive,shuffle_receipts=crossfit_predictions(permuted,available,families,teacher,tracks,video,permuted_history)
        output['MONO']=correct_belief(output['SEM_MEAN'],probability,base,active)
        assert np.array_equal(output['MONO'][~available],output['SEM_MEAN'][~available])
        output['MONO_ZERO']=correct_belief(output['ZERO_MEAN'],probability,base,active)
        output['BIN']=correct_belief(output['SEM_MEAN'],binned,bprior,bactive)
        output['SHUFFLED']=correct_belief(output['SEM_MEAN'],shuffled,sprior,sactive)
        output['NAIVE_MEAN']=output['SEM_MEAN'].copy();output['MONO_ONLY']=output['SEM_MEAN'].copy()
        output['NAIVE_MEAN'][active]=.5*(output['SEM_MEAN'][active]+probability[active])
        output['MONO_ONLY'][active]=probability[active]
        post_seconds=time.perf_counter()-post_begin
        append_history(history,cue,available,families,teacher);append_history(permuted_history,permuted,available,families,teacher)
        lookup={int(t):i for i,t in enumerate(tracks)};cols=np.asarray([lookup[int(t)] for _,t in obs]);exclusive=exclusive_group(texts,target)
        for name in METHODS:
            score=output[name][:,cols];mask=plain(score,exclusive);local=root/name/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',mask)
            if name in EXPANDED:np.save(local/'scores.npy',score)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,cue=cue,available=available,families=families,**output)
        receipt=dict(video=video,index=vi,fits=fits,current_teacher_examples=int(known.sum()),teacher_examples=sum(map(len,by)),
            supported_queries=int((families>=0).sum()),available_candidate_cues=int(available.sum()),
            active_monotone_nodes=int(active.sum()),crossfit_calibration=calibration,bin_calibration=bin_receipts,
            shuffled_calibration=shuffle_receipts,postprocess_seconds=post_seconds,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        receipts.append(receipt);atomic_json(local/'receipt.json',receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,bz=bz,by=by,history=history,permuted_history=permuted_history,
            receipts=receipts,next_video=vi+1,implementation=implementation)
        temporary=stream.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(stream)
        print(json.dumps(dict(event='SEMANTIC_COLOUR_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(videos))),flush=True)
    records=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool);destination=root/name
        for video in videos:
            local=np.load(destination/'videos'/video/'mask.npy')
            for qi,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=local[qi]
        seal=seal_mask(destination,m,mask,target=target,model=model,method='SEMANTIC_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,head_dimension=1081,same_raw_RGB_and_physical_cue_as_strong_teacher=True,
            no_new_teacher_or_encoder_calls=True,current_video_transductive=True,no_future_teacher=True,
            quality_replay_not_cold_timing=True,implementation=implementation))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_SEMANTIC_COLOUR',pair=pair,seed=seed,methods=records,receipts=receipts,
        source=str(source),logical_teacher_calls=sum(map(len,by)),GT_open_attempts=0,new_teacher_calls=0,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),wall_seconds=time.perf_counter()-begin,
        implementation=implementation,goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.pair,a.seed)
