"""Matched1080D feature ablation; frozen PCD plus312zeros, teacher-only fit."""
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
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

ZERO=OUT/'zero_colour_ablation'


def run(pair,seed):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,target=PAIRS[pair];m=metadata(target);root=ZERO/pair/f'seed{seed}';source=source_spec(pair)[0]
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);start_time=time.perf_counter();videos=sorted(m['videos'])
    total=np.zeros(1080);square=np.zeros(1080);rows=0;bx=[];bz=[];by=[];receipts=[];start=0;stream=root/'stream.pt'
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False)
        total,square,rows=saved['total'],saved['square'],saved['rows'];bx,bz,by=saved['bx'],saved['bz'],saved['by'];receipts,start=saved['receipts'],saved['next_video']
    for vi,video in enumerate(videos):
        if vi<start:continue
        stamp=time.perf_counter();folder=source/'source'/video;assert (folder/'receipt.json').is_file()
        with np.load(folder/'sparse_inputs.npz') as values:pcd,prior,tracks=values['pcd'],values['prior'],values['tracks']
        raw=np.concatenate([pcd,np.zeros(pcd.shape[:2]+(312,),dtype=pcd.dtype)],axis=2).astype('float32');current=raw.reshape(-1,1080)
        assert not raw[:,:,768:].any()
        total+=current.sum(0,dtype=np.float64);square+=np.square(current.astype(float)).sum(0);rows+=len(current)
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        z=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);teacher=np.full(z.shape,np.nan)
        for answer in json.loads((folder/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        known=np.isfinite(teacher);bx.append(raw[known]);bz.append(z[known]);by.append(teacher[known])
        scores,fit,head=fit_head(np.concatenate(bx),np.concatenate(bz),np.clip(np.concatenate(by),.02,.98),
            current,z.ravel(),mean,std,soft=True,seed=seed)
        torch.save(head,root/'ZERO_latest_head.pt')
        values={'ZERO_400':scores['400'].reshape(z.shape),'ZERO_FINAL':scores['CONVERGED'].reshape(z.shape)}
        values['ZERO_MEAN']=.5*(values['ZERO_400']+values['ZERO_FINAL'])
        bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'])
        lookup={int(t):i for i,t in enumerate(tracks)};cols=np.asarray([lookup[int(t)] for _,t in obs]);exclusive=exclusive_group([q['query_text'] for q in queries],target)
        for name,score in values.items():
            local=root/name/'videos'/video;local.mkdir(parents=True,exist_ok=True);expanded=score[:,cols];mask=plain(expanded,exclusive)
            np.save(local/'mask.npy',mask)
            if name=='ZERO_MEAN':np.save(local/'scores.npy',expanded)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        receipt=dict(video=video,index=vi,fits=dict(ZERO=fit),current_teacher_examples=int(known.sum()),teacher_examples=sum(map(len,by)),
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        receipts.append(receipt);atomic_json(root/'videos'/video/'receipt.json',receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,bz=bz,by=by,receipts=receipts,next_video=vi+1)
        temp=stream.with_suffix('.tmp');torch.save(saved,temp);temp.replace(stream)
        print(json.dumps(dict(event='ZERO_COLOUR_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(videos))),flush=True)
    records=[]
    for name in ('ZERO_400','ZERO_FINAL','ZERO_MEAN'):
        destination=root/name;mask=np.zeros(m['candidate_query_rows'],bool)
        for video in videos:
            local=np.load(destination/'videos'/video/'mask.npy')
            for q,e in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(destination,m,mask,target=target,model=model,method=name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,head_dimension=1080,zero_colour_channels=312,same_PCD_QA_and_initialization=True,
            no_new_teacher_or_encoder_calls=True,conditional_on_fixed_RGB_acquisition=True,no_future_teacher=True,quality_replay_not_cold_timing=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_ZERO_COLOUR_ABLATION',pair=pair,seed=seed,methods=records,receipts=receipts,
        source=str(source),logical_teacher_calls=sum(map(len,by)),GT_open_attempts=0,new_teacher_calls=0,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),wall_seconds=time.perf_counter()-start_time,
        implementation_sha256=file_sha(Path(__file__)),goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.pair,a.seed)
