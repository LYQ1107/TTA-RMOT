"""Canonical768D teacher heads, separately controlled classical graph metrics."""
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
from flex_uda.active.grounded_graph import refine_queries,positive_agreement_precision
from flex_uda.active.acquisition import exclusive_group
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

CANON=OUT/'canonical_pcd_graph'
METHODS=('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL','GRAPH_PCD','GRAPH_RGB',
    'GRAPH_COLOUR','GRAPH_HIDDEN','ANCHORS','GRAPH_GUARD')


def run(pair,seed):
    import torch
    from torch.nn import functional as F
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard=generation_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=CANON/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();implementation=file_sha(Path(__file__))
    model,target=PAIRS[pair];m=metadata(target);videos=sorted(m['videos']);source=source_spec(pair)[0]
    protocol=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
    total=np.zeros(768);square=np.zeros(768);rows=0;bx=[];bz=[];by=[];receipts=[];start=0;stream=root/'stream.pt'
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False);assert saved['implementation_sha256']==implementation
        total,square,rows=saved['total'],saved['square'],saved['rows'];bx,bz,by=saved['bx'],saved['bz'],saved['by'];receipts,start=saved['receipts'],saved['next_video']
    for vi,video in enumerate(videos):
        if vi<start:continue
        stamp=time.perf_counter();folder=source/'source'/video;assert (folder/'receipt.json').is_file()
        with np.load(folder/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        raw=np.asarray(pcd,dtype='float32');current=raw.reshape(-1,768)
        total+=current.sum(0,dtype=np.float64);square+=np.square(current.astype(float)).sum(0);rows+=len(current)
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        z=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);teacher=np.full(z.shape,np.nan)
        for answer in json.loads((folder/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        known=np.isfinite(teacher);bx.append(raw[known]);bz.append(z[known]);by.append(teacher[known]);fits={};heads={};hidden=None;hidden_seconds=0
        for label,soft in [('SOFT',True),('HARD',False)]:
            scores,fit,head=fit_head(np.concatenate(bx),np.concatenate(bz),np.clip(np.concatenate(by),.02,.98),
                current,z.ravel(),mean,std,soft=soft,seed=seed)
            fits[label]=fit;heads[label]={p:v.reshape(z.shape) for p,v in scores.items()};torch.save(head,root/(label+'_latest_head.pt'))
            if soft:
                hs=time.perf_counter()
                with torch.no_grad():
                    values=torch.tensor((current-mean)/std,dtype=torch.float32,device='cuda:0')
                    hidden=F.gelu(F.linear(values,head['model']['0.weight'],head['model']['0.bias'])).cpu().numpy().reshape(z.shape+(64,))
                hidden_seconds=time.perf_counter()-hs;del values
            del head
        output=dict(KD400=heads['SOFT']['400'],KD_FINAL=heads['SOFT']['CONVERGED'],
            KD_MEAN=.5*(heads['SOFT']['400']+heads['SOFT']['CONVERGED']),HARD400=heads['HARD']['400'],HARD_FINAL=heads['HARD']['CONVERGED'])
        bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        post={}
        for name,kind,graph,positive_guard in [('GRAPH_PCD','pcd',True,False),('GRAPH_RGB','combined',True,False),
            ('GRAPH_COLOUR','colour',True,False),('GRAPH_HIDDEN','hidden',True,False),('ANCHORS','pcd',False,False),('GRAPH_GUARD','pcd',True,True)]:
            t=time.perf_counter();output[name]=refine_queries(output['KD_MEAN'],pcd,colours,texts,teacher,kind=kind,hidden=hidden,
                graph=graph,positive_guard=positive_guard,early=heads['SOFT']['400'],final=heads['SOFT']['CONVERGED'],protocol=protocol)
            post[name]=time.perf_counter()-t
        if protocol=='CHECKLIST':assert np.array_equal(output['GRAPH_GUARD'],output['GRAPH_PCD'])
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)};cols=np.asarray([lookup[int(t)] for _,t in obs]);exclusive=exclusive_group(texts,target)
        for name in METHODS:
            expanded=output[name][:,cols];mask=plain(expanded,exclusive);local=root/name/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',mask);np.save(local/'scores.npy',expanded)
            immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        precision=positive_agreement_precision(heads['SOFT']['400'],heads['SOFT']['CONVERGED'],protocol)
        receipt=dict(video=video,index=vi,fits=fits,current_teacher_examples=int(known.sum()),teacher_examples=sum(map(len,by)),
            explicit_task_contract=protocol,positive_guard_boosted_unknown_nodes=int(((precision>.1)&~known).sum()),
            positive_guard_min_precision=float(precision.min()),positive_guard_max_precision=float(precision.max()),
            graph_seconds=post,hidden_feature_forward_seconds=hidden_seconds,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
        receipts.append(receipt);atomic_json(root/'videos'/video/'receipt.json',receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,bz=bz,by=by,receipts=receipts,next_video=vi+1,implementation_sha256=implementation)
        temporary=stream.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(stream)
        print(json.dumps(dict(event='CANONICAL_PCD_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(videos))),flush=True)
    records=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool);destination=root/name
        for video in videos:
            local=np.load(destination/'videos'/video/'mask.npy')
            for q,e in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(destination,m,mask,target=target,model=model,method='CANONICAL_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,head_dimension=768,natural_PCD_teacher_head=True,shared_soft_head_for_all_graphs=True,
            no_new_teacher_or_encoder_calls=True,explicit_task_contract=protocol,current_video_transductive=True,no_future_teacher=True,
            quality_replay_not_cold_timing=True))
        records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_CANONICAL_PCD_GRAPH',pair=pair,seed=seed,methods=records,receipts=receipts,
        source=str(source),logical_teacher_calls=sum(map(len,by)),GT_open_attempts=0,new_teacher_calls=0,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),wall_seconds=time.perf_counter()-begin,
        implementation_sha256=implementation,goal_status='ACTIVE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args();run(a.pair,a.seed)
