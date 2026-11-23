"""Strong frame-level KD: train on queried frames, predict every observation.

No whole-track semantic constancy is imposed on this teacher baseline.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import FrozenInputs,metadata,PAIRS,HISTORY,generation_guard
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_generate import distill,ACQUISITIONS
from scripts.a100_visual_classical_prepare import OUT
from flex_uda.active.acquisition import exclusive_group,expression_types


def run(pair,steps=400,group='predictions'):
    attempts=generation_guard();begin=time.perf_counter();model_name,target=PAIRS[pair];m=metadata(target)
    root=OUT/group/pair
    assert (OUT/'predictions'/pair/'completion.json').is_file()
    if (root/'frame_kd_completion.json').is_file():return
    evidence=OUT/'evidence'/pair
    plan=json.loads((evidence/'acquisition_plan.json').read_text())['positions']
    responses={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,(evidence/'responses.jsonl').read_text().splitlines())}
    inputs={}
    for video in sorted(m['videos']):
        cache=OUT/'frame_anchor_features'/pair/(video+'.npz')
        if cache.is_file():
            with np.load(cache) as a:inputs[video]={k:a[k].copy() for k in a.files}
            continue
        fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:anchors=a['anchors']
        positions={(int(f),int(t)):i for i,(f,t) in enumerate(fc.observations)}
        ids=np.asarray([positions[int(f),int(t)] for f,t in zip(anchors,fc.tracks)])
        pcd=np.load(HISTORY/'features'/target/model_name/'videos'/video/'pcd.npy',mmap_mode='r')
        n=len(fc.observations)
        features=np.stack([np.asarray(pcd[q*n+ids],dtype='float32') for q in range(len(fc.queries))])
        source=fc.margin[:,ids].copy()
        inputs[video]=dict(features=features,source_logits=source,prior=expit(source))
        cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**inputs[video])
        print(json.dumps(dict(event='FRAME_KD_ANCHORS',pair=pair,video=video)),flush=True)
    import torch
    from torch import nn
    device=torch.device('cuda:0');models={};training={}
    for policy in ACQUISITIONS:
        answers={v:np.full(x['prior'].shape,np.nan) for v,x in inputs.items()}
        for v in inputs:
            for q,t in plan[v][policy]:answers[v][q,t]=responses[v,q,t]['yes_probability']
        for label in ('HARD','SOFT'):
            name=policy+'__FRAME_KD_'+label+('' if steps==400 else '_'+str(steps));destination=root/name
            _,receipt=distill(inputs,answers,destination,soft=label=='SOFT',steps=steps)
            saved=torch.load(destination/'head.pt',map_location='cpu',weights_only=False)
            head=nn.Sequential(nn.Linear(768,64),nn.GELU(),nn.Linear(64,1)).to(device)
            head.load_state_dict(saved['model']);head.eval()
            models[name]=(head,torch.tensor(saved['mean'],device=device),torch.tensor(saved['std'],device=device))
            receipt.update(scope='frame-level KD on teacher queried frame, all-frame inference; frozen source PCD, actual unmodified raw source logit initialization',
                whole_track_label_broadcast=False)
            atomic_json(destination/'training_receipt.json',receipt);training[name]=receipt
    masks={name:np.zeros(m['candidate_query_rows'],dtype=bool) for name in models}
    for label in ('HARD','SOFT'):masks['COVER__FRAME_KD_'+label+('' if steps==400 else '_'+str(steps))+'_STRUCTURED']=np.zeros(m['candidate_query_rows'],dtype=bool)
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);pcd=np.load(HISTORY/'features'/target/model_name/'videos'/video/'pcd.npy',mmap_mode='r')
        source=fc.margin.ravel();score={name:np.empty(len(source),dtype='float32') for name in models}
        with torch.no_grad():
            for start in range(0,len(source),8192):
                stop=min(start+8192,len(source));x=torch.tensor(np.asarray(pcd[start:stop],dtype='float32'),device=device)
                z=torch.tensor(np.asarray(source[start:stop]),dtype=torch.float32,device=device)
                for name,(head,mean,std) in models.items():
                    score[name][start:stop]=torch.sigmoid(z+head((x-mean)/std).squeeze(-1)).cpu().numpy()
        for name,value in score.items():
            folder=root/name/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
            matrix=value.reshape(fc.margin.shape);mask=matrix>.5
            # Plain, strong frame-level KD has neither graph nor inherited
            # existence/exclusivity gates; it learns directly from teacher BCE.
            np.save(folder/'scores.npy',matrix);np.save(folder/'mask.npy',mask)
            for q,query in enumerate(fc.queries):masks[name][query['start']:query['stop']]=mask[q]
            if name.startswith('COVER__'):
                structured=mask.copy()
                if exclusive_group([q['query_text'] for q in fc.queries],target):
                    structured &= np.arange(len(fc.queries))[:,None]==matrix.argmax(0)[None]
                for q,query in enumerate(fc.queries):
                    if 'MOTION' in expression_types(query['query_text']):
                        p=expit(fc.raw_presence[q]);structured[q]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
                    masks[name+'_STRUCTURED'][query['start']:query['stop']]=structured[q]
                structured_folder=root/(name+'_STRUCTURED')/'videos'/video;structured_folder.mkdir(parents=True,exist_ok=True)
                np.save(structured_folder/'mask.npy',structured)
                (structured_folder/'scores.npy').symlink_to(folder/'scores.npy')
        print(json.dumps(dict(event='FRAME_KD_VIDEO_PREDICTED',pair=pair,video=video)),flush=True)
    records=[]
    for name,mask in masks.items():
        policy=name.split('__')[0];calls=sum(len(p[policy]) for p in plan.values())
        training_key=name.removesuffix('_STRUCTURED')
        seal=seal_mask(root/name,m,mask,target=target,model=model_name,method=name,GT_open_attempts=len(attempts),
            extra=dict(acquisition=policy,logical_teacher_calls=calls,training=training[training_key],seed=0,
                frame_level_inference=True,whole_track_label_broadcast=False,source_model_frozen=True,
                inherited_exclusivity_and_motion_gates=name.endswith('_STRUCTURED'),
                teacher_evidence_sha256=file_sha(evidence/'responses.jsonl')))
        records.append(dict(method=name,rows=seal['prediction_rows'],calls=calls))
    completion=(json.loads((root/'completion.json').read_text()) if (root/'completion.json').is_file() else dict(status='SEALED_FRAME_CONVERGENCE_BASELINES',pair=pair,methods=[],GT_open_attempts=0))
    completion['methods'].extend(records)
    completion['frame_level_KD_added_before_any_target_GT']=True;atomic_json(root/'completion.json',completion)
    assert not attempts
    atomic_json(root/'frame_kd_completion.json',dict(status='SEALED_STRONG_FRAME_KD',methods=records,
        wall_seconds=time.perf_counter()-begin,GT_open_attempts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--group',default='predictions')
    a=p.parse_args();run(a.pair,a.steps,a.group)
