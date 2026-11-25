"""Actual teacher KD and classical controls; seal all predictions without GT."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import metadata,FrozenInputs,PAIRS,generation_guard
from flex_uda.active.visual_classical import infer
from flex_uda.presence.predictions import seal_mask
from flex_uda.active.acquisition import exclusive_group,expression_types
from scripts.a100_visual_classical_prepare import OUT

ACQUISITIONS=('PRIOR','UNIFORM','UNCERTAINTY','COVER')
METHODS=('DIRECT_HARD','DIRECT_SOFT','DIRECT_NATIVE','KD_HARD','KD_SOFT','FULL','NO_CORRECTION','NO_GRAPH')


def distill(inputs,answers,output,*,soft,steps=400):
    """A real neural student head fitted to teacher labels on frozen PCD."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.manual_seed(0);np.random.seed(0)
    begin=time.perf_counter();device=torch.device('cuda:0')
    features=np.concatenate([x['features'].reshape(-1,768) for x in inputs.values()])
    priors=np.concatenate([x['prior'].ravel() for x in inputs.values()])
    target=np.concatenate([answers[v].ravel() for v in inputs])
    valid=np.flatnonzero(np.isfinite(target));assert len(valid)>0
    mean=features.mean(0);std=np.maximum(features.std(0),1e-3)
    x=torch.tensor((features-mean)/std,dtype=torch.float32,device=device)
    source_values=(np.concatenate([x['source_logits'].ravel() for x in inputs.values()])
        if all('source_logits' in x for x in inputs.values()) else np.clip(logit(np.clip(priors,.05,.95)),-2,2))
    source=torch.tensor(source_values,dtype=torch.float32,device=device)
    y=torch.tensor(target[valid] if soft else target[valid]>=.5,dtype=torch.float32,device=device)
    ids=torch.tensor(valid,dtype=torch.long,device=device)
    model=nn.Sequential(nn.Linear(768,64),nn.GELU(),nn.Linear(64,1)).to(device)
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=torch.Generator(device=device).manual_seed(0);losses=[]
    for step in range(steps):
        batch=torch.randperm(len(ids),generator=rng,device=device)[:512]
        prediction=source[ids[batch]]+model(x[ids[batch]]).squeeze(-1)
        loss=F.binary_cross_entropy_with_logits(prediction,y[batch])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step in (0,99,199,399,steps-1):losses.append(dict(step=step,loss=float(loss.detach())))
    with torch.no_grad():
        result=torch.sigmoid(source+model(x).squeeze(-1)).cpu().numpy()
        final_loss=float(F.binary_cross_entropy_with_logits(source[ids]+model(x[ids]).squeeze(-1),y))
    torch.cuda.synchronize();output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),mean=mean,std=std,optimizer_steps=steps,
        target='raw_soft_teacher' if soft else 'hard_teacher',source_model_frozen=True),output/'head.pt')
    receipt=dict(status='ACTUAL_KD_TRAINING_COMPLETE',soft=soft,optimizer_steps=steps,teacher_labels=len(valid),
        trainable_parameters=sum(p.numel() for p in model.parameters()),loss_trace=losses,
        final_full_teacher_loss=final_loss,GT_used=False,wall_seconds=time.perf_counter()-begin,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        scope='transductive frozen-PCD residual MLP distillation, not backbone or LoRA finetuning')
    atomic_json(output/'training_receipt.json',receipt)
    split={};offset=0
    for video,value in inputs.items():
        size=value['prior'].size;split[video]=result[offset:offset+size].reshape(value['prior'].shape);offset+=size
    del model,optimizer,x,source,y,ids;torch.cuda.empty_cache()
    return split,receipt


def decode(score,fc,*,native=None,seen=None):
    selected=np.asarray(score)>.5
    lookup={t:i for i,t in enumerate(fc.tracks)}
    track_columns=np.asarray([lookup[int(t)] for _,t in fc.observations])
    if native is not None:
        # Strong plain-teacher control: queried rows are exactly the raw
        # teacher decision, all unqueried rows exactly the source prediction.
        # The structured DIRECT_SOFT arm separately uses common constraints.
        return np.where(seen[:,track_columns],selected[:,track_columns],native)
    if exclusive_group([q['query_text'] for q in fc.queries],fc.target):
        selected &= np.arange(len(fc.queries))[:,None]==np.asarray(score).argmax(0)[None]
    mask=selected[:,track_columns]
    for q,query in enumerate(fc.queries):
        if 'MOTION' in expression_types(query['query_text']):
            p=expit(fc.raw_presence[q]);mask[q]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
    return mask


def generate(pair):
    attempts=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    evidence=OUT/'evidence'/pair
    assert (evidence/'completion.json').is_file()
    if (OUT/'predictions'/pair/'completion.json').is_file():
        print(json.dumps(dict(event='GENERATION_ALREADY_SEALED',pair=pair)),flush=True);return
    plan=json.loads((evidence/'acquisition_plan.json').read_text())
    responses={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,(evidence/'responses.jsonl').read_text().splitlines())}
    inputs={}
    for v in sorted(m['videos']):
        with np.load(OUT/'inputs'/pair/v/'inputs.npz') as a:inputs[v]={k:a[k].copy() for k in a.files}
    receipts=[];timings=[]
    for policy in ACQUISITIONS:
        answers={}
        for video,value in inputs.items():
            a=np.full(value['prior'].shape,np.nan)
            for q,t in plan['positions'][video][policy]:a[q,t]=responses[video,q,t]['yes_probability']
            answers[video]=a
        predictions={method:{} for method in METHODS};training={}
        for method in ('KD_HARD','KD_SOFT'):
            dest=OUT/'predictions'/pair/(policy+'__'+method)
            predictions[method],training[method]=distill(inputs,answers,dest,soft=method=='KD_SOFT')
        total_calls=sum(int(np.isfinite(a).sum()) for a in answers.values())
        global_masks={method:np.zeros(m['candidate_query_rows'],dtype=bool) for method in METHODS}
        for video,value in inputs.items():
            phase_begin=time.perf_counter();fc=FrozenInputs(pair,video)
            input_seconds=time.perf_counter()-phase_begin
            a=answers[video];prior=value['prior'];seen=np.isfinite(a)
            scores={method:predictions[method][video] for method in ('KD_HARD','KD_SOFT')}
            scores['DIRECT_SOFT']=np.where(seen,a,prior)
            scores['DIRECT_HARD']=np.where(seen,a>=.5,prior)
            scores['DIRECT_NATIVE']=scores['DIRECT_SOFT'].copy()
            for method,correction,graph in [('FULL',True,True),('NO_CORRECTION',False,True),('NO_GRAPH',True,False)]:
                phase_begin=time.perf_counter()
                scores[method]=np.stack([infer(prior[q],value['features'][q],a[q],correction=correction,graph=graph)
                    for q in range(len(fc.queries))])
                timings.append(dict(video=video,policy=policy,method=method,phase='classical_score_inference',wall_seconds=time.perf_counter()-phase_begin))
            phase_begin=time.perf_counter();native=fc.decode()[2]
            timings.append(dict(video=video,policy=policy,phase='shared_frozen_input_read_and_source_decode',
                wall_seconds=input_seconds+time.perf_counter()-phase_begin))
            for method,score in scores.items():
                folder=OUT/'predictions'/pair/(policy+'__'+method)/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
                mask=decode(score,fc,native=native if method=='DIRECT_NATIVE' else None,seen=seen)
                np.save(folder/'mask.npy',mask)
                np.savez_compressed(folder/'belief.npz',score=score,seen=seen,teacher_probability=a,tracks=value['tracks'])
                for q,query in enumerate(fc.queries):global_masks[method][query['start']:query['stop']]=mask[q]
            print(json.dumps(dict(event='CLASSICAL_GENERATION_VIDEO',pair=pair,policy=policy,video=video)),flush=True)
        for method,mask in global_masks.items():
            name=policy+'__'+method;dest=OUT/'predictions'/pair/name
            seal=seal_mask(dest,m,mask,target=target,model=model,method=name,GT_open_attempts=len(attempts),
                extra=dict(acquisition=policy,logical_teacher_calls=total_calls,seed=0,
                    teacher_evidence_sha256=file_sha(evidence/'responses.jsonl'),
                    same_fixed_evidence_within_acquisition=True,source_model_frozen=True,
                    training=training.get(method),task_protocol='PRESUPPOSED' if pair=='C4' else 'CHECKLIST',
                    new_early_absence_stop=False,shared_evidence_not_live_wall_benchmark=True,
                    feature_extraction_from_historical_unlabelled_cache=True))
            receipts.append(dict(method=name,rows=seal['prediction_rows'],calls=total_calls))
    assert not attempts
    atomic_json(OUT/'predictions'/pair/'completion.json',dict(status='ALL_PREDECLARED_METHODS_SEALED',pair=pair,
        methods=receipts,GT_open_attempts=0,wall_seconds=time.perf_counter()-begin,
        phase_timings=timings,scope='shared study execution, not per-method live cold wall benchmark'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();generate(a.pair)
