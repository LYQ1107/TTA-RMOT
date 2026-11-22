"""GT-free diagnostic: separate teacher replacement and feedback-time scopes.

This quality study reuses frozen features and recorded actual teacher answers.
PAST_ONLY predicts before opening the current video's answer file. Its purpose
is to measure transductive assistance, not to call different information scopes
a fair deployment comparison. All original masks/IDs/GT remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,FrozenInputs,generation_guard
from flex_uda.active.color_cohort import augmented_features
from flex_uda.active.acquisition import exclusive_group,expression_types
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

AUDIT=OUT/'benchmark_scope_audit'
FRESH=OUT/'cold_full_c4/C4_key_single_CLIP_KD_seed0'


def inputs(pair,video,m):
    bank=m['videos'][video];qs=[q for q in m['episodes'] if q['video_id']==video]
    obs=np.asarray(bank['observations']);tracks=sorted(set(obs[:,1].tolist()))
    if pair=='C4':
        folder=FRESH/'source'/video
        with np.load(folder/'sparse_inputs.npz') as a:
            pcd=a['pcd'];colour=a['colours'];prior=a['prior'];assert list(a['tracks'])==tracks
        # Original temporal context was preserved by the fresh sparse worker.
        samples=[]
        for track in tracks:
            ids=np.flatnonzero(obs[:,1]==track)
            samples.extend(ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))])
        frames=np.unique(obs[samples,0]);presence=np.load(folder/'presence_logits.npy')
        assert presence.shape==(len(qs),len(frames))
        answer_path=folder/'answers.json'
    else:
        fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:pcd=a['features'];prior=a['prior']
        with np.load(OUT/'color_features'/pair/video/'features.npz') as a:colour=a['colours']
        frames=np.asarray(sorted(fc.frame_index,key=fc.frame_index.get));presence=fc.raw_presence
        answer_path=OUT/'causal_teacher'/pair/'COVER'/(video+'.json')
    _,_,parsed=augmented_features(pcd,colour,[q['query_text'] for q in qs])
    raw=np.concatenate([pcd,parsed],axis=2).astype('float32')
    source=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.)
    columns=np.asarray([tracks.index(int(t)) for _,t in obs]);gates={}
    for qi,q in enumerate(qs):
        if 'MOTION' in expression_types(q['query_text']):
            gates[qi]=expit(np.interp(obs[:,0],frames,presence[qi]))>=.5
    return raw,source,answer_path,SimpleNamespace(bank=bank,queries=qs),columns,gates


def run(pair,scope):
    import torch
    from flex_uda.active.strong_prefix_teacher import fit_head
    guard=generation_guard();begin=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=AUDIT/pair/scope;root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').is_file():return
    model,target=PAIRS[pair];m=metadata(target);videos=sorted(m['videos'])
    methods=['PAST_ONLY'] if scope=='PAST_ONLY' else ['SOURCE_ONLY','DIRECT_TEACHER','CURRENT_ONLY','PREFIX_REPRO']
    moment_sum=np.zeros(1080);moment_square=np.zeros(1080);image_rows=0;bx=[];bz=[];by=[];receipts=[];start=0
    state=root/'stream.pt'
    if state.is_file():
        saved=torch.load(state,map_location='cpu',weights_only=False)
        start=saved['next_video'];moment_sum=saved['sum'];moment_square=saved['square'];image_rows=saved['image_rows']
        bx=saved['x'];bz=saved['z'];by=saved['y'];receipts=saved['receipts']
    for vi,video in enumerate(videos):
        if vi<start:continue
        t=time.perf_counter();raw,z,answer_path,context,columns,gates=inputs(pair,video,m)
        current=raw.reshape(-1,1080);moment_sum+=current.sum(0,dtype=np.float64)
        moment_square+=np.square(current.astype(float)).sum(0);image_rows+=len(current)
        mean=moment_sum/image_rows;std=np.maximum(np.sqrt(np.maximum(moment_square/image_rows-mean**2,0)),1e-3)
        exclusive=exclusive_group([q['query_text'] for q in context.queries],target)

        def output(method,prediction):
            expanded=prediction[:,columns];mask=plain(expanded,exclusive)
            for qi,gate in gates.items():mask[qi]&=gate
            folder=root/method;path=folder/'videos'/video;path.mkdir(parents=True,exist_ok=True)
            np.save(path/'mask.npy',mask);np.save(path/'scores.npy',expanded.astype('float32'))
            immediate_predictions(folder,context,mask)

        fits={};before=len(np.concatenate(by)) if by else 0
        if scope=='PAST_ONLY':
            if by:
                score,fit,_=fit_head(np.concatenate(bx),np.concatenate(bz),np.clip(np.concatenate(by),.02,.98),
                    current,z.ravel(),mean,std,soft=True,seed=0)
                prediction=score['CONVERGED'].reshape(z.shape);fits['PAST_ONLY']=fit
            else:prediction=expit(z)
            output('PAST_ONLY',prediction)
            prediction_written_before_current_answers=time.perf_counter()
        else:prediction_written_before_current_answers=None
        # PAST_ONLY has already sealed this video's output before this read.
        data=json.loads(answer_path.read_text());teacher=np.full(z.shape,np.nan)
        for row in data['answers']:teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
        known=np.isfinite(teacher);bx.append(raw[known]);bz.append(z[known]);by.append(teacher[known])
        if scope=='CURRENT_ONLY':
            output('SOURCE_ONLY',expit(z))
            output('DIRECT_TEACHER',np.where(known,teacher,expit(z)))
            for method,x,s,y in [('CURRENT_ONLY',raw[known],z[known],teacher[known]),
                                  ('PREFIX_REPRO',np.concatenate(bx),np.concatenate(bz),np.concatenate(by))]:
                score,fit,_=fit_head(x,s,np.clip(y,.02,.98),current,z.ravel(),mean,std,soft=True,seed=0)
                output(method,score['CONVERGED'].reshape(z.shape));fits[method]=fit
        receipt=dict(video=video,index=vi,previous_teacher_examples=before,current_teacher_examples=int(known.sum()),
            eligible_current_pairs=z.size,output_before_current_teacher_read=scope=='PAST_ONLY',
            pre_feedback_output_time=prediction_written_before_current_answers,fits=fits,
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-t)
        atomic_json(root/'videos'/video/'receipt.json',receipt);receipts.append(receipt)
        saved=dict(next_video=vi+1,sum=moment_sum,square=moment_square,image_rows=image_rows,x=bx,z=bz,y=by,receipts=receipts)
        temporary=state.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(state)
        print(json.dumps(dict(event='BENCHMARK_SCOPE_VIDEO',pair=pair,scope=scope,video=video,completed=vi+1,
            total=len(videos),current_answers=int(known.sum()),previous_answers=before)),flush=True)
    records=[]
    for method in methods:
        folder=root/method;mask=np.zeros(m['candidate_query_rows'],dtype=bool)
        for video in videos:
            local=np.load(folder/'videos'/video/'mask.npy')
            for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=local[qi]
        seal=seal_mask(folder,m,mask,target=target,model=model,method='BENCHMARK_SCOPE_'+method,GT_open_attempts=len(guard),
            extra=dict(feedback_scope=method,current_video_transductive=True,all_development_data=True,
                fixed_recorded_teacher_quality_diagnostic=True,new_teacher_calls=0,
                inherited_features_prepaid=True,not_cold_deployment_cost=True,
                PAST_ONLY_is_a_different_information_scope_not_a_matched_budget_superiority_test=True))
        records.append(dict(method=method,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_BENCHMARK_SCOPE',pair=pair,scope=scope,methods=records,
        teacher_records_read=sum(len(y) for y in by),GT_open_attempts=0,new_teacher_calls=0,
        receipts=receipts,wall_seconds=time.perf_counter()-begin,all_predictions_sealed_before_GT=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True)
    p.add_argument('--scope',choices=['CURRENT_ONLY','PAST_ONLY'],required=True);a=p.parse_args();run(a.pair,a.scope)
