"""Causal visual exemplar memory, with a fixed source model and teacher replay.

The matched KD scores were sealed by the upstream worker for this exact video
prefix. They are never recomputed with a later teacher-trained checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,FrozenInputs,PAIRS,generation_guard
from flex_uda.active.exemplar_memory import ExemplarMemory
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import read_guard,plain,immediate_predictions
from flex_uda.active.acquisition import exclusive_group,expression_types

POLICIES=('COVER','UNIFORM')
METHODS=('KERNEL','MEAN','SUPPORT_MIX','SUPPORT_GATED','SUPPORT_MIX_GRAPH')


def run(pair):
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    model,target=PAIRS[pair];m=metadata(target);root=OUT/'exemplar_prefix'/pair;begin=time.perf_counter()
    if (root/'completion.json').is_file():return
    rows=[];buffers={p:{k:[] for k in ('x','z','y')} for p in POLICIES}
    for index,video in enumerate(sorted(m['videos'])):
        upstream=OUT/'strict_prefix'/pair/'videos'/video/'prefix_receipt.json'
        while not upstream.is_file():time.sleep(5)
        started=time.perf_counter();visible.add(video)
        prefix=json.loads(upstream.read_text());assert prefix['visible_teacher_videos']==sorted(visible)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:
            features=a['features'];source=np.clip(logit(np.clip(a['prior'],.05,.95)),-2,2)
        answers={}
        for policy in POLICIES:
            teacher=np.full(source.shape,np.nan)
            shard=json.loads((OUT/'causal_teacher'/pair/policy/(video+'.json')).read_text())
            for answer in shard['answers']:teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
            known=np.isfinite(teacher)
            for k,value in [('x',features[known]),('z',source[known]),('y',teacher[known])]:buffers[policy][k].append(value)
            answers[policy]=teacher
        previous_path=root/'videos'/video/'prefix_receipt.json'
        if previous_path.is_file():
            previous=json.loads(previous_path.read_text())
            assert previous['source_KD_receipt_sha256']==file_sha(upstream)
            assert previous['visible_teacher_videos']==sorted(visible)
            rows.append(previous)
            continue
        fc=FrozenInputs(pair,video);track_index={t:i for i,t in enumerate(fc.tracks)}
        columns=np.asarray([track_index[int(t)] for _,t in fc.observations]);first=np.asarray([fc.by_track[t][0] for t in fc.tracks])
        exclusive=exclusive_group([q['query_text'] for q in fc.queries],target)
        fits={};receipts={};variances={}
        for policy in POLICIES:
            teacher=answers[policy]
            source_folder=OUT/'strict_prefix'/pair/(policy+'__TRACK_SOFT')/'videos'/video
            kd=np.load(source_folder/'scores.npy')[:,first];kz=logit(np.clip(kd,1e-6,1-1e-6))
            memory=ExemplarMemory();x=np.concatenate(buffers[policy]['x']);z=np.concatenate(buffers[policy]['z']);y=np.concatenate(buffers[policy]['y'])
            assert len(y)==prefix['teacher_examples'][policy]
            if len(y):
                fit_begin=time.perf_counter();fits[policy]=memory.fit(x,z,y)
                gp,v=memory.predict(features.reshape(-1,768),source.ravel());gp=gp.reshape(source.shape);v=v.reshape(source.shape)
                fits[policy]['fit_and_predict_wall_seconds']=time.perf_counter()-fit_begin
            else:gp=expit(source);v=np.ones_like(source);fits[policy]=dict(examples=0,fit_and_predict_wall_seconds=0.)
            g=logit(np.clip(gp,1e-6,1-1e-6))
            scores=dict(KERNEL=gp,MEAN=expit(.5*(g+kz)),SUPPORT_MIX=expit((1-v)*kz+v*g),
                SUPPORT_GATED=expit(source+(1-v)*(kz-source)))
            scores['SUPPORT_MIX_GRAPH']=np.stack([refine_distilled_prior(scores['SUPPORT_MIX'][q],features[q],teacher[q]) for q in range(len(source))])
            variances[policy]=dict(mean=float(v.mean()),quantiles=np.quantile(v,[0,.25,.5,.75,1]).tolist())
            receipts[policy]={}
            for method,score in scores.items():
                folder=root/(policy+'__'+method);local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
                expanded=score[:,columns];mask=plain(expanded,exclusive)
                for qi,q in enumerate(fc.queries):
                    if 'MOTION' in expression_types(q['query_text']):
                        p=expit(fc.raw_presence[qi]);mask[qi]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
                np.save(local/'mask.npy',mask);np.save(local/'scores.npy',expanded.astype('float32'))
                np.savez_compressed(local/'track_belief.npz',score=score,variance=v)
                receipts[policy][method]=immediate_predictions(folder,fc,mask)
        row=dict(video=video,index=index,visible_teacher_videos=sorted(visible),fits=fits,variance=variances,
            source_KD_receipt_sha256=file_sha(upstream),source_head_is_exact_current_prefix=True,
            no_future_teacher_feedback=True,GT_open_attempts=len(gt),output_hashes=receipts,
            process_wall_seconds=time.perf_counter()-started)
        atomic_json(root/'videos'/video/'prefix_receipt.json',row);rows.append(row)
        print(json.dumps(dict(event='EXEMPLAR_PREFIX_VIDEO',pair=pair,video=video,prefix=index+1,total=len(m['videos']),
            teacher_examples={p:fits[p]['examples'] for p in POLICIES},wall_seconds=row['process_wall_seconds'])),flush=True)
    records=[]
    for policy in POLICIES:
        for name in METHODS:
            folder=root/(policy+'__'+name);mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video in sorted(m['videos']):
                local=np.load(folder/'videos'/video/'mask.npy')
                for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
            calls=sum(len(x) for x in buffers[policy]['y'])
            seal=seal_mask(folder,m,mask,target=target,model=model,method='EXEMPLAR_'+policy+'__'+name,
                GT_open_attempts=len(gt),extra=dict(logical_teacher_calls=calls,video_stream_causal=True,
                current_video_transductive=True,no_future_teacher_feedback=True,no_batch_target_checkpoint=True,
                teacher_recording_replay=True,not_a_live_end_to_end_wall_benchmark=True,
                kernel='Gaussian on row-centered unit-normalized frozen PCD',bandwidth='median tenth training neighbour, visible prefix only',
                regularizer=.1,seed=0,upstream_KD='matching sealed current-prefix KD',
                support_mixture='(1-variance)*KD_logit+variance*kernel_logit',forbidden_read_attempts=len(forbidden)))
            records.append(dict(method=policy+'__'+name,rows=seal['prediction_rows'],calls=calls))
    assert not gt and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_CAUSAL_EXEMPLAR_PREFIX',pair=pair,methods=records,
        GT_open_attempts=0,forbidden_read_attempts=0,teacher_reads=reads,video_receipts=rows,
        wall_seconds=time.perf_counter()-begin,waiting_for_upstream_included=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();run(a.pair)
