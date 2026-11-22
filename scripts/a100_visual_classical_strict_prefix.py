"""Video-stream-causal RMOT with teacher-only adaptation and classic controls.

Past predictions are saved immediately. No batch target-trained checkpoint or
future teacher shard is an input. A video itself remains transductive.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import FrozenInputs,metadata,PAIRS,HISTORY,generation_guard
from flex_uda.active.prefix_teacher import PrefixTeacherHead
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.semantic_memory import blend_memory,categorical_smooth,binary_smooth,ema
from flex_uda.active.acquisition import expression_types,exclusive_group
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT

POLICIES=('PRIOR','UNIFORM','UNCERTAINTY','COVER')
METHODS=('TRACK_SOFT','FRAME_SOFT','FRAME_STRUCTURED','DIRECT_NATIVE','ENSEMBLE','LOCAL_GRAPH',
    'GRAPH_ENSEMBLE','FRAME_EMA','FRAME_HMM','MEMORY_HMM','RESET_MEMORY_HMM','GRAPH_MEMORY_HMM','GRAPH_NO_RESET_HMM')


def read_guard(pair,visible):
    attempts=[];reads=[]
    banned={'convergence','frame_convergence','predictions','refinement_v2','semantic_memory_v3','analysis','diagnosis','identity_diagnosis'}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));flags=args[2] if len(args)>2 else 0
        if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR):return
        try:parts=path.relative_to(OUT).parts
        except ValueError:return
        if not parts:return
        reason=None
        if parts[0] in banned:reason='batch-trained or evaluation-derived artifact'
        if parts[0]=='evidence':reason='aggregate future teacher pool'
        if parts[0]=='causal_teacher' and path.suffix=='.json':
            if path.stem not in visible:reason='future video teacher shard'
            else:reads.append(str(path))
        if parts[0]=='frame_anchor_features' and path.suffix=='.npz' and path.stem not in visible:
            reason='future video feature payload'
        if parts[0]=='inputs' and len(parts)>2 and parts[2] not in visible:
            reason='future video feature payload'
        if parts[0]=='color_features' and len(parts)>2 and parts[2] not in visible:
            reason='future video colour payload'
        if reason:
            attempts.append(dict(path=str(path),reason=reason));raise PermissionError(reason+': '+str(path))
    sys.addaudithook(audit)
    return attempts,reads


def plain(score,exclusive):
    mask=score>.5
    if exclusive:mask &= np.arange(len(score))[:,None]==score.argmax(0)[None]
    return mask


def temporal(frame,prior,fc,*,reset=False):
    exclusive=exclusive_group([q['query_text'] for q in fc.queries],fc.target)
    hazards=np.asarray([.1 if any(t in expression_types(q['query_text']) for t in ('MOTION','SPATIAL')) else .02 for q in fc.queries])
    score=np.zeros_like(frame,dtype=float);mask=np.zeros_like(frame,dtype=bool)
    for ti,track in enumerate(fc.tracks):
        ids=fc.by_track[track];cuts=np.r_[0,np.flatnonzero(np.diff(fc.observations[ids,0])>2)+1,len(ids)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            ix=ids[a:b];p=frame[:,ix].T
            if prior is not None:
                p=blend_memory(p,np.broadcast_to(prior[:,ti],p.shape),reset=reset,exclusive=exclusive)
            result,bits=categorical_smooth(p,.02) if exclusive else binary_smooth(p,hazards)
            score[:,ix]=result.T;mask[:,ix]=bits.T
    return score,mask


def current_predictions(fc,frame,track_probability,teacher):
    exclusive=exclusive_group([q['query_text'] for q in fc.queries],fc.target)
    index={t:i for i,t in enumerate(fc.tracks)};columns=np.asarray([index[int(t)] for _,t in fc.observations])
    track=track_probability[:,columns]
    graph=np.stack([refine_distilled_prior(track_probability[q],fc.local_features[q],teacher[q]) for q in range(len(fc.queries))])
    expanded=graph[:,columns]
    mixed=blend_memory(frame.T,track.T,exclusive=exclusive).T
    graph_mixed=blend_memory(frame.T,expanded.T,exclusive=exclusive).T
    score=dict(TRACK_SOFT=track,FRAME_SOFT=frame,FRAME_STRUCTURED=frame,ENSEMBLE=mixed,
        LOCAL_GRAPH=expanded,GRAPH_ENSEMBLE=graph_mixed)
    masks={k:plain(v,exclusive if k!='FRAME_SOFT' else False) for k,v in score.items()}
    native=fc.decode()[2];seen=np.isfinite(teacher)[:,columns]
    direct=np.where(seen,np.nan_to_num(teacher,nan=.5)[:,columns],native.astype(float))
    score['DIRECT_NATIVE']=direct;masks['DIRECT_NATIVE']=direct>.5
    smoothed=np.zeros_like(frame)
    for tid,ids in fc.by_track.items():
        cuts=np.r_[0,np.flatnonzero(np.diff(fc.observations[ids,0])>2)+1,len(ids)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            ix=ids[a:b];smoothed[:,ix]=ema(frame[:,ix].T).T
    score['FRAME_EMA']=smoothed;masks['FRAME_EMA']=plain(smoothed,exclusive)
    for name,prior,reset in [('FRAME_HMM',None,False),('MEMORY_HMM',track_probability,False),
        ('RESET_MEMORY_HMM',track_probability,True),('GRAPH_MEMORY_HMM',graph,True),('GRAPH_NO_RESET_HMM',graph,False)]:
        score[name],masks[name]=temporal(frame,prior,fc,reset=reset)
    for qi,q in enumerate(fc.queries):
        if 'MOTION' in expression_types(q['query_text']):
            p=expit(fc.raw_presence[qi]);gate=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
            for name in METHODS:
                if name not in ('FRAME_SOFT','DIRECT_NATIVE'):masks[name][qi]&=gate
    return score,masks


def immediate_predictions(folder,fc,mask):
    rows=[]
    for qi,q in enumerate(fc.queries):
        text=''.join(fc.bank['lines'][i]+'\n' for i in np.flatnonzero(mask[qi]))
        path=folder/'results'/q['video_id']/q['expression_id']/'predict.txt';path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file():assert path.read_text()==text,'Prefix prediction cannot change'
        else:path.write_text(text)
        rows.append(dict(query=q['expression_id'],sha256=file_sha(path)))
    return rows


def run(pair,policies=POLICIES,max_videos=0):
    import torch
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/'strict_prefix'/pair
    if max_videos:root=root/('pilot_'+str(max_videos))
    if (root/'completion.json').is_file():return
    videos=sorted(m['videos']);selected=videos[:max_videos] if max_videos else videos
    heads={p:{kind:PrefixTeacherHead('cuda:0',0) for kind in ('track','frame')} for p in policies}
    start=0;receipts=[];state_path=root/'stream_latest.pt'
    if state_path.is_file():
        saved=torch.load(state_path,map_location='cpu',weights_only=False);start=saved['next_video'];receipts=saved['receipts']
        visible.update(selected[:start])
        for p in policies:
            for kind in heads[p]:heads[p][kind].restore(saved['heads'][p][kind])
    root.mkdir(parents=True,exist_ok=True)
    for vi,video in enumerate(selected):
        if vi<start:continue
        video_begin=time.perf_counter();phases={};visible.add(video);fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:
            features=a['features'];source_prior=a['prior']
        fc.local_features=features
        # Pure source features cached without teacher training or global
        # normalization. They are loaded only for this current video.
        with np.load(OUT/'frame_anchor_features'/pair/(video+'.npz')) as a:
            anchors=a['features'];anchor_logits=a['source_logits']
        qn,tn=source_prior.shape;track_logits=np.clip(logit(np.clip(source_prior,.05,.95)),-2,2)
        phases['source_input_load']=time.perf_counter()-video_begin
        fit_begin=time.perf_counter();answers={};fits={};track_predictions={}
        for policy in policies:
            path=OUT/'causal_teacher'/pair/policy/(video+'.json');shard=json.loads(path.read_text())
            teacher=np.full((qn,tn),np.nan)
            for row in shard['answers']:teacher[int(row['local_query']),int(row['local_track'])]=row['yes_probability']
            answers[policy]=teacher;valid=np.isfinite(teacher)
            heads[policy]['track'].add(features[valid],track_logits[valid],teacher[valid])
            heads[policy]['frame'].add(anchors[valid],anchor_logits[valid],teacher[valid])
            fits[policy]={kind:head.fit() for kind,head in heads[policy].items()}
            track_predictions[policy]=heads[policy]['track'].predict(features.reshape(-1,768),track_logits.ravel()).reshape(qn,tn)
        phases['all_prefix_teacher_fit']=time.perf_counter()-fit_begin
        feature_begin=time.perf_counter();pcd=np.load(HISTORY/'features'/target/model/'videos'/video/'pcd.npy',mmap_mode='r')
        source=fc.margin.ravel();frames={p:np.empty(len(source),dtype='float32') for p in policies}
        for a in range(0,len(source),8192):
            b=min(a+8192,len(source));x=np.asarray(pcd[a:b],dtype='float32')
            for policy in policies:frames[policy][a:b]=heads[policy]['frame'].predict(x,source[a:b])
        phases['source_PCD_read_and_frame_inference']=time.perf_counter()-feature_begin
        classic_begin=time.perf_counter();outputs={}
        for policy in policies:
            score,masks=current_predictions(fc,frames[policy].reshape(fc.margin.shape),track_predictions[policy],answers[policy])
            outputs[policy]={}
            for name in METHODS:
                folder=root/(policy+'__'+name);local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
                np.save(local/'mask.npy',masks[name]);np.save(local/'scores.npy',score[name].astype('float32'))
                outputs[policy][name]=immediate_predictions(folder,fc,masks[name])
        phases['classical_inference_and_prediction_write']=time.perf_counter()-classic_begin
        row=dict(video=video,index=vi,visible_teacher_videos=sorted(visible),future_teacher_opens=0,phases=phases,
            teacher_examples={p:len(heads[p]['track'].y) for p in policies},fits=fits,output_hashes=outputs,
            no_batch_checkpoint_used=True,no_future_target_normalization=True,
            process_wall_seconds=time.perf_counter()-video_begin)
        atomic_json(root/'videos'/video/'prefix_receipt.json',row);receipts.append(row)
        state=dict(next_video=vi+1,receipts=receipts,heads={p:{k:h.state() for k,h in kinds.items()} for p,kinds in heads.items()})
        temporary=state_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(state_path)
        print(json.dumps(dict(event='STRICT_PREFIX_VIDEO',pair=pair,video=video,prefix=vi+1,total=len(selected),
            teacher_labels={p:len(heads[p]['track'].y) for p in policies},wall_seconds=row['process_wall_seconds'])),flush=True)
    assert not gt and not forbidden
    if max_videos:
        atomic_json(root/'completion.json',dict(status='CAUSAL_PREFIX_PILOT_ONLY',videos=selected,
            GT_open_attempts=0,forbidden_read_attempts=0,teacher_reads=reads))
        return
    records=[]
    for policy in policies:
        for name in METHODS:
            folder=root/(policy+'__'+name);mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video in selected:
                qs=[q for q in m['episodes'] if q['video_id']==video];local=np.load(folder/'videos'/video/'mask.npy')
                for qi,q in enumerate(qs):mask[q['start']:q['stop']]=local[qi]
            total=len(heads[policy]['track'].y)
            seal=seal_mask(folder,m,mask,target=target,model=model,method='CAUSAL_'+policy+'__'+name,GT_open_attempts=len(gt),
                extra=dict(video_stream_causal=True,current_video_transductive=True,frame_causal=False,
                    no_future_teacher_feedback=True,no_batch_target_checkpoint=True,global_target_normalization=False,
                    acquisition=policy,logical_teacher_calls=total,forbidden_read_attempts=len(forbidden),
                    physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),teacher_recording_replay=True,
                    not_a_live_end_to_end_wall_benchmark=True,source_feature_extraction_from_historical_cache=True))
            records.append(dict(method=policy+'__'+name,rows=seal['prediction_rows'],calls=total))
    atomic_json(root/'completion.json',dict(status='SEALED_STRICT_VIDEO_PREFIX',pair=pair,methods=records,
        GT_open_attempts=0,forbidden_read_attempts=0,teacher_reads=reads,video_receipts=receipts,
        wall_seconds=time.perf_counter()-begin,labels_only_from_teacher=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--max-videos',type=int,default=0)
    p.add_argument('--policies',default=','.join(POLICIES));a=p.parse_args();run(a.pair,tuple(a.policies.split(',')),a.max_videos)
