"""Same-feature KD vs classical colour cohorts, with video-causal teacher use."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit,expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.active.color_cohort import augmented_features,part_mask,PART_DIM
from flex_uda.active.prefix_teacher import PrefixTeacherHead
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.acquisition import exclusive_group,expression_types
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_strict_prefix import read_guard,plain,immediate_predictions
from scripts.a100_visual_classical_prepare import OUT

POLICIES=('COVER','UNIFORM')
METHODS=('RGB_KD','NO_GRAPH','COHORT','PCD_COHORT','COLOR_ONLY','STRONG_PRIOR')


def run(pair,policies=POLICIES,group='colour_prefix'):
    import torch
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model,target=PAIRS[pair];m=metadata(target);root=OUT/group/pair;begin=time.perf_counter()
    if (root/'completion.json').is_file():return
    videos=sorted(m['videos']);dimension=768+3*PART_DIM
    heads={p:PrefixTeacherHead('cuda:0',0,feature_dim=dimension) for p in policies};start=0;receipts=[]
    checkpoint=root/'stream_latest.pt'
    if checkpoint.is_file():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);start=saved['next_video'];receipts=saved['receipts'];visible.update(videos[:start])
        for p in policies:heads[p].restore(saved['heads'][p])
    root.mkdir(parents=True,exist_ok=True)
    for vi,video in enumerate(videos):
        if vi<start:continue
        feature_root=OUT/'color_features'/pair/video
        while not (feature_root/'receipt.json').is_file():time.sleep(5)
        began=time.perf_counter();visible.add(video);fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:original=a['features'];source=np.clip(logit(np.clip(a['prior'],.05,.95)),-2,2)
        with np.load(feature_root/'features.npz') as a:colours=a['colours'];assert np.array_equal(a['tracks'],fc.tracks)
        texts=[q['query_text'] for q in fc.queries];features,pcd,colour=augmented_features(original,colours,texts)
        applicable=[bool(part_mask(t).any()) for t in texts]
        exclusive=exclusive_group(texts,target);track_index={t:i for i,t in enumerate(fc.tracks)}
        columns=np.asarray([track_index[int(t)] for _,t in fc.observations]);fits={};outputs={};post_time={}
        for policy in policies:
            shard_path=OUT/'causal_teacher'/pair/policy/(video+'.json')
            while not shard_path.is_file():time.sleep(5)
            shard=json.loads(shard_path.read_text());teacher=np.full(source.shape,np.nan)
            for a in shard['answers']:teacher[int(a['local_query']),int(a['local_track'])]=a['yes_probability']
            known=np.isfinite(teacher);heads[policy].add(features[known],source[known],teacher[known]);fits[policy]=heads[policy].fit()
            kd=heads[policy].predict(features.reshape(-1,dimension),source.ravel()).reshape(source.shape)
            began_post=time.perf_counter();scores={name:kd.copy() for name in METHODS}
            for qi,use_colour in enumerate(applicable):
                if not use_colour:continue
                for name,geometry,graph,weight in [('NO_GRAPH',features[qi],False,.1),('COHORT',features[qi],True,.1),
                    ('PCD_COHORT',pcd[qi],True,.1),('COLOR_ONLY',colour[qi],True,.1),('STRONG_PRIOR',features[qi],True,1.)]:
                    requested=part_mask(texts[qi])>0
                    visible_parts=np.all(colours[:,requested].sum(2)>0,axis=1)
                    scores[name][qi]=refine_distilled_prior(kd[qi],geometry,teacher[qi],graph=graph,source_weight=weight,node_mask=visible_parts)
            outputs[policy]={}
            for name,score in scores.items():
                expanded=score[:,columns];mask=plain(expanded,exclusive)
                for qi,q in enumerate(fc.queries):
                    if 'MOTION' in expression_types(q['query_text']):
                        p=expit(fc.raw_presence[qi]);mask[qi]&=np.asarray([p[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
                folder=root/(policy+'__'+name);local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
                np.save(local/'mask.npy',mask);np.save(local/'scores.npy',expanded.astype('float32'))
                np.savez_compressed(local/'track_belief.npz',score=score)
                outputs[policy][name]=immediate_predictions(folder,fc,mask)
            post_time[policy]=time.perf_counter()-began_post
        row=dict(video=video,index=vi,visible_teacher_videos=sorted(visible),fits=fits,postprocess_wall_seconds=post_time,
            colour_applicable_queries=int(sum(applicable)),query_count=len(applicable),feature_dim=dimension,
            physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),colour_receipt_sha256=file_sha(feature_root/'receipt.json'),
            output_hashes=outputs,GT_open_attempts=len(gt),no_future_teacher_feedback=True,
            wall_seconds=time.perf_counter()-began)
        atomic_json(root/'videos'/video/'prefix_receipt.json',row);receipts.append(row)
        state=dict(next_video=vi+1,receipts=receipts,heads={p:h.state() for p,h in heads.items()})
        temporary=checkpoint.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint)
        print(json.dumps(dict(event='COLOUR_PREFIX_VIDEO',pair=pair,video=video,prefix=vi+1,total=len(videos),
            teacher_examples={p:len(h.y) for p,h in heads.items()},wall_seconds=row['wall_seconds'])),flush=True)
    methods=[]
    for policy in policies:
        for name in METHODS:
            folder=root/(policy+'__'+name);mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video in videos:
                local=np.load(folder/'videos'/video/'mask.npy')
                for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
            seal=seal_mask(folder,m,mask,target=target,model=model,method='COLOUR_'+policy+'__'+name,GT_open_attempts=len(gt),
                extra=dict(logical_teacher_calls=len(heads[policy].y),new_teacher_calls=0,feature_dim=dimension,
                same_augmented_features_for_KD_and_graph=True,video_stream_causal=True,current_video_transductive=True,
                no_future_teacher_feedback=True,global_target_normalization=False,teacher_only_fit_stop=True,
                pure_image_colour_features=True,not_live_end_to_end_wall=True,forbidden_read_attempts=len(forbidden)))
            methods.append(dict(method=policy+'__'+name,rows=seal['prediction_rows'],calls=len(heads[policy].y)))
    assert not gt and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_COLOUR_COHORT_PREFIX',pair=pair,methods=methods,
        GT_open_attempts=0,forbidden_read_attempts=0,teacher_reads=reads,video_receipts=receipts,
        wall_seconds=time.perf_counter()-begin,waiting_for_image_preparation_included=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1'])
    p.add_argument('--group',default='colour_prefix');p.add_argument('--policies',default=','.join(POLICIES))
    a=p.parse_args();run(a.pair,tuple(a.policies.split(',')),a.group)
