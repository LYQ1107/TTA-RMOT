"""Measured cold full-clip pipeline; no historical neural feature or QA bank.

Common physical work is performed once and explicitly shared. Each baseline
and full arm then fits its own same-seed head and measures its own postprocess.
This is component accounting on one full clip, not an invented whole-C4 cost.
"""
import time
PROCESS_BEGIN=time.perf_counter()
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,append_json
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.color_cohort import augmented_features,part_histograms,part_mask
from flex_uda.active.visual_classical import visual_graph,acquisition_order,refine_distilled_prior
from scripts.a100_visual_classical_prepare import OUT


class SparsePCD:
    """Retain exactly the old three uniformly spaced samples, not a 9GB bank."""
    def __init__(self,observations,nq):
        self.tracks=sorted(set(observations[:,1]));self.samples=[];self.mapping={}
        for ti,t in enumerate(self.tracks):
            ids=np.flatnonzero(observations[:,1]==t)
            chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
            self.samples.append(chosen)
            for si,row in enumerate(chosen):self.mapping[int(row)]=(ti,si)
        self.values=np.zeros((nq,len(self.tracks),3,768),dtype=np.float32)
        self.seen=np.zeros((nq,len(self.tracks),3),dtype=bool)

    def __call__(self,qa,ids,feature):
        for ci,row in enumerate(ids):
            if int(row) not in self.mapping:continue
            ti,si=self.mapping[int(row)];nq=feature.shape[1]
            self.values[qa:qa+nq,ti,si]=feature[ci].float().cpu().numpy().astype(np.float16).astype(np.float32)
            self.seen[qa:qa+nq,ti,si]=True

    def mean(self):
        for ti,rows in enumerate(self.samples):assert self.seen[:,ti,:len(rows)].all()
        return np.stack([self.values[:,ti,:len(rows)].mean(1) for ti,rows in enumerate(self.samples)],axis=1)


def run(name):
    guard=generation_guard();forbidden=[]
    def no_prepaid(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR):return
        if path.suffix in ('.npy','.npz') and ('features' in path.parts or path.name in ('initial.npz','expected_iou_utilities.npy','raw_k.npy')):
            if 'cold_live' not in path.parts:
                forbidden.append(str(path));raise PermissionError('No historical neural feature arrays in cold run')
        try:parts=path.relative_to(OUT).parts
        except ValueError:return
        if parts and parts[0]!='cold_live':
            forbidden.append(str(path));raise PermissionError('Cold run cannot consume prepared research artifacts: '+str(path))
    sys.addaudithook(no_prepaid)
    import torch
    from flex_uda.a100.runtime import load_student,source_head,teacher
    from flex_uda.presence.runtime import episode_from_bank
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.active.v6_frames import FrameCacheV6
    from flex_uda.active.strong_prefix_teacher import fit_head
    from scripts.a100_visual_classical_strict_prefix import plain
    from flex_uda.active.acquisition import exclusive_group,expression_types
    from flex_uda.presence.predictions import seal_mask
    from scripts.active_run import video_spec
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda:0');torch.cuda.set_device(device)
    root=OUT/'cold_live'/name;root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').is_file():return
    assert not (root/'raw.sqlite').is_file(),'Use a fresh named run to measure cold real teacher requests'
    m=metadata('SportsMOT');video=sorted(m['videos'])[0];bank=m['videos'][video]
    queries=[q for q in m['episodes'] if q['video_id']==video];observations=np.asarray(bank['observations']);nq=len(queries)
    frames=sorted(set(observations[:,0]));sink=SparsePCD(observations,nq);tracks=sink.tracks
    protocol=dict(name=name,pair='C4',video=video,frames=len(frames),observations=len(observations),queries=nq,
        sample_rule='first lexicographic C4 clip, full frames and all its provided queries; no GT selection',
        weights_and_features_cold=True,teacher_fresh_SQLite=True,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        target_GT_used=False,teacher='same inherited frozen binary CROP interface',source_feature_FP16_samples_then_FP32_mean=True,
        shared_measured_prefix=True,independent_same_seed_KD_fits_per_arm=True,
        no_whole_C4_cost_extrapolation=True,no_OS_cache_flushing=True)
    atomic_json(root/'protocol.json',protocol);phases={}
    def event(key,start):
        phases[key]=time.perf_counter()-start
        print(json.dumps(dict(event='COLD_LIVE_PHASE',phase=key,seconds=phases[key],elapsed=time.perf_counter()-PROCESS_BEGIN)),flush=True)
    start=time.perf_counter();student,tok,transform,student_receipt=load_student(root/'student_log',device);event('student_load_including_weight_verification',start)
    start=time.perf_counter();ep=episode_from_bank(video,bank,'SportsMOT',transform)
    cache=FrozenFeatureCache(student,ep,tok,device,max_bytes=512<<20)
    fc=FrameCacheV6(student,cache,bank,queries,frames,source_head(),root/'source_forward',store_pcd=False,pcd_sink=sink)
    fc.score();original=sink.mean();prior,_,native=fc.decode(1.,0.);source_receipt=fc.close();event('all_source_forward_text_and_initial_head',start)
    start=time.perf_counter();colours=[];anchors=[];images={};boxes=np.asarray([list(map(float,s.split(',')[2:6])) for s in bank['lines']])
    for rows in sink.samples:
        hist=[]
        for row in rows:
            frame=int(observations[row,0])
            if frame not in images:
                with Image.open(Path(bank['image_root'])/f'{frame:06d}.jpg') as im:images[frame]=im.convert('RGB')
            hist.append(part_histograms(images[frame],boxes[row]))
        hist=np.stack(hist);present=hist.sum(2)>0;count=present.sum(0)
        colour=hist.sum(0)/np.maximum(count[:,None],1)
        colours.append(np.sqrt(colour));anchors.append(int(observations[rows[np.argmax(np.prod(boxes[rows,2:4],axis=1))],0]))
    colours=np.asarray(colours);images.clear();event('real_RGB_part_features',start)
    start=time.perf_counter();texts=[q['query_text'] for q in queries]
    geometry,_,selected_colour=augmented_features(original,colours,texts)
    raw=np.concatenate([original,selected_colour],axis=2).astype('float32');source=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.)
    lengths=np.asarray([(observations[:,1]==t).sum() for t in tracks]);eligible=np.flatnonzero(lengths>=3)
    budget=int(np.floor(.3*nq*len(eligible)));quotas=np.full(nq,budget//nq);quotas[:budget%nq]+=1;requests=[]
    for qi,q in enumerate(queries):
        # This is the original frozen-PDC COVER policy, not the newer facility policy.
        _,dist=visual_graph(original[qi]);rng=np.random.default_rng(int(__import__('hashlib').sha256((video+str(qi)).encode()).hexdigest()[:8],16))
        order=acquisition_order(prior[qi],dist,eligible,'COVER',rng)
        for ti in order[:quotas[qi]]:requests.append(dict(target='SportsMOT',video_id=video,query_id=q['expression_id'],query_text=q['query_text'],
            track_id=int(tracks[ti]),frame_id=anchors[ti],view='CROP',local_query=qi,local_track=ti))
    atomic_json(root/'questions_sealed_before_teacher.json',dict(requests=requests,logical_budget=budget,GT_open_attempts=len(guard)))
    event('image_only_input_assembly_and_acquisition',start)
    fc.model=None;fc.cache=None;fc.texts=[]
    del cache,student,ep;torch.cuda.empty_cache()
    start=time.perf_counter();client=teacher([video_spec('SportsMOT',video,bank)],root/'raw.sqlite');event('teacher_load_including_weight_verification',start)
    start=time.perf_counter();answer=np.full(prior.shape,np.nan);responses=[]
    for offset in range(0,len(requests),8):
        batch=requests[offset:offset+8]
        values=client.ask_many([{k:v for k,v in r.items() if k not in ('local_query','local_track')} for r in batch])
        for r,v in zip(batch,values):
            answer[r['local_query'],r['local_track']]=v['yes_probability'];responses.append(v);append_json(root/'responses.jsonl',v)
    event('real_teacher_service_including_image_IO',start)
    start=time.perf_counter();teacher_receipt=client.close();event('teacher_final_frozen_weight_audit',start)
    assert teacher_receipt['cache_hits']==0 and teacher_receipt['new_calls']==budget
    del client;torch.cuda.empty_cache()
    current=raw.reshape(-1,raw.shape[-1]);mean=current.mean(0,dtype=np.float64);std=np.maximum(current.std(0,dtype=np.float64),1e-3)
    known=np.isfinite(answer);lookup={t:i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in observations]);exclusive=exclusive_group(texts,'SportsMOT')
    localmeta=dict(m,videos={video:bank},episodes=[],candidate_query_rows=nq*len(observations));at=0
    for q in queries:
        localmeta['episodes'].append(dict(q,start=at,stop=at+len(observations)));at+=len(observations)
    common_seconds=time.perf_counter()-PROCESS_BEGIN
    arms={};reference=None
    for arm,soft,classic in [('SOFT_KD',True,False),('COHORT',True,True),('HARD_KD',False,False)]:
        start=time.perf_counter();scores,fit,head=fit_head(raw[known],source[known],answer[known],current,source.ravel(),mean,std,soft=soft,seed=0)
        trained=scores['CONVERGED'].reshape(prior.shape)
        if arm=='SOFT_KD':reference=trained.copy()
        if arm=='COHORT':assert np.array_equal(reference,trained),'Both arms must receive the identical independently fitted KD head'
        train_seconds=time.perf_counter()-start;start=time.perf_counter();score=trained.copy()
        if classic:
            for qi,text in enumerate(texts):
                selected=part_mask(text)>0
                if selected.any():
                    valid=np.all(colours[:,selected].sum(2)>0,axis=1)
                    score[qi]=refine_distilled_prior(trained[qi],geometry[qi],answer[qi],source_weight=.1,node_mask=valid)
        expanded=score[:,columns];mask=plain(expanded,exclusive)
        for qi,q in enumerate(queries):
            if 'MOTION' in expression_types(q['query_text']):mask[qi]&=np.asarray([expit(fc.raw_presence[qi,fc.frame_index[int(f)]])>=.5 for f,_ in observations])
        folder=root/arm
        seal_mask(folder,localmeta,mask.ravel(),target='SportsMOT',model='Dance-best',method='COLD_'+arm,GT_open_attempts=len(guard),
            extra=dict(real_student_forward=True,real_teacher_forward=True,teacher_cache_hits=0,logical_teacher_calls=budget,
                cold_shared_prefix=True,not_independently_repeated_cold_arms=True,full_original_clip=True))
        torch.save(head,folder/'trained_head.pt');postprocess=time.perf_counter()-start
        arms[arm]=dict(train_seconds=train_seconds,postprocess_including_output_seal_seconds=postprocess,total_shared_prefix_plus_own_branch_seconds=common_seconds+train_seconds+postprocess,fit=fit)
        print(json.dumps(dict(event='COLD_ARM_COMPLETE',arm=arm,seconds=arms[arm]['total_shared_prefix_plus_own_branch_seconds'])),flush=True)
    assert not guard and not forbidden
    atomic_json(root/'completion.json',dict(status='PASS_REAL_COLD_FULL_CLIP',protocol=protocol,phases=phases,common_wall_seconds=common_seconds,
        arms=arms,student=student_receipt,teacher=teacher_receipt,source=source_receipt,
        GT_open_attempts=0,prepaid_artifact_read_attempts=0,total_experiment_wall_seconds=time.perf_counter()-PROCESS_BEGIN,
        accounting='one physically measured common prefix + own independently fitted arm; no synthetic kernel extrapolation',
        all_development_data=True,no_whole_C4_cost_claim=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='C4_first_full_clip_seed0');a=p.parse_args();run(a.name)
