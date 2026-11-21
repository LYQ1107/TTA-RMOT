"""Fresh whole-C4 source/teacher stream and physically timed same-evidence arms.

One common source/teacher stream is measured, then actual head fits and each
decoder are charged separately. No historical feature/QA payload is consumed.
Multiple arms share work for experimental economy; they are not independent
repeated cold deployments. Ground truth is prohibited in this worker.
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
from flex_uda.a100.mechanism import metadata,generation_guard,PAIRS
from flex_uda.active.color_cohort import augmented_features,part_histograms,part_mask
from flex_uda.active.visual_classical import visual_graph,acquisition_order,refine_distilled_prior
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_cold_slice import SparsePCD

LABELS=('RAW','CLIP','HARD')
METHODS=tuple(label+'_KD'+suffix for label in LABELS for suffix in ('_400','_4000',''))+tuple(
    label+'_'+name for label in ('RAW','CLIP') for name in ('COHORT','NO_GRAPH','STRONG_PRIOR'))+tuple(
    label+'_MEAN' for label in LABELS)+tuple(label+'_MEAN_'+variant for label in ('RAW','CLIP') for variant in ('COHORT','NO_GRAPH'))


def real_colours(bank,observations,samples,extension='.jpg'):
    boxes=np.asarray([list(map(float,s.split(',')[2:6])) for s in bank['lines']]);frames={};colours=[];anchors=[]
    for rows in samples:
        hist=[]
        for row in rows:
            t=int(observations[row,0])
            if t not in frames:
                with Image.open(Path(bank['image_root'])/f'{t:06d}{extension}') as im:frames[t]=im.convert('RGB')
            hist.append(part_histograms(frames[t],boxes[row]))
        values=np.asarray(hist);valid=values.sum(2)>0
        colours.append(np.sqrt(values.sum(0)/np.maximum(valid.sum(0)[:,None],1)))
        anchors.append(int(observations[rows[np.argmax(np.prod(boxes[rows,2:4],axis=1))],0]))
    return np.asarray(colours),anchors


def run(name,perception_mode='full',requested_methods=None,teacher_fraction=.3,pair='C4'):
    assert 0 < teacher_fraction <= 1
    model_name,target_name=PAIRS[pair]
    methods=tuple(requested_methods) if requested_methods else METHODS
    assert set(methods)<=set(METHODS)
    labels_needed={method.split('_',1)[0] for method in methods}
    guard=generation_guard();forbidden=[];root=OUT/'cold_full_c4'/name
    if (root/'completion.json').is_file():
        assert json.loads((root/'completion.json').read_text())['protocol'].get('source_perception_mode','full')==perception_mode
        assert json.loads((root/'completion.json').read_text())['protocol'].get('teacher_fraction',.3)==teacher_fraction
        assert json.loads((root/'completion.json').read_text())['protocol']['pair']==pair
        if requested_methods:assert {m['method'] for m in json.loads((root/'completion.json').read_text())['methods']}==set(methods)
        return
    root.mkdir(parents=True,exist_ok=True)
    def cold_only(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR):return
        try:parts=path.relative_to(OUT).parts
        except ValueError:parts=()
        if parts and parts[0]!='cold_full_c4':
            forbidden.append(str(path));raise PermissionError('Fresh stream cannot consume precomputed research artifacts: '+str(path))
        if path.suffix in ('.npy','.npz') and 'features' in path.parts and 'cold_full_c4' not in path.parts:
            forbidden.append(str(path));raise PermissionError('No historical neural features in fresh stream')
    sys.addaudithook(cold_only)
    import torch
    from flex_uda.a100.runtime import load_student,source_head,teacher
    from flex_uda.presence.runtime import episode_from_bank
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.active.v6_frames import FrameCacheV6
    from flex_uda.active.strong_prefix_teacher import fit_head
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    from flex_uda.active.acquisition import expression_types,exclusive_group
    from flex_uda.presence.predictions import seal_mask
    from scripts.active_run import video_spec
    torch.set_num_threads(4);torch.manual_seed(0);device=torch.device('cuda:0');torch.cuda.set_device(device)
    m=metadata(target_name);videos=sorted(m['videos']);ledger_path=root/'phase_ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.is_file() else dict(common_phases=[],head_fits=[],method_outputs=[],attempts=[])
    attempt=len(ledger['attempts']);ledger['attempts'].append(dict(attempt=attempt,started_unix=time.time(),pid=os.getpid(),
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    atomic_json(ledger_path,ledger)
    def common(phase,start,video=None):
        row=dict(phase=phase,seconds=time.perf_counter()-start,video=video,attempt=attempt)
        ledger['common_phases'].append(row);atomic_json(ledger_path,ledger)
        print(json.dumps(dict(event=f'FRESH_{pair}_COMMON_PHASE',**row)),flush=True)
    state_path=root/'stream.pt';start_video=0;bx=[];bz=[];by=[];receipts=[];moment_sum=np.zeros(1080);moment_square=np.zeros(1080);image_rows=0
    if state_path.is_file():
        saved=torch.load(state_path,map_location='cpu',weights_only=False);start_video=saved['next_video'];bx=saved['x'];bz=saved['z'];by=saved['y']
        assert saved.get('perception_mode','full')==perception_mode
        assert tuple(saved.get('requested_methods',METHODS))==methods
        assert saved.get('teacher_fraction',.3)==teacher_fraction
        assert saved.get('pair','C4')==pair
        receipts=saved['receipts'];moment_sum=saved['sum'];moment_square=saved['square'];image_rows=saved['image_rows']
    protocol=dict(name=name,pair=pair,videos=len(videos),queries=len(m['episodes']),seed=0,source='frozen official '+model_name,
        acquisition=f'COVER {teacher_fraction:g} query-balanced, three uniform observations and max-area anchor',
        teacher_fraction=teacher_fraction,
        teacher='frozen original Qwen CROP interface',source_features_actually_recomputed=True,initial_cache_prepaid=False,
        source_perception_mode=perception_mode,all_source_frames_recomputed=perception_mode=='full',
        sparse_rule='union of three uniformly spaced observations per provided TrackID; all candidates at selected frames',
        original_four_observation_context_preserved=True,
        sparse_frame_count_means_decoder_positions_not_all_backbone_context_images=True,
        sparse_motion_gate='linearly interpolate sampled source presence logits; nearest boundaries',
        shared_physical_source_and_teacher_stream=True,independent_cold_arms=False,
        raw_soft_bounded_soft_and_hard_actual_training=True,bounds=[.02,.98],
        video_stream_causal=True,current_video_transductive=True,no_target_GT=True,
        restart_attempt=attempt,OS_cache_not_flushed=True,host_has_other_research_jobs=True)
    protocol.update(requested_methods=list(methods),single_method_execution=len(methods)==1,
        raw_soft_bounded_soft_and_hard_actual_training=labels_needed==set(LABELS),actual_trained_label_types=sorted(labels_needed),
        shared_physical_source_and_teacher_stream=len(methods)>1,independent_cold_arms=len(methods)==1)
    atomic_json(root/'protocol.json',protocol)
    if attempt==0:assert not (root/'raw.sqlite').is_file(),'A fresh cold stream requires a new QA cache'
    common('imports_metadata_and_recovery_setup',PROCESS_BEGIN)
    t=time.perf_counter();student,tok,transform,student_receipt=load_student(root/'student_log',device,model_name=model_name,target=target_name);common('student_load_and_integrity_check',t)
    versions={n:p._version for n,p in student.named_parameters()}
    t=time.perf_counter();client=teacher([video_spec(target_name,v,b) for v,b in m['videos'].items()],root/'raw.sqlite');common('teacher_load_and_initial_integrity_check',t)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for vi,video in enumerate(videos):
        if vi<start_video:continue
        begin=time.perf_counter();bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video]
        observations=np.asarray(bank['observations']);sink=SparsePCD(observations,len(queries));tracks=sink.tracks;current_dir=root/'source'/video
        full_frames=np.unique(observations[:,0]);scoring_bank=bank;scoring_frames=full_frames;receiver=sink
        if perception_mode=='track-three':
            scoring_frames=np.unique(observations[np.concatenate(sink.samples),0])
            selected_rows=np.flatnonzero(np.isin(observations[:,0],scoring_frames))
            scoring_bank=dict(bank,observations=observations[selected_rows].tolist(),lines=[bank['lines'][i] for i in selected_rows])
            def receiver(qa,ids,feature):sink(qa,selected_rows[ids],feature)
        # Keep the original episode's four-observation history. Filtering its
        # history would alter PCD features rather than merely avoiding unused
        # decoder positions. Context images are genuinely forwarded and timed.
        t=time.perf_counter();ep=episode_from_bank(video,bank,target_name,transform);cache=FrozenFeatureCache(student,ep,tok,device,max_bytes=512<<20)
        fc=FrameCacheV6(student,cache,scoring_bank,queries,scoring_frames,source_head(model_name),current_dir,store_pcd=False,pcd_sink=receiver)
        fc.score(model_step=0);original=sink.mean();prior,_,_=fc.decode(1.,0.);fc.close()
        assert fc.tracks==list(map(int,tracks))
        common('real_all_frame_student_forward_and_initial_head' if perception_mode=='full' else 'real_sparse_frame_student_forward_and_initial_head',t,video)
        output_context=SimpleNamespace(bank=bank,queries=queries)
        output_frame_index={int(f):i for i,f in enumerate(full_frames)}
        output_presence=np.stack([np.interp(full_frames,fc.frames,fc.raw_presence[qi]) for qi in range(len(queries))])
        if perception_mode=='track-three':
            # This contains only interpolated sampled margins, for identical
            # counterfactual tie breaking. No unobserved neural margin is read.
            output_margin=np.zeros((len(queries),len(observations)),dtype='float32')
            for track in tracks:
                ids=np.flatnonzero(observations[:,1]==track);seen=fc.by_track[int(track)]
                for qi in range(len(queries)):
                    output_margin[qi,ids]=np.interp(observations[ids,0],fc.observations[seen,0],fc.margin[qi,seen])
            np.save(current_dir/'output_margin.npy',output_margin)
        t=time.perf_counter();colour,anchors=real_colours(bank,observations,sink.samples,'.jpg' if target_name=='SportsMOT' else '.png');texts=[q['query_text'] for q in queries]
        geometry,_,selected_colour=augmented_features(original,colour,texts);raw=np.concatenate([original,selected_colour],axis=2).astype('float32')
        source=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);current=raw.reshape(-1,1080)
        moment_sum+=current.sum(0,dtype=np.float64);moment_square+=np.square(current.astype(float)).sum(0);image_rows+=len(current)
        mean=moment_sum/image_rows;std=np.maximum(np.sqrt(np.maximum(moment_square/image_rows-mean**2,0)),1e-3)
        np.savez_compressed(current_dir/'sparse_inputs.npz',pcd=original,colours=colour,prior=prior,tracks=tracks,anchors=anchors)
        lengths=np.asarray([(observations[:,1]==track).sum() for track in tracks]);eligible=np.flatnonzero(lengths>=3)
        budget=int(np.floor(teacher_fraction*len(queries)*len(eligible)));quota=np.full(len(queries),budget//len(queries));quota[:budget%len(queries)]+=1
        requests=[]
        for qi,q in enumerate(queries):
            _,distance=visual_graph(original[qi]);seed=int(__import__('hashlib').sha256((video+str(qi)).encode()).hexdigest()[:8],16)
            order=acquisition_order(prior[qi],distance,eligible,'COVER',np.random.default_rng(seed))
            for ti in order[:quota[qi]]:requests.append(dict(target=target_name,video_id=video,query_id=q['expression_id'],query_text=q['query_text'],
                track_id=int(tracks[ti]),frame_id=anchors[ti],view='CROP',local_query=qi,local_track=ti))
        atomic_json(current_dir/'questions_before_answers.json',dict(requests=requests,GT_open_attempts=len(guard)))
        common('real_RGB_sparse_inputs_and_image_only_acquisition',t,video)
        t=time.perf_counter();answers=np.full(prior.shape,np.nan);returned=[]
        for offset in range(0,len(requests),8):
            batch=requests[offset:offset+8];responses=client.ask_many([{k:v for k,v in r.items() if k not in ('local_query','local_track')} for r in batch])
            for request,value in zip(batch,responses):
                qi,ti=request['local_query'],request['local_track'];answers[qi,ti]=value['yes_probability']
                row=dict(value,local_query=qi,local_track=ti);returned.append(row);append_json(root/'responses.jsonl',row)
        common('real_teacher_service_including_image_IO',t,video)
        atomic_json(current_dir/'answers.json',dict(answers=returned,logical_calls=len(returned),GT_open_attempts=len(guard)))
        known=np.isfinite(answers);bx.append(raw[known]);bz.append(source[known]);by.append(answers[known])
        x=np.concatenate(bx);z=np.concatenate(bz);y=np.concatenate(by);prediction={};fits={}
        for label in LABELS:
            if label not in labels_needed:continue
            t=time.perf_counter();target=np.clip(y,.02,.98) if label=='CLIP' else y
            scores,fit,head=fit_head(x,z,target,current,source.ravel(),mean,std,soft=label!='HARD',seed=0)
            torch.save(head,root/(label+'_latest_head.pt'));fit['measured_fit_and_head_save_seconds']=time.perf_counter()-t
            ledger['head_fits'].append(dict(video=video,label=label,attempt=attempt,receipt=fit));atomic_json(ledger_path,ledger);fits[label]=fit
            for point,value in scores.items():prediction[label+'_KD'+('' if point=='CONVERGED' else '_'+point)]=value.reshape(source.shape)
            prediction[label+'_MEAN']=.5*(prediction[label+'_KD_400']+prediction[label+'_KD'])
            del head
        columns=np.asarray([tracks.index(int(t)) for _,t in observations]);motion={}
        for qi,q in enumerate(queries):
            if 'MOTION' in expression_types(q['query_text']):motion[qi]=np.asarray([expit(output_presence[qi,output_frame_index[int(f)]])>=.5 for f,_ in observations])
        for method in methods:
            t=time.perf_counter()
            if method in prediction:score=prediction[method]
            else:
                label,variant=method.split('_',1);score=prediction[label+('_MEAN' if variant.startswith('MEAN_') else '_KD')].copy()
                for qi,text in enumerate(texts):
                    selected=part_mask(text)>0
                    if selected.any():score[qi]=refine_distilled_prior(score[qi],geometry[qi],answers[qi],
                        graph=not variant.endswith('NO_GRAPH'),source_weight=1. if variant=='STRONG_PRIOR' else .1,
                        node_mask=np.all(colour[:,selected].sum(2)>0,axis=1))
            expanded=score[:,columns];mask=plain(expanded,exclusive_group(texts,target_name))
            for qi,gate in motion.items():mask[qi]&=gate
            folder=root/'methods'/method;destination=folder/'videos'/video;destination.mkdir(parents=True,exist_ok=True)
            np.save(destination/'mask.npy',mask);np.save(destination/'scores.npy',expanded.astype('float32'));immediate_predictions(folder,output_context,mask)
            ledger['method_outputs'].append(dict(video=video,method=method,seconds=time.perf_counter()-t,attempt=attempt));atomic_json(ledger_path,ledger)
        receipt=dict(video=video,index=vi,visible_teacher_videos=videos[:vi+1],logical_teacher_calls=budget,teacher_examples=len(y),fits=fits,
            source_perception_mode=perception_mode,source_frames=len(scoring_frames),full_frames=len(full_frames),
            source_candidate_query_rows=len(fc.observations)*len(queries),full_candidate_query_rows=len(observations)*len(queries),
            actual_frozen_encoder_statistics=dict(cache.stats),original_temporal_context_preserved=True,
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-begin)
        atomic_json(current_dir/'receipt.json',receipt);receipts.append(receipt)
        state=dict(next_video=vi+1,x=bx,z=bz,y=by,receipts=receipts,sum=moment_sum,square=moment_square,image_rows=image_rows,perception_mode=perception_mode,requested_methods=methods,teacher_fraction=teacher_fraction,pair=pair)
        temporary=state_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(state_path)
        del fc,cache,ep,sink;torch.cuda.empty_cache()
        print(json.dumps(dict(event=f'FRESH_{pair}_VIDEO_COMPLETE',video=video,completed=vi+1,total=len(videos),
            logical_teacher_calls=budget,teacher_examples=len(y),video_wall_seconds=receipt['wall_seconds'])),flush=True)
    t=time.perf_counter();teacher_receipt=client.close();common('teacher_final_frozen_weight_audit',t)
    assert not guard and not forbidden
    assert all(not p.requires_grad and p._version==versions[n] for n,p in student.named_parameters())
    seals=[]
    for method in methods:
        t=time.perf_counter();folder=root/'methods'/method;mask=np.zeros(m['candidate_query_rows'],dtype=bool)
        for video in videos:
            local=np.load(folder/'videos'/video/'mask.npy')
            for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
        seal=seal_mask(folder,m,mask,target=target_name,model=model_name,method='FRESH_'+pair+'_'+method,GT_open_attempts=0,
            extra=dict(whole_C4_fresh_stream=pair=='C4',whole_target_fresh_stream=True,pair=pair,source_features_not_prepaid=True,teacher_answers_not_prepaid=True,
                logical_teacher_calls=sum(map(len,by)),same_teacher_and_source_stream_for_all_arms=True,
                current_video_transductive=True,no_future_teacher_feedback=True,teacher_target_clip=.02 if method.startswith('CLIP') else 0.))
        seals.append(dict(method=method,rows=seal['prediction_rows'],seal_seconds=time.perf_counter()-t))
    observed=time.perf_counter()-PROCESS_BEGIN
    ledger['attempts'][-1].update(finished_unix=time.time(),observed_worker_wall_seconds=observed)
    atomic_json(ledger_path,ledger)
    all_fit=sum(r['receipt']['measured_fit_and_head_save_seconds'] for r in ledger['head_fits'])
    all_outputs=sum(r['seconds'] for r in ledger['method_outputs'])+sum(r['seal_seconds'] for r in seals)
    # This residual includes real imports, metadata IO, progress/checkpoint
    # bookkeeping and any retries. It is not a guessed GPU-kernel integral.
    all_attempt_wall=sum(r.get('observed_worker_wall_seconds',0.) for r in ledger['attempts'])
    common_wall=all_attempt_wall-all_fit-all_outputs
    cost={}
    for item in seals:
        method=item['method'];label=method.split('_',1)[0]
        point='400' if method.endswith('_400') else '4000' if method.endswith('_4000') else 'CONVERGED'
        training=sum((r['receipt']['checkpoint_wall_seconds'][point] if point!='CONVERGED' else r['receipt']['measured_fit_and_head_save_seconds'])
            for r in ledger['head_fits'] if r['label']==label)
        outputs=sum(r['seconds'] for r in ledger['method_outputs'] if r['method']==method)+item['seal_seconds']
        cost[method]=dict(common_wall_seconds=common_wall,own_head_training_seconds=training,own_postprocess_and_output_seconds=outputs,
            shared_common_plus_own_branch_seconds=common_wall+training+outputs)
    if attempt==0:assert teacher_receipt['new_calls']==sum(map(len,by)) and teacher_receipt['cache_hits']==0
    atomic_json(root/'completion.json',dict(status=f'SEALED_WHOLE_{pair}_REAL_COLD_STREAM',protocol=protocol,methods=seals,
        source=student_receipt,teacher=teacher_receipt,logical_teacher_calls=sum(map(len,by)),
        GT_open_attempts=0,prepaid_artifact_reads=0,cost_by_method=cost,phase_ledger=str(ledger_path),
        actual_multiarm_worker_wall_seconds=all_attempt_wall,actual_worker_wall_seconds=all_attempt_wall,
        independent_cold_arm_deployments=len(methods)==1,
        single_method_execution=len(methods)==1,
        single_uninterrupted_attempt=attempt==0,all_development_data=True,
        scope='same physically measured source/teacher stream + actually trained per-label heads and decoder costs; GT evaluation excluded'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='C4_fresh_full_stream_seed0')
    p.add_argument('--perception-mode',choices=['full','track-three'],default='full')
    p.add_argument('--methods',help='Comma-separated actually executed methods; omit for all registered controls')
    p.add_argument('--teacher-fraction',type=float,default=.3)
    p.add_argument('--pair',choices=['C4','C1'],default='C4')
    a=p.parse_args();run(a.name,a.perception_mode,a.methods.split(',') if a.methods else None,a.teacher_fraction,a.pair)
