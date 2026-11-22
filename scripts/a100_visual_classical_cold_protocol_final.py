"""Fresh single-method target stream using the canonical converged head.

Ordinary: FINAL with the supplied nonempty/checklist contract, no unused RGB.
Proposed: FINAL with protocol pooling, then the frozen visual PAVA. Every
worker independently computes features, asks its frozen teacher and fits the
same weighted head. No historical feature/QA payload or target GT is consumed.
The wrapper times Popen-to-exit; source candidate generation is outside scope.
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

LABELS=('CLIP',)
BASE_METHODS=('CLIP_KD_NONEMPTY',)
METHODS=('CLIP_KD_NONEMPTY','CLIP_KD_PROTOCOL_CAL')


def real_colours(bank,observations,samples,extension='.jpg',*,compute_rgb=True):
    boxes=np.asarray([list(map(float,s.split(',')[2:6])) for s in bank['lines']]);frames={};colours=[];anchors=[]
    if not compute_rgb:
        anchors=[int(observations[rows[np.argmax(np.prod(boxes[rows,2:4],axis=1))],0]) for rows in samples]
        return np.zeros((len(samples),3,104),dtype=float),anchors
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


def run(name,perception_mode='full',requested_methods=None,teacher_fraction=.3,pair='C4',query_graph_scope='clothing',
        head_query_scope='clothing',task_protocol=None,set_decoder='original',graph_metric='combined',graph_prior='constant',head_source_logit='residual'):
    assert 0 < teacher_fraction <= 1
    assert perception_mode=='track-three' and teacher_fraction==.1
    assert tuple(requested_methods or ()) in (('CLIP_KD_NONEMPTY',),('CLIP_KD_PROTOCOL_CAL',))
    assert task_protocol in ('PRESUPPOSED','CHECKLIST')
    assert query_graph_scope=='extended' and graph_metric=='combined' and graph_prior=='constant' and set_decoder=='original'
    assert head_query_scope=='pcd' and head_source_logit=='zero'
    assert query_graph_scope in ('clothing','extended')
    assert head_query_scope in ('clothing','extended','zero','pcd')
    assert head_source_logit in ('residual','zero')
    assert graph_metric in ('combined','pcd','colour','hidden')
    assert graph_prior in ('constant','positive_agreement')
    if graph_metric!='combined':assert query_graph_scope=='extended'
    assert set_decoder in ('original','shared_singleton','retain_distilled_set')
    model_name,target_name=PAIRS[pair]
    methods=tuple(requested_methods) if requested_methods else BASE_METHODS
    assert set(methods)<=set(METHODS)
    calibrated=any(method.endswith('_CAL') for method in methods)
    if calibrated:
        assert graph_prior=='constant' and graph_metric=='combined' and set_decoder=='original'
        assert query_graph_scope=='extended' and head_query_scope in ('extended','pcd')
    if set_decoder!='original':
        assert task_protocol in ('CHECKLIST','PRESUPPOSED')
        assert set(methods)<={'CLIP_MEAN','CLIP_MEAN_COHORT'}
    if graph_prior!='constant' or graph_metric=='hidden':
        assert set(methods)<={'CLIP_MEAN','CLIP_MEAN_COHORT'}
    if graph_prior!='constant':assert task_protocol in ('CHECKLIST','PRESUPPOSED')
    dimension=768 if head_query_scope=='pcd' else 1080
    labels_needed={method.split('_',1)[0] for method in methods}
    guard=generation_guard();forbidden=[];root=OUT/'cold_full_c4'/name
    if (root/'completion.json').is_file():
        assert json.loads((root/'completion.json').read_text())['protocol'].get('source_perception_mode','full')==perception_mode
        assert json.loads((root/'completion.json').read_text())['protocol'].get('teacher_fraction',.3)==teacher_fraction
        assert json.loads((root/'completion.json').read_text())['protocol']['pair']==pair
        assert json.loads((root/'completion.json').read_text())['protocol'].get('query_graph_scope','clothing')==query_graph_scope
        previous=json.loads((root/'completion.json').read_text())['protocol']
        assert previous.get('head_query_scope','clothing')==head_query_scope
        assert previous.get('head_source_logit','residual')==head_source_logit
        assert previous.get('task_protocol')==task_protocol and previous.get('set_decoder','original')==set_decoder
        assert previous.get('graph_metric','combined')==graph_metric and previous.get('graph_prior','constant')==graph_prior
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
    from flex_uda.active.confidence_teacher import fit_head
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    from flex_uda.active.acquisition import expression_types,exclusive_group
    from flex_uda.presence.predictions import seal_mask
    from flex_uda.active.appearance_runtime import empty_appearance_history,regularize_appearance,commit_appearance_history
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
    state_path=root/'stream.pt';start_video=0;bx=[];bz=[];by=[];receipts=[];moment_sum=np.zeros(dimension);moment_square=np.zeros(dimension);image_rows=0
    appearance_history=empty_appearance_history()
    if state_path.is_file():
        saved=torch.load(state_path,map_location='cpu',weights_only=False);start_video=saved['next_video'];bx=saved['x'];bz=saved['z'];by=saved['y']
        assert saved.get('perception_mode','full')==perception_mode
        assert tuple(saved.get('requested_methods',BASE_METHODS))==methods
        assert saved.get('teacher_fraction',.3)==teacher_fraction
        assert saved.get('pair','C4')==pair
        assert saved.get('query_graph_scope','clothing')==query_graph_scope
        assert saved.get('head_query_scope','clothing')==head_query_scope
        assert saved.get('head_source_logit','residual')==head_source_logit
        assert saved.get('task_protocol')==task_protocol and saved.get('set_decoder','original')==set_decoder
        assert saved.get('graph_metric','combined')==graph_metric and saved.get('graph_prior','constant')==graph_prior
        receipts=saved['receipts'];moment_sum=saved['sum'];moment_square=saved['square'];image_rows=saved['image_rows']
        if calibrated:appearance_history=saved['appearance_history']
    protocol=dict(name=name,pair=pair,videos=len(videos),queries=len(m['episodes']),seed=0,source='frozen official '+model_name,
        acquisition=f'COVER {teacher_fraction:g} query-balanced, three uniform observations and max-area anchor',
        teacher_fraction=teacher_fraction,
        query_graph_scope=query_graph_scope,
        head_query_scope=head_query_scope,head_source_logit=head_source_logit,task_protocol=task_protocol,set_decoder=set_decoder,
        head_feature_dimension=dimension,graph_metric=graph_metric,graph_prior=graph_prior,
        acquisition_geometry='Frozen query-conditioned PCD only, unchanged by head colour or set-decoder controls',
        teacher='frozen original Qwen CROP interface',source_features_actually_recomputed=True,initial_cache_prepaid=False,
        source_perception_mode=perception_mode,all_source_frames_recomputed=perception_mode=='full',
        sparse_rule='union of three uniformly spaced observations per provided TrackID; all candidates at selected frames',
        original_four_observation_context_preserved=True,
        sparse_frame_count_means_decoder_positions_not_all_backbone_context_images=True,
        sparse_motion_gate='not applied: plain decoder matches quality study',
        shared_physical_source_and_teacher_stream=True,independent_cold_arms=False,
        raw_soft_bounded_soft_and_hard_actual_training=True,bounds=[.02,.98],
        video_stream_causal=True,current_video_transductive=True,no_target_GT=True,
        restart_attempt=attempt,OS_cache_not_flushed=True,host_has_other_research_jobs=True)
    protocol.update(requested_methods=list(methods),single_method_execution=len(methods)==1,
        raw_soft_bounded_soft_and_hard_actual_training=labels_needed==set(LABELS),actual_trained_label_types=sorted(labels_needed),
        shared_physical_source_and_teacher_stream=len(methods)>1,independent_cold_arms=len(methods)==1)
    protocol.update(appearance_regularization=calibrated,
        appearance_rule='fixed teacher-only TrackID-crossfit PAVA, ln3 bounded odds, after graph if requested' if calibrated else None)
    protocol.update(confidence_weighted_teacher_head=True,confidence_rule='1+3*abs(2*raw_teacher_p-1); normalized weighted CE/KL',
        quality_plain_decoder=True,source_motion_presence_gate_applied=False)
    protocol.update(shared_head_readout='CONVERGED_FINAL',task_contract_pooling=calibrated,
        task_contract_rule='PRESUPPOSED identity; CHECKLIST equal-edge global pool; then PAVA',
        RGB_statistics_computed=calibrated,unused_RGB_statistics_skipped=not calibrated,
        source_motion_presence_gate_applied=False,quality_plain_decoder=True,
        ordinary_same_task_nonempty_conditioning=not calibrated)
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
        t=time.perf_counter();colour,anchors=real_colours(bank,observations,sink.samples,'.jpg' if target_name=='SportsMOT' else '.png',compute_rgb=calibrated);texts=[q['query_text'] for q in queries]
        geometry,normalized_pcd,selected_colour=augmented_features(original,colour,texts);raw=np.concatenate([original,selected_colour],axis=2).astype('float32')
        if head_query_scope!='clothing':
            from flex_uda.active.context_cohort import distillation_features
            raw=distillation_features(original,colour,texts,head_query_scope)
        source=np.clip(logit(np.clip(prior,.05,.95)),-2.,2.);current=raw.reshape(-1,dimension)
        head_source=source if head_source_logit=='residual' else np.zeros_like(source)
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
        common('real_RGB_sparse_inputs_and_image_only_acquisition' if calibrated else 'anchor_and_PCD_only_acquisition_no_RGB_histograms',t,video)
        t=time.perf_counter();answers=np.full(prior.shape,np.nan);returned=[]
        for offset in range(0,len(requests),8):
            batch=requests[offset:offset+8];responses=client.ask_many([{k:v for k,v in r.items() if k not in ('local_query','local_track')} for r in batch])
            for request,value in zip(batch,responses):
                qi,ti=request['local_query'],request['local_track'];answers[qi,ti]=value['yes_probability']
                row=dict(value,local_query=qi,local_track=ti);returned.append(row);append_json(root/'responses.jsonl',row)
        common('real_teacher_service_including_image_IO',t,video)
        atomic_json(current_dir/'answers.json',dict(answers=returned,logical_calls=len(returned),GT_open_attempts=len(guard)))
        known=np.isfinite(answers);bx.append(raw[known]);bz.append(head_source[known]);by.append(answers[known])
        x=np.concatenate(bx);z=np.concatenate(bz);y=np.concatenate(by);prediction={};fits={}
        for label in LABELS:
            if label not in labels_needed:continue
            t=time.perf_counter();target=np.clip(y,.02,.98) if label=='CLIP' else y
            scores,fit,head=fit_head(x,z,target,current,head_source.ravel(),mean,std,sample_weight=1.+3.*np.abs(2.*y-1.),soft=label!='HARD',seed=0)
            fit['source_logit_residual_disabled']=head_source_logit=='zero'
            torch.save(head,root/(label+'_latest_head.pt'));fit['measured_fit_and_head_save_seconds']=time.perf_counter()-t
            ledger['head_fits'].append(dict(video=video,label=label,attempt=attempt,receipt=fit));atomic_json(ledger_path,ledger);fits[label]=fit
            for point,value in scores.items():prediction[label+'_KD'+('' if point=='CONVERGED' else '_'+point)]=value.reshape(source.shape)
            prediction[label+'_MEAN']=.5*(prediction[label+'_KD_400']+prediction[label+'_KD'])
            del head
        columns=np.asarray([tracks.index(int(t)) for _,t in observations])
        appearance_inputs=None
        for method in methods:
            t=time.perf_counter()
            score=prediction['CLIP_KD'].copy()
            if not calibrated:
                from flex_uda.active.protocol_calibration import nonempty_baseline
                score=nonempty_baseline(score,task_protocol)
            pooling_receipt=None
            if calibrated:
                pooling_begin=time.perf_counter()
                from flex_uda.active.protocol_calibration import global_pool,contract_graph
                if task_protocol=='CHECKLIST':
                    from flex_uda.active.topology_selection import graph_matrices
                    _,_,global_weights,valid,supported=graph_matrices(original,colour,texts)
                    pooled=global_pool(score,answers,global_weights,valid,supported)
                else:pooled=score
                score=contract_graph(score,pooled,task_protocol)
                pooling_receipt=dict(protocol=task_protocol,operator='GLOBAL' if task_protocol=='CHECKLIST' else 'IDENTITY',
                    actual_CPU_seconds=time.perf_counter()-pooling_begin,readout='CONVERGED_FINAL',target_GT_used=False)
            appearance_receipt=None
            if method.endswith('_CAL'):
                appearance_begin=time.perf_counter()
                score,appearance_inputs,appearance_receipt=regularize_appearance(score,colour,texts,lengths,answers,tracks,video,appearance_history)
                appearance_receipt['actual_CPU_seconds']=time.perf_counter()-appearance_begin
            expanded=score[:,columns];mask=plain(expanded,exclusive_group(texts,target_name))
            decoder_actions=None
            folder=root/'methods'/method;destination=folder/'videos'/video;destination.mkdir(parents=True,exist_ok=True)
            np.save(destination/'mask.npy',mask);np.save(destination/'scores.npy',expanded.astype('float32'));immediate_predictions(folder,output_context,mask)
            ledger['method_outputs'].append(dict(video=video,method=method,seconds=time.perf_counter()-t,attempt=attempt,
                task_contract_actions=decoder_actions,appearance_regularization=appearance_receipt,task_contract_pooling=pooling_receipt));atomic_json(ledger_path,ledger)
        if calibrated:commit_appearance_history(appearance_history,appearance_inputs,answers)
        receipt=dict(video=video,index=vi,visible_teacher_videos=videos[:vi+1],logical_teacher_calls=budget,teacher_examples=len(y),fits=fits,
            source_perception_mode=perception_mode,source_frames=len(scoring_frames),full_frames=len(full_frames),
            source_candidate_query_rows=len(fc.observations)*len(queries),full_candidate_query_rows=len(observations)*len(queries),
            actual_frozen_encoder_statistics=dict(cache.stats),original_temporal_context_preserved=True,
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-begin)
        atomic_json(current_dir/'receipt.json',receipt);receipts.append(receipt)
        state=dict(next_video=vi+1,x=bx,z=bz,y=by,receipts=receipts,sum=moment_sum,square=moment_square,image_rows=image_rows,perception_mode=perception_mode,requested_methods=methods,teacher_fraction=teacher_fraction,pair=pair,query_graph_scope=query_graph_scope,
            head_query_scope=head_query_scope,head_source_logit=head_source_logit,task_protocol=task_protocol,set_decoder=set_decoder)
        state.update(graph_metric=graph_metric,graph_prior=graph_prior)
        if calibrated:state['appearance_history']=appearance_history
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
                query_graph_scope=query_graph_scope,
                head_query_scope=head_query_scope,head_source_logit=head_source_logit,task_protocol=task_protocol,set_decoder=set_decoder,
                head_feature_dimension=dimension,graph_metric=graph_metric,graph_prior=graph_prior,
                appearance_regularization=method.endswith('_CAL'),
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
    p.add_argument('--query-graph-scope',choices=['clothing','extended'],default='clothing')
    p.add_argument('--head-query-scope',choices=['clothing','extended','zero','pcd'],default='clothing')
    p.add_argument('--task-protocol',choices=['CHECKLIST','PRESUPPOSED'])
    p.add_argument('--set-decoder',choices=['original','shared_singleton','retain_distilled_set'],default='original')
    p.add_argument('--graph-metric',choices=['combined','pcd','colour','hidden'],default='combined')
    p.add_argument('--graph-prior',choices=['constant','positive_agreement'],default='constant')
    p.add_argument('--head-source-logit',choices=['residual','zero'],default='residual')
    a=p.parse_args();run(a.name,a.perception_mode,a.methods.split(',') if a.methods else None,a.teacher_fraction,a.pair,a.query_graph_scope,
        a.head_query_scope,a.task_protocol,a.set_decoder,a.graph_metric,a.graph_prior,a.head_source_logit)
