"""Matched single-GPU offline UDA with explicit transductive-pilot permission."""
from collections import Counter,OrderedDict
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.data.manifests import ROOT,SOURCE,DANCE,read_jsonl,write_jsonl
from flex_uda.data.source_kitti_v2 import SourceSupervision
from flex_uda.data.target_dance import TargetDanceDataset
from flex_uda.model.flexhook_uda import attach_adapter,prepared_inputs,forward_uda,frozen_copy,ema_update
from flex_uda.losses.objectives import supervised_ce,presence_loss,select_pseudo,pseudo_bce,consistency,equivalent_query,exclusive_query_partner,exclusive_relation_loss
from flex_uda.adaptation.tracklet_memory import TrackletMemory

CHECKPOINT=Path('/data3/user/TTA-RMOT-assets/flexhook/SOTA_ckpts/refer-kitti-v2-best.pth')
SOURCE_SHA='71dbd691e7633a3d20facb1ebc405908863aff786e5af30899d4cb666250d613'
METHODS=('SOURCE_REHEARSAL','MEAN_TEACHER_UDA','WEAK_MIL_UDA','COMBINED_UDA')


def protect_target_annotations():
    attempts=[]
    def audit(event,values):
        if event!='open' or not isinstance(values[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(values[0])).absolute()
        if path.is_relative_to(DANCE) and (path.name in ('labels.json','gt.txt') or 'gt_template' in path.parts or 'labels_with_ids' in path.parts or 'expression' in path.parts):
            attempts.append(str(path));raise PermissionError('target training/runtime annotation read forbidden: '+str(path))
    sys.addaudithook(audit);return attempts


class Inputs:
    def __init__(self,model,tokenizer,device,source):
        self.model,self.tokenizer,self.device,self.source=model,tokenizer,device,source;self.caches=OrderedDict()

    def episode(self,kind,row):
        key=(kind,row['video_id'])
        if key not in self.caches:
            episode=(self.source.episode(row['video_id']) if kind=='source' else DanceEpisode(DANCE,Path(row['predicted_tracks']).parent,row['video_id']))
            cache=FrozenFeatureCache(self.model,episode,self.tokenizer,self.device,max_bytes=256<<20)
            self.caches[key]=cache
            while len(self.caches)>4:self.caches.popitem(last=False)
        self.caches.move_to_end(key);return self.caches[key]

    def batches(self,kind,row,sentence=None,batch_size=24):
        cache=self.episode(kind,row);frame=row['frame'];candidates=list(cache.episode.current_candidates(frame))
        for start in range(0,len(candidates),batch_size):
            selected=candidates[start:start+batch_size]
            prepared=prepared_inputs(cache,selected,sentence or row['query_text'],frame)
            labels=(torch.tensor([int(i in row['positive_track_ids']) for i,_ in selected],device=self.device)
                if kind=='source' else None)
            yield prepared,labels,selected,cache


def build_plans(source,targets,output):
    source_plan,coverage=source.plan();target_plan=[];banks={}
    for row in targets.rows:
        if row['video_id'] not in banks:
            from flex_tta.dance_io import load_tracker_lines
            banks[row['video_id']]=sorted(load_tracker_lines(Path(row['predicted_tracks']).parent,row['video_id']))
        frames=banks[row['video_id']]
        # Pre-fixed, GT-independent positions. Every actual query and video is retained.
        for q in (.1,.5,.9):target_plan.append(dict(row,frame=frames[round(q*(len(frames)-1))]))
    output=Path(output);write_jsonl(output/'source_train_plan.jsonl',source_plan);write_jsonl(output/'target_plan.jsonl',target_plan)
    atomic_json(output/'coverage_plan.json',dict(source_windows=len(source_plan),source_videos=len(source.rows),
        source_coverage=coverage,target_windows=len(target_plan),target_videos=len({r['video_id'] for r in target_plan}),
        target_expressions=len(targets),positions=[.1,.5,.9],target_GT_used=False))
    return source_plan,target_plan,coverage


def source_validate(model,tokenizer,device,source,plan):
    inputs=Inputs(model,tokenizer,device,source);losses=[]
    with torch.no_grad():
        for row in plan:
            for prepared,labels,_,_ in inputs.batches('source',row):
                value=forward_uda(model,prepared,return_features=False)
                losses.append(float(supervised_ce(value['all_logits'],labels)))
    return float(np.mean(losses))


def save_training(path,model,ema,optimizer,scaler,step,epoch,config,plan_sha,best_source,rng=None,tracking=None):
    state=dict(student=model.state_dict(),ema=ema.state_dict() if ema is not None else None,
        optimizer=optimizer.state_dict(),scheduler=dict(kind='constant',lr=config['optimizer']['lr']),scaler=scaler.state_dict(),
        step=step,epoch=epoch,config=config,source_checkpoint_sha256=SOURCE_SHA,plan_sha256=plan_sha,best_source_loss=best_source,
        rng=rng or dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),
        test_time_update=False,tracking=tracking)
    temporary=Path(path).with_suffix('.partial.pt');torch.save(state,temporary);os.replace(temporary,path)


@torch.no_grad()
def frozen_equivalence(model,inputs,row):
    cache=inputs.episode('target',row);frame=cache.episode.frames[0];test=dict(row,frame=frame)
    source=ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE/results'/row['video_id']/row['expression_id']/'candidate_margins.jsonl.gz'
    reference=None
    with gzip.open(source,'rt') as stream:
        for line in stream:
            record=json.loads(line)
            if record['frame']==frame:reference=record;break
    if reference is None:raise ValueError('sealed source frame missing')
    values=[];ids=[]
    for prepared,_,_,_ in inputs.batches('target',test):
        value=forward_uda(model,prepared,return_features=True)
        if value['pcd_features'].shape!=(len(prepared['track_ids']),1,768):raise ValueError('PCD feature dimensions differ')
        values.extend(value['margin'].tolist());ids.extend(value['track_ids'])
    if ids!=reference['ids']:raise ValueError('initial source candidate alignment differs')
    a=np.asarray(values);b=np.asarray(reference['adapted']);error=float(np.max(np.abs(a-b)));flips=int(np.sum((a>0)!=(b>0)))
    if error>1e-3 or flips:raise RuntimeError('zero-LoRA frozen equivalence failed')
    return dict(video_id=row['video_id'],frame=frame,candidates=len(ids),max_abs_margin_error=error,selection_mismatches=flips,precision='FP32')


def train(config,output,*,smoke=False,resume=False):
    if config['protocol']['target_split']=='val' and os.environ.get('UDA_VAL_DIAGNOSTIC_AUTHORIZED')!='YES':
        raise PermissionError('Val adaptation requires explicit transductive diagnostic authorization')
    if config['protocol']['name']!='TRANSDUCTIVE_VAL_DIAGNOSTIC':raise ValueError('this pilot cannot be relabelled as Train-to-Val UDA')
    output=Path(output).resolve();method=config['method'];weights=config['loss']
    if method not in METHODS:raise ValueError('unknown pre-registered method')
    if output.exists() and not resume:raise FileExistsError('keep existing experiment outputs: '+str(output))
    output.mkdir(parents=True,exist_ok=resume)
    attempts=protect_target_annotations();seed=config['protocol']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction((24576<<20)/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if file_sha(CHECKPOINT)!=SOURCE_SHA:raise ValueError('wrong source model')
    started=time.perf_counter()
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    _,model,_=_load_model(ROOT/'vendor/FlexHook',CHECKPOINT,DANCE,Path(read_jsonl(ROOT/config['data']['target_manifest'])[0]['predicted_tracks']).parent,
        output/'model_log',device,24,strict_checkpoint=True)
    source_teacher=frozen_copy(model)
    adapter=attach_adapter(model,device,rank=4);model.eval()
    uses_teacher=method in ('MEAN_TEACHER_UDA','COMBINED_UDA');uses_weak=method in ('WEAK_MIL_UDA','COMBINED_UDA')
    ema=frozen_copy(model) if uses_teacher else None
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'vendor/FlexHook/pretrained/roberta-base'))
    if tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('tokenizer regression')
    source=SourceSupervision(ROOT/config['data']['source_train_manifest'])
    targets=TargetDanceDataset(ROOT/config['data']['target_manifest'],split='val')
    source_plan,target_plan,source_coverage=build_plans(source,targets,output/'plans')
    source_val=SourceSupervision(ROOT/config['data']['source_val_manifest']);val_plan,_=source_val.plan(validation_queries_per_video=16)
    steps_per_epoch=max(len(source_plan),len(target_plan));total_steps=20 if smoke else config['train']['coverage_cycles']*steps_per_epoch
    plan_sha={name:file_sha(output/'plans'/name) for name in ('source_train_plan.jsonl','target_plan.jsonl')}
    trainable=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable)!=24576:raise ValueError('unmatched trainable parameter capacity')
    frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial=[p.detach().clone() for p in trainable]
    optimizer=torch.optim.AdamW(trainable,lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['train']['amp'])
    start_step=0;best_source=float('inf');counters=Counter();memory=TrackletMemory()
    src_seen=set();tgt_seen=set();grad_nonzero=0;target_grad_checks=[];prior_wall=0.
    if resume:
        resume_path=output/'checkpoint_latest.pt'
        if not resume_path.exists():
            resume_path=output/'checkpoint_best_source_selected.pt'
        state=torch.load(resume_path,map_location=device,weights_only=False)
        if resume_path.name!='checkpoint_latest.pt' and state['step']!=0:raise ValueError('only the saved untrained initial state may substitute for missing latest checkpoint')
        if state['config']!=config or state['plan_sha256']!=plan_sha:raise ValueError('resume protocol/data mismatch')
        model.load_state_dict(state['student'],strict=True)
        if ema is not None:ema.load_state_dict(state['ema'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);start_step=state['step'];best_source=state['best_source_loss']
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'].cpu());torch.cuda.set_rng_state(state['rng']['cuda'].cpu())
        frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        tracking=state.get('tracking') or {}
        counters.update(tracking.get('counters',{}));src_seen={tuple(r) for r in tracking.get('source_seen',[])}
        tgt_seen={tuple(r) for r in tracking.get('target_seen',[])};grad_nonzero=tracking.get('grad_nonzero',0)
        target_grad_checks=tracking.get('target_grad_checks',[]);memory.rows=tracking.get('memory',{})
        prior_wall=tracking.get('wall_seconds',0.)
    atomic_json(output/'config.json',config)
    if resume and (output/'provenance.json').exists():
        folder=output/'provenance_attempts';folder.mkdir(exist_ok=True)
        (output/'provenance.json').rename(folder/(str(time.time_ns())+'.json'))
    atomic_json(output/'provenance.json',dict(protocol=config['protocol'],source_checkpoint_sha256=SOURCE_SHA,
        experiment_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=os.environ.get('UDA_PHYSICAL_GPU'),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),plan_sha256=plan_sha,
        total_steps=total_steps,steps_per_epoch=steps_per_epoch,target_manifest=str(ROOT/config['data']['target_manifest']),
        target_GT_used=False,external_semantic_teacher=False,trainable_parameters=24576))
    if not resume:
        best_source=source_validate(model,tokenizer,device,source_val,val_plan)
        save_training(output/'checkpoint_best_source_selected.pt',model,ema,optimizer,scaler,0,0,config,plan_sha,best_source)
        append_json(output/'source_validation.jsonl',dict(step=0,source_val_CE=best_source,selected=True))
    inputs=Inputs(model,tokenizer,device,source);all_texts=sorted({r['query_text'] for r in targets.rows})
    if not resume:atomic_json(output/'frozen_equivalence.json',frozen_equivalence(model,inputs,targets.rows[0]))
    evidence_path=output/'pseudo_evidence.jsonl'
    for step in range(start_step,total_steps):
        src=source_plan[step%len(source_plan)];tgt=target_plan[step%len(target_plan)]
        optimizer.zero_grad(set_to_none=True);parts=Counter();source_rows=0;target_rows=0
        source_batches=list(inputs.batches('source',src));source_n=sum(len(x[2]) for x in source_batches)
        for prepared,labels,candidates,cache in source_batches:
            with torch.autocast('cuda',dtype=torch.float16,enabled=config['train']['amp']):
                value=forward_uda(model,prepared,return_features=False)
                loss=weights['source_ce']*supervised_ce(value['all_logits'],labels,value['regular'])
                if uses_weak:
                    state='SUPPORTED_PRESENT' if labels.any() else 'SUPPORTED_ABSENT' if src['confirmed_video_absent'] else 'UNKNOWN'
                    pl=presence_loss(value['margin'],state);loss=loss+weights['source_presence']*pl
                    counters['source_presence_'+state]+=1
                weighted=loss*(len(candidates)/source_n)
            scaler.scale(weighted).backward();parts['source_loss']+=float(weighted.detach());source_rows+=len(candidates)
        src_seen.add((src['video_id'],src['expression_id']))
        if method!='SOURCE_REHEARSAL':
            batches=list(inputs.batches('target',tgt));target_n=sum(len(x[2]) for x in batches)
            for prepared,_,candidates,cache in batches:
                with torch.no_grad():
                    anchor=forward_uda(source_teacher,prepared,return_features=False)['margin']
                    teacher=forward_uda(ema,prepared,return_features=False)['margin'] if ema is not None else anchor
                with torch.autocast('cuda',dtype=torch.float16,enabled=config['train']['amp']):
                    value=forward_uda(model,prepared,strong_view=uses_teacher,return_features=False)
                    margins=value['margin'];target_loss=margins.sum()*0
                    if uses_teacher:
                        state,stats=select_pseudo(teacher,anchor,margins)
                        soft=consistency(margins,teacher);hard=pseudo_bce(margins,state)
                        target_loss=target_loss+weights['consistency']*soft+weights['pseudo_bce']*hard
                        for k,v in stats.items():
                            if isinstance(v,(float,int)) and not isinstance(v,bool):counters[k]+=v
                        counters['effective_consistency_candidates']+=len(candidates)
                    else:state=torch.full_like(margins,-1,dtype=torch.long)
                    if uses_weak:
                        alt=equivalent_query(tgt['query_text']);relation=margins.sum()*0
                        if alt:
                            alternate=prepared_inputs(cache,candidates,alt,tgt['frame'])
                            second=forward_uda(model,alternate,return_features=False)['margin']
                            relation=relation+F.smooth_l1_loss(margins.float(),second.float())
                            counters['equivalent_query_candidates']+=len(candidates)
                        other=exclusive_query_partner(tgt['query_text'],all_texts)
                        if other:
                            alternate=prepared_inputs(cache,candidates,other,tgt['frame'])
                            second=forward_uda(model,alternate,return_features=False)['margin']
                            relation=relation+exclusive_relation_loss(margins,second)
                            counters['exclusive_query_candidates']+=len(candidates)
                        # No independently supported target-presence labels were
                        # supplied: MIL/presence stays UNKNOWN, never synthetic1.
                        target_loss=target_loss+weights['relation']*relation+weights['weak_mil']*presence_loss(margins,'UNKNOWN')
                        counters['presence_unknown_count']+=1
                        if method=='COMBINED_UDA':
                            temporal,number=memory.loss(tgt['video_id'],tgt['query_text'],tgt['frame'],prepared['track_ids'],margins)
                            target_loss=target_loss+weights['source_anchor']*F.smooth_l1_loss(margins.float(),anchor.float())+weights['temporal']*temporal
                            counters['temporal_valid_candidates']+=number
                    weighted=target_loss*(len(candidates)/target_n)
                if smoke and len(target_grad_checks)<4 and float(weighted.detach())!=0:
                    gradients=torch.autograd.grad(weighted,trainable,retain_graph=True,allow_unused=True)
                    norm=sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)**.5
                    target_grad_checks.append(norm)
                scaler.scale(weighted).backward();parts['target_loss']+=float(weighted.detach());target_rows+=len(candidates)
                if uses_teacher:
                    for j,(track,history) in enumerate(candidates):
                        append_json(evidence_path,dict(video_id=tgt['video_id'],track_id=track,query_id=tgt['expression_id'],frame_id=tgt['frame'],
                            teacher_probability=float(teacher[j].sigmoid()),student_probability=float(margins[j].detach().float().sigmoid()),
                            objectness=float(history[-1].raw_line.split(',')[6]),temporal_consistency=None,
                            augmentation_consistency=1-abs(float(teacher[j].sigmoid())-float(margins[j].detach().float().sigmoid())),
                            semantic_teacher_score=None,pseudo_state={-1:'UNKNOWN',0:'NEG',1:'POS'}[int(state[j])],weight=int(state[j]>=0)))
            tgt_seen.add((tgt['video_id'],tgt['expression_id']))
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(trainable,config['optimizer']['grad_clip'])
        if not torch.isfinite(norm):raise RuntimeError('nonfinite real UDA gradient')
        grad_nonzero+=int(float(norm)>0);scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<scale_before:counters['amp_skipped_updates']+=1
        else:
            counters['optimizer_updates']+=1
            if ema is not None:counters['EMA_updates']+=int(ema_update(ema,model)>0)
        counters['source_candidate_rows']+=source_rows;counters['target_candidate_rows']+=target_rows
        record=dict(step=step+1,coverage_cycle=step//steps_per_epoch,source_video=src['video_id'],target_video=tgt['video_id'],
            source_rows=source_rows,target_rows=target_rows,gradient_norm=float(norm),wall_seconds=time.perf_counter()-started,**dict(parts))
        append_json(output/'metrics_train.jsonl',record)
        if (step+1)%10==0 or step+1==total_steps:
            atomic_json(output/'progress.json',dict(status='RUNNING',step=step+1,total_steps=total_steps,source_query_pairs_seen=len(src_seen),target_query_pairs_seen=len(tgt_seen),**dict(counters)))
            print(json.dumps(record),flush=True)
        checkpoint=(step+1)%config['train']['save_every_steps']==0 or step+1==total_steps
        if checkpoint:
            measured=source_validate(model,tokenizer,device,source_val,val_plan)
            selected=measured<best_source
            tracking=dict(counters=dict(counters),source_seen=list(src_seen),target_seen=list(tgt_seen),
                grad_nonzero=grad_nonzero,target_grad_checks=target_grad_checks,memory=memory.rows,
                wall_seconds=prior_wall+time.perf_counter()-started)
            if selected:
                best_source=measured;save_training(output/'checkpoint_best_source_selected.pt',model,ema,optimizer,scaler,step+1,(step+1)/steps_per_epoch,config,plan_sha,best_source,tracking=tracking)
            append_json(output/'source_validation.jsonl',dict(step=step+1,source_val_CE=measured,selected=selected))
            save_training(output/'checkpoint_latest.pt',model,ema,optimizer,scaler,step+1,(step+1)/steps_per_epoch,config,plan_sha,best_source,tracking=tracking)
    torch.cuda.synchronize();frozen_unchanged=all(p._version==frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad)
    if not frozen_unchanged or attempts:raise RuntimeError('frozen source or target annotation isolation violation')
    delta=sum(float((p.detach()-old).float().square().sum()) for p,old in zip(trainable,initial))**.5
    if not delta or not grad_nonzero:raise RuntimeError('no actual adapter learning')
    if smoke and method!='SOURCE_REHEARSAL' and not any(x>0 and math.isfinite(x) for x in target_grad_checks):
        raise RuntimeError('no finite nonzero actual target unlabeled-loss gradient in smoke')
    result=dict(status='SMOKE_PASS' if smoke else 'COMPLETE_FIXED_COVERAGE_DIAGNOSTIC',method=method,protocol=config['protocol'],
        steps=total_steps,steps_per_coverage_cycle=steps_per_epoch,coverage_cycles=total_steps/steps_per_epoch,
        source_videos_seen=len({v for v,_ in src_seen}),source_query_pairs_seen=len(src_seen),
        target_videos_seen=len({v for v,_ in tgt_seen}),target_query_pairs_seen=len(tgt_seen),
        actual_source_coverage=source_coverage,source_plan_windows=len(source_plan),target_plan_windows=len(target_plan),
        trainable_parameters=24576,adapter_delta_L2=delta,finite_nonzero_gradient_steps=grad_nonzero,
        target_smoke_gradient_norms=target_grad_checks,frozen_source_unchanged=frozen_unchanged,target_GT_open_attempts=attempts,
        target_presence_supported_positive=0,target_presence_supported_negative=0,target_presence_UNKNOWN=True,
        presence_loss_coverage=0,external_semantic_teacher=False,wall_seconds=prior_wall+time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),counters=dict(counters),test_time_updates=0)
    atomic_json(output/'training_receipt.json',result);atomic_json(output/'progress.json',result)
    print(json.dumps(result,indent=2),flush=True);return result
