"""A separate raw-soft MLLM distillation path; the old UDA trainer is unchanged."""
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import random
import site
import subprocess
import sys
import time
import numpy as np
import torch
import compat.sitecustomize  # Reuse the already-validated canonical RoBERTa BPE shim.
from transformers import RobertaTokenizerFast
from flex_uda.training import (ROOT,SOURCE,DANCE,CHECKPOINT,SOURCE_SHA,Inputs,
    protect_target_annotations,source_validate,save_training,frozen_equivalence)
from flex_uda.data.source_kitti_v2 import SourceSupervision
from flex_uda.data.manifests import write_jsonl,read_jsonl
from flex_uda.teacher.mllm_pseudo_dataset import MLLMPseudoDataset
from flex_uda.teacher.psr_dataset import PSRPseudoDataset
from flex_uda.teacher.isolation import restrict_filesystem
from flex_uda.model.flexhook_uda import attach_adapter,prepared_inputs,forward_uda
from flex_uda.losses.objectives import supervised_ce
from flex_uda.losses.distillation import soft_distillation
from flex_uda.numerics import reconcile_resume
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha


def norm(values):
    return math.sqrt(sum(float(x.detach().float().square().sum()) for x in values if x is not None))


def target_batch(inputs,group):
    rows={r['track_id']:r for r in group['samples']}
    if not rows:return None
    cache=inputs.episode('target',group)
    candidates=list(cache.episode.current_candidates(group['frame'],set(rows)))
    if {i for i,_ in candidates}!=set(rows):raise ValueError('pseudo candidate missing from fixed MOT')
    for i,history in candidates:
        if any(o.frame_id>group['frame'] for o in history):raise ValueError('future student observation')
        box=history[-1].bbox_xyxy
        if not np.allclose(box,rows[i]['bbox_xyxy'],rtol=0,atol=1e-7):raise ValueError('teacher/student bbox mismatch')
    prepared=prepared_inputs(cache,candidates,group['query_text'],group['frame'])
    probability=torch.tensor([rows[i]['teacher_prob'] for i,_ in candidates],device=inputs.device)
    valid=torch.tensor([rows[i]['valid'] for i,_ in candidates],device=inputs.device,dtype=torch.bool)
    return prepared,probability,valid,[rows[i] for i,_ in candidates]


def study_frame_groups(group,field='samples'):
    by={}
    for row in group.get(field,[]):by.setdefault(row['frame_id'],[]).append(row)
    return [dict(group,frame=t,samples=rows) for t,rows in sorted(by.items())]


def study_loss(model,inputs,group,field,amp):
    rows=group.get(field,[]);valid_n=sum(r['valid'] for r in rows);total=None
    for part in study_frame_groups(group,field):
        batch=target_batch(inputs,part)
        if batch is None:continue
        prepared,p,valid,_=batch
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            value=forward_uda(model,prepared,return_features=False)
            loss=soft_distillation(value['margin'],p,valid)*(int(valid.sum())/max(valid_n,1))
        total=loss if total is None else total+loss
    return total,rows,valid_n


def cosine_between(a,b):
    na,nb=norm(a),norm(b)
    return sum(float((x*y).sum()) for x,y in zip(a,b) if x is not None and y is not None)/(na*nb) if na and nb else None


def enforce_inputs(output,pseudo):
    vendor=ROOT/'vendor/FlexHook'
    pretrained=[p.resolve() for p in (vendor/'pretrained').iterdir()]
    first=pseudo.runtime[0]
    frozen_reference=(ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE/results'/
        first['video_id']/first['expression_id']/'candidate_margins.jsonl.gz').resolve()
    return restrict_filesystem([vendor,*pretrained,ROOT/'compat',ROOT/'flex_tta',ROOT/'flex_uda',ROOT/'scripts',ROOT/'configs',
        ROOT/'manifests',SOURCE,SOURCE/'expression',SOURCE/'KITTI/training/image_02',CHECKPOINT,pseudo.directory,
        DANCE/'DanceTrack/training/image_02',
        Path(pseudo.runtime[0]['predicted_tracks']).parent,
        ROOT/'results/runs/teacher_probe_p1_20261026_v1/calibrate/T0/SOURCE_RELIABILITY/teacher_outputs.jsonl',
        Path('/data2/user/TTA-RMOT-assets/teacher-p1/calibration/input/source_manifest.jsonl'),
        frozen_reference,
        Path(sys.prefix),Path(sys.base_prefix),Path(site.getusersitepackages()),
        '/usr','/lib','/lib64','/bin','/etc','/proc','/sys'],
        [output,'/dev','/tmp'])


def source_scale_check(model,inputs,trainable):
    """Actual legal source-dev teacher outputs; no target labels or optimizer step."""
    manifest=read_jsonl('/data2/user/TTA-RMOT-assets/teacher-p1/calibration/input/source_manifest.jsonl')
    raw={r['sample_id']:r for r in read_jsonl(ROOT/'results/runs/teacher_probe_p1_20261026_v1/calibrate/T0/SOURCE_RELIABILITY/teacher_outputs.jsonl')}
    chosen={}
    for row in manifest:chosen.setdefault(row['video_id'],row)
    records=[]
    for sample in chosen.values():
        source=inputs.source.episode(sample['video_id']);frame=sample['frame_id']
        candidates=list(source.current_candidates(frame,{sample['track_id']}))
        if not candidates:raise ValueError('registered source calibration candidate absent')
        cache=inputs.episode('source',sample);prepared=prepared_inputs(cache,candidates,sample['query_text'],frame)
        truth=source.labels_for(candidates,sample['query_text'],inputs.device)
        result=forward_uda(model,prepared,return_features=False)
        source_loss=supervised_ce(result['all_logits'],truth)
        response=raw[sample['sample_id']]['source_full_expression_views']['original']
        p=torch.tensor([response['yes_probability']],device=inputs.device)
        kd=soft_distillation(result['margin'],p,torch.ones_like(p,dtype=torch.bool))
        sg=torch.autograd.grad(source_loss,trainable,retain_graph=True,allow_unused=True)
        tg=torch.autograd.grad(kd,trainable,allow_unused=True)
        records.append(dict(video=sample['video_id'],frame=frame,track_id=sample['track_id'],source_CE=float(source_loss),
            source_dev_teacher_KD=float(kd),source_grad_norm=norm(sg),KD_grad_norm=norm(tg),
            source_teacher_view='P1 legal source-dev original four-observation question; scale sanity only, not target inference'))
    if not all(math.isfinite(v) for r in records for k,v in r.items() if isinstance(v,float)) or not any(r['KD_grad_norm']>0 for r in records):
        raise RuntimeError('source-only KD scale validation failed')
    return dict(lambda_KD=1.0,selection='PRE_FIXED_EQUAL_DOMAIN_MEAN_WEIGHTS_RETAINED_AFTER_SOURCE_ONLY_FINITE_SCALE_CHECK',
        optimized_on_target_GT=False,source_video_count=len(records),records=records,
        limitation='Not an optimized lambda search; source teacher calibration uses its already frozen four-observation source view.')


@torch.no_grad()
def teacher_fit(model,inputs,pseudo,output):
    records=[];before=[];after=[];before_bce=[];after_bce=[]
    with gzip.open(output/'teacher_fit_after_training.jsonl.gz','wt') as handle:
        for group in pseudo.groups:
            batch=target_batch(inputs,group)
            if batch is None:continue
            prepared,p,valid,rows=batch;value=forward_uda(model,prepared,return_features=False)
            margins=value['margin'].float();original=torch.tensor([r['frozen_margin'] for r in rows],device=inputs.device)
            before_bce.append((float(soft_distillation(original,p,valid))*int(valid.sum()),int(valid.sum())))
            after_bce.append((float(soft_distillation(margins,p,valid))*int(valid.sum()),int(valid.sum())))
            for j,row in enumerate(rows):
                record=dict(sample_id=row['sample_id'],video_id=row['video_id'],query_id=row['query_id'],
                    frame_id=row['frame_id'],track_id=row['track_id'],valid=row['valid'],
                    teacher_probability=row['teacher_prob'],source_margin=row['frozen_margin'],student_margin=float(margins[j]))
                handle.write(json.dumps(record)+'\n')
                if row['valid']:
                    before.append((float(original[j].sigmoid())-row['teacher_prob'])**2)
                    after.append((float(margins[j].sigmoid())-row['teacher_prob'])**2)
    n=sum(r[1] for r in before_bce)
    result=dict(status='ACTUAL_LATEST_STUDENT_ON_TEACHER_TRAINING_ANCHORS_NOT_GT_ACCURACY',valid_samples=n,
        source_teacher_MSE=float(np.mean(before)),trained_teacher_MSE=float(np.mean(after)),
        source_teacher_BCE=sum(r[0] for r in before_bce)/n,trained_teacher_BCE=sum(r[0] for r in after_bce)/n,
        training_anchor_labels_not_independent_validation=True,checkpoint_selection_feedback=False)
    atomic_json(output/'teacher_fit_summary.json',result);return result


def run(config,output,*,scope='smoke',resume=False):
    output=Path(output).resolve()
    if config['protocol']['target_split']=='val' and os.environ.get('MLLM_VAL_DIAGNOSTIC_AUTHORIZED')!='YES':
        raise PermissionError('explicit Val transductive authorization required')
    study=config['protocol']['name'] in ('TRANSDUCTIVE_VAL_PSR_STUDY','FORMAL_TRAIN_TO_VAL_PSR_STUDY')
    study_methods=('NAIVE_PAIRED','UNIFORM_PAIRED','CONFIDENCE_PAIRED','PSR_DISCOVERY','PSR_GEOMETRY','PSR','NAIVE_TEMPORAL')
    if config['method'] not in (study_methods if study else ('MLLM_DISTILL','SOURCE_REHEARSAL')):raise ValueError('independent distillation method required')
    if output.exists() and not resume:raise FileExistsError('preserve existing run: '+str(output))
    output.mkdir(parents=True,exist_ok=resume);started=time.perf_counter()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    attempts=protect_target_annotations()
    pseudo=(PSRPseudoDataset(config['data']['pseudo_directory'],allow_val_diagnostic=True,expanded_kind=config['data'].get('expanded_kind','full')) if study
        else MLLMPseudoDataset(config['data']['pseudo_directory'],allow_val_diagnostic=True))
    if (not study and config['protocol']['name']!='TRANSDUCTIVE_VAL_MLLM_DISTILL_DIAGNOSTIC') or config['protocol']['target_split']!=pseudo.split:
        raise ValueError('this validated Val route cannot be relabelled as formal Train-to-Val')
    if study and (config['protocol']['name']=='FORMAL_TRAIN_TO_VAL_PSR_STUDY')!=(pseudo.split=='train'):
        raise ValueError('formal training cannot consume Val teacher labels')
    if file_sha(CHECKPOINT)!=SOURCE_SHA:raise ValueError('official V2 checkpoint changed')
    seed=config['protocol']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(((16384 if study else 24576)<<20)/props.total_memory,device)
    # Initialize only the empty CUDA context before the input whitelist. The
    # dfot driver needs host initialization paths; no model/target image or GT
    # is opened here. All actual model/data work remains kernel-isolated.
    torch.empty(1,device=device).zero_();torch.cuda.synchronize()
    isolation=enforce_inputs(output,pseudo)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    _,model,_=_load_model(ROOT/'vendor/FlexHook',CHECKPOINT,DANCE,Path(pseudo.runtime[0]['predicted_tracks']).parent,
        output/'model_log',device,24,strict_checkpoint=True)
    adapter=attach_adapter(model,device,rank=4);model.eval()
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'vendor/FlexHook/pretrained/roberta-base'))
    if tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('tokenizer regression')
    source=SourceSupervision(ROOT/config['data']['source_train_manifest'])
    source_val=SourceSupervision(ROOT/config['data']['source_val_manifest'])
    inputs=Inputs(model,tokenizer,device,source)
    params=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=24576:raise ValueError('adapter capacity changed')
    initial=[p.detach().clone() for p in params]
    versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(params,lr=config['optimizer']['lr'],weight_decay=config['optimizer']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=config['train']['amp'] and scope!='interface')
    provenance=dict(experiment_commit=commit,method=config['method'],scope=scope,protocol=config['protocol'],
        source_checkpoint_sha256=SOURCE_SHA,pseudo_seal_sha256=file_sha(pseudo.directory/'seal.json'),
        pseudo_manifest_sha256=pseudo.seal['files_sha256']['manifest.jsonl'],teacher_labels_sha256=pseudo.seal['files_sha256']['soft_labels.jsonl'],
        source_train_manifest_sha256=file_sha(ROOT/config['data']['source_train_manifest']),
        source_val_manifest_sha256=file_sha(ROOT/config['data']['source_val_manifest']),
        trainable_parameters=24576,physical_gpu=os.environ.get('MLLM_PHYSICAL_GPU'),gpu_uuid=str(props.uuid),isolation=isolation,
        target_GT_read=False,no_online_MLLM=True,test_time_updates=0,
        implementation_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in [Path(__file__),ROOT/'flex_uda/losses/distillation.py',ROOT/'flex_uda/teacher/mllm_pseudo_dataset.py',ROOT/'compat/sitecustomize.py',ROOT/'vendor/FlexHook/models/mymodel.py']})
    if study:
        provenance['implementation_sha256'].update({name:file_sha(ROOT/name) for name in ('flex_uda/teacher/psr_dataset.py','flex_uda/teacher/psr_supervision.py','flex_uda/teacher/psr_sampler.py','flex_uda/losses/psr_objectives.py')})
    atomic_json(output/'provenance.json',provenance);atomic_json(output/'config.json',config)
    atomic_json(output/'initial_frozen_equivalence.json',frozen_equivalence(model,inputs,pseudo.runtime[0]))
    calibration=source_scale_check(model,Inputs(model,tokenizer,device,source_val),params)
    atomic_json(output/'source_only_lambda_validation.json',calibration)
    if scope=='interface':
        checks=[]
        indices=set(np.linspace(0,len(pseudo.groups)-1,8,dtype=int).tolist())
        if study and pseudo.expanded:indices.add(next(i for i,g in enumerate(pseudo.groups) if g['expanded_samples']))
        for index in sorted(indices):
            group=pseudo.groups[index]
            batch=target_batch(inputs,study_frame_groups(group)[0]) if study and group['samples'] else target_batch(inputs,group)
            if batch is None:continue
            prepared,p,valid,rows=batch;optimizer.zero_grad(set_to_none=True)
            value=forward_uda(model,prepared,return_features=False);loss=soft_distillation(value['margin'],p,valid)
            loss.backward();gn=norm([x.grad for x in params])
            checks.append(dict(video=rows[0]['video_id'],frame=rows[0]['frame_id'],candidates=len(rows),valid=int(valid.sum()),
                logits_shape=list(value['all_logits'].shape),loss=float(loss),KD_gradient_norm=gn,optimizer_step_called=False))
            if study:
                optimizer.zero_grad(set_to_none=True)
                weak,weak_rows,weak_n=study_loss(model,inputs,group,'expanded_samples',False)
                if weak is not None:
                    weak.backward();weak_gn=norm([p.grad for p in params])
                    if weak_n and (not math.isfinite(weak_gn) or weak_gn<=0):raise RuntimeError('actual weak supervision cannot reach adapter')
                    checks[-1].update(expanded_candidates=weak_n,expanded_gradient_norm=weak_gn,expanded_loss=float(weak))
        if not checks or not any(r['KD_gradient_norm']>0 for r in checks):raise RuntimeError('KD cannot reach adapter')
        path=output/'temporary_interface_checkpoint.pt'
        saved={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        torch.save(dict(student_trainable=saved,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            config=config,pseudo_seal_sha256=provenance['pseudo_seal_sha256'],step=0,rng=torch.get_rng_state()),path)
        restored=torch.load(path,map_location=device,weights_only=False)
        if not all(torch.equal(saved[n],restored['student_trainable'][n]) for n in saved):raise RuntimeError('interface checkpoint roundtrip failed')
        temporary_sha=file_sha(path);path.unlink();optimizer.zero_grad(set_to_none=True)
        if any(not torch.equal(p,old) for p,old in zip(params,initial)):raise RuntimeError('no-step interface changed parameters')
        result=dict(status='INTERFACE_PASS_NO_OPTIMIZER_STEP',checks=checks,optimizer_updates=0,adapter_delta_L2=0.,
            temporary_checkpoint_roundtrip=True,temporary_checkpoint_sha256=temporary_sha,temporary_checkpoint_deleted=True)
    else:
        source_plan,source_coverage=source.plan();val_plan,_=source_val.plan(validation_queries_per_video=16)
        if len(source_plan)!=8973 or sum(r['provided_expressions'] for r in source_coverage)!=8896 or len(val_plan)!=64:
            raise RuntimeError('incomplete official source expression/validation schedule; no optimizer step permitted')
        plans=output/'plans';write_jsonl(plans/'source_train_plan.jsonl',source_plan)
        write_jsonl(plans/'target_plan.jsonl',[{k:v for k,v in r.items() if k!='samples'} for r in pseudo.groups])
        plan_sha={n:file_sha(plans/n) for n in ('source_train_plan.jsonl','target_plan.jsonl')}
        plan_sha.update(teacher_manifest=provenance['pseudo_manifest_sha256'],teacher_labels=provenance['teacher_labels_sha256'],pseudo_seal=provenance['pseudo_seal_sha256'])
        steps_per_cycle=max(len(source_plan),len(pseudo.groups))
        if study and steps_per_cycle!=8973:raise ValueError('PSR/Naive must share the registered8973 update budget')
        total=6 if scope=='smoke' else config['train']['coverage_cycles']*steps_per_cycle
        state_path=output/'checkpoint_latest.pt';step=0;best=float('inf');counters=Counter();seen_source=set();seen_target=set();seen_samples=set();seen_expanded=set()
        prior_wall=0.
        if resume:
            state=torch.load(state_path,map_location=device,weights_only=False)
            if state['config']!=config or state['plan_sha256']!=plan_sha or state['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('resume config/teacher/data/initialization mismatch')
            model.load_state_dict(state['student'],strict=True);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            step=state['step'];best=state['best_source_loss'];tracking=state['tracking']
            random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'].cpu());torch.cuda.set_rng_state(state['rng']['cuda'].cpu())
            counters.update(tracking['counters']);seen_source=set(map(tuple,tracking['source_seen']));seen_target=set(map(tuple,tracking['target_seen']));seen_samples=set(tracking['teacher_samples_seen'])
            seen_expanded=set(tracking.get('expanded_samples_seen',[]))
            prior_wall=tracking['wall_seconds']+reconcile_resume(output,state)
            versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
        else:
            best=source_validate(model,tokenizer,device,source_val,val_plan)
            if not math.isfinite(best):raise FloatingPointError('nonfinite legal source validation CE')
            save_training(output/'checkpoint_best_source_selected.pt',model,None,optimizer,scaler,0,0,config,plan_sha,best)
            append_json(output/'source_validation.jsonl',dict(step=0,source_val_CE=best,selected=True))
        retry=0
        while step<total:
            src=source_plan[step%len(source_plan)];target=pseudo.groups[step%len(pseudo.groups)]
            optimizer.zero_grad(set_to_none=True);source_loss_total=0.;source_rows=0
            batches=list(inputs.batches('source',src));source_n=sum(len(r[2]) for r in batches)
            for prepared,labels,candidates,_ in batches:
                with torch.autocast('cuda',dtype=torch.float16,enabled=scaler.is_enabled()):
                    value=forward_uda(model,prepared,return_features=False)
                    loss=supervised_ce(value['all_logits'],labels)*len(candidates)/source_n
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite source supervised objective')
                scaler.scale(loss).backward();source_loss_total+=float(loss.detach());source_rows+=len(candidates)
            source_grads=[None if p.grad is None else p.grad.detach().float().clone()/scaler.get_scale() for p in params]
            kd_value=0.;teacher_rows=[];valid_n=0;expanded_rows=[];expanded_n=0;expanded_value=0.;direct_grads=None
            if study:
                kd,teacher_rows,valid_n=study_loss(model,inputs,target,'samples',scaler.is_enabled())
                if kd is not None:
                    if not torch.isfinite(kd):raise FloatingPointError('nonfinite direct teacher objective')
                    scaler.scale(kd*config['loss']['target_kd']).backward();kd_value=float(kd.detach())
                direct_grads=[p.grad.detach().float().clone()/scaler.get_scale()-(s if s is not None else 0) if p.grad is not None else None for p,s in zip(params,source_grads)]
                weak,expanded_rows,expanded_n=study_loss(model,inputs,target,'expanded_samples',scaler.is_enabled())
                if weak is not None:
                    if not torch.isfinite(weak):raise FloatingPointError('nonfinite expanded weak objective')
                    scaler.scale(weak*config['loss']['expanded']).backward();expanded_value=float(weak.detach())
            elif config['method']=='MLLM_DISTILL':
                batch=target_batch(inputs,target)
                if batch is not None:
                    prepared,p,valid,teacher_rows=batch;valid_n=int(valid.sum())
                    with torch.autocast('cuda',dtype=torch.float16,enabled=scaler.is_enabled()):
                        value=forward_uda(model,prepared,return_features=False)
                        kd=soft_distillation(value['margin'],p,valid)*config['loss']['target_kd']
                    if not torch.isfinite(kd):raise FloatingPointError('nonfinite actual target KD objective')
                    scaler.scale(kd).backward();kd_value=float(kd.detach())
            scaler.unscale_(optimizer)
            target_grads=[p.grad.detach().float()-(s if s is not None else 0) if p.grad is not None else None for p,s in zip(params,source_grads)]
            source_norm=norm(source_grads);target_norm=norm(target_grads)
            expanded_grads=([None if t is None else t-(d if d is not None else 0) for t,d in zip(target_grads,direct_grads)] if study else [])
            total_norm=torch.nn.utils.clip_grad_norm_(params,config['optimizer']['grad_clip'])
            if not torch.isfinite(total_norm):
                if not scaler.is_enabled() or retry>=8:raise FloatingPointError('nonfinite gradient after same-sample AMP backoff')
                scale=scaler.get_scale();scaler.update(new_scale=scale/2);retry+=1
                append_json(output/'numerical_retries.jsonl',dict(step=step+1,retry=retry,previous_scale=scale,new_scale=scaler.get_scale(),same_source_and_target_replayed=True,optimizer_updated=False))
                continue
            dot=sum(float((a*b).sum()) for a,b in zip(source_grads,target_grads) if a is not None and b is not None)
            cosine=dot/(source_norm*target_norm) if source_norm and target_norm else None
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale_before:raise RuntimeError('unexpected skipped update after explicit finite gradient validation')
            counters['optimizer_updates']+=1;counters['source_candidate_rows']+=source_rows;counters['target_valid_exposures']+=valid_n
            counters['target_nonzero_gradient_steps']+=int(target_norm>1e-12)
            if study:
                counters['direct_valid_exposures']+=valid_n;counters['expanded_weak_exposures']+=expanded_n
                counters['direct_nonzero_gradient_steps']+=int(norm(direct_grads)>1e-12)
                counters['expanded_nonzero_gradient_steps']+=int(norm(expanded_grads)>1e-12)
                seen_expanded.update(r['sample_id'] for r in expanded_rows if r['valid'])
            seen_source.add((src['video_id'],src['expression_id']))
            if config['method']=='MLLM_DISTILL' or study:
                seen_target.add((target['video_id'],target['expression_id']))
                seen_samples.update(r['sample_id'] for r in teacher_rows if r['valid'])
            record=dict(step=step+1,source_video=src['video_id'],source_expression=src['expression_id'],
                target_video=target['video_id'],target_expression=target['expression_id'],target_frame=target['frame'],
                source_rows=source_rows,target_valid_rows=valid_n,source_loss=source_loss_total,target_KD_loss=kd_value,
                source_gradient_norm=source_norm,target_gradient_norm=target_norm,source_target_gradient_cosine=cosine,
                gradient_norm=float(total_norm),wall_seconds=time.perf_counter()-started)
            if study:
                record.update(expanded_KD_loss=expanded_value,expanded_weighted_loss=expanded_value*config['loss']['expanded'],
                    expanded_valid_rows=expanded_n,direct_gradient_norm=norm(direct_grads),expanded_gradient_norm=norm(expanded_grads),
                    source_direct_gradient_cosine=cosine_between(source_grads,direct_grads),source_expanded_gradient_cosine=cosine_between(source_grads,expanded_grads),
                    direct_expanded_gradient_cosine=cosine_between(direct_grads,expanded_grads),
                    direct_states=dict(Counter(r['state'] for r in teacher_rows)),expanded_states=dict(Counter(r['state'] for r in expanded_rows)),
                    UNKNOWN_not_default_negative=True)
            append_json(output/'metrics_train.jsonl',record);step+=1;retry=0
            if step%10==0 or step==total:
                atomic_json(output/'progress.json',dict(status='RUNNING',step=step,total_steps=total,unique_teacher_samples_seen=len(seen_samples),**dict(counters)))
                print(json.dumps(record),flush=True)
            if step%config['train']['save_every_steps']==0 or step==total:
                measured=source_validate(model,tokenizer,device,source_val,val_plan);selected=measured<best
                if not math.isfinite(measured):raise FloatingPointError('nonfinite source validation CE; no checkpoint selection')
                tracking=dict(counters=dict(counters),source_seen=sorted(seen_source),target_seen=sorted(seen_target),teacher_samples_seen=sorted(seen_samples),
                    expanded_samples_seen=sorted(seen_expanded),wall_seconds=prior_wall+time.perf_counter()-started,attempt_wall_seconds=time.perf_counter()-started)
                if selected:
                    best=measured;save_training(output/'checkpoint_best_source_selected.pt',model,None,optimizer,scaler,step,step/steps_per_cycle,config,plan_sha,best,tracking=tracking)
                append_json(output/'source_validation.jsonl',dict(step=step,source_val_CE=measured,selected=selected))
                save_training(state_path,model,None,optimizer,scaler,step,step/steps_per_cycle,config,plan_sha,best,tracking=tracking)
        restored=torch.load(state_path,map_location='cpu',weights_only=False)
        if restored['step']!=total or restored['plan_sha256']!=plan_sha:raise RuntimeError('checkpoint recovery roundtrip failed')
        del restored
        delta=norm([p.detach()-old for p,old in zip(params,initial)])
        if not delta or not counters['optimizer_updates']:raise RuntimeError('student parameters did not change')
        if (config['method']=='MLLM_DISTILL' or study) and not counters['target_nonzero_gradient_steps']:raise RuntimeError('no nonzero target KD gradients')
        if scope=='full' and (len(seen_source)!=len(source_plan) or
            (config['method']=='MLLM_DISTILL' or study) and (len(seen_target)!=len(pseudo.runtime) or len(seen_samples)!=pseudo.seal['valid_soft_samples'])):
            raise RuntimeError('incomplete actual source/target teacher coverage')
        if scope=='full' and study and len(seen_expanded)!=len(pseudo.expanded):raise RuntimeError('incomplete unique weak supervision coverage')
        if scope=='full' and study:
            from types import SimpleNamespace
            fit=teacher_fit(model,inputs,SimpleNamespace(groups=[part for group in pseudo.groups for part in study_frame_groups(group)]),output)
        else:fit=teacher_fit(model,inputs,pseudo,output) if scope=='full' and config['method']=='MLLM_DISTILL' else None
        result=dict(status='STUDENT_SMOKE_PASS' if scope=='smoke' else ('COMPLETE_FIXED_COVERAGE_PSR_STUDY' if study else 'COMPLETE_FIXED_COVERAGE_TRANSDUCTIVE_DIAGNOSTIC'),
            steps=total,steps_per_cycle=steps_per_cycle,counters=dict(counters),adapter_delta_L2=delta,
            source_query_pairs_seen=len(seen_source),source_videos_seen=len({x[0] for x in seen_source}),
            target_query_pairs_seen=len(seen_target),target_videos_seen=len({x[0] for x in seen_target}),
            unique_teacher_samples_seen=len(seen_samples),target_registered_windows=len(pseudo.groups),
            unique_expanded_samples_seen=len(seen_expanded),
            source_coverage=source_coverage,checkpoint_roundtrip=True,teacher_fit=fit,
            checkpoints={p.name:dict(path=str(p),sha256=file_sha(p)) for p in output.glob('checkpoint_*.pt')})
    torch.cuda.synchronize()
    frozen_unchanged=all(p._version==versions[n] for n,p in model.named_parameters() if not p.requires_grad)
    if not frozen_unchanged or attempts:raise RuntimeError('frozen-parameter or target-GT isolation violation')
    result.update(method=config['method'],scope=scope,protocol=config['protocol'],trainable_parameters=24576,
        frozen_parameters_unchanged=True,target_GT_open_attempts=attempts,wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),source_checkpoint_sha256=SOURCE_SHA,
        pseudo_seal_sha256=provenance['pseudo_seal_sha256'],optimizer_at_test_time=False,student_calls_MLLM=False)
    atomic_json(output/'training_receipt.json',result);atomic_json(output/'progress.json',result)
    print(json.dumps(result,indent=2),flush=True);return result
