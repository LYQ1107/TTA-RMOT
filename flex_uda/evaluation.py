"""Read-only FP32 evaluation, optionally batching query-separable decode calls.

Only the original decoder is used. Query batching is a numerical execution
optimization, never a new model or cross-expression adaptation. Its masks
exclude other queries, and every selected checkpoint gets a real-frame parity
test before this mode is allowed. Near-zero decisions use the isolated path.
"""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from transformers import RobertaTokenizerFast

from flex_uda.data.manifests import ROOT,DANCE
from flex_uda.data.target_dance import TargetDanceDataset
from flex_uda.model.flexhook_uda import attach_adapter,prepared_inputs
from flex_uda.training import CHECKPOINT,SOURCE_SHA,protect_target_annotations
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.pf_receipts import EpisodeStore,atomic_json,file_sha

FROZEN=ROOT/'results/runs/pf_zero_completebank_full_20261025_merged/ZERO_UPDATE'


@torch.no_grad()
def query_margins(model,prepared,texts,*,batched=True):
    """Distinct real-query features; original masked decode, no candidate softmax."""
    old=model.sample_expression_num
    try:
        if batched:
            model.sample_expression_num=len(texts)
            text=torch.cat([t for t,m in texts],dim=1).expand(len(prepared['track_ids']),-1,-1,-1).contiguous()
            mask=torch.cat([m for t,m in texts],dim=1).expand(len(prepared['track_ids']),-1,-1).contiguous()
            logits,_=model.decode(prepared['outputs'],text,mask,prepared['positions'],prepared['boxes'])
            if logits.shape[-2:]!=(len(texts),2):raise ValueError('unexpected multi-query decode shape')
            return logits[:,-1,:,1]-logits[:,-1,:,0]
        result=[];model.sample_expression_num=1
        for text,mask in texts:
            logits,_=model.decode(prepared['outputs'],text.expand(len(prepared['track_ids']),-1,-1,-1).contiguous(),
                mask.expand(len(prepared['track_ids']),-1,-1).contiguous(),prepared['positions'],prepared['boxes'])
            result.append(logits[:,-1,0,1]-logits[:,-1,0,0])
        return torch.stack(result,dim=1)
    finally:
        model.sample_expression_num=old


@torch.no_grad()
def parity(model,cache,rows,batch_size=24):
    """Pre-fixed first/middle/last real MOT frames; no labels or score selection."""
    episode=cache.episode;frames=sorted({episode.frames[0],episode.frames[len(episode.frames)//2],episode.frames[-1]})
    texts=[cache._text(r['query_text']) for r in rows];records=[];passed=True
    for frame in frames:
        candidates=list(episode.current_candidates(frame))
        for start in range(0,len(candidates),batch_size):
            prepared=prepared_inputs(cache,candidates[start:start+batch_size],rows[0]['query_text'],frame)
            multi=query_margins(model,prepared,texts,batched=True)
            separate=query_margins(model,prepared,texts,batched=False)
            error=float((multi-separate).abs().max());flips=int(((multi>0)!=(separate>0)).sum())
            # Pre-registered engineering tolerance, not tuned with target GT.
            passed=passed and error<=1e-3 and flips==0 and bool(torch.isfinite(multi).all())
            records.append(dict(frame=frame,candidates=len(prepared['track_ids']),queries=len(rows),
                max_abs_margin_error=error,selection_mismatches=flips))
    return dict(status='PASS' if passed else 'FALLBACK_TO_ISOLATED_QUERY',video=episode.video,
        fixed_frame_rule='first/middle/last MOT-active frame',precision='FP32',records=records)


def load_selected(checkpoint,output,tracker_root,device):
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    if file_sha(CHECKPOINT)!=SOURCE_SHA:raise ValueError('source initialization changed')
    _,model,_=_load_model(ROOT/'vendor/FlexHook',CHECKPOINT,DANCE,tracker_root,output/'model_log',device,24,strict_checkpoint=True)
    adapter=attach_adapter(model,device,rank=4)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['source_checkpoint_sha256']!=SOURCE_SHA or state['test_time_update']:raise ValueError('wrong checkpoint protocol')
    model.load_state_dict(state['student'],strict=True)
    info=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),selected_step=state['step'],
        source_val_CE=state['best_source_loss'],selection='SOURCE_VAL_CE_ONLY',protocol=state['config']['protocol'])
    del state
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'vendor/FlexHook/pretrained/roberta-base'))
    if tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('tokenizer regression')
    return model,tokenizer,info


def evaluate(checkpoint,method,output,*,videos=None,batch_size=24,parity_only=False):
    output=Path(output).resolve();checkpoint=Path(checkpoint).resolve();folder=output/method
    rows=TargetDanceDataset(ROOT/'manifests/dance_val_runtime.jsonl',split='val').rows
    all_videos=sorted({r['video_id'] for r in rows});videos=videos or all_videos
    if not set(videos)<=set(all_videos):raise ValueError('unknown target video')
    attempts=protect_target_annotations();started=time.perf_counter();device=torch.device('cuda:0')
    start_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Inference has no student/EMA/optimizer or backward graphs. Its own
    # memory budget should not unnecessarily inherit the larger train budget.
    torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,tokenizer,info=load_selected(checkpoint,folder,Path(rows[0]['predicted_tracks']).parent,device)
    start_provenance=folder/('parity_start_provenance.json' if parity_only else 'inference_start_provenance.json')
    if start_provenance.exists():
        archive=folder/'inference_provenance_attempts';archive.mkdir(exist_ok=True)
        start_provenance.rename(archive/(str(time.time_ns())+'_'+start_provenance.name))
    atomic_json(start_provenance,dict(experiment_commit=start_commit,recording_mode='worker-entry snapshot',
        checkpoint_sha256=info['checkpoint_sha256'],method=method,optimizer_steps=0,target_GT_used=False))
    versions={n:p._version for n,p in model.named_parameters()};receipts=[];parities=[];cache_stats=[];fallback_queries=0
    manifest_sha=file_sha(ROOT/'manifests/dance_val_runtime.jsonl')
    for video in videos:
        queries=[r for r in rows if r['video_id']==video]
        episode=DanceEpisode(DANCE,Path(queries[0]['predicted_tracks']).parent,video)
        cache=FrozenFeatureCache(model,episode,tokenizer,device,max_bytes=512<<20)
        report=parity(model,cache,queries,batch_size);parities.append(report)
        atomic_json(folder/'parity'/f'{video}.json',report)
        if parity_only:break
        batched=report['status']=='PASS'
        config=dict(protocol=info['protocol'],checkpoint_sha256=info['checkpoint_sha256'],manifest_sha256=manifest_sha,
            threshold=0,precision='FP32',optimizer_steps=0,frontend='OFFICIAL_DETR_NEURALSORT',
            query_batch_mode='MASKED_WITH_ISOLATED_NEAR_ZERO_FALLBACK' if batched else 'ISOLATED_QUERY',fallback_abs_margin=1e-3)
        stores=[EpisodeStore(folder,video,r['expression_id'],config,resume=True) for r in queries]
        completed=[s.completed() for s in stores]
        if all(completed):receipts.extend(completed);continue
        pending=[i for i,r in enumerate(completed) if r is None]
        texts=[cache._text(r['query_text']) for r in queries]
        counts={i:dict(frames=0,candidate_rows=0,selected_rows=0,optimizer_steps=0) for i in pending}
        video_start=time.perf_counter()
        with ExitStack() as stack:
            writers={i:stack.enter_context(stores[i].attempt()) for i in pending}
            source_streams={i:stack.enter_context(gzip.open(FROZEN/'results'/video/queries[i]['expression_id']/'candidate_margins.jsonl.gz','rt')) for i in pending}
            for frame in episode.frames:
                candidates=list(episode.current_candidates(frame));ids=[];score_chunks=[]
                for start in range(0,len(candidates),batch_size):
                    chosen=candidates[start:start+batch_size]
                    prepared=prepared_inputs(cache,chosen,queries[0]['query_text'],frame)
                    with torch.no_grad():
                        value=query_margins(model,prepared,texts,batched=batched)
                        if batched:
                            # A larger batched attention can change FP32 rounding.
                            # Recompute every near-threshold query in its original
                            # one-query shape; never alter the decision threshold.
                            near=(value.abs()<=1e-3).any(dim=0).nonzero().flatten().tolist()
                            for q in near:
                                value[:,q]=query_margins(model,prepared,[texts[q]],batched=False)[:,0]
                            fallback_queries+=len(near)
                    if not torch.isfinite(value).all():raise RuntimeError('nonfinite sealed evaluation scores')
                    ids.extend(prepared['track_ids']);score_chunks.append(value.cpu().numpy())
                scores=np.concatenate(score_chunks,axis=0)
                for q in pending:
                    frozen=json.loads(next(source_streams[q]))
                    if frozen['frame']!=frame or frozen['ids']!=ids:raise ValueError('sealed source universe mismatch')
                    predict,margins=writers[q];values=scores[:,q].tolist()
                    for track,score in zip(ids,values):
                        if score>0:predict.write(episode.observations_by_frame[frame][track].raw_line+'\n')
                    margins.write(json.dumps(dict(frame=frame,ids=ids,source=frozen['adapted'],adapted=values))+'\n')
                    counts[q]['frames']+=1;counts[q]['candidate_rows']+=len(ids);counts[q]['selected_rows']+=sum(v>0 for v in values)
            for q in pending:
                if next(source_streams[q],None) is not None:raise ValueError('source stream has unconsumed frames')
        video_wall=time.perf_counter()-video_start
        for q,store in enumerate(stores):
            receipt=completed[q]
            if receipt is None:
                receipt=store.finish(dict(full_frame_coverage=True,wall_seconds=video_wall/len(pending),
                    wall_time_attribution='shared video inference divided across simultaneous pending expressions',
                    stats=counts[q],counter_delta=dict(frames=counts[q]['frames'],candidates=counts[q]['candidate_rows'],
                        selected=counts[q]['selected_rows'],optimizer_steps=0)))
            receipts.append(receipt)
        cache_stats.append(dict(video=video,**cache.stats))
        print(json.dumps(dict(event='video_sealed',method=method,video=video,episodes=17,wall_seconds=video_wall)),flush=True)
    torch.cuda.synchronize()
    if attempts or any(p._version!=versions[n] for n,p in model.named_parameters()):raise RuntimeError('evaluation mutated model or opened targetGT')
    result=dict(status='PARITY_ONLY' if parity_only else 'SEALED',method=method,**info,
        experiment_commit=start_commit,
        videos=videos,valid_episodes=len(receipts),episodes=receipts,parities=parities,
        target_GT_open_attempts=attempts,optimizer_steps=0,test_time_updates=0,precision='FP32',threshold=0,
        query_batch_near_zero_fallbacks=fallback_queries,cache_stats=cache_stats,wall_seconds=time.perf_counter()-started,
        physical_gpu=__import__('os').environ.get('UDA_PHYSICAL_GPU'),gpu_uuid=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    result['allocator_budget_bytes']=16384<<20
    name='parity_receipt.json' if parity_only else 'run_metadata.json'
    atomic_json(folder/name,result)
    if not parity_only and set(videos)==set(all_videos):
        audit=[]
        for r in receipts:
            lines=Path(r['prediction_path']).read_text().splitlines()
            audit.append(dict(video=r['video'],expression=r['expression_id'],frames=r['stats']['frames'],
                candidates=r['stats']['candidate_rows'],selected_rows=len(lines),selected_ids=len({x.split(',')[1] for x in lines}),
                wall_seconds=r['wall_seconds']))
        if len(audit)!=425 or len({(r['video'],r['expression']) for r in audit})!=425:raise ValueError('incomplete evaluation')
        atomic_json(folder/'full_audit.json',dict(status='PASS',method=method,valid_episodes=425,unique_videos=25,
            duplicated_episodes=0,missing_episodes=0,target_gt_opened=False,tracker_geometry_changed=False,tracker_ids_changed=False,
            episodes=audit,**{k:sum(e[k] for e in audit) for k in ('frames','candidates','selected_rows','selected_ids')}))
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','parities','cache_stats')},indent=2))
    return result
