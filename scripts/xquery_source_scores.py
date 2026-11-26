"""One-GPU, frozen checkpoint KITTI scoring; no Dance inputs or teacher."""
from collections import OrderedDict,defaultdict
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *

def worker(smoke=False):
    import compat.sitecustomize
    import torch
    import torchvision.transforms as T
    from transformers import RobertaTokenizerFast
    from flex_tta.runtime.dance_episode import DanceEpisode,Observation
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.model.flexhook_uda import attach_adapter,adapter_kwargs,prepared_inputs
    from flex_uda.evaluation import query_margins
    from flex_uda.training import CHECKPOINT,SOURCE_SHA as CKPT_SHA
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    from scripts.count_structure_source import SOURCE_TRACKS
    from flex_tta.mot_interface.trackers import merge_kitti_class_splits
    setup();require_plan();attempts=v2.deny_gt();started=time.perf_counter();dest=RUN/'source'
    if (dest/'scores_seal.json').exists() and not smoke:
        result=json.loads((dest/'scores_seal.json').read_text())
        for r in result['models'].values():
            if file_sha(r['array_path'])!=r['array_sha256']:raise ValueError('changed completed source scores')
        print('SOURCE_SCORES_ALREADY_COMPLETE_NO_GPU',flush=True);return
    meta=json.loads((dest/'metadata.json').read_text());device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if file_sha(CHECKPOINT)!=CKPT_SHA:raise ValueError('source initializer changed')
    _,base,_=_load_model(ROOT/'vendor/FlexHook',CHECKPOINT,SOURCE,SOURCE_TRACKS,dest/'model_log',device,8,strict_checkpoint=True)
    original=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'];models={};provenance={}
    for m,r in checkpoints().items():
        if file_sha(r['path'])!=r['sha256']:raise ValueError('changed final checkpoint '+m)
        state=torch.load(r['path'],map_location='cpu',weights_only=False)
        if state['source_checkpoint_sha256']!=CKPT_SHA or state['test_time_update'] or state['step']!=8973:raise ValueError('wrong checkpoint protocol')
        for k,v in original.items():
            if k.startswith(('backbone.','text_encoder.')) and not torch.equal(v,state['student'][k]):raise ValueError('cannot share changed frozen encoder '+m+' '+k)
        model=copy.deepcopy(base);model.backbone=base.backbone;model.text_encoder=base.text_encoder
        attach_adapter(model,device,**adapter_kwargs(state['config']));model.load_state_dict(state['student'],strict=True)
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        models[m]=model;provenance[m]=dict(**r,step=state['step'],frozen_encoders_equal_source=True)
        del state;torch.cuda.empty_cache()
    del original
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'vendor/FlexHook/pretrained/roberta-base'))
    if tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('tokenizer mismatch')
    versions={m:{n:p._version for n,p in model.named_parameters()} for m,model in models.items()}
    arrays={}
    if not smoke:
        for m in models:
            p=dest/(m+'.npy');arrays[m]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype='float64',shape=(2355528,))
    allparity=[];video_receipts=[];margin_count={m:0 for m in models}
    for video,bank in meta['videos'].items():
        done=dest/'video_receipts'/(video+'.json')
        if not smoke and done.exists():video_receipts.append(json.loads(done.read_text()));continue
        begin=time.perf_counter();episode=DanceEpisode.__new__(DanceEpisode);episode.video=video
        episode.image_root=SOURCE/'KITTI/training/image_02'/video;episode.observations_by_frame=defaultdict(dict);episode.observations_by_track=defaultdict(list)
        for r in merge_kitti_class_splits(SOURCE_TRACKS/video/'car/predict.txt',SOURCE_TRACKS/video/'pedestrian/predict.txt'):
            t=r.frame_id-1;x,y,w,h=r.values[2:6];obs=Observation(t,r.track_id,(x,y,x+w,y+h),','.join(map(str,r.values)))
            episode.observations_by_frame[t][r.track_id]=obs;episode.observations_by_track[r.track_id].append(obs)
        episode.frames=sorted(episode.observations_by_frame);episode.track_index={i:{o.frame_id:j for j,o in enumerate(rows)} for i,rows in episode.observations_by_track.items()}
        episode.image_cache=OrderedDict();episode.image_cache_size=12;episode.width,episode.height=episode._read_resolution()
        episode.transform=T.Compose([T.Resize((224,672)),T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
        cache=FrozenFeatureCache(base,episode,tokenizer,device,max_bytes=512<<20)
        queries=[q for q in meta['episodes'] if q['video_id']==video];texts=[cache._text(q['query_text']) for q in queries]
        observations=bank['observations'];byframe=defaultdict(list)
        for j,(t,i) in enumerate(observations):byframe[t].append((j,i))
        frames=list(byframe);checkframes={frames[0],frames[len(frames)//2],frames[-1]}
        for t in frames[:1] if smoke else frames:
            idx=byframe[t];allowed={i for j,i in idx};candidates=list(episode.current_candidates(t,allowed));positions={i:j for j,i in idx}
            for a in range(0,len(candidates),8):
                chosen=candidates[a:a+8];prepared=prepared_inputs(cache,chosen,queries[0]['query_text'],t)
                with torch.no_grad():
                    for m,model in models.items():
                        for qa in range(0,len(queries),8):
                            qt=texts[qa:qa+8];value=query_margins(model,prepared,qt,batched=True)
                            if qa==0 and a==0 and t in checkframes:
                                isolated=query_margins(model,prepared,qt,batched=False)
                                error=float((value-isolated).abs().max());flips=int(((value>0)!=(isolated>0)).sum())
                                allparity.append(dict(model=m,video=video,frame=t,max_error=error,sign_flips=flips))
                                if error>1e-3 or flips:raise RuntimeError('source masked-query parity failed; preserve failure, use isolated protocol before resume')
                            # Same pre-existing isolated near-zero fallback as target inference.
                            near=(value.abs()<=1e-3).any(dim=0).nonzero().flatten().tolist()
                            for j in near:value[:,j]=query_margins(model,prepared,[qt[j]],batched=False)[:,0]
                            if not torch.isfinite(value).all():raise FloatingPointError('nonfinite source inference')
                            if not smoke:
                                cpu=value.cpu().numpy()
                                for j,i in enumerate(prepared['track_ids']):
                                    for k,q in enumerate(queries[qa:qa+8]):arrays[m][q['start']+positions[i]]=float(cpu[j,k])
                            margin_count[m]+=int(value.numel())
                if smoke:break
        torch.cuda.synchronize()
        if not smoke:
            for array in arrays.values():array.flush()
            record=dict(video=video,observations=len(observations),queries=len(queries),candidate_query_rows=len(observations)*len(queries),wall_seconds=time.perf_counter()-begin)
            atomic_json(done,record);video_receipts.append(record)
        print(json.dumps(dict(event='XQUERY_SOURCE_VIDEO',video=video,smoke=smoke,elapsed=time.perf_counter()-begin)),flush=True)
        if smoke:break
    torch.cuda.synchronize()
    if attempts or any(p._version!=versions[m][n] for m,model in models.items() for n,p in model.named_parameters()):raise RuntimeError('source inference changed parameters or opened Dance GT')
    result=dict(status='SOURCE_SMOKE_PASS' if smoke else 'ALL5_STUDENT_SOURCE_SCORES_SEALED',models=provenance,
        target_GT_open_attempts=len(attempts),student_updates=0,teacher_calls=0,precision='FP32',parities=allparity,
        wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        physical_gpu=os.environ.get('XQUERY_PHYSICAL_GPU'),GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'),
        videos=video_receipts,margin_count_new=margin_count,source_candidate_query_rows=2355528)
    if not smoke:
        for m in models:result['models'][m].update(array_path=str(dest/(m+'.npy')),array_sha256=file_sha(dest/(m+'.npy')))
    atomic_json(dest/('smoke_receipt.json' if smoke else 'scores_seal.json'),result)
    atomic_json(PUBLIC/('source_smoke_receipt.json' if smoke else 'source_scores_seal.json'),result)
    print(json.dumps(dict(event='XQUERY_SOURCE_WORKER_FINISHED',status=result['status'],seconds=result['wall_seconds'])),flush=True)

def launch():
    from scripts.launch_uda_workers import audit_gpus
    setup();require_plan();dest=RUN/'source';attempt=dest/'launchers'/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    while True:
        try:chosen,gpus,processes=audit_gpus(1,minimum_free_mib=20480);break
        except RuntimeError as e:print(json.dumps(dict(event='XQUERY_GPU_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    gpu=chosen[0];atomic_json(attempt/'admission.json',dict(selected=gpu,all_gpus=gpus,existing_processes=processes,max_GPUs=1,sharing_authorized=True,foreign_processes_stopped=0))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],XQUERY_PHYSICAL_GPU=str(gpu['physical_gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    jobs=[]
    for smoke in (True,False):
        if smoke and (dest/'smoke_receipt.json').exists():continue
        phase='smoke' if smoke else 'full';log=attempt/(phase+'.txt');begin=time.perf_counter()
        with log.open('w') as h:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',*(['--smoke'] if smoke else [])],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
            print(json.dumps(dict(event='XQUERY_SOURCE_CHILD_STARTED',phase=phase,pid=p.pid,physical_gpu=gpu['physical_gpu'],log=str(log))),flush=True)
            code=p.wait()
        jobs.append(dict(phase=phase,pid=p.pid,exit_code=code,wall_seconds=time.perf_counter()-begin,stdout=str(log)))
        atomic_json(attempt/'launcher_status.json',dict(status='FAILED' if code else 'RUNNING' if smoke else 'COMPLETE',jobs=jobs,wall_seconds=time.perf_counter()-start))
        if code:raise RuntimeError('source scoring failed: '+str(log))
    atomic_json(PUBLIC/'source_GPU_cost.json',dict(jobs=jobs,physical_gpu=gpu['physical_gpu'],actual_parent_wall_seconds=time.perf_counter()-start,
        GPU_process_hours=sum(r['wall_seconds'] for r in jobs)/3600,student_updates=0,teacher_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    worker(a.smoke) if a.worker else launch()
