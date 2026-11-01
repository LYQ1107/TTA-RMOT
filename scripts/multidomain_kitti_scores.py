"""Strict official checkpoints + official KITTI test loader, FP32, no adaptation.

Frozen encoder reuse is checked against the unchanged vendor __getitem__/forward.
Only missing source tails are written; historical source values stay bit-exact.
"""
from argparse import ArgumentParser,Namespace
from collections import OrderedDict,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import *


def config_for(checkpoint,folder):
    os.chdir(VENDOR);sys.path.insert(0,str(VENDOR));os.environ.setdefault('LOCAL_RANK','0')
    from config import get_config
    cli=Namespace(cfg=str(VENDOR/'configs/infer/kitti2.yaml'),opts=['data_root',str(RUN/'runtime/kitti')],
        batch_size=8,val_batch_size=8,visual='rope-swin-tiny',text='roberta',dataset='kitti-2',
        freeze_text=False,freeze_visual=False,track_root=str(TRACKS),data_path=None,zip=False,cache_mode='part',
        pretrained=None,resume=str(checkpoint),accumulation_steps=None,use_checkpoint=False,disable_amp=True,
        amp_opt_level=None,output=str(folder),tag=None,eval=True,throughput=False,
        fused_window_process=False,fused_layernorm=False,optim=None,lre=None)
    cfg=get_config(cli)
    cfg.defrost();cfg.sample_expression_num=8;cfg.val_sample_expression_num=8;cfg.freeze()
    return cfg


def worker(name,smoke):
    import compat.sitecustomize
    import torch
    from flex_tta.runtime.dance_episode import DanceEpisode,Observation
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.model.flexhook_uda import prepared_inputs
    from flex_uda.evaluation import query_margins
    require_plan();attempts=guard();started=time.perf_counter()
    if not smoke:
        gate=RUN/'p0_resume_publication.json'
        if not gate.exists() or not json.loads(gate.read_text())['remote_verified']:
            raise PermissionError('P0_SMOKE_AWAITING_PUBLICATION: preserve completed smoke video; publish inventory/costs before P1')
    meta=json.loads((RUN/'kitti/metadata.json').read_text());folder=RUN/'kitti'/name
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint,expected=CHECKPOINTS[name]
    if file_sha(checkpoint)!=expected:raise ValueError('Official checkpoint SHA changed')
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=config_for(checkpoint,folder/'model_log')
    from models import build_model
    from data.mydataloader import RMOT_Dataset
    model=build_model(cfg).to(device)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    loaded=model.load_state_dict(state,strict=True);del state
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    versions={n:p._version for n,p in model.named_parameters()}
    ds=RMOT_Dataset('test',cfg)
    if ds.tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('Canonical tokenizer regression')
    # Index original vendor samples solely for real-image forward parity checks.
    lookup={}
    for k,key in enumerate(ds.data_keys):
        parts=key.split('_');trackkey='_'.join(parts[:2]);indices=ds.sample_dict[key]
        frame=int(ds.data[trackkey]['bbox'][indices[-1]][0]);lookup.setdefault((parts[0],frame,int(parts[1])),k)

    class OfficialGeometryEpisode(DanceEpisode):
        def geometry_inputs(self,candidates,device):
            positions=[];boxes=[];ids=[]
            for track,history in candidates:
                b=[];p=[]
                for obs in history:
                    x1,y1,x2,y2=obs.bbox_xyxy
                    bbox=[x1/self.width,y1/self.height,x2/self.width,y2/self.height]
                    # Match vendor float32 cxcywh operation order, not a new preprocessing.
                    value=torch.tensor(bbox,dtype=torch.float32)
                    value[2:]=value[2:]-value[:2];value[:2]=value[:2]+value[2:]/2
                    b.append(value);p.append(self._grid(obs.bbox_xyxy))
                positions.append(torch.stack(p));boxes.append(torch.stack(b));ids.append(track)
            return torch.stack(positions).to(device),torch.stack(boxes).to(device),ids

    path=folder/'complete.npy'
    fresh=not path.exists()
    scores=np.lib.format.open_memmap(path,mode='w+' if fresh else 'r+',dtype='float64',shape=(meta['candidate_query_rows'],))
    if fresh:
        scores[:]=np.nan
        if name=='V2-best':
            old=np.load(OLD_SOURCE/'FROZEN.npy',mmap_mode='r')
            hist=np.load(RUN/'kitti/historical.npy');positions=np.load(RUN/'kitti/historical_indices.npy')
            scores[hist]=old[positions]
        scores.flush()
    parity=[];receipts=[];margin_count=0;smoke_rows=0;smoke_seconds=0
    for video,bank in meta['videos'].items():
        if smoke and video!=next(iter(meta['videos'])):break
        done=folder/'video_receipts'/(video+'.json')
        if done.exists():receipts.append(json.loads(done.read_text()));continue
        begin=time.perf_counter();episode=OfficialGeometryEpisode.__new__(OfficialGeometryEpisode)
        episode.video=video;episode.image_root=Path(bank['image_root'])
        episode.observations_by_frame=defaultdict(dict);episode.observations_by_track=defaultdict(list)
        for line,(t,i) in zip(bank['lines'],bank['observations']):
            row=list(map(float,line.split(',')));x,y,w,h=row[2:6]
            o=Observation(t,i,(x,y,x+w,y+h),line)
            episode.observations_by_frame[t][i]=o;episode.observations_by_track[i].append(o)
        episode.frames=sorted(episode.observations_by_frame)
        episode.track_index={i:{o.frame_id:j for j,o in enumerate(rows)} for i,rows in episode.observations_by_track.items()}
        episode.image_cache=OrderedDict();episode.image_cache_size=12;episode.width,episode.height=episode._read_resolution()
        episode.transform=ds.transform
        cache=FrozenFeatureCache(model,episode,ds.tokenizer,device,max_bytes=512<<20)
        queries=[q for q in meta['episodes'] if q['video_id']==video];texts=[cache._text(q['query_text']) for q in queries]
        obs=bank['observations'];byframe=defaultdict(list)
        for j,(t,i) in enumerate(obs):
            if name!='V2-best' or smoke or bank['historical_positions'][j]<0:byframe[t].append((j,i))
        frames=sorted(byframe);checks={frames[0],frames[len(frames)//2],frames[-1]}
        compute_begin=time.perf_counter();historical_errors=[];computed_here=0;resumed_complete_frames=0
        for t in frames:
            idx=byframe[t];allowed={i for j,i in idx};candidates=list(episode.current_candidates(t,allowed));position={i:j for j,i in idx}
            if name!='V2-best' and not smoke:
                positions=np.asarray([j for j,i in idx])
                if all(np.isfinite(scores[q['start']+positions]).all() for q in queries):
                    resumed_complete_frames+=1;continue
            for a in range(0,len(candidates),8):
                chosen=candidates[a:a+8];prepared=prepared_inputs(cache,chosen,queries[0]['query_text'],t)
                if a==0 and t in checks:
                    # Exact vendor loader geometry/tokenization and original model forward.
                    key=next(((video,t,i) for i in prepared['track_ids'] if (video,t,i) in lookup),None)
                    if key is not None:
                        item=ds[lookup[key]];images,grid,boxes,qtext,token,mask=item[:6]
                        candidate=[(key[2],episode.history(key[2],t))];p0=prepared_inputs(cache,candidate,qtext[0],t)
                        grid_error=float((p0['positions'][0].cpu()-grid).abs().max())
                        box_error=float((p0['boxes'][0].cpu()-boxes).abs().max())
                        if grid_error>1e-6 or box_error!=0:raise ValueError('Official dataloader preprocessing mismatch')
                        model.sample_expression_num=len(qtext)
                        with torch.no_grad():native,_=model(images.unsqueeze(0).to(device),grid.unsqueeze(0).to(device),boxes.unsqueeze(0).to(device),token.unsqueeze(0).to(device),mask.unsqueeze(0).to(device))
                        native=native[:,-1,:,1]-native[:,-1,:,0]
                        cached=query_margins(model,p0,[cache._text(s) for s in qtext],batched=True)
                        err=float((native-cached).abs().max());flips=int(((native>0)!=(cached>0)).sum())
                        parity.append(dict(video=video,frame=t,vendor_forward_max_error=err,sign_flips=flips,grid_error=grid_error,box_error=box_error))
                        if err>1e-3 or flips:raise RuntimeError('STOP: official forward/cache parity failed')
                with torch.no_grad():
                    for qa in range(0,len(queries),8):
                        qt=texts[qa:qa+8];value=query_margins(model,prepared,qt,batched=True)
                        if qa==0 and a==0 and t in checks:
                            isolated=query_margins(model,prepared,qt,batched=False)
                            err=float((value-isolated).abs().max());flips=int(((value>0)!=(isolated>0)).sum())
                            if err>1e-3 or flips:raise RuntimeError('STOP: masked-query/isolated parity failed')
                        for j in (value.abs()<=1e-3).any(dim=0).nonzero().flatten().tolist():
                            value[:,j]=query_margins(model,prepared,[qt[j]],batched=False)[:,0]
                        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen inference')
                        cpu=value.cpu().numpy()
                        for j,i in enumerate(prepared['track_ids']):
                            for k,q in enumerate(queries[qa:qa+8]):
                                dest=q['start']+position[i]
                                if name=='V2-best' and bank['historical_positions'][position[i]]>=0:
                                    historical_errors.append(abs(float(cpu[j,k])-scores[dest]))
                                else:scores[dest]=float(cpu[j,k])
                        margin_count+=int(value.numel())
                        computed_here+=int(value.numel())
            # Frame is the unfinished-unit recovery boundary; preserve completed work on SIGTERM.
            scores.flush()
        torch.cuda.synchronize();compute=time.perf_counter()-compute_begin;scores.flush()
        record=dict(video=video,model=name,full_candidate_query_rows=len(obs)*len(queries),
            newly_computed_margin_count=computed_here,resumed_complete_frames_skipped=resumed_complete_frames,
            historical_values_written=0 if name=='V2-best' else None,
            historical_recompute_max_abs_error=max(historical_errors,default=None),
            compute_seconds=compute,wall_seconds=time.perf_counter()-begin,cache=cache.stats,
            complete_values_finite=all(np.isfinite(scores[q['start']:q['stop']]).all() for q in queries))
        if not record['complete_values_finite']:raise ValueError('Video scores incomplete')
        atomic_json(done,record);receipts.append(record)
        smoke_rows=record['newly_computed_margin_count'];smoke_seconds=compute
        print(json.dumps(dict(event='OFFICIAL_KITTI_VIDEO_COMPLETE',**record)),flush=True)
    if any(p._version!=versions[n] for n,p in model.named_parameters()) or attempts:
        raise RuntimeError('Weights changed or GT opened')
    result=dict(status='SMOKE_PASS' if smoke else 'COMPLETE_SCORES_SEALED',model=name,
        checkpoint=str(checkpoint),checkpoint_sha256=expected,strict_load=str(loaded),
        target_config=str(VENDOR/'configs/infer/kitti2.yaml'),target_config_sha256=file_sha(VENDOR/'configs/infer/kitti2.yaml'),
        official_cli_overrides=dict(visual='rope-swin-tiny',text='roberta'),
        category_tokens='No added class token; vendor KITTI parser retains car/ped track ID merge; all expressions scored on all candidates',
        precision='FP32',tf32=False,optimizer_updates=0,teacher_calls=0,GT_open_attempts=len(attempts),
        physical_gpu=os.environ.get('MULTIDOMAIN_PHYSICAL_GPU'),GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'),
        wall_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        new_margin_computations=margin_count,videos=receipts,official_loader_parity=parity)
    if smoke:
        estimate=smoke_seconds/max(smoke_rows,1)*meta['candidate_query_rows']/3600
        result.update(estimated_full_scoring_GPU_hours=estimate,smoke_video_complete=True,
            cost_gate_pass=estimate<=6,remaining_source_tails=meta['missing_tail_candidate_query_rows'] if name=='V2-best' else None)
        atomic_json(folder/'smoke_receipt.json',result)
        if estimate>6:raise RuntimeError('STOP: >6 GPU-hour estimate; await user decision')
    else:
        if not np.isfinite(scores).all():raise ValueError('Full array incomplete')
        hist=np.load(RUN/'kitti/historical.npy');history=scores[hist]
        np.save(folder/'historical.npy',history)
        if name=='V2-best':
            old=np.load(OLD_SOURCE/'FROZEN.npy',mmap_mode='r');idx=np.load(RUN/'kitti/historical_indices.npy')
            if not np.array_equal(history,old[idx]):raise ValueError('STOP: historical source scores not bit-exact')
            result['historical_score_bit_exact']=True
        result.update(complete_path=str(path),complete_sha256=file_sha(path),complete_rows=len(scores),
            historical_path=str(folder/'historical.npy'),historical_sha256=file_sha(folder/'historical.npy'),historical_rows=len(history))
        atomic_json(folder/'scores_seal.json',result)
    print(json.dumps(dict(event='OFFICIAL_KITTI_WORKER_FINISHED',model=name,smoke=smoke,status=result['status'],seconds=result['wall_seconds'],estimated_GPU_hours=result.get('estimated_full_scoring_GPU_hours'))),flush=True)


def launch():
    from scripts.launch_uda_workers import audit_gpus
    require_plan()
    if not (RUN/'kitti/metadata.json').exists():prepare_kitti()
    pending=[m for m in CHECKPOINTS if not (RUN/'kitti'/m/'scores_seal.json').exists()]
    if not pending:return
    chosen,gpus,processes=audit_gpus(len(pending),minimum_free_mib=20480)
    attempt=RUN/'kitti/launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'admission.json',dict(selected=chosen,all_gpus=gpus,existing_processes=processes,
        max_GPUs=2,sharing_authorized=True,no_foreign_process_stopped=True))
    start=time.perf_counter();jobs=[]
    # One blocking parent owns both workers through smoke then unfinished videos.
    for phase in ('smoke','full'):
        children=[]
        for name,gpu in zip(pending,chosen):
            folder=RUN/'kitti'/name
            if (folder/('smoke_receipt.json' if phase=='smoke' else 'scores_seal.json')).exists():continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],MULTIDOMAIN_PHYSICAL_GPU=str(gpu['physical_gpu']),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            log=attempt/(name+'_'+phase+'.txt');handle=log.open('x');begin=time.perf_counter()
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',name,*(['--smoke'] if phase=='smoke' else [])],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((p,handle,name,begin,log,gpu))
            print(json.dumps(dict(event='OFFICIAL_KITTI_CHILD_STARTED',phase=phase,model=name,pid=p.pid,physical_gpu=gpu['physical_gpu'],stdout=str(log))),flush=True)
        codes=[]
        def finish(child):
            p,h,name,begin,log,gpu=child;code=p.wait();h.close()
            return dict(phase=phase,model=name,physical_gpu=gpu['physical_gpu'],exit_code=code,
                wall_seconds=time.perf_counter()-begin,stdout=str(log),pid=p.pid)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(finish,c) for c in children]):
                job=future.result();codes.append(job['exit_code']);jobs.append(job)
                atomic_json(attempt/'launcher_status.json',dict(status='FAILED' if any(codes) else 'RUNNING',jobs=jobs,wall_seconds=time.perf_counter()-start))
                print(json.dumps(dict(event='OFFICIAL_KITTI_CHILD_FINISHED',**job)),flush=True)
                if job['exit_code']:print(Path(job['stdout']).read_text()[-10000:],flush=True)
        if any(codes):raise RuntimeError('Stop phase; preserve healthy completed video receipts and investigate first failure')
    atomic_json(attempt/'launcher_status.json',dict(status='COMPLETE',jobs=jobs,wall_seconds=time.perf_counter()-start))
    atomic_json(PUBLIC/'KITTI_GPU_cost.json',dict(jobs=jobs,maximum_simultaneous_GPUs=2,wall_seconds=time.perf_counter()-start,
        GPU_process_hours=sum(r['wall_seconds'] for r in jobs)/3600,teacher_calls=0,optimizer_updates=0))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--worker',choices=list(CHECKPOINTS));p.add_argument('--smoke',action='store_true');a=p.parse_args()
    worker(a.worker,a.smoke) if a.worker else launch()
