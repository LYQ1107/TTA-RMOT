"""SportsMOT official-model P0 smoke, then explicitly gated full score export.

Default is P0 only. Full mode requires a separately published P0 receipt.
Frozen-encoder reuse must match the original real-image forward numerically.
"""
from argparse import ArgumentParser, Namespace
from collections import OrderedDict, defaultdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.multidomain_runtime import RUN, PUBLIC, VENDOR, CHECKPOINTS, require_plan, require_remaining_p0, guard
from scripts.multidomain_sports_prepare import TRACKS
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def config(checkpoint, folder):
    os.chdir(VENDOR);sys.path.insert(0,str(VENDOR));os.environ.setdefault('LOCAL_RANK','0')
    from config import get_config
    cfg = get_config(Namespace(cfg=str(VENDOR/'configs/infer/mcbyte-sports-infer-tiny.yaml'),
        opts=['data_root',str(RUN/'runtime/sports'),'val_data_root',str(RUN/'runtime/sports')],
        batch_size=8,val_batch_size=50,visual=None,text=None,dataset=None,
        freeze_text=False,freeze_visual=False,track_root=str(TRACKS),data_path='12',
        zip=False,cache_mode='part',pretrained=None,resume=str(checkpoint),
        accumulation_steps=None,use_checkpoint=False,disable_amp=True,amp_opt_level=None,
        output=str(folder),tag=None,eval=True,throughput=False,fused_window_process=False,
        fused_layernorm=False,optim=None,lre=None))
    return cfg


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['V2-best','Dance-best','LaMOT-best'],default='LaMOT-best')
    parser.add_argument('--cpu-strict-only',action='store_true')
    parser.add_argument('--full',action='store_true')
    args = parser.parse_args()
    require_plan()
    if args.full:
        require_remaining_p0()
        smoke=json.loads((RUN/'sports'/args.model/'smoke_receipt.json').read_text())
        if smoke['status']!='SMOKE_PASS' or not smoke['cost_gate_pass']:
            raise PermissionError('Own complete-video P0 must pass before full inference')
    import compat.sitecustomize
    import torch
    assets = json.loads((PUBLIC/'sports_assets.json').read_text())
    checkpoint, expected = ((Path(assets['checkpoint']['path']),assets['checkpoint']['sha256'])
        if args.model=='LaMOT-best' else CHECKPOINTS[args.model])
    if file_sha(checkpoint)!=expected:
        raise ValueError('Official weight SHA changed')
    attempts=guard([RUN/'runtime/sports'])
    folder=RUN/'sports'/args.model;folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();cfg=config(checkpoint,folder/'model_log')
    from models import build_model
    model=build_model(cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model']
    strict=model.load_state_dict(state,strict=True);del state
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    load_receipt=dict(status='STRICT_LOAD_PASS',model=args.model,checkpoint=str(checkpoint),
        sha256=expected,strict_load=str(strict),backbone=cfg.VISUAL,text_encoder=cfg.TEXT,
        target_config=str(VENDOR/'configs/infer/mcbyte-sports-infer-tiny.yaml'),
        target_config_sha256=file_sha(VENDOR/'configs/infer/mcbyte-sports-infer-tiny.yaml'),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        GT_open_attempts=len(attempts),seconds=time.perf_counter()-start)
    atomic_json(folder/'strict_load.json',load_receipt)
    if args.cpu_strict_only:
        print(json.dumps(load_receipt),flush=True);return
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory,device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model.to(device);versions={n:p._version for n,p in model.named_parameters()}
    from data.mydataloader_mix import RMOT_Dataset_mix
    ds=RMOT_Dataset_mix('test',cfg)
    if ds.tokenizer.encode('person',add_special_tokens=False)!=[5970]:
        raise ValueError('Canonical tokenizer mismatch; stop without remapping')
    meta=json.loads((RUN/'sports/metadata.json').read_text())
    selected=list(meta['videos']) if args.full else [next(iter(meta['videos']))]
    receipts=[]
    for video in selected:
        receipt=score_video(model,ds,meta,video,device,args.model,folder,expected,str(strict),attempts,versions,load_receipt,
            enforce_cost_gate=not args.full)
        receipts.append(receipt)
    if args.full:
        path=folder/'complete.npy'
        values=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(meta['candidate_query_rows'],))
        for video in selected:
            scored=np.load(folder/'video_scores'/(video+'.npy'),mmap_mode='r')
            for i,q in enumerate(q for q in meta['episodes'] if q['video_id']==video):
                values[q['start']:q['stop']]=scored[i]
        if not np.isfinite(values).all():raise ValueError('Full Sports score bank incomplete')
        values.flush();mask=np.load(RUN/'sports/historical.npy')
        np.save(folder/'historical.npy',values[mask])
        result=dict(status='COMPLETE_SCORES_SEALED',model=args.model,checkpoint=str(checkpoint),
            checkpoint_sha256=expected,strict_load=str(strict),complete_path=str(path),
            complete_sha256=file_sha(path),complete_rows=len(values),
            historical_path=str(folder/'historical.npy'),historical_sha256=file_sha(folder/'historical.npy'),
            historical_rows=int(mask.sum()),videos=receipts,GT_open_attempts=len(attempts),
            wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            physical_gpu=os.environ.get('MULTIDOMAIN_PHYSICAL_GPU'),optimizer_updates=0,teacher_calls=0)
        atomic_json(folder/'scores_seal.json',result)
    else:
        result=dict(receipts[0],wall_seconds=time.perf_counter()-start)
        atomic_json(folder/'smoke_receipt.json',result)
    print(json.dumps(dict(event='SPORTS_WORKER_FINISHED',model=args.model,full=args.full,
        status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)


def score_video(model,ds,meta,video,device,model_name,folder,expected,strict,attempts,versions,load_receipt,*,enforce_cost_gate=True):
    import torch
    from flex_tta.runtime.dance_episode import DanceEpisode,Observation
    from flex_tta.runtime.frozen_features import FrozenFeatureCache
    from flex_uda.model.flexhook_uda import prepared_inputs
    from flex_uda.evaluation import query_margins
    from PIL import Image
    receipt_path=folder/'video_receipts'/(video+'.json')
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old['checkpoint_sha256']!=expected or file_sha(old['scores_path'])!=old['scores_sha256']:
            raise ValueError('Preserve changed Sports video receipt')
        return old
    start=time.perf_counter();bank=meta['videos'][video]
    queries=[q for q in meta['episodes'] if q['video_id']==video]
    official_texts=set(ds.target_expressions[video])
    if official_texts!={q['query_text'] for q in queries}:
        raise ValueError('Official loader/text-only projection mismatch')
    # The target's original tokenization must agree, without new truncation fixes.
    for q in queries:
        encoded=ds.tokenizer([q['query_text']],padding='max_length',return_tensors='pt',max_length=25)
        other=ds.tokenizer([q['query_text']],padding='max_length',truncation=True,return_tensors='pt',max_length=25)
        if not torch.equal(encoded['input_ids'],other['input_ids']):
            raise ValueError('Stock tokenizer semantics differ; stop this combination')
    lookup={}
    for j,key in enumerate(ds.data_keys):
        parts=key.split('#')
        if parts[0]!=video:continue
        k='#'.join(parts[:2]);indices=ds.sample_dict[key]
        frame=int(ds.data[k]['bbox'][indices[-1]][0]);lookup.setdefault((frame,int(parts[1])),j)

    class SportsEpisode(DanceEpisode):
        def _image(self,frame_id):
            if frame_id in self.image_cache:
                value=self.image_cache.pop(frame_id);self.image_cache[frame_id]=value;return value
            with Image.open(self.image_root/f'{frame_id:06d}.jpg') as image:
                value=self.transform(image)
            self.image_cache[frame_id]=value
            while len(self.image_cache)>self.image_cache_size:self.image_cache.popitem(last=False)
            return value
        def geometry_inputs(self,candidates,device):
            positions=[];boxes=[];ids=[]
            for track,history in candidates:
                b=[];p=[]
                for obs in history:
                    x1,y1,x2,y2=obs.bbox_xyxy
                    value=torch.tensor([x1/self.width,y1/self.height,x2/self.width,y2/self.height],dtype=torch.float32)
                    value[2:]=value[2:]-value[:2];value[:2]=value[:2]+value[2:]/2
                    b.append(value);p.append(self._grid(obs.bbox_xyxy))
                positions.append(torch.stack(p));boxes.append(torch.stack(b));ids.append(track)
            return torch.stack(positions).to(device),torch.stack(boxes).to(device),ids

    episode=SportsEpisode.__new__(SportsEpisode);episode.video=video
    episode.image_root=Path(bank['image_root']);episode.width=bank['width'];episode.height=bank['height']
    episode.observations_by_frame=defaultdict(dict);episode.observations_by_track=defaultdict(list)
    for line,(frame,track) in zip(bank['lines'],bank['observations']):
        x,y,w,h=list(map(float,line.split(',')))[2:6]
        obs=Observation(frame,track,(x,y,x+w,y+h),line)
        episode.observations_by_frame[frame][track]=obs;episode.observations_by_track[track].append(obs)
    episode.frames=sorted(episode.observations_by_frame)
    episode.track_index={i:{o.frame_id:j for j,o in enumerate(rows)} for i,rows in episode.observations_by_track.items()}
    episode.image_cache=OrderedDict();episode.image_cache_size=12;episode.transform=ds.transform
    cache=FrozenFeatureCache(model,episode,ds.tokenizer,device,max_bytes=512<<20)
    texts=[cache._text(q['query_text']) for q in queries]
    checks={episode.frames[0],episode.frames[len(episode.frames)//2],episode.frames[-1]}
    positions={tuple(o):j for j,o in enumerate(bank['observations'])}
    scores=np.full((len(queries),len(positions)),np.nan,dtype=np.float64);parities=[]
    torch.cuda.synchronize();compute_start=time.perf_counter()
    for frame in episode.frames:
        candidates=list(episode.current_candidates(frame))
        for a in range(0,len(candidates),8):
            chosen=candidates[a:a+8];prepared=prepared_inputs(cache,chosen,queries[0]['query_text'],frame)
            if a==0 and frame in checks:
                key=next(((frame,i) for i in prepared['track_ids'] if (frame,i) in lookup),None)
                if key:
                    item=ds[lookup[key]];images,grid,boxes,qtext,token,mask=item[:6]
                    one=prepared_inputs(cache,[(key[1],episode.history(key[1],frame))],qtext[0],frame)
                    ge=float((one['positions'][0].cpu()-grid).abs().max())
                    be=float((one['boxes'][0].cpu()-boxes).abs().max())
                    if ge>1e-6 or be!=0:raise ValueError('Official Sports grid/box mismatch')
                    model.sample_expression_num=len(qtext)
                    with torch.no_grad(): native,_=model(images.unsqueeze(0).to(device),grid.unsqueeze(0).to(device),boxes.unsqueeze(0).to(device),token.unsqueeze(0).to(device),mask.unsqueeze(0).to(device))
                    native=native[:,-1,:,1]-native[:,-1,:,0]
                    cached=query_margins(model,one,[cache._text(s) for s in qtext],batched=True)
                    error=float((native-cached).abs().max());flips=int(((native>0)!=(cached>0)).sum())
                    parities.append(dict(frame=frame,grid_error=ge,box_error=be,margin_error=error,sign_flips=flips))
                    if error>1e-3 or flips:raise RuntimeError('Official Sports full-forward parity failed')
            with torch.no_grad():
                for qa in range(0,len(queries),8):
                    qt=texts[qa:qa+8];value=query_margins(model,prepared,qt,batched=True)
                    if a==0 and frame in checks:
                        isolated=query_margins(model,prepared,qt,batched=False)
                        if float((value-isolated).abs().max())>1e-3 or int(((value>0)!=(isolated>0)).sum()):
                            raise RuntimeError('Official query isolation parity failed')
                    for j in (value.abs()<=1e-3).any(dim=0).nonzero().flatten().tolist():
                        value[:,j]=query_margins(model,prepared,[qt[j]],batched=False)[:,0]
                    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite frozen margin')
                    cpu=value.cpu().numpy()
                    for j,track in enumerate(prepared['track_ids']):
                        scores[qa:qa+len(qt),positions[frame,track]]=cpu[j]
    torch.cuda.synchronize();seconds=time.perf_counter()-compute_start
    if not np.isfinite(scores).all() or attempts or any(p._version!=versions[n] for n,p in model.named_parameters()):
        raise RuntimeError('Incomplete smoke, GT access or weights changed')
    path=folder/'video_scores'/(video+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,scores)
    estimate=seconds/max(scores.size,1)*meta['candidate_query_rows']/3600
    status=('SMOKE_PASS' if estimate<=6 else 'STOP_OVER_6_GPU_HOURS') if enforce_cost_gate else 'VIDEO_SCORED'
    result=dict(status=status,model=model_name,video=video,
        complete_video=True,candidate_query_rows=int(scores.size),prediction_rows=int((scores>0).sum()),
        checkpoint_sha256=expected,strict_load=str(strict),dataloader='data.mydataloader_mix.RMOT_Dataset_mix(test)',
        stock_target_config=load_receipt['target_config'],original_infer_sh_command='mcbyte-sports-infer-tiny.yaml + LaMOT-best + --data-path 12 --val-batch-size 50 --eval; FP32 per PREREG',
        dataset_class_token='0; no category token added',tokenizer_person_ids=[5970],FP32=True,
        frames=bank['frame_count'],candidate_observations=len(positions),compute_seconds=seconds,
        estimated_full_scoring_GPU_hours=estimate,cost_gate_pass=estimate<=6,
        wall_seconds=time.perf_counter()-start,physical_gpu=os.environ.get('MULTIDOMAIN_PHYSICAL_GPU'),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        parity=parities,GT_open_attempts=0,optimizer_updates=0,teacher_calls=0,
        scores_path=str(path),scores_sha256=file_sha(path),cache=cache.stats)
    atomic_json(receipt_path,result)
    print(json.dumps(dict(event='SPORTS_VIDEO_COMPLETE',**result)),flush=True)
    if enforce_cost_gate and estimate>6:raise SystemExit(2)
    return result


if __name__=='__main__':main()
