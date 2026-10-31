#!/usr/bin/env python3
"""Frozen V2 margins at Naive sampling frames only; Train inputs, no labels."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import compat.sitecustomize
import numpy as np
import torch
from transformers import RobertaTokenizerFast
from flex_uda.data.manifests import DANCE,SOURCE,read_jsonl,official_splits
from flex_uda.data.train2val import ASSETS
from flex_uda.training import CHECKPOINT,SOURCE_SHA,protect_target_annotations
from flex_uda.train2val_isolation import install_train_only_audit
from flex_uda.teacher.isolation import restrict_filesystem
from flex_uda.model.flexhook_uda import prepared_inputs,forward_uda
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.frozen_features import FrozenFeatureCache
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def run(videos,output):
    started=time.perf_counter();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    allrows=read_jsonl(ASSETS/'runtime_train.jsonl');train=official_splits()['dance']['train']
    if not set(videos)<=set(train):raise PermissionError('Train videos only')
    grouped=defaultdict(list)
    for r in allrows:
        if r['video_id'] in videos:grouped[r['video_id']].append(r)
    attempts=protect_target_annotations()
    inputs_audit=install_train_only_audit(ASSETS,output)
    if file_sha(CHECKPOINT)!=SOURCE_SHA:raise ValueError('source checkpoint changed')
    device=torch.device('cuda:0');props=torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction((8192<<20)/props.total_memory,device)
    torch.empty(1,device=device).zero_();torch.cuda.synchronize()
    vendor=ROOT/'vendor/FlexHook';pretrained=[p.resolve() for p in (vendor/'pretrained').iterdir()]
    isolation=restrict_filesystem([vendor,*pretrained,ROOT/'compat',ROOT/'flex_tta',ROOT/'flex_uda',ROOT/'scripts',
        ASSETS,CHECKPOINT,*(DANCE/'DanceTrack/training/image_02'/v for v in videos),
        Path(sys.prefix),Path(sys.base_prefix),Path(site.getusersitepackages()),'/usr','/lib','/lib64','/bin','/etc','/proc','/sys'],
        [output,'/dev','/tmp'])
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,model,_=_load_model(vendor,CHECKPOINT,DANCE,ASSETS/'tracker_outputs',output/'model_log',device,24,strict_checkpoint=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    tokenizer=RobertaTokenizerFast.from_pretrained(str(vendor/'pretrained/roberta-base'))
    if tokenizer.encode('person',add_special_tokens=False)!=[5970]:raise ValueError('tokenizer mismatch')
    records=[]
    for video,queries in sorted(grouped.items()):
        done=output/'receipts'/(video+'.json')
        if done.exists():
            r=json.loads(done.read_text())
            for p,sha in r['files_sha256'].items():
                if file_sha(p)!=sha:raise ValueError('changed sparse source scores')
            records.append(r);continue
        begin=time.perf_counter();episode=DanceEpisode(DANCE,ASSETS/'tracker_outputs',video)
        cache=FrozenFeatureCache(model,episode,tokenizer,device,max_bytes=512<<20)
        first=queries[0];a=first['frame_indices']['start'];b=first['frame_indices']['stop']-1
        anchors={a+round((b-a)*q) for q in (.1,.3,.5,.7,.9)}
        frames=sorted({episode.frames[0]}|{t+d for t in anchors for d in (-1,0,1)})
        buffers={r['expression_id']:[] for r in queries}
        # Original isolated-query FP32 decode, shared frozen backbone cache.
        # No changed attention shape or new candidate sampling heuristic.
        with torch.no_grad():
            for frame in frames:
                candidates=list(episode.current_candidates(frame))
                for row in queries:
                    ids=[];values=[]
                    for start in range(0,len(candidates),24):
                        prepared=prepared_inputs(cache,candidates[start:start+24],row['query_text'],frame)
                        margin=forward_uda(model,prepared,return_features=False)['margin'].float()
                        if not torch.isfinite(margin).all():raise FloatingPointError('nonfinite source margins')
                        ids.extend(prepared['track_ids']);values.extend(margin.tolist())
                    buffers[row['expression_id']].append(dict(frame=frame,ids=ids,source=values,adapted=values))
        hashes={}
        for query,rows in buffers.items():
            path=output/'results'/video/query/'candidate_margins.jsonl.gz';path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.partial.gz')
            with gzip.open(temporary,'wt') as handle:
                for r in rows:handle.write(json.dumps(r)+'\n')
            temporary.replace(path);hashes[str(path)]=file_sha(path)
        record=dict(video=video,frames=frames,queries=len(queries),files_sha256=hashes,
            worker_seconds=time.perf_counter()-begin,source_checkpoint_sha256=SOURCE_SHA,
            target_GT_open_attempts=len(attempts),forbidden_input_open_attempts=len(inputs_audit),
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),precision='FP32_ISOLATED_QUERY',updates=0)
        atomic_json(done,record);records.append(record)
        print(json.dumps(dict(event='TRAIN2VAL_SPARSE_VIDEO_COMPLETE',video=video,seconds=record['worker_seconds'])),flush=True)
    if attempts or inputs_audit:raise RuntimeError('source-score input isolation violation')
    atomic_json(output/('worker_'+str(os.getpid())+'.json'),dict(status='SEALED_TRAIN_SPARSE_SCORES',videos=videos,
        worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        source_checkpoint_sha256=SOURCE_SHA,isolation=isolation,target_GT_open_attempts=0,forbidden_input_open_attempts=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--videos',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.videos,a.output)
