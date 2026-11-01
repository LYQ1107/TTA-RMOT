"""Verify inputs, reproduce old X0/X1 and freeze teacher-only manifests."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *
from flex_uda.data.manifests import write_jsonl

def prepare():
    setup();require_plan();attempts=deny_annotations();start=time.perf_counter()
    meta,arrays,_=inputs(True);source,src_arrays=source_inputs(True);lines=v2.candidate_lines(meta)
    target=list(cs.video_banks(meta));proof=[]
    for m,array in arrays.items():
        masks={d:np.zeros(len(array),bool) for d in ('X0_C3','X1_C3_CAUSAL')}
        for video,queries,tracks,frames,ids,unique,inverse in target:
            matrix=np.stack([array[q['start']:q['stop']] for q in queries]);a,b,_=raw_decodes(matrix,inverse)
            for j,q in enumerate(queries):
                masks['X0_C3'][q['start']:q['stop']]=a[j];masks['X1_C3_CAUSAL'][q['start']:q['stop']]=b[j]
        for d,mask in masks.items():
            name=m+'__'+d;seal=v2.write_selection(name,mask,meta,lines,protocol='BYTE_REPRODUCED_XQUERY_CONTROL')
            for e in seal['episodes']:
                old=OLD/'evaluation'/name/'results'/e['video']/e['expression']/'predict.txt'
                if file_sha(old)!=e['prediction_sha256']:raise ValueError('STOP: X0/X1 byte mismatch '+str(old))
            proof.append(dict(method=name,files=len(seal['episodes']),rows=seal['prediction_rows'],old_path=str(OLD/'evaluation'/name)))
    atomic_json(RUN/'reproduction.json',dict(status='PASS',sets=12,files=5100,methods=proof,GT_open_attempts=len(attempts)))
    atomic_json(PUBLIC/'reproduction.json',json.loads((RUN/'reproduction.json').read_text()))
    manifest=[];source_budget=[]
    def add(dataset,video,queries,byframe,image_root,modes):
        images=sorted(int(p.stem) for p in Path(image_root).glob('*.png'));first,last=images[0],images[-1]
        for mode,fractions in modes:
            selected=sampled_frames(first,last,byframe,fractions);unique=sorted(set(selected))
            planned=sum(len(byframe[t])*len(queries) for t in unique);cap=dataset=='KITTI' and planned>50000
            if dataset=='KITTI':source_budget.append(dict(video=video,planned_before_cap=planned,truncated=cap,cap_track_ids=40 if cap else None))
            for t in unique:
                candidates=sorted(byframe[t],key=lambda r:r['track_id']);candidates=candidates[:40] if cap else candidates
                image=Path(image_root)/f'{t:06d}.png'
                manifest.append(dict(dataset=dataset,video_id=video,gate_mode=mode,frame_id=t,anchor_slots=[i for i,f in enumerate(selected) if f==t],
                    full_video_first=first,full_video_last=last,image_path=str(image),image_sha256=file_sha(image),
                    candidates=candidates,queries=[dict(query_id=q['expression_id'],query_text=q['query_text']) for q in queries],truncated=cap))
    for video,queries,tracks,frames,ids,unique,inverse in target:
        by=defaultdict(list)
        for t,rows in tracks.items():
            for i,line in rows.items():by[int(t)].append(dict(track_id=int(i),bbox_xywh=list(map(float,line.split(',')))[2:6],MOT_row=line))
        add('Dance',video,queries,by,queries[0]['image_root'],[('normal',NORMAL),('early',EARLY)])
    for video,bank in sorted(source['videos'].items()):
        by=defaultdict(list);queries=sorted([q for q in source['episodes'] if q['video_id']==video],key=lambda q:q['expression_id'])
        for (t,i),line in zip(bank['observations'],bank['lines']):by[int(t)].append(dict(track_id=int(i),bbox_xywh=list(map(float,line.split(',')))[2:6],MOT_row=line))
        add('KITTI',video,queries,by,SOURCE/'KITTI/training/image_02'/video,[('normal',NORMAL)])
    path=RUN/'teacher/manifest.jsonl';write_jsonl(path,manifest)
    plan=dict(status='GT_FREE_MANIFEST_FROZEN_AFTER_X0_X1_GATE',manifest_path=str(path),manifest_sha256=file_sha(path),
        frames=len(manifest),planned_logical_calls=sum(len(r['queries'])*len(r['candidates']) for r in manifest),
        planned_by_dataset_mode={dataset+'_'+mode:sum(len(r['queries'])*len(r['candidates']) for r in manifest if r['dataset']==dataset and r['gate_mode']==mode) for dataset,mode in [('Dance','normal'),('Dance','early'),('KITTI','normal')]},
        source_budget=source_budget,GT_open_attempts=len(attempts),CPU_seconds=time.perf_counter()-start,
        backend_sha256=file_sha(ROOT/'flex_uda/teacher/yestrack_backend.py'),builder_sha256=file_sha(ROOT/'flex_uda/teacher/visual_input_builder.py'),
        model_revision='89644892e4d85e24eaac8bacfd4f463576704203',zero_E_policy='EMPTY_NO_TEACHER_EVIDENCE_USER_AUTHORIZED')
    if attempts:raise RuntimeError('annotation opened in preparation')
    atomic_json(RUN/'teacher_plan.json',plan);atomic_json(PUBLIC/'teacher_plan.json',plan)
    print(json.dumps(dict(event='GATED_PREPARE_PASS',**plan)),flush=True)

if __name__=='__main__':prepare()
