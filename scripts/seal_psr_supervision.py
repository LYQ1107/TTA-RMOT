#!/usr/bin/env python3
"""Seal real single-frame raw teacher responses and bounded weak expansion."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from flex_uda.data.manifests import read_jsonl,write_jsonl
from flex_uda.teacher.mllm_pseudo_dataset import REVISION,digest
from flex_uda.teacher.psr_supervision import direct_evidence,expand_local
from flex_uda.training import protect_target_annotations
from flex_tta.runtime.dance_episode import DanceEpisode
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def histogram(image,box):
    x1,y1,x2,y2=box
    crop=np.asarray(image)[max(0,int(y1)):min(image.height,int(np.ceil(y2))),max(0,int(x1)):min(image.width,int(np.ceil(x2)))]
    if not crop.size:return None
    return np.concatenate([np.histogram(crop[:,:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(float)


def seal(folder,raw_directory):
    folder,raw_directory=Path(folder).resolve(),Path(raw_directory).resolve();started=time.perf_counter();attempts=protect_target_annotations()
    if (folder/'seal.json').exists():
        seal=json.loads((folder/'seal.json').read_text())
        for name,sha in seal['files_sha256'].items():
            if file_sha(folder/name)!=sha:raise ValueError('preserve already sealed supervision')
        print(json.dumps(dict(status='PRESERVED_SEALED_PSR_CACHE',policy=folder.name)));return seal
    planned=json.loads((folder.parent/'manifest_receipt.json').read_text());manifest=read_jsonl(folder/'manifest.jsonl')
    raw=read_jsonl(raw_directory/'teacher_outputs.jsonl');receipt=json.loads((raw_directory/'receipt.json').read_text())
    if receipt['status']!='SEALED_FIXED_PROBE' or receipt['GT_access_audit']['target_GT_open_attempts'] or not receipt['frozen']['frozen_parameters_unchanged']:
        raise ValueError('require full frozen GT-free actual teacher output')
    if receipt['method']!='YES_FULL' or receipt['condition']!='T0' or receipt['numeric_dtype']!='bfloat16' or receipt['quantization'] is not None:
        raise ValueError('only the registered frozen single-frame original teacher is allowed')
    if receipt['manifest_sha256']!=file_sha(folder/'manifest.jsonl'):raise ValueError('teacher used a different manifest')
    outputs={r['sample_id']:r for r in raw}
    if set(outputs)!={r['sample_id'] for r in manifest}:raise ValueError('incomplete teacher outputs')
    direct=[];sha=file_sha(folder/'manifest.jsonl')
    for sample in manifest:
        raw_row=outputs[sample['sample_id']];response=raw_row.get('raw_response')
        if any(raw_row[k]!=sample[k] for k in ('video_id','query_id','frame_id','track_id')):raise ValueError('teacher candidate identity mismatch')
        row=direct_evidence(sample,response,REVISION,sha)
        row.update(raw_request_id=response.get('request_id') if response else None,
            teacher_input_ROI_sha256=response.get('input_view',{}).get('image_sha256',[]) if response else [],
            teacher_prompt=response.get('prompt') if response else None,
            full_image_sha256=planned['image_files_sha256'][f"{sample['video_id']}/{sample['frame_id']}"],
            raw_teacher_output_path=str(raw_directory/'teacher_outputs.jsonl'),raw_teacher_output_sha256=receipt['predictions_sha256'])
        row['cache_identity']=dict(model_revision=REVISION,prompt_hash=digest(row['teacher_prompt']),image_hash=row['full_image_sha256'],
            bbox_hash=digest(row['bbox_xyxy']),video_id=row['video_id'],query_id=row['query_id'],frame_id=row['frame_id'],track_id=row['track_id'])
        row['cache_key']=digest(row['cache_identity']);direct.append(row)
    runtime=read_jsonl(folder/'runtime.jsonl');grouped=defaultdict(list)
    for row in direct:grouped[row['video_id']].append(row)
    full=[];geometry=[];rejections=[];audits=[];input_stage=Path(planned['teacher_input_staging'])
    for video,rows in sorted(grouped.items()):
        ep=DanceEpisode(input_stage,input_stage/'tracker_outputs',video)
        obs={t:{i:dict(bbox_xyxy=list(o.bbox_xyxy),raw_mot_line=o.raw_line) for i,o in values.items()} for t,values in ep.observations_by_frame.items()}
        needed={(r['anchor_frame']+d,r['track_id']) for r in rows for d in (-1,0,1)};appearance={}
        for frame in sorted({t for t,_ in needed}):
            path=input_stage/'DanceTrack/training/image_02'/video/f'{frame:06d}.png'
            with Image.open(path) as opened:
                image=opened.convert('RGB')
                for t,track in sorted(needed):
                    if t!=frame or track not in obs.get(t,{}):continue
                    vector=histogram(image,obs[t][track]['bbox_xyxy'])
                    if vector is not None:appearance[t,track]=vector
        expanded,rejected,audit=expand_local(rows,obs,appearance,full=True)
        basic,basic_rejected,basic_audit=expand_local(rows,obs,appearance,full=False)
        full.extend(expanded);geometry.extend(basic);rejections.extend(rejected);audits.append(dict(video=video,full=audit,geometry=basic_audit))
    write_jsonl(folder/'soft_labels.jsonl',direct);write_jsonl(folder/'expanded.jsonl',full);write_jsonl(folder/'expanded_geometry.jsonl',geometry)
    atomic_json(folder/'expansion_audit.json',dict(per_video=audits,rejected=rejections,
        direct_states=dict(Counter(r['state'] for r in direct)),expanded_full=len(full),expanded_geometry=len(geometry),
        direct_labels_are_raw_soft=True,weak_interpolation_not_ground_truth=True,extra_teacher_calls=0))
    names=('manifest.jsonl','runtime.jsonl','windows.jsonl','soft_labels.jsonl','expanded.jsonl','expanded_geometry.jsonl')
    result=dict(status='SEALED_PSR_DIRECT_AND_WEAK_CACHE',target_split=planned['target_split'],sampling_policy=folder.name,
        files_sha256={name:file_sha(folder/name) for name in names},tracker_files_sha256=planned['tracker_files_sha256'],
        valid_soft_samples=sum(r['valid'] for r in direct),direct_samples=len(direct),expanded_full=len(full),expanded_geometry=len(geometry),
        teacher_revision=REVISION,teacher_parameter_SHA=receipt['frozen'],teacher_receipt=str(raw_directory/'receipt.json'),
        teacher_receipt_sha256=file_sha(raw_directory/'receipt.json'),teacher_actual_budget=receipt['budget'],
        teacher_worker_seconds=receipt['worker_elapsed_seconds'],direct_manifest_sha256=sha,
        target_GT_open_attempts=attempts,teacher_GT_access_audit=receipt['GT_access_audit'],
        unknown_not_negative=True,only_explicit_positive_two_endpoint_expansion=True,CPU_seal_seconds=time.perf_counter()-started)
    for name in names:(folder/name).chmod(0o444)
    atomic_json(folder/'seal.json',result);(folder/'seal.json').chmod(0o444)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files_sha256','tracker_files_sha256','teacher_parameter_SHA')},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pseudo-directory',type=Path,required=True);p.add_argument('--raw-directory',type=Path,required=True);a=p.parse_args();seal(a.pseudo_directory,a.raw_directory)
