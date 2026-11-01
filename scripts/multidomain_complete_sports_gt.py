"""Independent, post-seal official GT completion with the authorized exact gate.

Never imported or called by a score worker. It uses original LaMOT targets and
original SportsMOT GT rows, not McByte boxes/IDs or any model predictions as GT.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,require_plan
from scripts.multidomain_sports_prepare import LAMOT,IMAGES,link_new
from scripts.multidomain_diagnostics import base,verify
from scripts.diag_attrib_v2_common import write_new
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def numeric_rows(lines):
    return [tuple(map(float,line.split(','))) for line in lines if line.strip()]


def select_original_rows(raw_lines,targets):
    wanted={int(t):{int(i) for i in ids} for t,ids in targets.items()}
    found=defaultdict(set);out=[]
    for line in raw_lines:
        values=line.strip().split(',')
        frame,track=int(float(values[0])),int(float(values[1]))
        if track in wanted.get(frame,set()):
            out.append(line.strip());found[frame].add(track)
    missing=[(t,i) for t,ids in wanted.items() for i in ids-found[t]]
    if missing:raise ValueError(f'Official referring targets absent from original Sports boxes: {missing[:10]}')
    return out


def main():
    started=time.perf_counter()
    publication=require_plan()
    if publication.get('revision')!='v3 official 418 exact GT-completion gate':
        raise PermissionError('Publish the user-authorized GT-completion amendment first')
    # No GT values opened until both cross-domain score sources and the common
    # in-domain source have all complete, immutable 466-episode predictions.
    prediction_inputs=[]
    for model in ('V2-best','Dance-best'):
        seal=base('SportsMOT',model)/'GT_free_predictions.json'
        record=json.loads(seal.read_text())
        if record['GT_open_attempts']:raise PermissionError('GT-free prediction audit failed')
        for method in record['methods']:
            verified=verify('SportsMOT',method,model)
            if verified['oracle']:raise ValueError('Oracle output used as pre-GT evidence')
        prediction_inputs.append(dict(path=str(seal),sha256=file_sha(seal)))
    meta=json.loads((RUN/'sports/metadata.json').read_text())
    originals=LAMOT/'gt_templates/SportsMOT'
    expected={q['video_id']+'+'+q['expression_id'] for q in meta['episodes']}
    templates={str(p.relative_to(originals).parent).replace('/','+',1):p for p in originals.rglob('gt.txt')}
    if len(templates)!=418 or not set(templates)<=expected or len(expected-set(templates))!=48:
        raise ValueError('Official 418/48 discovery changed; do not silently alter the completion protocol')
    ann={}
    for path in sorted((LAMOT/'annotations_v1/val/SportsMOT').glob('*.json')):
        payload=json.loads(path.read_text())
        video=path.stem.rsplit('_',1)[0]
        key=video+'+'+payload['language'].replace(' ','-')
        ann[key]=(path,payload)
    if set(ann)!=expected:raise ValueError('Original referring annotation/seqmap mismatch')
    raw={};raw_records=[]
    for video in meta['videos']:
        p=IMAGES/video/'gt/gt.txt'
        if not p.is_file():raise FileNotFoundError('Original official SportsMOT boxes missing: '+str(p))
        raw[video]=p.read_text().splitlines()
        raw_records.append(dict(video=video,path=str(p),sha256=file_sha(p),rows=len(raw[video]),
            source='PREEXISTING_ORIGINAL_SPORTSMOT_VAL_NOT_LOCATEMOT_PREDICTIONS',
            official_dataset='https://github.com/MCG-NJU/SportsMOT'))
    derived={};checks=[];failures=[];annotation_records=[]
    for key in sorted(expected):
        path,payload=ann[key];video=key.split('+',1)[0]
        lines=select_original_rows(raw[video],payload['targets'])
        derived[key]=lines
        annotation_records.append(dict(sequence=key,path=str(path),sha256=file_sha(path)))
        if key not in templates:continue
        original=templates[key];left=numeric_rows(original.read_text().splitlines());right=numeric_rows(lines)
        passed=left==right
        r=dict(sequence=key,official_path=str(original),official_sha256=file_sha(original),
            official_rows=len(left),reconstructed_rows=len(right),all_fields_and_order_numeric_exact=passed)
        checks.append(r)
        if not passed:
            first=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
            failures.append(dict(sequence=key,first_different_row=first,
                official_row=left[first] if first<len(left) else None,
                reconstructed_row=right[first] if first<len(right) else None))
    result=dict(status='STOP_EXACT_REPRODUCTION_FAILED' if failures else 'PASS_418_EXACT_48_COMPLETED',
        independent_post_seal_process=True,prereg_publication=publication['remote_commit'],
        prediction_inputs=prediction_inputs,official_templates=418,missing_templates=48,
        exact_passed=sum(x['all_fields_and_order_numeric_exact'] for x in checks),
        exact_failed=len(failures),checks=checks,failures=failures,raw_box_sources=raw_records,
        annotation_sources=annotation_records,prediction_feedback=False,GT_used_for_model=False,
        training_updates=0,teacher_calls=0,raw_templates_overwritten=False)
    if failures:
        result.update(CPU_wall_seconds=time.perf_counter()-started,new_GPU_hours=0)
        atomic_json(PUBLIC/'sports_GT_completion.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('checks','raw_box_sources','annotation_sources','failures')}),flush=True)
        raise SystemExit(2)
    evaluation=RUN/'eval_assets/SportsMOT';files=[]
    for key in sorted(expected):
        video,expression=key.split('+',1);p=evaluation/'gt_template'/video/expression/'gt.txt'
        if key in templates:link_new(p,templates[key])
        else:write_new(p,''.join(line+'\n' for line in derived[key]))
        files.append(dict(sequence=key,path=str(p),sha256=file_sha(p),rows=len(derived[key]),
            provenance='OFFICIAL_ORIGINAL_418' if key in templates else 'AUTHORIZED_OFFICIAL_TARGETS_X_ORIGINAL_BOXES_48'))
    result.update(evaluation_root=str(evaluation),files=files,
        generated_GT_rows=sum(r['rows'] for r in files if r['provenance'].endswith('_48')),
        official_GT_rows=sum(r['rows'] for r in files if r['provenance']=='OFFICIAL_ORIGINAL_418'),
        CPU_wall_seconds=time.perf_counter()-started,new_GPU_hours=0)
    atomic_json(PUBLIC/'sports_GT_completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','raw_box_sources','annotation_sources','files','failures')}),flush=True)


if __name__=='__main__':main()
