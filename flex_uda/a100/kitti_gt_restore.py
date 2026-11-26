"""Evaluation-only recovery of missing KITTI templates by original byte hash.

Only a fully sealed study may enter this module's recovery function. Existing
raw annotations stay on read-only NAS. No new prediction/metric is consulted:
acceptance requires every reconstructed file to match its original A800 hash.
The initial recipe follows TempRMOT's documented normalized top-left xywh
writer. Bounded serialization/order alternatives are asset forensics, never
threshold/annotation selection by a method's score.
"""
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.runtime import ROOT,NAS

SIX=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'
OUT=SIX/'offline_GT/kitti_original_hash_restore'
RAW=NAS/'data2/user/LocateMOT/data/refer_kitti_v2'
REFERENCE=NAS/'data2/user/reference_repos/TempRMOT/inference.py'
SOURCES={
    '0005':'ab2ed142a7a1098dd22ec7f67c444dc6e3a83fe1b59906926cfb1aef64dd79a8',
    '0011':'ccf57cb6ec06b56caf0971f19000a4a81028d68227a51a149d5b492a28baeee1',
    '0013':'b2cd3f5f43852515d2b8ac5358ae93bd6fae3fbad0355a3365c8d13930c9835d',
    '0019':'1c30ac2f5835f94f338c612950dc0cd62bedd067c4f08db0dcecf85ccbf0e978'}


def sha(data):return hashlib.sha256(data).hexdigest()


def sealed_study(folder):
    path=Path(folder)
    for study,count in ((SIX/'query_consistency',252),(SIX/'view_fusion',468),(SIX,234)):
        if not path.is_relative_to(study):continue
        barrier=study/'all_predictions_sealed_before_GT.json'
        if not barrier.is_file():break
        row=json.loads(barrier.read_text());assert row['methods']==count,row
        return dict(path=str(barrier),sha256=sha(barrier.read_bytes()),outputs=row['methods'])
    raise PermissionError('KITTI GT recovery requires this study raw/count prediction seal barrier')


def expected():
    rows=[];sources=[]
    for video,digest in SOURCES.items():
        path=ROOT/'results/runs/track_ranking_20261102/targets/KITTI-V2'/video/'receipt.json'
        data=path.read_bytes();assert sha(data)==digest,('Original GT input metadata changed',video)
        for record in json.loads(data)['GT_inputs']:
            old=Path(record['path']);assert old.name=='gt.txt' and old.parts[-3]==video
            rows.append(dict(video=video,expression=old.parts[-2],sha256=record['sha256'],bytes=record['bytes']))
        sources.append(dict(path=str(path),sha256=digest))
    assert len(rows)==862 and len({(r['video'],r['expression']) for r in rows})==862
    return rows,sources


@lru_cache(None)
def dimensions(video):
    paths=json.loads((ROOT/'configs/a100_paths.json').read_text())
    root=Path(paths['datasets']['KITTI-V2']['video_roots'][video])
    with Image.open(root/'000000.png') as im:return im.size


@lru_cache(None)
def boxes(video,frame):
    path=RAW/'labels_with_ids/image_02'/video/f'{frame:06d}.txt'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);values=np.loadtxt(path).reshape(-1,6)
    width,height=dimensions(video)
    return [(int(row[1]),row[2:6]*np.array([width,height,width,height])) for row in values]


@lru_cache(None)
def raw_rows(video,expression):
    path=RAW/'expression'/video/(expression+'.json');payload=json.loads(path.read_text());result=[]
    for key,ids in payload['label'].items():
        frame=int(key);wanted={int(t) for t in ids}
        for track,box in boxes(video,frame):
            if track in wanted:result.append((frame+1,track,box))
    return result


def encode(rows,recipe):
    order=recipe['order']
    if order=='frame_track':rows=sorted(rows,key=lambda r:(r[0],r[1]))
    elif order=='track_frame':rows=sorted(rows,key=lambda r:(r[1],r[0]))
    def number(x):
        kind=recipe['box_format']
        return str(round(float(x),int(kind[1:]))) if kind.startswith('r') else format(x,kind)
    lines=[]
    for frame,track,box in rows:
        prefix=[format(frame,recipe['id_format']),format(track,recipe['id_format'])]
        lines.append(','.join(prefix+[number(x) for x in box])+recipe['tail'])
    ending=recipe['line_ending'];text=ending.join(lines)+(ending if lines else '')
    return text.encode()


def recipes():
    # First entry is exactly TempRMOT inference.py:write_gt's scalar writer.
    for order,fmt,tail,ids,end in itertools.product(
            ('input','frame_track','track_frame'),('','.2f','.3f','.6f','r2','r3','r6','g'),
            (',1, 1, 1',',1,1,1',',1,1,1.0',',1.0,1.0,1.0',',1,1',',1,1,-1',',1,1,-1,-1'),
            ('d','.1f'),('\n','\r\n')):
        yield dict(order=order,box_format=fmt,tail=tail,id_format=ids,line_ending=end)


def recover(folder):
    barrier=sealed_study(folder)
    plan=json.loads((ROOT/'docs/a100_six_direction_20261122/kitti_GT_restoration_registration_v1.json').read_text())
    assert plan['restoration_module_sha256']==sha(Path(__file__).read_bytes())
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'restore.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        completed=OUT/'completion.json';failed=OUT/'attempt_failure.json'
        if completed.is_file():
            record=json.loads(completed.read_text());assert record['status']=='PASS_ALL862_ORIGINAL_GT_BYTE_HASHES'
            return OUT
        if failed.is_file():raise RuntimeError('Original KITTI GT recovery requires root cause: '+str(failed))
        rows,sources=expected();probe=next(r for r in rows if r['bytes']>0);witness=raw_rows(probe['video'],probe['expression'])
        candidates=[];attempts=0
        for recipe in recipes():
            payload=encode(witness,recipe);attempts+=1
            if len(payload)==probe['bytes'] and sha(payload)==probe['sha256']:candidates.append(recipe)
        winner=None;outputs=None;candidate_checks=[]
        for recipe in candidates:
            current=[]
            for row in rows:
                payload=encode(raw_rows(row['video'],row['expression']),recipe)
                if len(payload)!=row['bytes'] or sha(payload)!=row['sha256']:break
                current.append((row,payload))
            candidate_checks.append(dict(recipe=recipe,matching_prefix_files=len(current)))
            if len(current)==862:winner,outputs=recipe,current;break
        if winner is None:
            report=dict(status='NO_EXACT_ORIGINAL_GT_RESTORATION_DO_NOT_SCORE_KITTI',barrier=barrier,
                original_metadata_sources=sources,serialization_recipes_tried=attempts,probe_hash_matched_recipes=len(candidates),
                all_file_checks=candidate_checks,GT_membership_IDs_boxes_or_frames_not_selected_by_prediction_scores=True,
                no_prediction_or_metric_input=True,geometry='Original normalized TOP-LEFT xywh, real image dimensions; original IDs, frame0+1.',
                historical_GT_and_NAS_unchanged=True,goal_status='ACTIVE')
            atomic_json(failed,report)
            raise RuntimeError('No byte-exact original KITTI templates; cannot substitute approximate GT: '+str(failed))
        for row,payload in outputs:
            path=OUT/'gt_template'/row['video']/row['expression']/'gt.txt';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert path.read_bytes()==payload
            else:path.write_bytes(payload)
        record=dict(status='PASS_ALL862_ORIGINAL_GT_BYTE_HASHES',barrier=barrier,evaluation_root=str(OUT),
            original_metadata_sources=sources,reference_writer_source=str(REFERENCE),reference_writer_sha256=sha(REFERENCE.read_bytes()),
            recipe=winner,matching_original_files=862,mismatched_files=0,total_bytes=sum(len(p) for _,p in outputs),
            no_prediction_or_metric_input=True,GT_used_only_in_isolated_post_seal_evaluation=True,
            NAS_and_historical_outputs_unchanged=True,restored_GT_not_to_Git=True,goal_status='ACTIVE')
        atomic_json(completed,record)
        atomic_json(ROOT/'docs/a100_six_direction_20261122/kitti_original_GT_restoration.json',record)
        return OUT
