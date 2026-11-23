"""Image geometry and raw per-frame teacher targets for a matched comparator.

No evaluator, target annotation or fitted model is an input. Observations use
the bank's internal frame indices, not the possibly one-based MOT text index.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def normalize_boxes(boxes,width,height):
    box=np.asarray(boxes,float)
    return np.stack((np.clip((box[...,0]+box[...,2])/(2*width),0,1),
        np.clip((box[...,1]+box[...,3])/(2*height),0,1),
        (box[...,2]-box[...,0])/width,(box[...,3]-box[...,1])/height),axis=-1)


def geometry(bank,target):
    frame=min(int(f) for f,_ in bank['observations'])
    extension='.jpg' if target=='SportsMOT' else '.png'
    with Image.open(Path(bank['image_root'])/f'{frame:06d}{extension}') as image:width,height=image.size
    xywh=np.asarray([[float(v) for v in line.split(',')[2:6]] for line in bank['lines']])
    boxes=np.concatenate((xywh[:,:2],xywh[:,:2]+xywh[:,2:]),axis=1)
    assert len(boxes)==len(bank['observations']) and np.isfinite(boxes).all()
    return normalize_boxes(boxes,width,height),width,height


def extra_features(coords,direction):
    coords=np.asarray(coords,float);direction=np.asarray(direction)
    oriented=np.where(direction>0,coords[...,0],np.where(direction<0,1-coords[...,0],.5))
    return np.concatenate((coords,np.broadcast_to(oriented,coords.shape[:-1])[...,None],
        np.broadcast_to(direction!=0,coords.shape[:-1])[...,None]),axis=-1)


def moments(raw,coords,columns,directions):
    """Exact moments of Q x observations without allocating Q x N x 1087."""
    nq,nt,dimension=raw.shape;lengths=np.bincount(columns,minlength=nt)
    total=np.einsum('qtd,t->d',raw,lengths,dtype=float)
    square=np.einsum('qtd,t->d',np.square(raw.astype(float)),lengths,dtype=float)
    extra=np.zeros(6);extra_square=np.zeros(6)
    for direction in directions:
        values=extra_features(coords,direction);extra+=values.sum(0);extra_square+=np.square(values).sum(0)
    return np.r_[total,extra],np.r_[square,extra_square],nq*len(coords)


def frame_supervision(records,queries,tracks,width,height):
    """Raw answers, exact-request deduplication, then mean across paid views.

    CLIP answers label their requested anchor frame for this deliberately
    strong ordinary comparator; this is not a claim of true frame annotation.
    Repeated identical inputs/answers do not add independent weight.
    """
    qi={q['expression_id']:i for i,q in enumerate(queries)};ti={int(t):i for i,t in enumerate(tracks)}
    requests={}
    for row in records:
        request=row['input'];key=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key not in requests:requests[key]=dict(row=row,probabilities=set())
        requests[key]['probabilities'].add(float(row['yes_probability']))
    grouped=defaultdict(list)
    for value in requests.values():
        row=value['row'];key=(qi[row['query_id']],ti[int(row['track_id'])],int(row['frame_id']))
        grouped[key].append((row['effective_view'],float(np.median(sorted(value['probabilities']))),row['input']['bbox']))
    keys=sorted(grouped);y=[];boxes=[]
    for key in keys:
        rows=grouped[key];views=defaultdict(list)
        for view,probability,box in rows:views[view].append(probability)
        assert all(np.allclose(box,rows[0][2],rtol=0,atol=1e-6) for _,_,box in rows)
        y.append(np.mean([np.mean(p) for p in views.values()]));boxes.append(rows[0][2])
    return np.asarray(keys,int),np.asarray(y,float),normalize_boxes(boxes,width,height),dict(
        actual_question_records=len(records),unique_exact_requests=len(requests),frame_target_cells=len(keys),
        target='Raw mean of actually requested view answers at each requested frame; no DS target or manual label',
        CLIP_assigned_to_requested_anchor=True)


def references(teacher_rows,obs,tracks,directions,coords):
    """Anchor to paid observation states; unseen cells use the source's 3 samples."""
    lookup={int(t):i for i,t in enumerate(tracks)};reference=np.zeros((len(directions),len(tracks)))
    for ti,track in enumerate(tracks):
        ids=np.flatnonzero(obs[:,1]==track)
        chosen=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
        x=float(coords[chosen,0].mean());reference[:,ti]=np.where(directions>0,x,1-x)
    by_view=defaultdict(list)
    # The caller attaches local query indices, retaining the original text gate.
    for row in teacher_rows:by_view[row['local_query'],lookup[int(row['track'])],row['view']].append(row['x'])
    cells=defaultdict(list)
    for (q,t,v),values in by_view.items():cells[q,t].append(float(np.mean(values)))
    for (q,t),values in cells.items():reference[q,t]=np.mean(values)
    return reference
