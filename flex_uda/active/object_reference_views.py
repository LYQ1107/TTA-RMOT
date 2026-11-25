"""Transport each compatible teacher observation before view averaging.

Logit transport is nonlinear in probability. Moving it across a probability
average is generally invalid when views have different stable intercepts.
This independent refinement never changes the live mean-first experiment.
"""
from collections import defaultdict
import hashlib
import json
import numpy as np
from flex_uda.active.object_reference import transport_targets
from flex_uda.active.relational_features import normalize_boxes


def reference_frame_supervision(records, queries, tracks, width, height, directions, beta):
    qi = {q['expression_id']:i for i,q in enumerate(queries)}
    ti = {int(t):i for i,t in enumerate(tracks)}
    requests = {}
    for row in records:
        key = hashlib.sha256(json.dumps(row['input'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key not in requests:
            requests[key] = dict(row=row,probabilities=set())
        requests[key]['probabilities'].add(float(row['yes_probability']))
    grouped = defaultdict(list)
    for value in requests.values():
        row = value['row']
        key = (qi[row['query_id']],ti[int(row['track_id'])],int(row['frame_id']))
        p = float(np.median(sorted(value['probabilities'])))
        grouped[key].append((row['effective_view'],p,row['input']['bbox']))
    keys = sorted(grouped)
    raw,reference,control,boxes = [],[],[],[]
    multi_view_frames = eligible_frames = 0
    for key in keys:
        rows = grouped[key]
        assert all(np.allclose(box,rows[0][2],rtol=0,atol=1e-6) for _,_,box in rows)
        views = defaultdict(list)
        ref_views = defaultdict(list)
        control_views = defaultdict(list)
        direction = int(directions[key[0]])
        for view,p,box in rows:
            views[view].append(p)
            if direction:
                # Pure horizontal relation gates have no compatible CLIP
                # answers in the frozen six-setting input audit. Fail rather
                # than silently assign a clip an unsupported state semantics.
                if view not in ('CROP','FRAME_BOX'):
                    raise ValueError('Eligible relation needs a single-frame teacher observation')
                coords = normalize_boxes([box],width,height)
                canonical = float(transport_targets(np.array([p]),coords,np.array([direction]),beta)[0])
                ref_views[view].append(canonical)
                control_views[view].append(float(np.clip(p,.02,.98)))
        base = float(np.mean([np.mean(v) for v in views.values()]))
        raw.append(base)
        if direction:
            reference.append(float(np.mean([np.mean(v) for v in ref_views.values()])))
            control.append(float(np.mean([np.mean(v) for v in control_views.values()])))
            eligible_frames += 1
            multi_view_frames += len(views) > 1
        else:
            reference.append(float(np.clip(base,.02,.98)))
            control.append(float(np.clip(base,.02,.98)))
        boxes.append(rows[0][2])
    return dict(cells=np.asarray(keys,int),raw_frame_probability=np.asarray(raw),
        reference_frame_probability=np.asarray(reference),control_frame_probability=np.asarray(control),
        coords=normalize_boxes(boxes,width,height),
        eligible_frame_cells=eligible_frames,eligible_multi_view_frame_cells=multi_view_frames,
        unique_exact_requests=len(requests))


def group_canonical_frames(frames):
    cells = frames['cells']
    unique,inverse = np.unique(cells[:,:2],axis=0,return_inverse=True)
    weight = 1.+3.*np.abs(2.*frames['raw_frame_probability']-1.)
    mass = np.bincount(inverse,weights=weight,minlength=len(unique))
    reference = np.bincount(inverse,weights=weight*frames['reference_frame_probability'],minlength=len(unique))/mass
    control = np.bincount(inverse,weights=weight*frames['control_frame_probability'],minlength=len(unique))/mass
    return dict(cells=unique,reference_target=reference,unshifted_target=control,
        sample_weight=mass,frame_to_group=inverse,raw_frame_weight=weight)
