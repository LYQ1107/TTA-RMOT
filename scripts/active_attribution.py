"""Independent post-seal recovery/FP attribution, never imported by TTA workers."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def selection_changes(iou,before,after):
    from scripts.analyze_pf_full_offline import matched_pairs
    b,a=set(before),set(after)
    bm,am=matched_pairs(iou,sorted(b)),matched_pairs(iou,sorted(a))
    common=a & b
    result=dict(baseline_matched_gt=len(bm),adapted_matched_gt=len(am),
        recovered_gt=len(set(am.values())-set(bm.values())),
        lost_gt=len(set(bm.values())-set(am.values())),
        new_false_positive_rows=len((a-b)-am.keys()),
        removed_false_positive_rows=len((b-a)-bm.keys()),
        unchanged_rows_new_FP=len((common & bm.keys())-am.keys()),
        unchanged_rows_removed_FP=len((common & am.keys())-bm.keys()),
        baseline_false_positive_rows=len(b)-len(bm),adapted_false_positive_rows=len(a)-len(am))
    net=result['adapted_false_positive_rows']-result['baseline_false_positive_rows']
    decomposed=(result['new_false_positive_rows']-result['removed_false_positive_rows']+
        result['unchanged_rows_new_FP']-result['unchanged_rows_removed_FP'])
    if net!=decomposed:raise ValueError('FP attribution accounting mismatch')
    return result


def attribute(folder,target):
    """Fixed IoU>=.5 assignment complements, not replaces, official TrackEval."""
    from flex_uda.presence.predictions import verify_predictions
    from flex_uda.presence.runtime import target_metadata
    from scripts.presence_finish_a import references
    from scripts.multidomain_diagnostics import target_root
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,query_presence_summary
    folder=Path(folder);seal=verify_predictions(folder)
    native=dict(references(target,seal['model']))['NATIVE']
    source_seal_path=Path(native) / 'predictions_seal.json'
    source=json.loads(source_seal_path.read_text())
    if source.get('GT_open_attempts',0):raise ValueError('Native generation GT isolation failed')
    for entry in source['episodes']:
        if file_sha(entry['prediction_path'])!=entry['prediction_sha256']:
            raise ValueError('Changed sealed Native prediction')
    baseline={(e['video'],e['expression']):e for e in source['episodes']}
    meta=target_metadata(target)
    expected={(q['video_id'],q['expression_id']) for q in meta['episodes']}
    if {(e['video'],e['expression']) for e in seal['episodes']}!=expected:
        raise ValueError('Attribution requires complete target predictions')
    # GT is first opened below, only after all current and Native seals pass.
    gt_root=target_root(target);episodes=[]
    by_video=defaultdict(list)
    for entry in seal['episodes']:by_video[entry['video']].append(entry)
    for video,entries in by_video.items():
        bank=meta['videos'][video];frames=defaultdict(list);boxes=[];keys={}
        for j,line in enumerate(bank['lines']):
            f=line.split(',');t,i=int(float(f[0])),int(float(f[1]))
            x,y,w,h=map(float,f[2:6]);boxes.append([x,y,x+w,y+h])
            frames[t].append(j);keys[t,i]=j
        boxes=np.asarray(boxes)
        def selected(entry):
            result=set()
            for line in Path(entry['prediction_path']).read_text().splitlines():
                if not line.strip():continue
                f=line.split(',');key=int(float(f[0])),int(float(f[1]))
                j=keys[key]
                if line!=bank['lines'][j]:raise ValueError('Prediction changed fixed MOT row')
                result.add(j)
            if len(result)!=entry['selected_rows']:raise ValueError('Duplicate prediction row')
            return result
        for entry in entries:
            expression=entry['expression'];truth=load_eval_gt(gt_root,video,expression)
            a,b=selected(entry),selected(baseline[video,expression])
            row=dict(video=video,expression=expression,gt_observations=sum(map(len,truth.values())),
                baseline_selected_rows=len(b),adapted_selected_rows=len(a))
            for frame,indices in frames.items():
                matrix=overlaps(boxes[indices],[box for _,box in truth.get(frame,[])])
                changes=selection_changes(matrix,[k for k,j in enumerate(indices) if j in b],
                    [k for k,j in enumerate(indices) if j in a])
                for name,value in changes.items():row[name]=row.get(name,0)+value
            episodes.append(row)
    groups=query_presence_summary(episodes)
    totals={name:sum(e[name] for e in episodes) for name in episodes[0] if name not in ('video','expression')}
    result=dict(target=target,method=seal['method'],model=seal['model'],episodes=episodes,
        **totals,query_presence=groups,
        net_gt_recovery=totals['recovered_gt']-totals['lost_gt'],
        net_FP_change=totals['adapted_false_positive_rows']-totals['baseline_false_positive_rows'],
        prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'),
        native_prediction_seal_sha256=file_sha(source_seal_path),
        definition='offline max-cardinality framewise IoU>=0.5 assignment, not official HOTA matching',
        FP_accounting='new-row FP minus removed-row FP plus unchanged-row new FP minus unchanged-row removed FP',
        GT_feedback_to_runtime=False,all_predictions_sealed_before_GT=True)
    atomic_json(folder / 'offline_attribution.json',result)
    return result
