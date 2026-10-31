"""ORACLE / GT: descriptive teacher metrics, NEVER a teacher chooser."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import *
from scripts.count_structure_oracle import before_GT
from scripts.count_structure_probe import decisions
from flex_uda.data.manifests import read_jsonl

def evaluate():
    setup();before_GT()
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt,auroc
    plan=json.loads((RUN/'probe_plan.json').read_text());seal=json.loads((RUN/'probe_seal.json').read_text())
    frames=read_jsonl(plan['manifest_path']);results=[];per_frame=[]
    gt=lru_cache(None)(lambda v,q:load_eval_gt(v2.DATA,v,q))
    for model,receipt in seal['models'].items():
        by=defaultdict(list)
        for row in read_jsonl(receipt['outputs_path']):by[row['method'],row['video_id'],row['frame_id']].append(row)
        for method in plan['models'][model]['methods']:
            scored=[];frame_rows=[]
            for frame in frames:
                v,t=frame['video_id'],frame['frame_id'];raw=by[method,v,t];score=decisions(raw,method,frame)
                for q in frame['queries']:
                    truth=gt(v,q['query_id']).get(t,[]);candidates=frame['candidates'];boxes=[]
                    for c in candidates:
                        x,y,w,h=c['bbox_xywh'];boxes.append([x,y,x+w,y+h])
                    iou=overlaps(boxes,[box for _,box in truth]);_,positives=matched_gt(iou,list(range(len(candidates))))
                    predictions=[score[v,t,c['track_id'],q['query_id']] for c in candidates]
                    # T3 parse failure is UNKNOWN even for an empty candidate
                    # frame; do not turn a failed answer into an empty set.
                    parsed=next((r for r in raw if r.get('query_id')==q['query_id']),None) if method=='T3_FRAME_SOM' else None
                    valid=all(p is not None for p in predictions) and (parsed is None or parsed['valid'])
                    count=sum(p>.5 for p in predictions) if valid else None
                    f=dict(model=model,method=method,video=v,frame=t,query=q['query_id'],seen=q['seen_train_text'],
                        GT_count=len(truth),predicted_count=count,valid=valid)
                    frame_rows.append(f);per_frame.append(f)
                    for j,p in enumerate(predictions):scored.append(dict(p=p,y=j in positives,seen=q['seen_train_text']))
            for scope,flag in [('ALL',None),('SEEN_EXACT_PHRASE',True),('UNSEEN_EXACT_PHRASE',False)]:
                rows=[r for r in scored if flag is None or r['seen']==flag];valid=[r for r in rows if r['p'] is not None]
                y=np.asarray([r['y'] for r in valid],dtype=bool);p=np.asarray([r['p'] for r in valid],dtype=float);selected=p>.5
                fs=[r for r in frame_rows if flag is None or r['seen']==flag];vf=[r for r in fs if r['valid']]
                correct=int((selected==y).sum());presence_correct=sum((r['predicted_count']>0)==(r['GT_count']>0) for r in vf)
                result=dict(role='ORACLE / GT, ONLY POST_SEAL DESCRIPTION',model=model,method=method,text_scope=scope,status='EVALUATED_FIXED_PROBE',
                    candidate_pairs=len(rows),valid_candidate_pairs=len(valid),unknown_candidate_pairs=len(rows)-len(valid),
                    candidate_positive_labels=int(y.sum()),candidate_negative_labels=int((~y).sum()),
                    AUROC_valid=auroc(p,y),accuracy_at_0p5_valid=correct/len(valid) if valid else None,
                    accuracy_all_unknown_counted_incorrect=correct/len(rows) if rows else None,
                    positive_recall_valid=int((selected&y).sum())/int(y.sum()) if y.sum() else None,
                    negative_FPR_valid=int((selected&~y).sum())/int((~y).sum()) if (~y).sum() else None,
                    frame_queries=len(fs),valid_frame_queries=len(vf),unknown_frame_queries=len(fs)-len(vf),
                    presence_accuracy_all=presence_correct/len(fs),
                    presence_accuracy_valid=presence_correct/len(vf) if vf else None,
                    quantity_MAE_valid=float(np.mean([abs(r['predicted_count']-r['GT_count']) for r in vf])) if vf else None,
                    quantity_exact_accuracy_all=sum(r['predicted_count']==r['GT_count'] for r in vf)/len(fs),
                    GT_present_frame_queries=sum(r['GT_count']>0 for r in fs),
                    total_GT=sum(r['GT_count'] for r in fs),total_selected_valid=sum(r['predicted_count'] for r in vf),
                    T3_scores_are_binary_not_soft_probabilities=method=='T3_FRAME_SOM')
                results.append(result)
    if seal['skipped_8B']:
        for method in ('T2_CROP_CHOICE','T3_FRAME_SOM'):
            r={k:None for k in results[0]};r.update(role='NOT_RUN_NO_GT_OUTPUT',model='8B',method=method,text_scope='ALL',
                status='SKIPPED_NO_VERIFIED_LOCAL_BASE_WEIGHTS_NO_DOWNLOAD');results.append(r)
    v2.write_csv(PUBLIC/'teacher_probe.csv',results)
    atomic_json(PUBLIC/'teacher_probe.json',dict(role='ORACLE / GT, only offline description, no method choice',rows=results,
        parse_failures_remain_UNKNOWN=True,unknown_rows_not_synthesized_as_NEG=True,
        valid_conditional_metrics_disclosed_with_coverage=True,all_request_accuracy_counts_failures_incorrect=True,
        labels='fixed MOT candidate one-to-one max-cardinality IoU>=.5; frame quantities use full official GT count',
        does_not_select_teacher=True,not_full_RMOT_HOTA=True))
    atomic_json(RUN/'probe_offline_frames.json',dict(role='ORACLE / GT ONLY',frames=per_frame))
    print(json.dumps(dict(event='COUNT_PROBE_GT_DESCRIPTION_COMPLETE',rows=len(results))),flush=True)

if __name__=='__main__':evaluate()
