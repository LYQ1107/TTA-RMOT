"""Frozen cache combinations and separate post-seal GT description. No calls."""
from collections import defaultdict
from functools import lru_cache
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *
from flex_uda.data.manifests import read_jsonl,write_jsonl
METHODS=('T1_ARGMAX','T1_ARGMAX_REJECT','T2_ARGMAX')

def combo_masks(p1,p2):
    p1=np.asarray(p1);p2=np.asarray(p2);a=p1.argmax(axis=0);b=p2.argmax(axis=0)
    arg=np.arange(17)[:,None]==a;reject=arg&(p1.max(0)[None,:]>=.5)
    return dict(zip(METHODS,(arg,reject,np.arange(17)[:,None]==b)))

def generate():
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter()
    oldseal=json.loads((cs.RUN/'probe_seal.json').read_text());plan=json.loads((cs.RUN/'probe_plan.json').read_text())
    model=oldseal['models']['2B'];path=Path(model['outputs_path'])
    if file_sha(path)!=model['outputs_sha256']:raise ValueError('changed old teacher output')
    if file_sha(plan['manifest_path'])!=plan['manifest_sha256']:raise ValueError('changed fixed50frame probe')
    frames=read_jsonl(plan['manifest_path']);by=defaultdict(list)
    for r in read_jsonl(path):by[r['method'],r['video_id'],r['frame_id']].append(r)
    outputs=[]
    for f in frames:
        v,t=f['video_id'],f['frame_id'];p1={(r['track_id'],r['query_id']):r['yes_probability'] for r in by['T1_CROP_YESNO',v,t]}
        p2={r['track_id']:r['option_probabilities'] for r in by['T2_CROP_CHOICE',v,t]}
        queries=f['queries'];candidates=f['candidates']
        a=np.asarray([[p1[c['track_id'],q['query_id']] for c in candidates] for q in queries]).reshape(17,len(candidates))
        b=np.asarray([p2[c['track_id']] for c in candidates],dtype=float).reshape(len(candidates),18).T
        masks=combo_masks(a,b)
        for method,mask in masks.items():
            outputs.append(dict(video_id=v,frame_id=t,method=method,track_ids=[c['track_id'] for c in candidates],
                query_ids=[q['query_id'] for q in queries],selected=mask.tolist()))
    dest=RUN/'teacher_combo.jsonl';write_jsonl(dest,outputs)
    result=dict(status='ALL3_TEACHER_COMBOS_SEALED_BEFORE_GT',outputs_path=str(dest),outputs_sha256=file_sha(dest),
        original_outputs_path=str(path),original_outputs_sha256=model['outputs_sha256'],manifest_path=plan['manifest_path'],manifest_sha256=plan['manifest_sha256'],
        methods=METHODS,frames=50,videos=5,candidates=378,candidate_query_pairs=6426,target_GT_open_attempts=len(attempts),
        new_teacher_calls=0,GPU_calls=0,CPU_seconds=time.perf_counter()-start)
    if attempts:raise RuntimeError('teacher combination read target GT')
    atomic_json(RUN/'teacher_combo_seal.json',result);atomic_json(PUBLIC/'teacher_combo_seal.json',result)
    print(json.dumps(dict(event='XQUERY_TEACHER_COMBOS_SEALED',target_GT_open_attempts=len(attempts))),flush=True)

def evaluate():
    setup();require_plan()
    seal=json.loads((RUN/'teacher_combo_seal.json').read_text())
    if file_sha(seal['outputs_path'])!=seal['outputs_sha256']:raise ValueError('changed post-seal predictions')
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt,auroc
    frames=read_jsonl(seal['manifest_path']);pred={(r['method'],r['video_id'],r['frame_id']):r for r in read_jsonl(seal['outputs_path'])}
    gt=lru_cache(None)(lambda v,q:load_eval_gt(v2.DATA,v,q));results=[];details=[]
    for method in METHODS:
        scored=[];frame_rows=[]
        for f in frames:
            v,t=f['video_id'],f['frame_id'];r=pred[method,v,t]
            boxes=[]
            for c in f['candidates']:
                x,y,w,h=c['bbox_xywh'];boxes.append([x,y,x+w,y+h])
            for j,q in enumerate(f['queries']):
                truth=gt(v,q['query_id']).get(t,[]);_,positive=matched_gt(overlaps(boxes,[b for _,b in truth]),list(range(len(boxes))))
                selected=np.asarray(r['selected'][j],dtype=bool);count=int(selected.sum())
                row=dict(method=method,video=v,frame=t,query=q['query_id'],seen=q['seen_train_text'],GT_count=len(truth),predicted_count=count)
                frame_rows.append(row);details.append(row)
                for k,p in enumerate(selected):scored.append(dict(selected=bool(p),y=k in positive,seen=q['seen_train_text']))
        for scope,flag in [('ALL',None),('SEEN_EXACT_PHRASE',True),('UNSEEN_EXACT_PHRASE',False)]:
            rows=[r for r in scored if flag is None or r['seen']==flag];fs=[r for r in frame_rows if flag is None or r['seen']==flag]
            y=np.asarray([r['y'] for r in rows],dtype=bool);s=np.asarray([r['selected'] for r in rows],dtype=bool)
            results.append(dict(method=method,text_scope=scope,role='POST_SEAL_OFFLINE_GT_DESCRIPTION_ONLY',candidate_pairs=len(rows),
                AUROC_hard_binary_scores=auroc(s,y),candidate_accuracy=float((s==y).mean()),positive_recall=float((s&y).sum()/y.sum()) if y.any() else None,
                negative_FPR=float((s&~y).sum()/(~y).sum()) if (~y).any() else None,positive_candidate_labels=int(y.sum()),
                frame_queries=len(fs),presence_accuracy=sum((r['predicted_count']>0)==(r['GT_count']>0) for r in fs)/len(fs),
                quantity_MAE=float(np.mean([abs(r['predicted_count']-r['GT_count']) for r in fs])),
                quantity_exact_accuracy=sum(r['predicted_count']==r['GT_count'] for r in fs)/len(fs),selected_rows=int(s.sum()),
                unknown_candidate_pairs=0,new_teacher_calls=0))
    v2.write_csv(PUBLIC/'teacher_combo.csv',results);atomic_json(PUBLIC/'teacher_combo.json',dict(rows=results,does_not_select_teacher=True,
        new_teacher_calls=0,not_full_RMOT=True,source_outputs_SHA=seal['original_outputs_sha256']))
    atomic_json(RUN/'teacher_combo_offline_frames.json',details)
    print(json.dumps(dict(event='XQUERY_TEACHER_COMBO_OFFLINE_COMPLETE',rows=len(results))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else generate()
