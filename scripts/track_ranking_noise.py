"""Use the already sealed CROP answers to measure conditional probe error rates."""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *
from scripts import decompose_common as dc


def main():
    os.environ['CUDA_VISIBLE_DEVICES']='';require_plan();start=time.perf_counter()
    plan=json.loads((RUN/'plan.json').read_text())
    for r in plan['inputs']:checked(r)
    probe=json.loads((dc.RUN/'probe/plan.json').read_text());sealed=json.loads((dc.RUN/'probe/seal.json').read_text())
    answers={}
    for worker in sealed['workers']:
        for r in json_rows(worker['outputs_path']):
            if r['method']!='CROP':continue
            k=r['target'],r['video_id'],r['frame_id'],r['track_id'],r['query_id']
            if k in answers:raise ValueError('Repeated CROP response')
            answers[k]=r['yes_probability']>=.5
    frames=list(json_rows(probe['manifest_path']));wanted={}
    for frame in frames:
        for q in frame['queries']:wanted.setdefault((frame['target'],frame['video_id'],q['query_id']),set()).add(frame['frame_id'])
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt
    roots={t:dc.previous.target_root(t) for t in TARGETS};gt_inputs={}
    @lru_cache(maxsize=None)
    def gt(target,video,query):
        path=roots[target]/'gt_template'/video/query/'gt.txt';gt_inputs[str(path)]=record(path)
        return {t:v for t,v in load_eval_gt(roots[target],video,query).items() if t in wanted[target,video,query]}
    counts={t:dict(TP=0,FN=0,FP=0,TN=0) for t in TARGETS};total=0
    for frame in frames:
        target,video,t=frame['target'],frame['video_id'],frame['frame_id'];cs=frame['candidates']
        boxes=[[x,y,x+w,y+h] for x,y,w,h in [c['bbox_xywh'] for c in cs]]
        for q in frame['queries']:
            truth=gt(target,video,q['query_id']).get(t,[]);_,idx=matched_gt(overlaps(boxes,[b for _,b in truth]),list(range(len(cs))))
            for j,c in enumerate(cs):
                y=j in idx;a=answers[target,video,t,c['track_id'],q['query_id']]
                counts[target]['TP' if y and a else 'FN' if y else 'FP' if a else 'TN']+=1;total+=1
    if total!=len(answers):raise ValueError('Unmatched original CROP probe answers')
    rows=[]
    for target,c in counts.items():
        rows.append(dict(target=target,**c,positive_recall=c['TP']/(c['TP']+c['FN']),
            FNR=c['FN']/(c['TP']+c['FN']),FPR=c['FP']/(c['FP']+c['TN']),
            candidate_pairs=sum(c.values()),role=ROLE))
    csv_rows(PUBLIC/'noise_rates.csv',rows)
    dump(RUN/'noise_rates.json',dict(role=ROLE,rates={r['target']:r for r in rows},
        all_original_decompose_CROP_candidates=True,GT_inputs=list(gt_inputs.values()),
        new_teacher_calls=0,new_model_calls=0,GPU_hours=0,CPU_wall_seconds=time.perf_counter()-start))
    print(json.dumps(dict(event='TRACK_NOISE_RATES_SEALED',rates=rows,total_original_CROP_pairs=total)),flush=True)


if __name__=='__main__':main()
