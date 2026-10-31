"""Allowed source-only two-scalar Platt calibration, never target GT."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import RUN,PUBLIC,v2,setup,require_plan
from scripts.analyze_pf_full_offline import overlaps,matched_gt
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
SOURCE=Path('/data3/user/TTA-RMOT-assets/flexhook/refer-kitti-v2')

def xyxy(box):
    x,y,w,h=box;return [x,y,x+w,y+h]

def calibrate():
    setup();require_plan();access=v2.deny_gt();started=time.perf_counter()
    saved=RUN/'source_calibration.json'
    if saved.exists():
        r=json.loads(saved.read_text())
        if r['status']=='SOURCE_ONLY_PLATT_FROZEN':return r
    stats=json.loads((v2.RUN/'source_statistics.json').read_text());path=Path(stats['score_path'])
    if file_sha(path)!=stats['score_sha256']:raise ValueError('source margin SHA differs')
    scores=json.loads(path.read_text());labels=json.loads((SOURCE/'labels.json').read_text())
    all_x=[];all_y=[];inputs=[dict(path=str(path),sha256=file_sha(path))];queries=0
    for video,by_id in sorted(scores.items()):
        geom=defaultdict(dict)
        for track,frames in labels[video].items():
            for t,row in frames.items():
                if row.get('expression_raw'):geom[int(t)][int(track)]=xyxy(row['bbox'])
        truths={}
        for p in sorted((SOURCE/'expression'/video).glob('*.json')):
            item=json.loads(p.read_text());text=item['sentence'];truth=item.get('label',{})
            if text in truths and truths[text]!=truth:raise ValueError('ambiguous duplicate source sentence')
            truths[text]=truth;inputs.append(dict(path=str(p),sha256=file_sha(p)))
        observations=defaultdict(dict)
        for track,frames in by_id.items():
            for t,query in frames.items():observations[int(t)][int(track)]=query
        query_texts=sorted({q for rows in observations.values() for qs in rows.values() for q in qs})
        for text in query_texts:
            queries+=1
            for t,rows in sorted(observations.items()):
                ids=sorted(i for i,qs in rows.items() if text in qs)
                if not ids:continue
                gt_ids=[int(i) for i in truths[text].get(str(t),[])];gt_boxes=[xyxy(labels[video][str(i)][str(t)]['bbox']) for i in gt_ids]
                iou=overlaps([geom[t][i] for i in ids],gt_boxes)
                _,matched=matched_gt(iou,list(range(len(ids))))
                for j,i in enumerate(ids):
                    no,yes=rows[i][text];all_x.append(float(yes)-float(no));all_y.append(j in matched)
        print(json.dumps(dict(event='COUNT_SOURCE_VIDEO',video=video,rows=len(all_x))),flush=True)
    x=np.asarray(all_x,dtype=np.float64);y=np.asarray(all_y,dtype=np.float64)
    if len(x)!=2355528 or queries!=862:raise ValueError('must calibrate the exact official2355528 source rows')
    def objective(theta):
        z=theta[0]*x+theta[1];res=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)),np.array([np.mean(res*x),np.mean(res)])
    fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',options=dict(maxiter=1000,gtol=1e-9,ftol=1e-12))
    if not fit.success or not np.isfinite(fit.x).all():raise RuntimeError('source Platt did not converge: '+fit.message)
    result=dict(status='SOURCE_ONLY_PLATT_FROZEN',a=float(fit.x[0]),b=float(fit.x[1]),pi_s=float(y.mean()),
        source_candidate_rows=len(x),source_queries=queries,positive_rows=int(y.sum()),
        source_mean=float(x.mean()),source_unscaled_NLL=objective([1.,0.])[0],source_Platt_NLL=float(fit.fun),
        optimizer='L-BFGS-B two scalar source calibration only; no student or teacher updates',iterations=int(fit.nit),
        source_model='official frozen KITTI-V2; same calibration reused for all four margin libraries',
        labels='IoU>=.5 one-to-one max-cardinality; source query GT boxes; normalized-coordinate IoU is pixel-coordinate invariant',
        source_GT=str(SOURCE/'labels.json'),source_GT_sha256=file_sha(SOURCE/'labels.json'),inputs=inputs,
        target_GT_open_attempts=len(access),student_updates=0,teacher_updates=0,CPU_wall_seconds=time.perf_counter()-started)
    if access:raise RuntimeError('source calibration touched target GT')
    atomic_json(saved,result);atomic_json(PUBLIC/'source_calibration.json',result)
    return result

if __name__=='__main__':calibrate()
