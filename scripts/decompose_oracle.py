"""Independent offline ORACLE generation after frozen-input prediction seals."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *


def assert_same(new,old):
    old_seal=json.loads((old/'predictions_seal.json').read_text()) if (old/'predictions_seal.json').exists() else None
    if old_seal is not None:
        by={(e['video'],e['expression']):e for e in old_seal['episodes']}
    for e in new['episodes']:
        p=old/'results'/e['video']/e['expression']/'predict.txt'
        if file_sha(p)!=e['prediction_sha256']:
            raise ValueError('STOP: mandatory raw-row equality failed: '+str(p))
        if old_seal is not None and file_sha(p)!=by[e['video'],e['expression']]['prediction_sha256']:
            raise ValueError('Changed historical prediction')


def oracle(combo):
    require_plan();begin=time.perf_counter()
    _,model,target=next(p for p in PAIRS if p[0]==combo)
    # Each oracle only runs after all six corresponding GT-free predictions are
    # sealed; no oracle-derived information is made available to their process.
    for name in CALIBRATORS:verify(combo,name)
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt
    meta,arrays,groups,_=load_target(target);x=arrays[model]
    root=previous.target_root(target);g=meta['group_count']
    truth_count=np.zeros(g,dtype=np.int32);mean_count=np.zeros(g,dtype=np.int32)
    track_mask=np.zeros(len(x),dtype=bool);episode_records=[];gt_paths=[]
    by_video=video_queries(meta)
    for video,queries in by_video.items():
        rows=physical(meta,target,video);ids=np.asarray([r['track_id'] for r in rows])
        unique,inverse=np.unique(ids,return_inverse=True)
        boxes=np.asarray([[r['bbox_xywh'][0],r['bbox_xywh'][1],sum(r['bbox_xywh'][::2]),sum(r['bbox_xywh'][1::2])] for r in rows])
        for q in queries:
            truth=load_eval_gt(root,video,q['expression_id']);total=sum(map(len,truth.values()))
            gt_paths.append(str(root/'gt_template'/video/q['expression_id']/'gt.txt'))
            average=int(np.floor(total/len(truth)+.5)) if truth else 0
            matched=np.zeros(len(rows),dtype=bool)
            for t,a,b,group in q['groups']:
                gt=truth.get(t,[]);truth_count[group]=len(gt);mean_count[group]=average
                ia,ib=a-q['start'],b-q['start']
                iou=overlaps(boxes[ia:ib],[box for _,box in gt]);_,idx=matched_gt(iou,list(range(ib-ia)))
                if idx:matched[ia+np.asarray(sorted(idx))]=True
            successes=np.bincount(inverse,weights=matched,minlength=len(unique))
            positive=2*successes>=np.bincount(inverse,minlength=len(unique))
            track_mask[q['start']:q['stop']]=positive[inverse]
            episode_records.append(dict(video=video,expression=q['expression_id'],GT_observations=total,GT_frames=len(truth),
                                       present=total>0,rounded_mean=average,positive_tracks=int(positive.sum())))
        print(json.dumps(dict(event='DECOMPOSE_ORACLE_VIDEO',combo=combo,video=video)),flush=True)
    ranks=rank_within_groups(x,groups,g)
    masks={'G0_GLOBAL':top_budget(x,sum(q['GT_observations'] for q in episode_records))[0],
           'G1_QUERY':np.zeros(len(x),dtype=bool),
           'G2_V_MEAN':ranks<mean_count[groups],
           'G3_F_MEAN':ranks<np.where(truth_count>0,mean_count,0)[groups],
           'G4_F_FRAME':ranks<truth_count[groups], 'TRACK_ORACLE':track_mask}
    truth_by={(q['video'],q['expression']):q for q in episode_records}
    for q in meta['episodes']:
        a,b=q['start'],q['stop'];n=truth_by[q['video_id'],q['expression_id']]['GT_observations']
        masks['G1_QUERY'][a:b]=top_budget(x[a:b],n)[0]
    seals={n:seal(combo,n,masks[n],meta,oracle=True) for n in ORACLES}
    old=previous.base(target,model)/'R_SxK_ORACLE'
    assert_same(seals['G4_F_FRAME'],old)
    checks=[dict(method='G4_F_FRAME',old=str(old),episodes=len(meta['episodes']),raw_bytes_equal=True)]
    if combo=='C1':
        for name,old_name in [('G2_V_MEAN','V_MEAN'),('G3_F_MEAN','F_MEAN'),('G4_F_FRAME','F_FRAME'),('TRACK_ORACLE','TRACK_ORACLE')]:
            old=COUNT/'evaluation'/('ORACLE__'+old_name)
            assert_same(seals[name],old)
            checks.append(dict(method=name,old=str(old),episodes=425,raw_bytes_equal=True))
    record=dict(status='ORACLE_SEALED_G4_EXACT_ROWS_PASS',combo=combo,target=target,source=model,episodes=episode_records,
                self_checks=checks,GT_only_independent_offline=True,GT_paths=gt_paths,
                GT_protocol=previous.SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
                CPU_wall_seconds=time.perf_counter()-begin)
    atomic_json(RUN/'oracle'/f'{combo}.json',record)
    print(json.dumps(dict(event='DECOMPOSE_ORACLE_DONE',combo=combo,checks=checks,seconds=record['CPU_wall_seconds'])),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--combo',choices=[r[0] for r in PAIRS],required=True);a=p.parse_args();oracle(a.combo)
