"""ORACLE / GT ONLY: granularity and descriptive structure after GT-free seals."""
from collections import Counter,defaultdict
import bisect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import *

def summary(values):
    x=np.asarray(values,dtype=float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
        **{f'p{p}':float(np.percentile(x,p)) if len(x) else None for p in (10,50,90)})

def before_GT():
    require_plan();s=json.loads((RUN/'selectors_seal.json').read_text());p=json.loads((RUN/'probe_seal.json').read_text())
    if s['status']!='ALL28_GT_FREE_SELECTORS_SEALED' or s['target_GT_open_attempts'] or p['status']!='ALL_AVAILABLE_MODELS_PROBE_SEALED_BEFORE_GT':
        raise ValueError('seal all GT-free selectors and all available teacher probes before target GT')
    for r in s['methods']:v2.verify_seal(r['method'])
    for r in p['models'].values():
        if file_sha(r['outputs_path'])!=r['outputs_sha256'] or r['target_GT_open_attempts']:raise ValueError('changed/invalid probe seal')

def generate():
    setup();before_GT()
    # GT routines deliberately imported ONLY after all independent GT-free jobs.
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt
    started=time.perf_counter();opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).absolute()
            if p.is_relative_to(v2.DATA) and p.name=='gt.txt':opened.append(str(p))
    sys.addaudithook(audit)
    meta,scores,groups=load_inputs(verify=False);g=meta['query_frame_groups'];size=len(groups)
    truth_count=np.zeros(g,dtype=np.int32);video_flags=np.zeros(g,dtype=bool);mean_count=np.zeros(g,dtype=np.int32)
    matched_observations=np.zeros(size,dtype=bool);track_oracle=np.zeros(size,dtype=bool)
    original_naive=Path('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE/manifest.jsonl');teacher_frames=defaultdict(set)
    for line in original_naive.read_text().splitlines():
        r=json.loads(line);teacher_frames[r['video_id'],r['query_id']].add(r['frame_id'])
    temporal={(m,b):Counter() for m in ('FROZEN','SAME_SET_D1') for b in ('0-1','2-25','26-100','>100')}
    episodes=[];track_records=[];co_video=[];co_frame=Counter()
    for video,queries,tracks,frames,ids,unique,inverse in video_banks(meta):
        box={t:{} for t in tracks}
        for t,rows in tracks.items():
            for i,line in rows.items():
                x,y,w,h=list(map(float,line.split(',')))[2:6];box[t][i]=[x,y,x+w,y+h]
        positive_count=np.zeros(len(unique),dtype=np.int32);co=np.zeros(queries[0]['frame_count'],dtype=np.int32);video_q=0
        for q in queries:
            truth=load_eval_gt(v2.DATA,video,q['expression_id']);total=sum(map(len,truth.values()));present=bool(total);video_q+=int(present)
            annotated=sorted(teacher_frames[video,q['expression_id']]);local=np.zeros(len(ids),dtype=bool)
            for pos,(t,a,b,group) in enumerate(q['groups']):
                gt=truth.get(t,[]);truth_count[group]=len(gt);co[pos]+=bool(gt)
                ia,ib=a-q['start'],b-q['start'];order=ids[ia:ib].tolist()
                iou=overlaps([box[t][i] for i in order],[bb for _,bb in gt])
                _,matched=matched_gt(iou,list(range(len(order))))
                if matched:local[ia+np.asarray(sorted(matched))]=True
                k=bisect.bisect_left(annotated,t)
                distance=min([abs(t-annotated[j]) for j in (k-1,k) if 0<=j<len(annotated)])
                bucket='0-1' if distance<=1 else '2-25' if distance<=25 else '26-100' if distance<=100 else '>100'
                for m in ('FROZEN','SAME_SET_D1'):
                    selected=np.flatnonzero(scores[m][a:b]>0).tolist();found,_=matched_gt(iou,selected)
                    r=temporal[m,bucket];r['frames']+=1;r['GT']+=len(gt);r['predictions']+=len(selected)
                    r['TP']+=len(found);r['FP']+=len(selected)-len(found);r['FN']+=len(gt)-len(found)
            matched_observations[q['start']:q['stop']]=local
            matched=np.bincount(inverse,weights=local,minlength=len(unique));obs=np.bincount(inverse,minlength=len(unique))
            positive=2*matched>=obs;positive_count+=positive.astype(np.int32)
            track_oracle[q['start']:q['stop']]=positive[inverse]
            gt_frames=len(truth);average=int(np.floor(total/gt_frames+.5)) if gt_frames else 0
            for _,a,b,group in q['groups']:video_flags[group]=present;mean_count[group]=average
            episodes.append(dict(video=video,expression=q['expression_id'],query_text=q['query_text'],GT_observations=total,
                GT_frames=gt_frames,video_frames=q['frame_count'],present=present,rounded_mean_GT=average,
                tracks=len(unique),positive_tracks=int(positive.sum()),positive_track_ratio=float(positive.mean()),
                GT_frame_fraction=gt_frames/q['frame_count'],role='ORACLE / GT, diagnostic only'))
        co_video.append(dict(video=video,present_queries=video_q,frame_simultaneous_query_counts=dict(Counter(map(int,co)))))
        co_frame.update(map(int,co));track_records.extend(dict(video=video,track_id=int(i),positive_queries=int(c)) for i,c in zip(unique,positive_count))
        print(json.dumps(dict(event='COUNT_ORACLE_VIDEO',video=video)),flush=True)
    if int(truth_count.sum())!=212541 or sum(e['present'] for e in episodes)!=40:raise ValueError('official GT protocol changed')
    native=np.bincount(groups,weights=scores['FROZEN']>0,minlength=g).astype(np.int32)
    budgets=dict(V_NATIVE1=np.where(video_flags,np.maximum(native,1),0),V_MEAN=np.where(video_flags,mean_count,0),
        F_NATIVE1=np.where(truth_count>0,np.maximum(native,1),0),F_MEAN=np.where(truth_count>0,mean_count,0),F_FRAME=truth_count)
    rank=v2.rank_within_groups(scores['FROZEN'],groups,g);lines=v2.candidate_lines(meta);methods=[]
    for name in ORACLES:
        mask=track_oracle if name=='TRACK_ORACLE' else rank<budgets[name][groups]
        method='ORACLE__'+name;v2.write_selection(method,mask,meta,lines,protocol='ORACLE / GT, DIAGNOSTIC ONLY, NEVER A GT_FREE_METHOD')
        methods.append(method)
    original=OLD/'evaluation/M_FROZEN__ORACLE_GT'
    seal=v2.assert_equivalent('ORACLE__F_FRAME',original)
    for e in seal['episodes']:
        previous=original/'results'/e['video']/e['expression']/'predict.txt'
        if file_sha(previous)!=e['prediction_sha256']:raise ValueError('STOP: F_FRAME raw rows/order must reproduce exactly')
    from scripts.pf_diag_common import official_metrics
    metrics=official_metrics(original/'results')
    if seal['prediction_rows']!=201009 or metrics['HOTA']!=45.794:raise ValueError('STOP: F_FRAME must reproduce201009rows/HOTA45.794')
    present=[e for e in episodes if e['present']];whole=sum(e['positive_track_ratio']>=.8 for e in present)
    counts=Counter(r['positive_queries'] for r in track_records);ntracks=len(track_records)
    query_stats=[]
    for expression in sorted({e['expression'] for e in episodes}):
        all_q=[e for e in episodes if e['expression']==expression];ps=[e for e in all_q if e['present']]
        query_stats.append(dict(expression=expression,query_text=all_q[0]['query_text'],present_videos=len(ps),
            positive_track_ratio_given_present=summary([e['positive_track_ratio'] for e in ps]),
            GT_frame_fraction_given_present=summary([e['GT_frame_fraction'] for e in ps]),
            GT_frame_fraction_all_videos=summary([e['GT_frame_fraction'] for e in all_q])))
    buckets=[]
    for (m,b),counts_ in temporal.items():
        r=dict(margin=m,distance_to_nearest_actual_Naive_teacher_frame=b,**counts_)
        r.update(precision=r['TP']/r['predictions'] if r['predictions'] else None,recall=r['TP']/r['GT'] if r['GT'] else None)
        buckets.append(r)
    structure=dict(role='ORACLE / GT, ONLY DESCRIPTIVE',query_statistics=query_stats,episodes=episodes,
        present_episodes=len(present),whole_outfit_qualifying_episodes=whole,whole_outfit_fraction=whole/len(present),
        whole_outfit_structure_holds=whole/len(present)>.5,positive_query_count_per_track_histogram=dict(counts),tracks=ntracks,
        mutual_exclusion_fractions=dict(zero=counts[0]/ntracks,one=counts[1]/ntracks,two_or_more=sum(c for k,c in counts.items() if k>=2)/ntracks),
        mutual_exclusion_structure_holds=(counts[0]+counts[1])/ntracks>=.9,track_records=track_records,
        distinct_present_queries_per_video=co_video,frame_simultaneous_present_query_count_histogram=dict(co_frame),
        temporal_distance_buckets=buckets,teacher_annotation_manifest=dict(path=str(original_naive),sha256=file_sha(original_naive)),
        definitions='P=one-to-one IoU>=.5 matches on >=50% of physical track observations; all q and full timelines, no GT filtering of queries',
        target_GT_open_count=len(opened),GT_paths=opened,CPU_wall_seconds=time.perf_counter()-started)
    atomic_json(PUBLIC/'structure_stats.json',structure)
    atomic_json(RUN/'oracle_seal.json',dict(status='ALL6_ORACLE_SEALED_F_FRAME_REPRODUCED',methods=methods,role='ORACLE / GT ONLY',
        F_FRAME_exact_row_equivalence=str(original),F_FRAME_byte_identical_episodes=425,F_FRAME_rows=201009,F_FRAME_HOTA=45.794,
        target_GT_open_count=len(opened),GT_feedback_to_selectors_or_teachers=False,CPU_wall_seconds=time.perf_counter()-started))

if __name__=='__main__':generate()
