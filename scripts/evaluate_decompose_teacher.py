"""Independent post-seal GT probe evaluator; never imported by the teacher."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from flex_uda.data.manifests import read_jsonl


def evaluate():
    require_plan();plan=json.loads((RUN/'probe/plan.json').read_text());sealed=json.loads((RUN/'probe/seal.json').read_text())
    if sealed['status']!='ALL_PAIRED_PROBES_SEALED_BEFORE_GT':raise ValueError('Seal every paired probe before GT')
    if file_sha(plan['manifest_path'])!=plan['manifest_sha256']:raise ValueError('Changed probe set')
    if file_sha(plan['expression_types']['path'])!=plan['expression_types']['sha256']:raise ValueError('Changed text-only taxonomy')
    raw={};raw_paths=[]
    for worker in sealed['workers']:
        if worker['GT_open_attempts'] or not worker['isolation']['kernel_read_whitelist'] or file_sha(worker['outputs_path'])!=worker['outputs_sha256']:
            raise ValueError('Invalid or unsealed teacher output')
        raw_paths.append(dict(path=worker['outputs_path'],sha256=worker['outputs_sha256']))
        for r in read_jsonl(worker['outputs_path']):
            key=r['target'],r['method'],r['video_id'],r['frame_id'],r['track_id'],r['query_id']
            if key in raw:raise ValueError('Duplicate paired teacher result')
            raw[key]=r
    frames=read_jsonl(plan['manifest_path'])
    expected=sum(len(f['queries'])*len(f['candidates'])*3 for f in frames)
    if len(raw)!=expected:raise ValueError('Incomplete paired teacher results')
    # Only after the complete raw-output checks import GT routines.
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt,auroc
    gt=lru_cache(None)(lambda target,v,q:load_eval_gt(previous.target_root(target),v,q))
    datasets={t:load_target(t) for t in TARGETS}
    q_index={t:{(q['video_id'],q['expression_id']):q for q in data[0]['episodes']} for t,data in datasets.items()}
    physical_index={}
    for t,data in datasets.items():
        for v in data[0]['videos']:
            physical_index[t,v]={(r['frame_id'],r['track_id']):i for i,r in enumerate(physical(data[0],t,v))}
    candidate_rows=defaultdict(list);frame_rows=defaultdict(list)
    for frame in frames:
        target,v,t=frame['target'],frame['video_id'],frame['frame_id'];candidates=frame['candidates']
        arrays=datasets[target][1];boxes=[]
        for c in candidates:
            x,y,w,h=c['bbox_xywh'];boxes.append([x,y,x+w,y+h])
        for q in frame['queries']:
            qid=q['query_id'];truth=gt(target,v,qid).get(t,[]);iou=overlaps(boxes,[b for _,b in truth])
            _,positive=matched_gt(iou,list(range(len(candidates))))
            labels=np.asarray([j in positive for j in range(len(candidates))],dtype=bool)
            offset=q_index[target][v,qid]['start'];tags=types(q['query_text'])
            scores={}
            for method in ('CROP','CLIP','FRAME_BOX'):
                scores[method]=np.asarray([raw[target,method,v,t,c['track_id'],qid]['yes_probability'] for c in candidates])
            for model in MODELS:
                scores[model]=np.asarray([arrays[model][offset+physical_index[target,v][t,c['track_id']]] for c in candidates])
            for method,p in scores.items():
                selected=p>=.5 if method in ('CROP','CLIP','FRAME_BOX') else p>0
                order=sorted(range(len(candidates)),key=lambda j:(-p[j],candidates[j]['track_id']))
                found,_=matched_gt(iou,order[:1]);hit=int(bool(found));n_gt=len(truth)
                f=dict(target=target,method=method,video=v,frame=t,query=qid,types=tags,
                       GT_count=n_gt,predicted_count=int(selected.sum()),hit_at_1=hit,
                       GT_recall_at_1=hit/n_gt if n_gt else None)
                frame_rows[target,method].append(f)
                candidate_rows[target,method].extend(dict(p=float(value),y=bool(y),types=tags,video=v,query=qid) for value,y in zip(p,labels))
    output=[];bootstrap={};video_stats={}
    for target in TARGETS:
        for method in ('CROP','CLIP','FRAME_BOX',*MODELS):
            fs=frame_rows[target,method];cs=candidate_rows[target,method]
            videos=sorted({r['video'] for r in fs});per_video=[]
            for v in videos:
                valid=[r for r in fs if r['video']==v and r['GT_count']>0]
                per_video.append([sum(r['GT_recall_at_1'] for r in valid),sum(r['hit_at_1'] for r in valid),len(valid)])
            video_stats[target,method]=np.asarray(per_video,dtype=float)
            for scope in ('ALL',*WORDS,'OTHER'):
                ff=[r for r in fs if scope=='ALL' or scope in r['types']]
                cc=[r for r in cs if scope=='ALL' or scope in r['types']]
                valid=[r for r in ff if r['GT_count']>0]
                output.append(dict(target=target,scorer=method,expression_type=scope,videos=len({r['video'] for r in ff}),
                    frame_queries=len(ff),GT_present_frame_queries=len(valid),candidate_query_pairs=len(cc),
                    GT_R_at_1=float(np.mean([r['GT_recall_at_1'] for r in valid])) if valid else None,
                    Hit_at_1=float(np.mean([r['hit_at_1'] for r in valid])) if valid else None,
                    AUROC=auroc([r['p'] for r in cc],[r['y'] for r in cc]),
                    presence_accuracy=float(np.mean([(r['predicted_count']>0)==(r['GT_count']>0) for r in ff])) if ff else None,
                    count_MAE=float(np.mean([abs(r['predicted_count']-r['GT_count']) for r in ff])) if ff else None,
                    teacher_threshold='p_yes>=0.5' if method in ('CROP','CLIP','FRAME_BOX') else 'margin>0',
                    role='POST_SEAL_FIXED_PROBE_NOT_FULL_RMOT',GT_feedback=False))
        if target!='KITTI-V2':
            n=len(video_stats[target,'CROP']);draws=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=2000)
            for method in ('CROP','CLIP','FRAME_BOX'):
                for model in MODELS:
                    a,b=video_stats[target,method],video_stats[target,model]
                    for field,column in [('GT_R_at_1',0),('Hit_at_1',1)]:
                        def pooled(x,w):return (w@x[:,column])/np.maximum(w@x[:,2],1)
                        delta=pooled(a,np.ones(n))-pooled(b,np.ones(n));samples=pooled(a,draws)-pooled(b,draws)
                        bootstrap[target+'__'+method+'_minus_'+model+'__'+field]=dict(target=target,first=method,second=model,
                            metric=field,delta=float(delta),CI95=np.percentile(samples,[2.5,97.5]).tolist(),
                            clusters=n,samples=2000,seed=0,unit='whole video, all frame/query dependence preserved')
    write_csv(PUBLIC/'teacher_probe.csv',output)
    missing=set(json.loads((OLD_PUBLIC/'sports_template_inventory.json').read_text())['missing_GT'])
    sports418=[]
    for method in ('CROP','CLIP','FRAME_BOX',*MODELS):
        fs=[r for r in frame_rows['SportsMOT',method] if r['video']+'+'+r['query'] not in missing]
        cs=[r for r in candidate_rows['SportsMOT',method] if r['video']+'+'+r['query'] not in missing]
        for scope in ('ALL',*WORDS,'OTHER'):
            ff=[r for r in fs if scope=='ALL' or scope in r['types']]
            cc=[r for r in cs if scope=='ALL' or scope in r['types']]
            valid=[r for r in ff if r['GT_count']>0]
            sports418.append(dict(target='SportsMOT',scorer=method,expression_type=scope,
                GT_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE',
                videos=len({r['video'] for r in ff}),frame_queries=len(ff),GT_present_frame_queries=len(valid),candidate_query_pairs=len(cc),
                GT_R_at_1=float(np.mean([r['GT_recall_at_1'] for r in valid])) if valid else None,
                Hit_at_1=float(np.mean([r['hit_at_1'] for r in valid])) if valid else None,
                AUROC=auroc([r['p'] for r in cc],[r['y'] for r in cc]),
                presence_accuracy=float(np.mean([(r['predicted_count']>0)==(r['GT_count']>0) for r in ff])) if ff else None,
                count_MAE=float(np.mean([abs(r['predicted_count']-r['GT_count']) for r in ff])) if ff else None))
        videos=sorted({r['video'] for r in fs})
        per_video=[]
        for v in videos:
            valid=[r for r in fs if r['video']==v and r['GT_count']>0]
            per_video.append([sum(r['GT_recall_at_1'] for r in valid),sum(r['hit_at_1'] for r in valid),len(valid)])
        video_stats['SportsMOT418',method]=np.asarray(per_video,dtype=float)
    for method in ('CROP','CLIP','FRAME_BOX'):
        for model in MODELS:
            a,b=video_stats['SportsMOT418',method],video_stats['SportsMOT418',model]
            n=len(a);draws=np.random.default_rng(0).multinomial(n,np.ones(n)/n,size=2000)
            for field,column in [('GT_R_at_1',0),('Hit_at_1',1)]:
                def pooled(x,w):return (w@x[:,column])/np.maximum(w@x[:,2],1)
                samples=pooled(a,draws)-pooled(b,draws)
                bootstrap['SportsMOT418__'+method+'_minus_'+model+'__'+field]=dict(target='SportsMOT',
                    GT_protocol='V4_GT_RESTRICTED_TO_OFFICIAL_418_COVERAGE',first=method,second=model,metric=field,
                    delta=float(pooled(a,np.ones(n))-pooled(b,np.ones(n))),CI95=np.percentile(samples,[2.5,97.5]).tolist(),clusters=n,samples=2000,seed=0)
    write_csv(PUBLIC/'sports418_teacher.csv',sports418)
    atomic_json(PUBLIC/'teacher_bootstrap.json',bootstrap)
    write_jsonl_path=RUN/'probe/offline_frames.jsonl'
    from flex_uda.data.manifests import write_jsonl
    write_jsonl(write_jsonl_path,[r for values in frame_rows.values() for r in values])
    atomic_json(RUN/'probe/evaluation.json',dict(status='ALL_FIXED_PAIRED_PROBES_OFFLINE_EVALUATED',rows=output,
        teacher_outputs=raw_paths,offline_frame_path=str(write_jsonl_path),offline_frame_sha256=file_sha(write_jsonl_path),
        GT_open_only_here=True,teacher_GT_open_attempts=0,GT_feedback=False,
        GT_R_at_1_definition='mean over all GT-present frame/query pairs of top1 IoU>=.5 hit / GT_count; zero if no candidate',
        Hit_at_1_definition='mean top1 binary IoU>=.5 hit on the same GT-present frame/query pairs',
        AUROC_labels='fixed candidate one-to-one maximum-cardinality IoU>=.5 matches, score-independent; same as count_structure probe',
        rank_ties='descending score then ascending numeric track ID',
        main_GT_protocol='SportsMOT v4 / unchanged KITTI and Dance',
        sports_418_sensitivity='v4 GT restricted to the official418 IDs; same sealed teacher calls, no extra teacher invocation'))
    print(json.dumps(dict(event='DECOMPOSE_PAIRED_TEACHER_EVALUATED',rows=len(output),raw_pairs=len(raw))),flush=True)


if __name__=='__main__':evaluate()
