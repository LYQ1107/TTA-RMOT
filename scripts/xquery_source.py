"""Source-only runtime preparation, GT matching and exact rejection thresholds."""
from collections import defaultdict,Counter
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *
from scripts.count_structure_source import source_geometry,xyxy,SOURCE_TRACKS
from scripts.analyze_pf_full_offline import overlaps,matched_gt
from flex_tta.mot_interface.trackers import merge_kitti_class_splits

def prepare():
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter();dest=RUN/'source';dest.mkdir(exist_ok=True)
    if file_sha(SOURCE_SCORE)!=SOURCE_SHA:raise ValueError('source frozen2355528 SHA differs')
    if (dest/'metadata.json').exists():
        meta=json.loads((dest/'metadata.json').read_text())
        if file_sha(dest/'FROZEN.npy')!=meta['frozen_array_sha256']:raise ValueError('changed source bank')
        return meta
    raw=json.loads(SOURCE_SCORE.read_text());offset=0;episodes=[];videos={};values=[]
    for video,tracks in sorted(raw.items()):
        geometry,files=source_geometry(video);rows=merge_kitti_class_splits(*files)
        lines={(r.frame_id-1,r.track_id):','.join(map(str,r.values)) for r in rows}
        observations=sorted((int(t),int(i)) for i,fs in tracks.items() for t in fs)
        texts={q for fs in tracks.values() for qs in fs.values() for q in qs}
        queries=[]
        for p in sorted((SOURCE/'expression'/video).glob('*.json')):
            q=json.loads(p.read_text())['sentence']
            if q in texts:queries.append(dict(video_id=video,expression_id=p.stem,query_text=q))
        if len({q['query_text'] for q in queries})!=len(queries) or len(queries)!=len(texts):raise ValueError('source expression alias mismatch')
        videos[video]=dict(observations=observations,lines=[lines[t,i] for t,i in observations],
            normalized_boxes=[geometry[t][i] for t,i in observations],tracker_files=[dict(path=str(p),sha256=file_sha(p)) for p in files])
        for q in queries:
            a=offset
            for t,i in observations:
                no,yes=tracks[str(i)][str(t)][q['query_text']];values.append(float(yes)-float(no));offset+=1
            episodes.append(dict(**q,start=a,stop=offset))
    if offset!=2355528 or len(episodes)!=862:raise ValueError('must use original2355528 source rows/862 queries')
    np.save(dest/'FROZEN.npy',np.asarray(values,dtype=np.float64))
    meta=dict(episodes=episodes,videos=videos,candidate_query_rows=offset,source_score_path=str(SOURCE_SCORE),source_score_sha256=SOURCE_SHA,
        frozen_array_sha256=file_sha(dest/'FROZEN.npy'),source_frame_rule='image=rawMOT-1; evaluation MOT=rawMOT',
        source_coverage='exact historical official scores; no supplemental last observations',target_GT_open_attempts=len(attempts),CPU_seconds=time.perf_counter()-start)
    atomic_json(dest/'metadata.json',meta)
    atomic_json(PUBLIC/'source_inputs.json',dict(status='PREPARED_NO_TARGET_GT',source_candidate_query_rows=offset,source_queries=862,
        source_score_path=str(SOURCE_SCORE),source_score_sha256=SOURCE_SHA,metadata_path=str(dest/'metadata.json'),metadata_sha256=file_sha(dest/'metadata.json'),
        student_checkpoints=checkpoints(),target_GT_open_attempts=len(attempts)))
    if attempts:raise RuntimeError('source input prep touched target GT')
    print(json.dumps(dict(event='XQUERY_SOURCE_PREPARED',rows=offset,queries=862)),flush=True);return meta

def labels():
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter();dest=RUN/'source';meta=json.loads((dest/'metadata.json').read_text())
    if (dest/'GT_seal.json').exists():return json.loads((dest/'GT_seal.json').read_text())
    gtlabels=json.loads((SOURCE/'labels.json').read_text());y=np.zeros(meta['candidate_query_rows'],dtype=bool);trackpairs=[];byvideo=defaultdict(Counter)
    for q in meta['episodes']:
        v=q['video_id'];obs=meta['videos'][v]['observations'];boxes=meta['videos'][v]['normalized_boxes']
        truth=json.loads((SOURCE/'expression'/v/(q['expression_id']+'.json')).read_text()).get('label',{})
        byframe=defaultdict(list)
        for j,(t,i) in enumerate(obs):byframe[t].append(j)
        for t,idx in byframe.items():
            gids=truth.get(str(t),[]);gboxes=[xyxy(gtlabels[v][str(i)][str(t)]['bbox']) for i in gids]
            _,matched=matched_gt(overlaps([boxes[j] for j in idx],gboxes),list(range(len(idx))))
            for j in matched:y[q['start']+idx[j]]=True
        ids=np.asarray([i for t,i in obs]);unique,inverse=np.unique(ids,return_inverse=True)
        fraction=cs.track_mean(y[q['start']:q['stop']],inverse,len(unique))
        for j,i in enumerate(unique):
            positive=bool(fraction[j]>=.5);trackpairs.append(dict(video=v,expression=q['expression_id'],track_id=int(i),positive=positive,matched_fraction=float(fraction[j])))
            byvideo[v][int(i)]+=positive
    if int(y.sum())!=354658:raise ValueError('source labels must reproduce prior IoU matching354658')
    np.save(dest/'labels.npy',y)
    hist=Counter(n for rows in byvideo.values() for n in rows.values());nt=sum(hist.values())
    result=dict(status='SOURCE_GT_ONLY_SEALED',source_GT_sha256=file_sha(SOURCE/'labels.json'),labels_sha256=file_sha(dest/'labels.npy'),
        positive_observations=int(y.sum()),query_track_pairs=trackpairs,
        track_positive_query_histogram=dict(sorted(hist.items())),tracks=nt,
        fractions=dict(zero=sum(n for k,n in hist.items() if k==0)/nt,one=sum(n for k,n in hist.items() if k==1)/nt,
                       two_or_more=sum(n for k,n in hist.items() if k>=2)/nt),
        target_GT_open_attempts=len(attempts),CPU_seconds=time.perf_counter()-start)
    atomic_json(dest/'GT_seal.json',result);print(json.dumps(dict(event='XQUERY_SOURCE_GT_SEALED',tracks=nt,histogram=dict(hist))),flush=True);return result

def thresholds():
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter();dest=RUN/'source';meta=json.loads((dest/'metadata.json').read_text())
    gt=json.loads((dest/'GT_seal.json').read_text());ys=[r['positive'] for r in gt['query_track_pairs']];result={}
    seal=json.loads((dest/'scores_seal.json').read_text())
    for m in MARGINS:
        p=dest/(m+'.npy');expected=meta['frozen_array_sha256'] if m=='FROZEN' else seal['models'][m]['array_sha256']
        if file_sha(p)!=expected:raise ValueError('changed source scores '+m)
        x=np.load(p,mmap_mode='r');means=[]
        for q in meta['episodes']:
            ids=np.asarray([i for t,i in meta['videos'][q['video_id']]['observations']]);unique,inverse=np.unique(ids,return_inverse=True)
            means.extend(cs.track_mean(x[q['start']:q['stop']],inverse,len(unique)).tolist())
        result[m]=dict(exact_f1_threshold(means,ys),source_scores_path=str(p),source_scores_sha256=expected)
    record=dict(status='ALL6_SOURCE_ONLY_EXACT_THRESHOLDS',models=result,target_GT_open_attempts=len(attempts),
        source_matching='IoU>=.5 max-cardinality per frame, positive track if matched fraction>=.5',
        target_GT_used=False,student_updates=0,optimizer_steps=0,CPU_seconds=time.perf_counter()-start)
    if attempts:raise RuntimeError('threshold process touched Dance GT')
    atomic_json(RUN/'thresholds.json',record);atomic_json(PUBLIC/'thresholds.json',record)
    print(json.dumps(dict(event='XQUERY_SOURCE_THRESHOLDS_COMPLETE',thresholds={k:v['threshold'] for k,v in result.items()})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','labels','thresholds']);a=p.parse_args()
    {'prepare':prepare,'labels':labels,'thresholds':thresholds}[a.phase]()
