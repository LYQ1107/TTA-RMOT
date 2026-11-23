"""All three official models on one target. Per-video receipts are recovery units."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *
from scripts import decompose_common as dc


def evaluate_video(target,video,queries,meta,arrays,noise):
    dest=RUN/'targets'/target/video;dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        for p in r['outputs']:checked(p)
        if r['noise_rates_sha256']!=sha(RUN/'noise_rates.json'):raise ValueError('Noise rates changed after a completed video')
        print(json.dumps(dict(event='TRACK_REUSE_COMPLETED_VIDEO',target=target,video=video)),flush=True);return r
    start=time.perf_counter()
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt
    root=dc.previous.target_root(target);rows=dc.physical(meta,target,video)
    frames=np.asarray([r['frame_id'] for r in rows]);ids=np.asarray([r['track_id'] for r in rows])
    boxes=np.asarray([[x,y,x+w,y+h] for x,y,w,h in [r['bbox_xywh'] for r in rows]])
    unique,counts=np.unique(ids,return_counts=True);keep_ids=unique[counts>=3]
    tracks=[np.flatnonzero(ids==i) for i in keep_ids]
    tracks=[a[np.argsort(frames[a],kind='stable')] for a in tracks]
    eligible=np.isin(ids,keep_ids);frame_order=np.argsort(frames,kind='stable')
    _,starts=np.unique(frames[frame_order],return_index=True)
    frame_parts=np.split(frame_order,starts[1:]);frame_ids=[int(frames[a[0]]) for a in frame_parts]
    retained_frames=[a[eligible[a]] for a in frame_parts]
    length=int(np.floor(2*FPS[target]+.5));segments=[];segment_tracks=[]
    for j,idx in enumerate(tracks):
        for local in frame_segments(frames[idx],length):segments.append(idx[local]);segment_tracks.append(j)
    layout=dict(target=target,video=video,role=ROLE,fps=FPS[target],segment_length_frames=length,
        raw_tracks=len(unique),retained_tracks=len(tracks),excluded_tracks=int((counts<3).sum()),
        raw_observations=len(rows),retained_observations=int(eligible.sum()),excluded_observations=int((~eligible).sum()),
        frames=len(frame_parts),frames_with_zero_retained_candidates=sum(not len(a) for a in retained_frames),
        frame_candidate_counts=[len(a) for a in retained_frames],frame_ids=frame_ids,
        track_ids=keep_ids.tolist(),track_observation_counts=[len(a) for a in tracks],
        segment_count=len(segments),segment_observation_counts=[len(a) for a in segments],
        segment_spans_frames=[int(frames[a[-1]]-frames[a[0]]+1) for a in segments],
        segment_track_ids=[int(keep_ids[j]) for j in segment_tracks])
    dump(dest/'layout.json',layout)
    common=['target','video','query','model','role']
    columns={
      'units.csv':['target','video','query','role','track_id','observations','first_frame','last_frame','matched_observations','matched_fraction','Y','time_varying','segments'],
      'auroc_frame.csv':common+['frame','objects','positives','AUROC','prefix_ceiling_F1','prefix_best_k'],
      'auroc_segment.csv':common+['objects','positives','AUROC'],
      'auroc_track.csv':common+['aggregation','objects','positives','AUROC'],
      'prefix_ceiling.csv':common+['granularity','aggregation','F1','frame_units','GT_present','positive_tracks'],
      'bisect_track.csv':common+['verifier','method','tracks','positive_tracks','questions','F1','selected_count','asked_positions','asked_track_ids','answers','GT_present'],
      'presence_scores.csv':common+['aggregation','score','GT_present','positive_tracks'],
      'sequence_metrics.csv':common+['tracks','segments','frames','GT_present','GT_observations','positive_observations','positive_tracks','positive_segments','time_varying_tracks',
        'frame_defined','frame_undefined','FRAME_AUROC_qmean','SEGMENT_AUROC',*[m+'_AUROC' for m in AGGS],'common_AUROC_q','FRAME_prefix_qmean','TRACK_prefix',
        'oracle_bisect_questions','oracle_bisect_F1','oracle_dense_F1','noisy_bisect_questions','noisy_bisect_F1','noisy_dense_F1']}
    handles={name:file_writer(dest/name,fields) for name,fields in columns.items()}
    def write(name,r):handles[name][1].writerow(r)
    ycache=np.lib.format.open_memmap(dest/'observation_labels.npy',mode='w+',dtype=bool,shape=(len(queries),len(rows)))
    gt_inputs=[];fnr=noise['FNR'];fpr=noise['FPR'];model_seals=[]
    for qi,q in enumerate(queries):
        query=q['expression_id'];gt_path=root/'gt_template'/video/query/'gt.txt';gt_inputs.append(record(gt_path))
        truth=load_eval_gt(root,video,query);g_total=sum(map(len,truth.values()));gt_present=int(g_total>0)
        labels=np.zeros(len(rows),dtype=bool)
        for t,idx in zip(frame_ids,frame_parts):
            gt=truth.get(t,[])
            if not gt:continue
            _,chosen=matched_gt(overlaps(boxes[idx],[box for _,box in gt]),list(range(len(idx))))
            if chosen:labels[idx[np.asarray(sorted(chosen),dtype=int)]]=True
        ycache[qi]=labels
        successes=np.asarray([int(labels[idx].sum()) for idx in tracks]);sizes=np.asarray([len(idx) for idx in tracks])
        Y=2*successes>=sizes;varying=(10*successes>sizes)&(10*successes<9*sizes)
        seg_success=np.asarray([int(labels[idx].sum()) for idx in segments]);seg_sizes=np.asarray([len(idx) for idx in segments])
        seg_Y=2*seg_success>=seg_sizes
        noisy=flip_labels(Y,target,video,query,fnr,fpr)
        for j,idx in enumerate(tracks):
            write('units.csv',dict(target=target,video=video,query=query,role=ROLE,track_id=int(keep_ids[j]),observations=len(idx),
                first_frame=int(frames[idx[0]]),last_frame=int(frames[idx[-1]]),matched_observations=int(successes[j]),
                matched_fraction=float(successes[j]/sizes[j]),Y=int(Y[j]),time_varying=int(varying[j]),segments=segment_tracks.count(j)))
        for model in MODELS:
            base=dict(target=target,video=video,query=query,model=model,role=ROLE)
            scores=np.asarray(arrays[model][q['start']:q['stop']]);track_scores={m:[] for m in AGGS}
            for idx in tracks:
                aa=aggregate(scores[idx])
                for m in AGGS:track_scores[m].append(aa[m])
            segment_scores=np.asarray([float(scores[idx].mean()) for idx in segments])
            seg_auc=auc(segment_scores,seg_Y);track_aucs={m:auc(track_scores[m],Y) for m in AGGS}
            write('auroc_segment.csv',base|dict(objects=len(segments),positives=int(seg_Y.sum()),AUROC=seg_auc))
            for m in AGGS:
                write('auroc_track.csv',base|dict(aggregation=m,objects=len(tracks),positives=int(Y.sum()),AUROC=track_aucs[m]))
                write('presence_scores.csv',base|dict(aggregation=m,score=max(track_scores[m]) if tracks else None,
                    GT_present=gt_present,positive_tracks=int(Y.sum())))
            frame_aucs=[];frame_prefix=[]
            for t,idx in zip(frame_ids,retained_frames):
                yy=labels[idx];a=auc(scores[idx],yy);p,k=prefix_ceiling(scores[idx],yy,ids[idx])
                if a is not None:frame_aucs.append(a)
                frame_prefix.append(p)
                write('auroc_frame.csv',base|dict(frame=t,objects=len(idx),positives=int(yy.sum()),AUROC=a,prefix_ceiling_F1=p,prefix_best_k=k))
            track_prefix,_=prefix_ceiling(track_scores['MEAN'],Y,keep_ids)
            frame_auc=float(np.mean(frame_aucs)) if frame_aucs else None
            frame_ceiling=float(np.mean(frame_prefix))
            for grain,value in [('FRAME',frame_ceiling),('TRACK',track_prefix)]:
                write('prefix_ceiling.csv',base|dict(granularity=grain,aggregation='MEAN' if grain=='TRACK' else 'NATIVE_MARGIN',
                    F1=value,frame_units=len(frame_ids),GT_present=gt_present,positive_tracks=int(Y.sum())))
            controls={}
            for verifier,ans in [('ORACLE',Y),('NOISY_SIMULATION',noisy)]:
                result=bisect_result(track_scores['MEAN'],Y,ans,keep_ids)
                dense=dict(questions=len(tracks),F1=binary_f1(np.flatnonzero(ans),Y),selected_count=int(ans.sum()),
                    asked_positions='',asked_track_ids='',answers='')
                if verifier=='ORACLE' and dense['F1']!=1:raise ValueError('Perfect dense track verifier must select exactly Y')
                controls[verifier]=dict(bisect=result,dense=dense)
                for name,r in [('BISECT_EXACT',result),('DENSE',dense)]:
                    write('bisect_track.csv',base|dict(verifier=verifier,method=name,tracks=len(tracks),positive_tracks=int(Y.sum()),**r,GT_present=gt_present))
            write('sequence_metrics.csv',base|dict(tracks=len(tracks),segments=len(segments),frames=len(frame_ids),
                GT_present=gt_present,GT_observations=g_total,positive_observations=int(labels[eligible].sum()),positive_tracks=int(Y.sum()),positive_segments=int(seg_Y.sum()),
                time_varying_tracks=int(varying.sum()),frame_defined=len(frame_aucs),frame_undefined=len(frame_ids)-len(frame_aucs),
                FRAME_AUROC_qmean=frame_auc,SEGMENT_AUROC=seg_auc,**{m+'_AUROC':track_aucs[m] for m in AGGS},
                common_AUROC_q=int(frame_auc is not None and seg_auc is not None and track_aucs['MEAN'] is not None),
                FRAME_prefix_qmean=frame_ceiling,TRACK_prefix=track_prefix,
                oracle_bisect_questions=controls['ORACLE']['bisect']['questions'],oracle_bisect_F1=controls['ORACLE']['bisect']['F1'],oracle_dense_F1=controls['ORACLE']['dense']['F1'],
                noisy_bisect_questions=controls['NOISY_SIMULATION']['bisect']['questions'],noisy_bisect_F1=controls['NOISY_SIMULATION']['bisect']['F1'],noisy_dense_F1=controls['NOISY_SIMULATION']['dense']['F1']))
    ycache.flush();del ycache
    for stream,_ in handles.values():stream.close()
    outputs=[record(dest/name) for name in columns]+[record(dest/'observation_labels.npy'),record(dest/'layout.json')]
    receipt_record=dict(status='ALL_MODELS_VIDEO_DIAGNOSTIC_COMPLETE',target=target,video=video,role=ROLE,queries=len(queries),
        layout=layout,GT_inputs=gt_inputs,outputs=outputs,noise_rates_sha256=sha(RUN/'noise_rates.json'),
        new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0,CPU_wall_seconds=time.perf_counter()-start)
    dump(receipt,receipt_record)
    print(json.dumps(dict(event='TRACK_VIDEO_COMPLETE',target=target,video=video,queries=len(queries),tracks=len(tracks),
        segments=len(segments),seconds=receipt_record['CPU_wall_seconds'])),flush=True)
    return receipt_record


def main(target):
    os.environ['CUDA_VISIBLE_DEVICES']='';require_plan();start=time.perf_counter()
    plan=json.loads((RUN/'plan.json').read_text());rates=json.loads((RUN/'noise_rates.json').read_text())['rates'][target]
    meta,arrays,_,records=dc.load_target(target);receipts=[]
    for video,queries in dc.video_queries(meta).items():receipts.append(evaluate_video(target,video,queries,meta,arrays,rates))
    result=dict(status='TARGET_ALL_THREE_MODELS_COMPLETE',target=target,role=ROLE,videos=len(receipts),
        queries=sum(r['queries'] for r in receipts),video_receipts=[record(RUN/'targets'/target/r['video']/'receipt.json') for r in receipts],
        inputs=list(records.values()),new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0,CPU_wall_seconds=time.perf_counter()-start)
    dump(RUN/'targets'/target/'completion.json',result)
    print(json.dumps(dict(event='TRACK_TARGET_COMPLETE',target=target,queries=result['queries'],seconds=result['CPU_wall_seconds'])),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--target',choices=TARGETS,required=True);main(p.parse_args().target)
