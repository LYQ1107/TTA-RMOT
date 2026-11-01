"""Fixed official-checkpoint rank/count diagnostics. Predict and GT stages are separate processes."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import *
from scripts.diag_attrib_v2_common import rank_within_groups,top_budget,write_csv,write_new
from scripts.diag_attrib_audit import prediction

V2=ROOT/'results/runs/diag_attrib_20261031_v2'
DANCE=Path('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')
DANCE_TRACKS=Path('/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs')
CONFIGS=('NATIVE','IN_DOMAIN','R_SxK_I','R_IxK_S','R_SxK_ORACLE','R_IxK_ORACLE','B0_GLOBAL')
DEFAULT_SOURCE={'Dance':'V2-best','KITTI-V2':'Dance-best','SportsMOT':'V2-best'}
IN_DOMAIN_MODEL={'Dance':'Dance-best','KITTI-V2':'V2-best','SportsMOT':'LaMOT-best'}


def target_root(target):
    if target=='Dance':return DANCE
    if target=='KITTI-V2':return SOURCE
    publication=require_plan()
    if publication['revision']!=SPORTS_GT_V4_REVISION:
        raise PermissionError('Sports evaluation requires the newly authorized v4 GT protocol')
    gate=json.loads((PUBLIC/'sports_GT_reconstruction_v4.json').read_text())
    if gate['status']!=SPORTS_GT_V4_STATUS or gate['prereg_publication']!=publication['remote_commit']:
        raise PermissionError('Sports GT v4 reconstruction/publication has not passed')
    return Path(gate['evaluation_root'])


def inputs(target,source_model=None):
    require_plan()
    model=source_model or DEFAULT_SOURCE[target]
    if target=='Dance':
        from flex_tta.dance_io import load_tracker_lines
        meta=json.loads((V2/'score_bank/bank.json').read_text());meta['group_count']=meta['query_frame_groups']
        videos={}
        for v in sorted({q['video_id'] for q in meta['episodes']}):
            tracks=load_tracker_lines(DANCE_TRACKS,v)
            obs=[(t,i) for t,rows in tracks.items() for i in rows]
            videos[v]=dict(observations=obs,lines=[tracks[t][i] for t,i in obs])
        meta['videos']=videos
        paths=[V2/'score_bank/FROZEN.npy',V2/'score_bank/OFFICIAL_DANCE_COMPLETE.npy']
        for p in paths:
            if file_sha(p)!=meta['cache_files_sha256'][p.name]:raise ValueError('Changed sealed Dance scores')
        groups=np.load(V2/'score_bank/groups.npy',mmap_mode='r')
        if model!='V2-best':
            seal=json.loads((RUN/'dance'/model/'scores_seal.json').read_text())
            p=Path(seal['complete_path'])
            if seal['GT_open_attempts'] or file_sha(p)!=seal['complete_sha256']:raise ValueError('Unsealed Dance source scores')
            paths[0]=p
    else:
        key='sports' if target=='SportsMOT' else 'kitti'
        meta=json.loads((RUN/key/'metadata.json').read_text());paths=[]
        groups=np.load(RUN/key/'groups.npy',mmap_mode='r')
        for m in (model,IN_DOMAIN_MODEL[target]):
            seal=json.loads((RUN/key/m/'scores_seal.json').read_text())
            p=Path(seal['complete_path'])
            if seal['GT_open_attempts'] or file_sha(p)!=seal['complete_sha256']:raise ValueError('Unsealed KITTI scores')
            paths.append(p)
        for q in meta['episodes']:
            obs=np.asarray(meta['videos'][q['video_id']]['observations']);frame=obs[:,0]+(1 if target=='KITTI-V2' else 0)
            unique,start=np.unique(frame,return_index=True);stop=np.r_[start[1:],len(frame)]
            q['groups']=[[int(t),q['start']+int(a),q['start']+int(b),int(groups[q['start']+a])] for t,a,b in zip(unique,start,stop)]
    arrays=[np.load(p,mmap_mode='r') for p in paths]
    return meta,arrays,groups,paths


def base(target,source_model=None):
    original=RUN/'diagnostics'/target
    if target!='SportsMOT' and (source_model is None or source_model==DEFAULT_SOURCE[target]):return original
    return original/(source_model or DEFAULT_SOURCE[target])


def seal_prediction(target,name,mask,meta,*,oracle=False,source_model=None):
    folder=base(target,source_model)/name;episodes=[]
    for q in meta['episodes']:
        v,e=q['video_id'],q['expression_id'];bank=meta['videos'][v]
        lines=[bank['lines'][k] for k in np.flatnonzero(mask[q['start']:q['stop']])]
        path=folder/'results'/v/e/'predict.txt';write_new(path,''.join(x+'\n' for x in lines))
        episodes.append(dict(video=v,expression=e,prediction_path=str(path),prediction_sha256=file_sha(path),selected_rows=len(lines)))
    record=dict(status='SEALED',target=target,method=name,episodes=episodes,episode_count=len(episodes),
        prediction_rows=sum(e['selected_rows'] for e in episodes),GT_open_attempts=None if oracle else 0,
        oracle=oracle,role='ORACLE_ONLY_NOT_A_METHOD' if oracle else 'POST_HOC_OFFICIAL_RANK_COUNT_DIAGNOSTIC',
        optimizer_updates=0,teacher_calls=0,immutable_MOT=True)
    atomic_json(folder/'predictions_seal.json',record)
    return record


def verify(target,name,source_model=None):
    folder=base(target,source_model)/name;s=json.loads((folder/'predictions_seal.json').read_text())
    if s['episode_count']!={'Dance':425,'KITTI-V2':862,'SportsMOT':466}[target]:raise ValueError('Incomplete episode coverage')
    for e in s['episodes']:
        if file_sha(e['prediction_path'])!=e['prediction_sha256']:raise ValueError('Prediction changed after seal')
    return s


def generate(target,source_model=None):
    attempts=guard();started=time.perf_counter();meta,(source,inside),groups,paths=inputs(target,source_model);g=meta['group_count']
    model=source_model or DEFAULT_SOURCE[target]
    rs=rank_within_groups(source,groups,g);ri=rank_within_groups(inside,groups,g)
    ks=np.bincount(groups,weights=source>0,minlength=g).astype(np.int32)
    ki=np.bincount(groups,weights=inside>0,minlength=g).astype(np.int32)
    masks=dict(NATIVE=source>0,IN_DOMAIN=inside>0,R_SxK_I=rs<ki[groups],R_IxK_S=ri<ks[groups],
        B0_GLOBAL=top_budget(source,int(ki.sum()))[0])
    for name,mask in masks.items():seal_prediction(target,name,mask,meta,source_model=source_model)
    extra=[]
    if target in ('KITTI-V2','SportsMOT') or model=='LaMOT-best':
        key={'KITTI-V2':'kitti','SportsMOT':'sports','Dance':'dance'}[target]
        hist=np.load(RUN/key/'historical.npy')
        for m,a in [(IN_DOMAIN_MODEL[target],inside),(model,source)]:
            name='HIST_'+m;seal_prediction(target,name,(a>0)&hist,meta,source_model=source_model);extra.append(name)
    if target=='KITTI-V2':
        old=ROOT/'results/runs/xquery_decode_20261101/source_boundary/evaluation/FROZEN__NATIVE'
        sealed=verify(target,'HIST_V2-best',source_model)
        for e in sealed['episodes']:
            p=old/'results'/e['video']/e['expression']/'predict.txt'
            if prediction(p)!=prediction(e['prediction_path']):raise ValueError('STOP: historical KITTI native differs')
        if sealed['prediction_rows']!=750834:raise ValueError('Historical KITTI row count differs')
    if attempts:raise ValueError('GT opened by rank/count decoder')
    record=dict(status='ALL_GT_FREE_PREDICTIONS_SEALED',target=target,methods=list(masks),historical_methods=extra,
        score_paths=[dict(path=str(p),sha256=file_sha(p)) for p in paths],GT_open_attempts=0,
        CPU_wall_seconds=time.perf_counter()-started)
    atomic_json(base(target,source_model)/'GT_free_predictions.json',record)
    print(json.dumps(dict(event='MULTIDOMAIN_GT_FREE_SEALED',target=target,methods=list(masks),GT_open_attempts=0)),flush=True)


def oracle(target,source_model=None):
    # No caller from inference imports this function or its GT readers.
    record=json.loads((base(target,source_model)/'GT_free_predictions.json').read_text())
    if record['GT_open_attempts']:raise ValueError('Decoder GT isolation failed')
    for name in record['methods']:verify(target,name,source_model)
    from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt
    meta,(source,inside),groups,_=inputs(target,source_model);root=target_root(target)
    g=meta['group_count'];count=np.zeros(g,dtype=np.int32);labels=np.zeros(len(groups),dtype=bool)
    all_gt=[];trackrecords=[];fractions=[];retrieval={m:dict(valid_frames=0,recall_at_1_sum=0.,hit_at_1_sum=0,
        GT_valid_frames=0,GT_recall_at_1_sum=0.) for m in ('source','in_domain')}
    for v,bank in meta['videos'].items():
        obs=np.asarray(bank['observations']);ids=obs[:,1];unique,inverse=np.unique(ids,return_inverse=True)
        positives=np.zeros(len(unique),dtype=int);box=[]
        for line in bank['lines']:
            x,y,w,h=list(map(float,line.split(',')))[2:6];box.append([x,y,x+w,y+h])
        box=np.asarray(box)
        for q in [q for q in meta['episodes'] if q['video_id']==v]:
            truth=load_eval_gt(root,v,q['expression_id']);total=sum(map(len,truth.values()))
            all_gt.append(dict(video=v,expression=q['expression_id'],GT_observations=total,GT_frames=len(truth),video_frames=q['frame_count'],present=total>0))
            for r in retrieval.values():r['GT_valid_frames']+=len(truth)
            if total:fractions.append(len(truth)/q['frame_count'])
            matched=np.zeros(len(ids),dtype=bool)
            for t,a,b,group in q['groups']:
                count[group]=len(truth.get(t,[]));ia,ib=a-q['start'],b-q['start']
                iou=overlaps(box[ia:ib],[bb for _,bb in truth.get(t,[])])
                _,matched_idx=matched_gt(iou,list(range(ib-ia)))
                if matched_idx:matched[ia+np.asarray(sorted(matched_idx))]=True
                y=(iou>=.5).any(axis=1) if iou.shape[1] else np.zeros(ib-ia,dtype=bool)
                labels[a:b]=y;p=int(y.sum())
                if p:
                    for name,array in [('source',source),('in_domain',inside)]:
                        rank=sorted(range(b-a),key=lambda j:(-array[a+j],int(ids[ia+j])))
                        hit=int(y[rank[0]]);retrieval[name]['valid_frames']+=1
                        retrieval[name]['recall_at_1_sum']+=hit/p;retrieval[name]['hit_at_1_sum']+=hit
                        retrieval[name]['GT_recall_at_1_sum']+=hit/len(truth[t])
            successes=np.bincount(inverse,weights=matched,minlength=len(unique));n=np.bincount(inverse,minlength=len(unique))
            positives+=(2*successes>=n)
        trackrecords.extend(dict(video=v,track_id=int(i),positive_queries=int(n)) for i,n in zip(unique,positives))
        print(json.dumps(dict(event='MULTIDOMAIN_ORACLE_VIDEO',target=target,video=v)),flush=True)
    for name,array in [('R_SxK_ORACLE',source),('R_IxK_ORACLE',inside)]:
        rank=rank_within_groups(array,groups,g);seal_prediction(target,name,rank<count[groups],meta,oracle=True,source_model=source_model)
    for r in retrieval.values():
        r.update(R_at_1=r['recall_at_1_sum']/max(r['valid_frames'],1),Hit_at_1=r['hit_at_1_sum']/max(r['valid_frames'],1),
            GT_R_at_1=r['GT_recall_at_1_sum']/max(r['GT_valid_frames'],1))
    hist=Counter(min(2,r['positive_queries']) for r in trackrecords);n=len(trackrecords)
    from scripts.analyze_pf_full_offline import auroc
    result=dict(status='OFFLINE_ORACLE_COMPLETE',target=target,official_GT_detections=sum(e['GT_observations'] for e in all_gt),
        GT_protocol=SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
        episodes=all_gt,track_records=trackrecords,structure=dict(physical_tracks=n,zero=hist[0]/n,one=hist[1]/n,two_or_more=hist[2]/n,
            empty_query_fraction=sum(not e['present'] for e in all_gt)/len(all_gt),GT_frame_fraction_present_mean=float(np.mean(fractions)),
            GT_frame_fraction_present_p50=float(np.median(fractions)),query_present_episodes=sum(e['present'] for e in all_gt),query_empty_episodes=sum(not e['present'] for e in all_gt)),
        retrieval=retrieval,source_AUROC=auroc(source,labels),in_domain_AUROC=auroc(inside,labels),
        GT_feedback_to_inference=False,teacher_calls=0,optimizer_updates=0)
    if target=='Dance' and (result['official_GT_detections']!=212541 or result['structure']['query_present_episodes']!=40):
        raise ValueError('STOP: Dance official protocol alignment differs')
    atomic_json(base(target,source_model)/'oracle.json',result)


def reusable(target,name,source_model=None):
    model=source_model or DEFAULT_SOURCE[target]
    if target=='Dance' and model=='V2-best':
        mapping=dict(NATIVE='M_FROZEN__FROZEN',IN_DOMAIN='OFFICIAL_DANCE_COMPLETE',R_SxK_I='M_FROZEN__OFFICIAL_DANCE_COMPLETE',
            R_IxK_S='M_OFFICIAL_DANCE_COMPLETE__FROZEN',R_SxK_ORACLE='M_FROZEN__ORACLE_GT',R_IxK_ORACLE='M_OFFICIAL_DANCE_COMPLETE__ORACLE_GT',B0_GLOBAL='B0_NATIVE_OFFICIAL_COMPLETE_COUNT')
        return V2/'evaluation'/mapping[name]
    if target=='Dance' and name=='IN_DOMAIN':return V2/'evaluation/OFFICIAL_DANCE_COMPLETE'
    if target=='Dance' and name=='HIST_Dance-best':
        return V2/'evaluation/HIST_OFFICIAL_DANCE_COMPLETE__OFFICIAL_DANCE_COMPLETE'
    if target=='Dance' and name=='R_IxK_ORACLE':
        return V2/'evaluation/M_OFFICIAL_DANCE_COMPLETE__ORACLE_GT'
    if target=='KITTI-V2' and name=='HIST_V2-best':return OLD_SOURCE.parent/'source_boundary/evaluation/FROZEN__NATIVE'
    if target=='KITTI-V2' and model!='Dance-best' and name=='R_IxK_ORACLE':
        return base(target)/name
    if target=='SportsMOT' and name in ('IN_DOMAIN','R_IxK_ORACLE','HIST_LaMOT-best') and model!='V2-best':
        previous=base(target,'V2-best')/name
        if (previous/'official_completion.json').exists():return previous
    if target=='KITTI-V2' and name=='IN_DOMAIN' and (base(target)/'V2_COMPLETE_REFERENCE/official_completion.json').exists():
        return base(target)/'V2_COMPLETE_REFERENCE'
    return None


def evaluate(target,names=None,source_model=None):
    from scripts.pf_diag_common import official_metrics
    reference_only=names is not None
    if names is None:
        oracle_record=json.loads((base(target,source_model)/'oracle.json').read_text())
        prep=json.loads((base(target,source_model)/'GT_free_predictions.json').read_text());names=[*CONFIGS,*prep['historical_methods']]
    def job(name):
        seal=verify(target,name,source_model);folder=base(target,source_model)/name;done=folder/'official_completion.json'
        if done.exists():
            r=json.loads(done.read_text())
            if file_sha(folder/'results/pedestrian_detailed.csv')!=r['detailed_sha256']:raise ValueError('Changed evaluated CSV')
            return r
        old=reusable(target,name,source_model);begin=time.perf_counter();reused=False
        if old is not None:
            olddone=json.loads((old/'official_completion.json').read_text())
            if target=='SportsMOT' and olddone.get('GT_protocol')!=SPORTS_GT_PROTOCOL:
                raise ValueError('Do not mix SportsMOT GT protocols')
            for f,k in [('pedestrian_summary.txt','summary_sha256'),('pedestrian_detailed.csv','detailed_sha256')]:
                if file_sha(old/'results'/f)!=olddone[k]:raise ValueError('Changed reused evaluation')
            for e in seal['episodes']:
                if prediction(e['prediction_path'])!=prediction(old/'results'/e['video']/e['expression']/'predict.txt'):
                    raise ValueError('STOP: proposed reference reuse differs, preserve and report '+name)
            for f in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(old/'results'/f,folder/'results'/f)
            reused=True;code=0
        else:
            root=target_root(target);seq={'Dance':'dance','KITTI-V2':'kitti-2','SportsMOT':'SportsMOT'}[target]
            for e in seal['episodes']:
                link=folder/'results'/e['video']/e['expression']/'gt.txt'
                if not link.exists():link.symlink_to(root/'gt_template'/e['video']/e['expression']/'gt.txt')
            command=[sys.executable,'./TrackEval/scripts/run_mot_challenge.py','--METRICS','HOTA','CLEAR','Identity',
                '--SEQMAP_FILE',str(VENDOR/'seqmaps'/(seq+'.txt')),'--SKIP_SPLIT_FOL','True','--GT_FOLDER',str(root),
                '--TRACKERS_FOLDER',str(folder/'results'),'--GT_LOC_FORMAT','{gt_folder}{video_id}/{expression_id}/gt.txt',
                '--TRACKERS_TO_EVAL',str(folder/'results'),'--USE_PARALLEL','True','--NUM_PARALLEL_CORES','2','--PLOT_CURVES','False']
            with (folder/'trackeval.log').open('w') as h:code=subprocess.call(command,cwd=VENDOR,stdout=h,stderr=subprocess.STDOUT)
            if code:raise RuntimeError('Official TrackEval failed: '+str(folder/'trackeval.log'))
        metrics=official_metrics(folder/'results')
        if int(metrics['Dets'])!=seal['prediction_rows']:raise ValueError('Official rows differ from sealed candidates')
        if target=='KITTI-V2' and name=='HIST_V2-best' and metrics['HOTA']!=42.526:raise ValueError('STOP: historical KITTI 42.526 not reproduced')
        r=dict(status='PASS',target=target,method=name,metrics=metrics,prediction_rows=seal['prediction_rows'],exit_code=code,
            GT_protocol=SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
            reused_after_all_episode_row_equality=reused,reuse_folder=str(old) if reused else None,wall_seconds=time.perf_counter()-begin,
            summary_sha256=file_sha(folder/'results/pedestrian_summary.txt'),detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'),
            prediction_seal_sha256=file_sha(folder/'predictions_seal.json'))
        atomic_json(done,r);print(json.dumps(dict(event='MULTIDOMAIN_OFFICIAL_COMPLETE',**r)),flush=True);return r
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in as_completed([pool.submit(job,m) for m in names])]
    atomic_json(base(target,source_model)/('reference_completion.json' if reference_only else 'completion.json'),dict(status='SOURCE_REFERENCE_COMPLETE' if reference_only else 'ALL_AUTHORIZED_CONFIGURATIONS_COMPLETE',target=target,methods=results,
        target_GT_only_independent_offline=True,GT_feedback=False))


def reference_predict(target):
    if target!='KITTI-V2':raise ValueError('Only KITTI source-tail completion needs a new reference')
    require_plan();attempts=guard();meta=json.loads((RUN/'kitti/metadata.json').read_text())
    record=json.loads((RUN/'kitti/V2-best/scores_seal.json').read_text())
    if file_sha(record['complete_path'])!=record['complete_sha256'] or not record['historical_score_bit_exact']:
        raise ValueError('Complete source seal/history gate failed')
    a=np.load(record['complete_path'],mmap_mode='r');h=np.load(RUN/'kitti/historical.npy')
    seal_prediction(target,'V2_COMPLETE_REFERENCE',a>0,meta)
    seal_prediction(target,'HIST_V2-best',(a>0)&h,meta)
    if attempts:raise ValueError('Source reference generator attempted GT')
    atomic_json(base(target)/'reference_predictions.json',dict(status='SEALED_BEFORE_GT',methods=['V2_COMPLETE_REFERENCE','HIST_V2-best'],GT_open_attempts=0,
        source_score_seal_sha256=file_sha(RUN/'kitti/V2-best/scores_seal.json')))


def run_phase(phase,target,source_model=None):
    # evaluate's second positional argument is an optional method list, not a
    # source model. Keep existing reference-evaluation callers compatible.
    if phase=='evaluate':return evaluate(target,source_model=source_model)
    if phase=='reference_predict':return reference_predict(target)
    return {'predict':generate,'oracle':oracle}[phase](target,source_model)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('phase',choices=('predict','oracle','evaluate','reference_predict','reference_evaluate'));p.add_argument('--target',choices=('Dance','KITTI-V2','SportsMOT'),required=True)
    p.add_argument('--source-model',choices=('V2-best','Dance-best','LaMOT-best'));a=p.parse_args()
    if a.phase=='reference_evaluate':
        r=json.loads((base(a.target)/'reference_predictions.json').read_text())
        if r['GT_open_attempts']:raise ValueError('Source reference GT isolation failed')
        evaluate(a.target,r['methods'])
    else:run_phase(a.phase,a.target,a.source_model)
