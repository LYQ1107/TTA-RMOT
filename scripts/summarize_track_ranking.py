"""All fixed aggregation variants, paired-video uncertainty and literal gates."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *

NAMES=('units.csv','auroc_frame.csv','auroc_segment.csv','auroc_track.csv','prefix_ceiling.csv',
       'bisect_track.csv','presence_scores.csv','sequence_metrics.csv')


def numeric(v):
    return float(v) if pd.notna(v) else None


def merge_csv(name,inputs,path):
    path.parent.mkdir(parents=True,exist_ok=True);rows=0;header=None
    with path.open('w',newline='') as out:
        writer=csv.writer(out,lineterminator='\n')
        for directory in inputs:
            with (directory/name).open(newline='') as source:
                reader=csv.reader(source);h=next(reader)
                if header is None:header=h;writer.writerow(h)
                elif header!=h:raise ValueError('Changed video schema: '+name)
                for r in reader:writer.writerow(r);rows+=1
    return record(path)|dict(rows=rows)


def ci(samples):
    samples=np.asarray(samples);good=samples[np.isfinite(samples)]
    return dict(CI95=np.percentile(good,[2.5,97.5]).tolist() if len(good) else None,
                valid_bootstrap_samples=len(good),samples=2000,seed=0,unit='whole video')


def paired_samples(part,field,videos,draws):
    part=part.dropna(subset=[field]);count=part.groupby('video')[field].count().reindex(videos,fill_value=0).to_numpy(dtype=float)
    sums=part.groupby('video')[field].sum().reindex(videos,fill_value=0).to_numpy(dtype=float)
    den=draws@count
    return np.divide(draws@sums,den,out=np.full(len(draws),np.nan),where=den>0)


def decisions(frame_auc,track_auc,ceiling):
    if frame_auc is None or track_auc is None:
        first='规则未覆盖：共同q没有双类AUROC';delta=None
    else:
        delta=track_auc-frame_auc
        first=('轨迹级排序可靠' if track_auc>=.75 else '轨迹级更好但仍不够可靠') if delta>=.05 else '轨迹级没有明显改善'
    second='前缀约束可接受' if ceiling>=.8 else '前缀约束代价过大'
    return first,second,delta


def summarize_protocol(protocol,data,all_videos,draws):
    auc_rows=[];prefix_rows=[];bisect_rows=[];variability=[];comparison=[];boot=[]
    for (target,model),g in data.groupby(['target','model'],sort=True):
        common=g[g.common_AUROC_q==1];base=dict(protocol=protocol,target=target,model=model,role=ROLE,
            sequences=len(g),common_sequences=len(common),GT_present_sequences=int(g.GT_present.sum()))
        frame_count=int(g.frame_defined.sum());frame_undef=int(g.frame_undefined.sum())
        raw=(g.FRAME_AUROC_qmean*g.frame_defined).sum()/frame_count if frame_count else None
        auc_rows.append(base|dict(granularity='FRAME',aggregation='MARGIN',scope='ALL_DEFINED_FRAMES',
            mean_AUROC=numeric(raw),defined_units=frame_count,undefined_units=frame_undef))
        for scope,d in [('ALL_DEFINED_Q',g),('COMMON_Q',common)]:
            for grain,agg,field in [('FRAME','MARGIN','FRAME_AUROC_qmean'),('SEGMENT','MEAN','SEGMENT_AUROC'),
                                    *[('TRACK',a,a+'_AUROC') for a in AGGS]]:
                auc_rows.append(base|dict(granularity=grain,aggregation=agg,scope=scope,mean_AUROC=numeric(d[field].mean()),
                    defined_units=int(d[field].notna().sum()),undefined_units=int(d[field].isna().sum())))
        for scope,d in [('ALL',g),('GT_PRESENT',g[g.GT_present==1]),('GT_EMPTY',g[g.GT_present==0]),
                        ('Y_POSITIVE',g[g.positive_tracks>0]),('Y_ZERO',g[g.positive_tracks==0]),('COMMON_AUROC_Q',common)]:
            for grain,field in [('FRAME','FRAME_prefix_qmean'),('TRACK','TRACK_prefix')]:
                prefix_rows.append(base|dict(scope=scope,granularity=grain,sequences_in_scope=len(d),mean_prefix_F1=numeric(d[field].mean())))
            for verifier,prefix in [('ORACLE','oracle'),('NOISY_SIMULATION','noisy')]:
                n=numeric(d.tracks.mean());qs=numeric(d[prefix+'_bisect_questions'].mean());f=numeric(d[prefix+'_bisect_F1'].mean());dense=numeric(d[prefix+'_dense_F1'].mean())
                bisect_rows.append(base|dict(scope=scope,verifier=verifier,sequences_in_scope=len(d),mean_tracks=n,
                    mean_bisect_questions=qs,mean_dense_questions=n,questions_ratio=qs/n if n else None,
                    mean_bisect_F1=f,mean_dense_F1=dense,F1_ratio=f/dense if dense else None))
        for scope,d in [('ALL',g),('GT_PRESENT',g[g.GT_present==1]),('GT_EMPTY',g[g.GT_present==0])]:
            total=int(d.tracks.sum());var=int(d.time_varying_tracks.sum());fraction=var/total if total else None
            variability.append(base|dict(scope=scope,q_track_pairs=total,time_varying_pairs=var,fraction=fraction,
                annotation='该目标需要片段级而非轨迹级的判断' if scope=='ALL' and fraction is not None and fraction>.2 else ''))
        fa=numeric(common.FRAME_AUROC_qmean.mean());ta=numeric(common.MEAN_AUROC.mean());ceiling=float(g.TRACK_prefix.mean())
        t1,t2,delta=decisions(fa,ta,ceiling)
        comparison.append(base|dict(FRAME_AUROC=fa,SEGMENT_AUROC=numeric(common.SEGMENT_AUROC.mean()),TRACK_MEAN_AUROC=ta,
            AUROC_delta=delta,TRACK_prefix_F1=ceiling,FRAME_prefix_F1=float(g.FRAME_prefix_qmean.mean()),
            TRACK_prefix_GT_present=numeric(g[g.GT_present==1].TRACK_prefix.mean()),
            T1=t1,T2=t2,joint_pass=t1=='轨迹级排序可靠' and t2=='前缀约束可接受'))
        if target!='KITTI-V2':
            d=common.copy();d['difference']=d.MEAN_AUROC-d.FRAME_AUROC_qmean
            p=g.copy();p['difference']=p.TRACK_prefix-p.FRAME_prefix_qmean
            for metric,part,point in [('TRACK_MEAN_minus_FRAME_AUROC',d,delta),
                    ('TRACK_minus_FRAME_prefix_F1',p,float(p.difference.mean()))]:
                boot.append(base|dict(metric=metric,point=point,clusters=len(all_videos[target]),
                    **ci(paired_samples(part,'difference',all_videos[target],draws[target]))))
    return dict(auroc=auc_rows,prefix=prefix_rows,bisect=bisect_rows,variability=variability,
                comparison=comparison,bootstrap=boot)


def main():
    require_plan();start=time.perf_counter();receipts=[];directories=[]
    for target in TARGETS:
        completion=json.loads((RUN/'targets'/target/'completion.json').read_text())
        if completion['status']!='TARGET_ALL_THREE_MODELS_COMPLETE':raise ValueError('Incomplete target')
        for entry in completion['video_receipts']:
            path=checked(entry);r=json.loads(path.read_text())
            for output in r['outputs']:checked(output)
            receipts.append(r);directories.append(path.parent)
    machine=RUN/'machine';index={name:merge_csv(name,directories,machine/name) for name in NAMES}
    data=pd.read_csv(machine/'sequence_metrics.csv',dtype={'video':str})
    all_videos={t:sorted({r['video'] for r in receipts if r['target']==t}) for t in TARGETS}
    draws={t:np.random.default_rng(0).multinomial(len(v),np.ones(len(v))/len(v),size=2000) for t,v in all_videos.items() if t!='KITTI-V2'}
    main=summarize_protocol('FULL_MAIN_V4',data,all_videos,draws)
    from scripts import decompose_common as dc
    missing=set(json.loads((dc.OLD_PUBLIC/'sports_template_inventory.json').read_text())['missing_GT'])
    sports=data[data.target=='SportsMOT'].copy();sports['sequence_id']=sports.video+'+'+sports['query']
    subset=sports[~sports.sequence_id.isin(missing)]
    if len(subset[['video','query']].drop_duplicates())!=418:raise ValueError('Sports sensitivity must contain exactly original 418 IDs')
    sensitive=summarize_protocol('SPORTS418_V4_CONTENT',subset,all_videos,draws)
    for name in ('auroc','prefix','bisect','variability','comparison'):
        csv_rows(PUBLIC/(name+'_summary.csv'),main[name]+sensitive[name])
    matrix=[];sensitivity_matrix=[]
    for combo,model,target in PAIRS:
        row=next(r for r in main['comparison'] if r['target']==target and r['model']==model)
        matrix.append(dict(combo=combo,**row))
        if target=='SportsMOT':
            rr=next(r for r in sensitive['comparison'] if r['model']==model)
            sensitivity_matrix.append(dict(combo=combo,**rr))
    csv_rows(PUBLIC/'verdict_matrix.csv',matrix);csv_rows(PUBLIC/'sports418_verdicts.csv',sensitivity_matrix)
    passed=sum(r['joint_pass'] for r in matrix);unimproved=sum(r['T1']=='轨迹级没有明显改善' for r in matrix)
    total='两级搜索的前提成立' if passed>=4 else '轨迹聚合不能解决排序问题' if unimproved>=4 else '部分成立'
    target_variability=[]
    for target in TARGETS:
        row=next(r for r in main['variability'] if r['target']==target and r['model']=='V2-best' and r['scope']=='ALL')
        target_variability.append({k:row[k] for k in ('target','q_track_pairs','time_varying_pairs','fraction','annotation')})
    bootstrap=dict(role=ROLE,samples=2000,seed=0,unit='whole video; paired FRAME/TRACK, q-mean estimands',
        KITTI='4 videos, point estimates only',Sports418='same v4 content and original 45-video draw weights',
        comparisons=main['bootstrap']+sensitive['bootstrap'],presence=[])
    presence=pd.read_csv(machine/'presence_scores.csv',dtype={'video':str});presence=presence[presence.target=='Dance']
    presence_rows=[]
    for (video,model,agg),d in presence.groupby(['video','model','aggregation'],sort=True):
        if len(d)!=17:raise ValueError('Dance video must preserve its 17 queries')
        presence_rows.append(dict(target='Dance',video=video,model=model,aggregation=agg,role=ROLE,expressions=17,
            GT_present=int(d.GT_present.sum()),GT_empty=int((d.GT_present==0).sum()),AUROC=auc(d.score,d.GT_present)))
    csv_rows(machine/'presence_auc.csv',presence_rows);index['presence_auc.csv']=record(machine/'presence_auc.csv')|dict(rows=len(presence_rows))
    presence_df=pd.DataFrame(presence_rows);presence_summary=[]
    for (model,agg),d in presence_df.groupby(['model','aggregation'],sort=True):
        r=dict(model=model,aggregation=agg,role=ROLE,videos=len(d),defined_videos=int(d.AUROC.notna().sum()),
            undefined_videos=int(d.AUROC.isna().sum()),mean_AUROC=numeric(d.AUROC.mean()))
        r.update(ci(paired_samples(d,'AUROC',all_videos['Dance'],draws['Dance'])));presence_summary.append(r)
    csv_rows(PUBLIC/'presence_summary.csv',presence_summary);bootstrap['presence']=presence_summary
    csv_rows(PUBLIC/'object_counts_by_video.csv',[{k:r['layout'][k] for k in ('target','video','raw_tracks','retained_tracks','excluded_tracks',
        'raw_observations','retained_observations','excluded_observations','frames','frames_with_zero_retained_candidates','segment_count','fps','segment_length_frames')} for r in receipts])
    objects=[]
    for target in TARGETS:
        rr=[r['layout'] for r in receipts if r['target']==target]
        for grain,values in [('FRAME',[n for r in rr for n in r['frame_candidate_counts']]),('TRACK',[r['retained_tracks'] for r in rr])]:
            objects.append(dict(target=target,granularity=grain,role=ROLE,physical_units=len(values),mean=float(np.mean(values)),
                p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90)),minimum=min(values),maximum=max(values),
                excluded_physical_tracks=sum(r['excluded_tracks'] for r in rr),excluded_physical_observations=sum(r['excluded_observations'] for r in rr)))
    csv_rows(PUBLIC/'object_counts.csv',objects)
    dump(PUBLIC/'bootstrap.json',bootstrap)
    dump(PUBLIC/'verdicts.json',dict(overall=total,matrix=matrix,joint_pass_count=passed,no_improvement_count=unimproved,
        target_time_variability=target_variability,Sports418=sensitivity_matrix,role=ROLE,new_teacher_calls=0,new_model_calls=0,GPU_hours=0))
    dump(RUN/'summary_completion.json',dict(status='ALL_TARGETS_AGGREGATIONS_AND_SENSITIVITY_COMPLETE',role=ROLE,
        machine_files=index,video_receipts=[record(p/'receipt.json') for p in directories],
        sequences=len(data)//3,model_sequence_rows=len(data),models=3,CPU_wall_seconds=time.perf_counter()-start,
        new_teacher_calls=0,new_model_calls=0,GPU_hours=0))
    print(json.dumps(dict(event='TRACK_RANKING_VERDICTS',overall=total,matrix=matrix,variable_targets=target_variability)),flush=True)


if __name__=='__main__':main()
