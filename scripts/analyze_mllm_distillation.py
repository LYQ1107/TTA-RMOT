#!/usr/bin/env python3
"""Independent offline GT attribution, only after complete sealed predictions."""
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.pf_diag_common import (DATA,TRACKS,FROZEN,official_metrics,official_video_values,
    write_csv,load_source_video,candidate_labels)
from scripts.analyze_pf_video_uncertainty import paired_clusters,pooled_hota
from scripts.analyze_pf_full_offline import analyze as old_attribution,load_eval_gt,overlaps,matched_gt,auroc
from scripts.analyze_uda_full import presence_from_stream
from scripts.analyze_uda_text_signal_offline import scores_for
from scripts.diag_pf_query_information import average_precision,binary_summary,frame_metrics,add,finalize
from scripts.evaluate_teacher_probe import comparison_counts
from flex_uda.teacher.mllm_pseudo_dataset import MLLMPseudoDataset

METHODS=('FROZEN','SOURCE_REHEARSAL','MLLM_DISTILL','FROZEN_AT_MLLM_COUNT','MLLM_AT_FROZEN_COUNT')


def binary(scores,labels):
    return dict(candidate_pairs=len(labels),positive_candidates=int(labels.sum()),AUROC=auroc(scores,labels),
        AP=average_precision(scores,labels),selected_count=int((scores>0).sum()))


def calibration(probability,labels):
    """Fixed bins, independent offline labels; never fit a threshold/temperature."""
    p=np.asarray(probability,dtype=float);y=np.asarray(labels,dtype=bool)
    bins=np.minimum((p*10).astype(int),9);parts=[]
    for k in range(10):
        mask=bins==k
        if mask.any():parts.append(dict(bin=k,n=int(mask.sum()),mean_probability=float(p[mask].mean()),official_positive_fraction=float(y[mask].mean())))
    clipped=np.clip(p,1e-12,1-1e-12)
    return dict(Brier=float(np.square(p-y).mean()),log_loss=float(-np.mean(y*np.log(clipped)+(~y)*np.log1p(-clipped))),
        ECE10=float(sum(r['n']*abs(r['mean_probability']-r['official_positive_fraction']) for r in parts)/len(y)),bins=parts,
        warning='official referring labels may be incomplete; this is protocol calibration, not exhaustive human semantic correctness; no recalibration trained')


def text_comparisons(totals,per_video):
    draws=np.random.default_rng(0).integers(0,25,size=(2000,25));comparisons={}
    for method in METHODS[1:3]:
        fields={}
        for first,second,denominator in (('correct_top1',None,'valid_frames'),
            ('correct_top1','all_other_top1','valid_frames'),
            ('mixed_source_AUROC','mixed_visual_AUROC','mixed_frames'),
            ('mixed_source_AUROC','mixed_other_AUROC','mixed_frames'),
            ('hard_disjoint_correct_top1','hard_disjoint_top1','hard_disjoint_frames')):
            def effect(m,ix):
                den=sum(per_video[m][i].get(denominator,0) for i in ix)
                return sum(per_video[m][i].get(first,0)-(per_video[m][i].get(second,0) if second else 0) for i in ix)/den if den else np.nan
            point=effect(method,range(25))-effect('FROZEN',range(25))
            boot=np.asarray([effect(method,d)-effect('FROZEN',d) for d in draws]);boot=boot[np.isfinite(boot)]
            fields[first+('_minus_'+second if second else '')]=dict(delta=float(point) if np.isfinite(point) else None,
                video_cluster_CI95=np.percentile(boot,[2.5,97.5]).tolist() if len(boot) else None,valid_resamples=len(boot))
        comparisons[method+'_vs_FROZEN']=fields
    return dict(methods={m:finalize(t) for m,t in totals.items()},comparisons=comparisons,
        visual_control='mean over17 real-query scores for each candidate, not an independent visual network',
        text_swap='real other expressions can match the same person; disjoint and nonempty-disjoint controls are separately reported',
        uncertainty='25-video paired bootstrap2000 seed0; descriptive same-Val diagnosis, not frame-independent inference',
        GT_used_only_after_complete_prediction_seal=True)


def dense_ranking_and_transfer(root,pseudo,public):
    arrays={m:[] for m in METHODS[:3]};ys=[];primary_masks=[];direct_masks=[];anchor_masks=[];teacher=[]
    samples=defaultdict(list)
    for row in pseudo.rows:samples[row['video_id']].append(row)
    windows={(r['video_id'],r['expression_id'],r['frame']) for r in pseudo.groups}
    recalls={m:Counter() for m in METHODS[:3]};denominator=0;fp=Counter();teacher_gt=Counter();per_video=[]
    recall_denominators=Counter();recall_scope={m:Counter() for m in METHODS[:3]}
    text_totals={m:{} for m in METHODS[:3]};text_videos={m:[] for m in METHODS[:3]}
    for video in sorted({r['video_id'] for r in pseudo.runtime}):
        bank=load_source_video(video);labels,_=candidate_labels(bank)
        scores={'FROZEN':bank['scores']}
        for method in METHODS[1:3]:scores[method]=scores_for(root,method,bank)
        direct=np.zeros_like(labels);anchor=np.zeros_like(labels);primary=np.zeros_like(labels)
        p=np.full(labels.shape,np.nan);qcol={q:j for j,(q,_) in enumerate(bank['expressions'])}
        positions={(int(t),int(i)):j for j,(t,i) in enumerate(zip(bank['frame'],bank['ids']))}
        for row in samples[video]:
            i=positions[row['frame_id'],row['track_id']];j=qcol[row['query_id']]
            direct[i,j]=True;p[i,j]=row['teacher_prob']
        truth={q:load_eval_gt(DATA,video,q) for q,_ in bank['expressions']}
        teacher_gt['full_video_GT_observations']+=sum(len(v) for gt in truth.values() for v in gt.values())
        teacher_gt['anchor_GT_observations']+=sum(len(gt.get(t,[])) for q,gt in truth.items()
            for v,query,t in windows if v==video and query==q)
        # Include zero-candidate GT frames in the macro recall denominator.
        denominator+=sum(len(gt) for gt in truth.values())
        for q,gt in truth.items():
            for t in gt:recall_denominators['teacher_anchor' if (video,q,t) in windows else 'non_teacher_anchor']+=1
        boxes={}
        from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes
        boxes=load_tracker_boxes(TRACKS,video)
        for t,a,b in bank['slices']:
            ids=bank['ids'][a:b].tolist();n=b-a
            for q,j in qcol.items():
                gt=truth[q].get(t,[]);primary[a:b,j]=bool(gt);anchor[a:b,j]=(video,q,t) in windows
                if not gt:
                    base=scores['FROZEN'][a:b,j]>0;now=scores['MLLM_DISTILL'][a:b,j]>0
                    fp['new_FP_prediction_rows']+=int((now&~base).sum());fp['removed_FP_prediction_rows']+=int((base&~now).sum())
                    fp['assignment_FP_change']+=int(now.sum()-base.sum())
                    continue
                iou=overlaps([boxes[t][i] for i in ids],[box for _,box in gt])
                for method in METHODS[:3]:
                    order=sorted(range(n),key=lambda k:(-scores[method][a+k,j],ids[k]))
                    for k in (1,3,5):
                        found,_=matched_gt(iou,order[:k]);recalls[method][str(k)]+=len(found)/len(gt)
                        scope='teacher_anchor' if (video,q,t) in windows else 'non_teacher_anchor'
                        recall_scope[method][scope+'_'+str(k)]+=len(found)/len(gt)
                src=np.where(scores['FROZEN'][a:b,j]>0,'SUPPORT_POS','SUPPORT_NEG')
                new=np.where(scores['MLLM_DISTILL'][a:b,j]>0,'SUPPORT_POS','SUPPORT_NEG')
                changes=comparison_counts(src,new,labels[a:b,j],[dict(indices=np.arange(n),iou=iou)])
                for key in ('new_FP_prediction_rows','removed_FP_prediction_rows','existing_rows_became_FP_due_to_rematching',
                    'existing_rows_ceased_FP_due_to_rematching','assignment_FP_change','recovered_GT','lost_GT'):
                    fp[key]+=changes[key]
                all_found,_=matched_gt(iou,list(range(n)));sampled=np.flatnonzero(direct[a:b,j]).tolist()
                sampled_found,_=matched_gt(iou,sampled)
                chosen=[i for i in sampled if p[a+i,j]>.5];accepted,_=matched_gt(iou,chosen)
                teacher_gt.update(MOT_active_frame_GT_observations=len(gt),MOT_recoverable_GT=len(all_found),
                    sampled_recoverable_GT=len(sampled_found),teacher_POS_GT_observations=len(accepted))
                if (video,q,t) in windows:teacher_gt['anchor_MOT_recoverable_GT']+=len(all_found)
        for method in METHODS[:3]:
            arrays[method].append(scores[method].reshape(-1));per_video.append(dict(video=video,method=method,
                **binary(scores[method].reshape(-1),labels.reshape(-1))))
            local={}
            for _,a,b in bank['slices']:
                rows,_=frame_metrics(scores[method][a:b],labels[a:b])
                for row in rows:add(local,row);add(text_totals[method],row)
            text_videos[method].append(local)
        ys.append(labels.reshape(-1));primary_masks.append(primary.reshape(-1));direct_masks.append(direct.reshape(-1));anchor_masks.append(anchor.reshape(-1));teacher.append(p.reshape(-1))
        print(json.dumps(dict(event='offline_dense_transfer_video',video=video)),flush=True)
    labels=np.concatenate(ys);primary=np.concatenate(primary_masks);direct=np.concatenate(direct_masks);anchor=np.concatenate(anchor_masks);p=np.concatenate(teacher)
    if int(direct.sum())!=8381 or int(labels[direct].sum())!=485:raise ValueError('fixed teacher/GT identity alignment differs from sealed P1')
    if fp['assignment_FP_change']!=fp['new_FP_prediction_rows']-fp['removed_FP_prediction_rows']+fp['existing_rows_became_FP_due_to_rematching']-fp['existing_rows_ceased_FP_due_to_rematching']:
        raise ValueError('FP row/rematching accounting mismatch')
    masks={'all425_official':np.ones(len(labels),bool),'GT_present_frame_queries':primary,
        'direct_teacher_pairs':direct,'direct_teacher_pairs_GT_present':direct&primary,
        'same_anchor_unasked_candidates':anchor&~direct,'non_teacher_anchor_frames':~anchor,
        'non_teacher_anchor_frames_GT_present':~anchor&primary}
    result={};table=[]
    for method,parts in arrays.items():
        score=np.concatenate(parts);result[method]={}
        for name,mask in masks.items():
            item=binary(score[mask],labels[mask]);result[method][name]=item;table.append(dict(method=method,scope=name,**item))
        result[method]['GT_recall_at_K_macro_all_GT_frames']={k:float(v/denominator) for k,v in recalls[method].items()}
        result[method]['GT_recall_at_K_by_direct_anchor_frame']={scope:{str(k):float(recall_scope[method][scope+'_'+str(k)]/den) for k in (1,3,5)} for scope,den in recall_denominators.items()}
    tmask=direct&np.isfinite(p);positive=p[tmask]>.5;y=labels[tmask]
    teaching=dict(**binary(p[tmask],y),diagnostic_POS_threshold=.5,threshold_used_in_training=False,
        Accepted_POS_Precision=float(y[positive].mean()) if positive.any() else None,
        Positive_Discovery_Recall=float((positive&y).sum()/y.sum()),
        diagnostic_below_half_GT_match_fraction=float(y[~positive].mean()),valid_supervised_pairs=8381,
        explicit_hard_POS_labels_created=0,explicit_hard_NEG_labels_created=0,
        unasked_candidate_query_rows=int((~direct).sum()),unasked_are_NOT_NEG=True,
        teacher_GT_coverage_all_video=teacher_gt['teacher_POS_GT_observations']/teacher_gt['full_video_GT_observations'],
        teacher_GT_coverage_at_fixed_anchors=teacher_gt['teacher_POS_GT_observations']/teacher_gt['anchor_GT_observations'],
        GT_counts=dict(teacher_gt),warning='official unmatched is not exhaustive semantic absence; p>.5 is offline diagnosis only, all valid soft probabilities trained')
    teaching['selected_count']=int(positive.sum())
    teaching['selected_count_definition']='offline p_yes>0.5 only; no hard selection/masking in training'
    teaching['calibration_against_official_labels']=calibration(p[tmask],y)
    teaching['explicit_NEG_false_negative_rate']=None
    teaching['explicit_NEG_false_negative_rate_reason']='no explicit hard NEG labels were created'
    teaching['unasked_fraction']=float((~direct).mean())
    teaching['invalid_input_or_response_pairs']=0
    teaching['prior_P1_threshold_states_not_used']=pseudo.seal['p1_threshold_states_NOT_used']
    teaching['soft_below_half_GT_candidates']=int((y&~positive).sum())
    teaching['anchor_GT_failure_accounting']=dict(
        GT_without_MOT=teacher_gt['anchor_GT_observations']-teacher_gt['anchor_MOT_recoverable_GT'],
        MOT_recoverable_but_not_sampled=teacher_gt['anchor_MOT_recoverable_GT']-teacher_gt['sampled_recoverable_GT'],
        sampled_recoverable_but_teacher_below_half=teacher_gt['sampled_recoverable_GT']-teacher_gt['teacher_POS_GT_observations'],
        teacher_above_half_recovered=teacher_gt['teacher_POS_GT_observations'])
    text_result=text_comparisons(text_totals,text_videos);atomic_json(public/'text_signal_offline.json',text_result)
    result.update(teacher=teaching,FP_row_and_rematching=dict(fp),
        GT_recall_denominator_frame_queries=denominator,recall_definition='macro one-to-one IoU05 GT recall including GT frames without MOT candidates',
        recall_scope_frame_denominators=dict(recall_denominators),text_information='text_signal_offline.json',
        scopes_warning='all425 official scoring vs GT-present conditional semantic scoring have different class prevalence; sparse teacher compared only on its actual8381pairs',
        GT_read_only_after_complete_student_seal=True)
    atomic_json(public/'dense_ranking_transfer.json',result);write_csv(public/'candidate_ranking_by_scope.csv',table);write_csv(public/'candidate_quality_by_video.csv',per_video)
    return result


def optimization_diagnosis(training,public):
    rows=[json.loads(s) for s in (training/'metrics_train.jsonl').read_text().splitlines()]
    if len(rows)!=8973:raise ValueError('incomplete actual training trace')
    names=('source_loss','target_KD_loss','source_gradient_norm','target_gradient_norm','gradient_norm')
    if any(not math.isfinite(r[k]) for r in rows for k in names):raise ValueError('nonfinite trained objective/gradient')
    cosine=[r['source_target_gradient_cosine'] for r in rows if r['source_target_gradient_cosine'] is not None]
    ratio=[r['target_gradient_norm']/r['source_gradient_norm'] for r in rows if r['source_gradient_norm']>0]
    fit_rows=[json.loads(s) for s in gzip.open(training/'teacher_fit_after_training.jsonl.gz','rt')];fits={}
    for name,select in [('all_valid',lambda r:r['valid']),('teacher_p_above_half',lambda r:r['valid'] and r['teacher_probability']>.5),
        ('teacher_p_below_or_equal_half',lambda r:r['valid'] and r['teacher_probability']<=.5)]:
        chosen=[r for r in fit_rows if select(r)];values={}
        for tag,field in [('frozen','source_margin'),('latest_student','student_margin')]:
            x=np.asarray([r[field] for r in chosen]);p=np.asarray([r['teacher_probability'] for r in chosen]);prob=1/(1+np.exp(-np.clip(x,-700,700)))
            values[tag]=dict(MSE=float(np.mean((prob-p)**2)),BCE=float(np.mean(np.maximum(x,0)-x*p+np.log1p(np.exp(-np.abs(x))))),
                agreement_with_teacher_above_half=float(np.mean((x>0)==(p>.5))),student_POS_rate=float(np.mean(x>0)))
        fits[name]=dict(samples=len(chosen),**values)
    result=dict(status='TRAINING_OBJECTIVE_AND_NO_GT_TEACHER_FIT_DIAGNOSIS',optimizer_steps=len(rows),
        finite_nonzero_KD_gradient_steps=sum(r['target_gradient_norm']>0 for r in rows),
        source_KD_gradient_conflict_fraction=float(np.mean(np.asarray(cosine)<0)),
        source_KD_gradient_cosine_quantiles=np.quantile(cosine,[.1,.5,.9]).tolist(),
        KD_to_source_gradient_ratio_quantiles=np.quantile(ratio,[.1,.5,.9]).tolist(),
        fit_to_actual_teacher_training_anchors=fits,fit_checkpoint='fixed latest8973, not GT-selected; primary evaluated checkpoint remains source-val CE selected',
        no_target_GT_used=True,source_validation=json.loads((training/'provenance.json').read_text())['source_val_manifest_sha256'])
    atomic_json(public/'optimization_diagnosis.json',result);return result


def analyze(root,training,public):
    root,training,public=map(lambda p:Path(p).resolve(),(root,training,public));public.mkdir(parents=True,exist_ok=True)
    if not (root/'controls_sealed.json').exists():raise ValueError('seal predictions and all GT-free controls first')
    metrics={};values={};reports={};presence_rows=[];official=[]
    for method in METHODS:
        folder=root/method;dest=public/method;dest.mkdir(exist_ok=True)
        audit=json.loads((folder/'full_audit.json').read_text())
        if audit['valid_episodes']!=425 or audit['unique_videos']!=25:raise ValueError('full425 protocol required')
        metrics[method]=official_metrics(folder/'results');values[method]=official_video_values(folder/'results/pedestrian_detailed.csv')
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(folder/'results'/name,dest/name)
        shutil.copy2(folder/'trackeval.log',dest/'trackeval_stdout.txt');shutil.copy2(folder/'full_audit.json',dest/'full_audit.json')
        official.append(dict(method=method,**{k:metrics[method][k] for k in ('HOTA','DetA','AssA','IDF1','IDSW','DetRe','DetPr','Dets','CLR_TP','CLR_FP')}))
    if abs(metrics['FROZEN']['HOTA']-8.2180)>.0001:raise ValueError('different Frozen evaluation protocol')
    for method in METHODS[1:3]:
        with (public/method/'offline_analysis_stdout.txt').open('w') as log,redirect_stdout(log):
            report=old_attribution(root,method,FROZEN,DATA,TRACKS,public/method/'offline_attribution.json')
        reports[method]=report;rows,scores=presence_from_stream(root/method,report)
        write_csv(public/method/'episode_presence.csv',rows);atomic_json(public/method/'presence_quality.json',scores)
        for name,group in report['query_presence'].items():presence_rows.append(dict(method=method,group=name,**group))
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={};video_rows=[]
    for left,right in [('MLLM_DISTILL','FROZEN'),('MLLM_DISTILL','SOURCE_REHEARSAL'),
        ('MLLM_DISTILL','FROZEN_AT_MLLM_COUNT'),('MLLM_AT_FROZEN_COUNT','FROZEN')]:
        x,y=values[left][1],values[right][1];per=paired_clusters(x,y,np.eye(25));boot=paired_clusters(x,y,draws)
        comparisons[left+'_vs_'+right]=dict(delta_HOTA=float(paired_clusters(x,y,np.ones(25))),
            CI95=np.percentile(boot,[2.5,97.5]).tolist(),improved_videos=int((per>1e-9).sum()),declined_videos=int((per< -1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(values[left][0],per)])
    for method,(videos,v) in values.items():
        point=pooled_hota(v,np.eye(25))
        video_rows.extend(dict(method=method,video=video,HOTA=float(h)) for video,h in zip(videos,point))
    pseudo=MLLMPseudoDataset('/data2/user/TTA-RMOT-assets/mllm-distill/val_pseudo_v1',allow_val_diagnostic=True)
    ranking=dense_ranking_and_transfer(root,pseudo,public);optimization=optimization_diagnosis(training,public)
    train=json.loads((training/'training_receipt.json').read_text());run=json.loads((root/'MLLM_DISTILL/run_metadata.json').read_text())
    selected={k:run[k] for k in ('checkpoint','checkpoint_sha256','selected_step','source_val_CE','selection')}
    costs=dict(training_GPU_hours=train['wall_seconds']/3600,inference_GPU_hours=run['worker_seconds']/3600,
        training_actual_wall_seconds=train['wall_seconds'],inference_scheduler_wall_seconds=run['actual_scheduler_wall_seconds'],
        inference_allocated_worker_seconds=run['worker_seconds'],trainable_parameters=24576,optimizer_updates=train['counters']['optimizer_updates'],
        unique_teacher_pairs=8381,new_teacher_GPU_calls=0,teacher_cost_dependency='inherited P1 A required8381single-frame calls; not a zero-cost independent teacher',
        teacher_forward_equivalent_seconds=pseudo.seal['inherited_teacher_budget']['equivalent_amortized_forward_seconds'],
        peak_train_allocated_bytes=train['peak_allocated_bytes'],peak_inference_allocated_bytes=run['peak_allocated_bytes'],
        scheduler_query_frame_FPS=433636/run['actual_scheduler_wall_seconds'],
        scheduler_MOT_active_video_frame_FPS=(433636/17)/run['actual_scheduler_wall_seconds'],
        timing_scope='shared GPU worker allocations, not exclusive utilization/energy; offline CPU analysis and all preparation attempts reported separately')
    worker_rows=[]
    for path in sorted(root.glob('launchers/*/launcher_status.json')):
        for worker in json.loads(path.read_text())['workers']:
            shard=json.loads((root/'shards'/f"shard{worker['shard']:02d}"/'MLLM_DISTILL/run_metadata.json').read_text())
            frames=sum(r['stats']['frames'] for r in shard['episodes'])
            worker_rows.append(dict(shard=worker['shard'],physical_GPU=worker['physical_gpu'],GPU_UUID=worker['uuid'],
                videos=len(shard['videos']),episodes=len(shard['episodes']),query_frames=frames,
                wall_seconds=worker['elapsed_seconds'],query_frame_FPS=frames/worker['elapsed_seconds'],
                active_video_frame_FPS=(frames/17)/worker['elapsed_seconds'],
                peak_allocated_bytes=shard['peak_allocated_bytes'],exit_code=worker['exit_code']))
    costs['inference_workers']=worker_rows
    with (ROOT/'results/uda/full/cost_comparison.csv').open() as stream:
        costs['historical_reused_comparator_costs']=[r for r in csv.DictReader(stream) if r['method'] in ('SOURCE_REHEARSAL','MEAN_TEACHER_UDA')]
    costs['source_score_bank_preparation']='P1 sampling depends on the existing Frozen source bank; setup time is outside the incremental current run, not claimed free in a standalone system'
    write_csv(public/'per_worker_compute.csv',worker_rows)
    base=ranking['FROZEN']['all425_official'];new=ranking['MLLM_DISTILL']['all425_official']
    point_positive=(metrics['MLLM_DISTILL']['HOTA']>metrics['FROZEN']['HOTA'] and new['AP']>base['AP'] and
        comparisons['MLLM_DISTILL_vs_FROZEN_AT_MLLM_COUNT']['delta_HOTA']>0 and comparisons['MLLM_AT_FROZEN_COUNT_vs_FROZEN']['delta_HOTA']>0)
    intervals_positive=all(comparisons[k]['CI95'][0]>0 for k in ('MLLM_DISTILL_vs_FROZEN','MLLM_DISTILL_vs_FROZEN_AT_MLLM_COUNT','MLLM_AT_FROZEN_COUNT_vs_FROZEN'))
    verdict='PROMISING_REQUIRES_PRE_REGISTERED_3SEED_CONFIRMATION' if point_positive and intervals_positive else 'INCONCLUSIVE_NO_AUTOMATIC_OURS' if point_positive else 'NO_GO_FIRST_NAIVE_BASELINE_COMPLETE_FAILURE_DIAGNOSIS'
    summary=dict(status='COMPLETE_25VAL_TRANSDUCTIVE_FIRST_ROUND',videos=25,expressions=425,protocol='NOT_TRAIN_TO_VAL_NOT_BLIND_TEST',
        official_metrics=metrics,comparisons=comparisons,selected_checkpoint=selected,cost=costs,scientific_verdict=verdict,
        target_GT_used_by_training=False,target_GT_used_for_threshold_or_checkpoint=False,
        CountMatched='both directions POST-HOC only, no GT construction or deployable threshold',
        bootstrap='25 complete videos,2000 resamples,seed0; descriptive development uncertainty, not frame independence or training-seed confirmation',
        historical_MeanTeacher=dict(HOTA=8.2494,status='REFERENCE_ONLY_DIFFERENT_TARGET_SUPERVISION_AND_SAMPLING_BUDGET'),
        optimization_diagnosis='optimization_diagnosis.json',dense_ranking_and_transfer='dense_ranking_transfer.json',
        formal_40Train_to_Val_completed=False,formal_Train_MOT_missing=True,pretraining_overlap='UNKNOWN')
    atomic_json(public/'summary.json',summary);atomic_json(public/'scientific_verdict.json',dict(status=verdict,
        positive_point_conditions=point_positive,video_intervals_support_positive=intervals_positive,automatic_Ours_authorized=False))
    write_csv(public/'official_comparison.csv',official);write_csv(public/'per_video_metrics.csv',video_rows);write_csv(public/'query_presence.csv',presence_rows)
    atomic_json(public/'compute_cost.json',costs);atomic_json(public/'selected_checkpoint.json',selected)
    print(json.dumps(dict(status=summary['status'],metrics=metrics['MLLM_DISTILL'],verdict=verdict),indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-root',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--public-root',type=Path,default=ROOT/'results/mllm_distillation/full')
    a=p.parse_args();analyze(a.evaluation_root,a.training_root,a.public_root)
