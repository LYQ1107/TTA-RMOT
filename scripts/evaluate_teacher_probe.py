#!/usr/bin/env python3
"""Independent, post-sealing sparse teacher evaluation. Never imported by teacher.

Target annotations are read ONLY inside main evaluation after validate_seal.
No teacher rule, prompt, model, online threshold, or student is modified here.
"""
from collections import Counter,defaultdict
import argparse
import csv
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from scripts.audit_teacher_leakage import METHODS,OUTPUT,RUN,validate_seal
from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt,auroc
from scripts.diag_pf_query_information import average_precision
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

DATA=Path('/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance')


def write_csv(path,rows):
    rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def divide(a,b):return float(a/b) if b else None


def softmatch_weights(probability,mean,variance):
    confidence=np.maximum(probability,1-probability)
    # Official USB half-Gaussian, n_sigma=2; source statistics are frozen.
    return np.exp(-np.minimum(confidence-mean,0.)**2/(2*max(variance,1e-12)/4))


def weighted_binary(scores,labels,weights):
    """Tie-correct AP/AUC for video-count bootstrap, not independent frames."""
    s=np.asarray(scores);y=np.asarray(labels,dtype=bool);w=np.asarray(weights,dtype=float)
    order=np.argsort(-s,kind='stable');s=s[order];y=y[order];w=w[order]
    if not len(s):return None,None
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(w*y)[ends];fp=np.cumsum(w*~y)[ends]
    pos=tp[-1];neg=fp[-1]
    ap=float(np.sum(np.diff(np.r_[0,tp])*tp/np.maximum(tp+fp,1e-12))/pos) if pos else None
    auc=float(np.sum(np.diff(np.r_[0,fp])*(tp+np.r_[0,tp[:-1]])/2)/(pos*neg)) if pos and neg else None
    return ap,auc


def tie_prefix(scores,limit,track_ids):
    """Deterministic diagnostic top-k; no GT-dependent tie breaking."""
    return np.lexsort((np.asarray(track_ids),-np.asarray(scores)))[:limit].tolist()


def precision_coverage(scores,labels,accepted):
    chosen=np.flatnonzero(accepted)
    if not len(chosen):return []
    order=chosen[np.argsort(-scores[chosen],kind='stable')]
    ends=np.r_[np.flatnonzero(np.diff(scores[order])),len(order)-1]
    tp=np.cumsum(labels[order])[ends];n=ends+1
    return [dict(threshold=float(scores[order[e]]),accepted_count=int(k),true_positive_count=int(t),
        candidate_coverage=k/len(labels),precision=t/k,GT_candidate_coverage=divide(t,int(labels.sum())))
        for e,k,t in zip(ends,n,tp)]


def aggregate(method,indices,scores,states,labels,groups,weights=None):
    ix=np.asarray(indices,dtype=int);p=states[ix]=='SUPPORT_POS';n=states[ix]=='SUPPORT_NEG';y=labels[ix]
    result=dict(method=method,candidate_pairs=len(ix),positive_candidate_opportunities=int(y.sum()),
        AUROC=auroc(scores[ix],y),AUPRC=average_precision(scores[ix],y) if len(ix) else None,
        POS_count=int(p.sum()),POS_true_candidates=int((p&y).sum()),POS_precision=divide((p&y).sum(),p.sum()),
        NEG_count=int(n.sum()),NEG_true_candidates_killed=int((n&y).sum()),NEG_wrong_fraction=divide((n&y).sum(),n.sum()),
        UNKNOWN_count=int((~p&~n).sum()),UNKNOWN_rate=divide((~p&~n).sum(),len(ix)),
        GT_candidate_coverage=divide((p&y).sum(),y.sum()),GT_candidate_killed_fraction=divide((n&y).sum(),y.sum()))
    eligible=set(ix.tolist());stats=Counter(dict(GT_observations=0,recoverable_sampled_GT=0,
        accepted_matched_GT=0,assignment_FP=0));r1=[];r3=[];r5=[]
    for group in groups:
        ids=group['indices']
        if not len(ids) or int(ids[0]) not in eligible:continue
        local=states[ids]=='SUPPORT_POS';selected=np.flatnonzero(local).tolist()
        matched,_=matched_gt(group['iou'],selected);all_matched,_=matched_gt(group['iou'],list(range(len(ids))))
        stats.update(GT_observations=len(group['gt']),recoverable_sampled_GT=len(all_matched),
            accepted_matched_GT=len(matched),assignment_FP=len(selected)-len(matched))
        if group['gt']:
            for k,values in [(1,r1),(3,r3),(5,r5)]:
                top=tie_prefix(scores[ids],k,group['track_ids']);found,_=matched_gt(group['iou'],top)
                values.append(len(found)/len(group['gt']))
    result.update(stats)
    result.update(GT_positive_coverage=divide(stats['accepted_matched_GT'],stats['GT_observations']),
        recoverable_GT_coverage=divide(stats['accepted_matched_GT'],stats['recoverable_sampled_GT']),
        one_to_one_POS_precision=divide(stats['accepted_matched_GT'],int(p.sum())),
        Recall_at1=float(np.mean(r1)) if r1 else None,Recall_at3=float(np.mean(r3)) if r3 else None,
        Recall_at5=float(np.mean(r5)) if r5 else None,GT_present_frame_queries=len(r1))
    if weights is not None:
        w=weights[ix]
        result.update(soft_weighted_POS_precision=divide((w*p*y).sum(),(w*p).sum()),
            soft_weighted_GT_candidate_coverage=divide((w*p*y).sum(),y.sum()),
            soft_weight_mean=float(w.mean()) if len(w) else None)
    return result


def comparison_counts(base_states,states,labels,groups):
    recovered=lost=fp_delta=added_fp=removed_fp=became_fp=ceased_fp=0
    for group in groups:
        ix=group['indices'];base=np.flatnonzero(base_states[ix]=='SUPPORT_POS').tolist()
        now=np.flatnonzero(states[ix]=='SUPPORT_POS').tolist()
        b,bp=matched_gt(group['iou'],base);a,ap=matched_gt(group['iou'],now)
        recovered+=len(a-b);lost+=len(b-a);fp_delta+=(len(now)-len(a))-(len(base)-len(b))
        base_fp=set(base)-bp;now_fp=set(now)-ap;existing=set(base)&set(now)
        added_fp+=len((set(now)-set(base))&now_fp)
        removed_fp+=len((set(base)-set(now))&base_fp)
        became_fp+=len(existing&now_fp-base_fp);ceased_fp+=len(existing&base_fp-now_fp)
    if fp_delta!=added_fp-removed_fp+became_fp-ceased_fp:raise ValueError('FP rematching accounting mismatch')
    old=base_states=='SUPPORT_POS';new=states=='SUPPORT_POS'
    return dict(recovered_GT=recovered,lost_GT=lost,new_prediction_rows=int((new&~old).sum()),
        removed_prediction_rows=int((old&~new).sum()),new_FP_prediction_rows=added_fp,
        removed_FP_prediction_rows=removed_fp,existing_rows_became_FP_due_to_rematching=became_fp,
        existing_rows_ceased_FP_due_to_rematching=ceased_fp,
        new_rows_without_any_GT_IoU_match=int((new&~old&~labels).sum()),
        removed_rows_without_any_GT_IoU_match=int((old&~new&~labels).sum()),assignment_FP_change=fp_delta,
        interpretation='new/removed FP rows use their respective one-to-one IoU05 assignment; existing-row rematching is separate; any-IoU semantic unmatched rows also reported; NOT full TrackEval')


def bootstrap_metrics(scores,states,labels,primary,video_index,groups,config):
    """All resampling units are complete videos; no frame-level p-values."""
    videos=sorted(set(video_index));vi=np.asarray([videos.index(v) for v in video_index])
    draws=np.random.default_rng(config['seed']).multinomial(len(videos),np.full(len(videos),1/len(videos)),size=config['resamples'])
    ids=np.flatnonzero(primary);result={};samples={}
    for method,score in scores.items():
        ap=[];auc=[];precision=[];recall=[]
        pos=states[method]=='SUPPORT_POS';per_video_gt=np.zeros(len(videos));per_video_found=np.zeros(len(videos))
        for g in groups:
            if not g['gt']:continue
            j=videos.index(g['video']);top=tie_prefix(score[g['indices']],1,g['track_ids'])
            got,_=matched_gt(g['iou'],top);per_video_gt[j]+=1;per_video_found[j]+=len(got)/len(g['gt'])
        for draw in draws:
            w=draw[vi[ids]];a,u=weighted_binary(score[ids],labels[ids],w);ap.append(a);auc.append(u)
            precision.append(divide((w*pos[ids]*labels[ids]).sum(),(w*pos[ids]).sum()))
            recall.append(divide(np.dot(draw,per_video_found),np.dot(draw,per_video_gt)))
        samples[method]=dict(AUPRC=ap,AUROC=auc,POS_precision=precision,Recall_at1=recall)
        result[method]={}
        for name,values in samples[method].items():
            valid=[v for v in values if v is not None]
            result[method][name]=dict(valid_resamples=len(valid),CI95=np.percentile(valid,[2.5,97.5]).tolist() if valid else None)
    comparisons={}
    for method in scores:
        if method=='FROZEN_FLEXHOOK':continue
        values=[]
        for a,b in zip(samples[method]['AUPRC'],samples['FROZEN_FLEXHOOK']['AUPRC']):
            if a is not None and b is not None:values.append(a-b)
        comparisons[method+'_minus_FROZEN_FLEXHOOK_AUPRC']=dict(point_delta=None,valid_resamples=len(values),
            CI95=np.percentile(values,[2.5,97.5]).tolist() if values else None)
    return dict(unit='complete_video',videos=len(videos),resamples=config['resamples'],seed=config['seed'],
        scope='official_GT_present_sampled_frame_queries',descriptive_development_probe=True,by_method=result,
        paired_comparisons=comparisons),draws,vi


def matched_coverage_comparison(left,right,scores,states,labels,primary,draws,vi,groups):
    """POST-HOC comparator downsampling, never a deployable selection threshold."""
    chosen=np.flatnonzero(primary&(states[left]=='SUPPORT_POS'));n=len(chosen)
    available=np.flatnonzero(primary&(states[right]=='SUPPORT_POS'))
    if not n or len(available)<n:
        return dict(status='INSUFFICIENT_MATCHED_POS_COVERAGE',left_count=n,right_available=len(available))
    order=available[np.argsort(-scores[right][available],kind='stable')];control=order[:n]
    a=np.zeros(len(labels),dtype=bool);b=a.copy();a[chosen]=True;b[control]=True
    w0=labels.astype(float);delta=float(labels[chosen].mean()-labels[control].mean());boot=[]
    for draw in draws:
        w=draw[vi];pa=divide((w*a*w0).sum(),(w*a).sum());pb=divide((w*b*w0).sum(),(w*b).sum())
        if pa is not None and pb is not None:boot.append(pa-pb)
    full_right=states[right]=='SUPPORT_POS'
    left_matched_GT=right_full_matched_GT=matched_control_GT=0
    for group in groups:
        ids=group['indices']
        got,_=matched_gt(group['iou'],np.flatnonzero(a[ids]).tolist());left_matched_GT+=len(got)
        got,_=matched_gt(group['iou'],np.flatnonzero(full_right[ids]).tolist());right_full_matched_GT+=len(got)
        got,_=matched_gt(group['iou'],np.flatnonzero(b[ids]).tolist());matched_control_GT+=len(got)
    return dict(status='POST_HOC_COUNT_MATCHED_DIAGNOSTIC_NOT_DEPLOYABLE',left_count=n,right_available=len(available),
        left_precision=float(labels[chosen].mean()),matched_right_precision=float(labels[control].mean()),
        precision_gain=delta,CI95=np.percentile(boot,[2.5,97.5]).tolist() if boot else None,
        valid_resamples=len(boot),GT_coverage_retained_vs_full_right=divide(left_matched_GT,right_full_matched_GT),
        left_distinct_GT_observations=left_matched_GT,right_full_distinct_GT_observations=right_full_matched_GT,
        matched_control_distinct_GT_observations=matched_control_GT,
        comparison_uses_same_candidates=True,threshold_feedback_to_teacher=False)


def query_swap_metrics(method,scores,labels,samples,primary):
    by_candidate=defaultdict(list)
    for i,s in enumerate(samples):by_candidate[s['video_id'],s['frame_id'],s['track_id']].append(i)
    rows=[]
    for (video,frame,track),ix in sorted(by_candidate.items()):
        pos=[i for i in ix if labels[i]];neg=[i for i in ix if not labels[i]]
        # Other true queries are NOT declared negatives. Only official eligible-GT
        # matching vs official-unmatched queries on this exact sampled box.
        if not pos or not neg:continue
        delta=[scores[i]-scores[j] for i in pos for j in neg]
        hard_delta=[scores[i]-scores[j] for i in pos for j in neg if primary[j]]
        rows.append(dict(method=method,video=video,frame=frame,track_id=track,matched_queries=len(pos),
            official_unmatched_queries=len(neg),mean_correct_minus_other=float(np.mean(delta)),
            pairwise_correct_greater_fraction=float(np.mean(np.asarray(delta)>0)),
            tied_fraction=float(np.mean(np.asarray(delta)==0)),
            GT_present_other_query_pairs=len(hard_delta),
            GT_present_correct_minus_other=float(np.mean(hard_delta)) if hard_delta else None,
            GT_present_pairwise_correct_greater_fraction=float(np.mean(np.asarray(hard_delta)>0)) if hard_delta else None))
    return rows


def evaluate(output=OUTPUT,run=RUN):
    started=time.perf_counter();output=Path(output);seal=validate_seal(output)
    # This is the first target-GT access in this independent P1 evaluator.
    samples=[json.loads(s) for s in (output/'manifest.jsonl').read_text().splitlines()]
    by_id={s['sample_id']:i for i,s in enumerate(samples)};n=len(samples)
    compact=[json.loads(s) for s in (output/'teacher_outputs.jsonl').read_text().splitlines()]
    rows={m:{} for m in METHODS}
    for row in compact:rows[row['method']][row['sample_id']]=row
    scores={'FROZEN_FLEXHOOK':np.asarray([s['frozen_margin'] for s in samples])}
    states={'FROZEN_FLEXHOOK':np.where(scores['FROZEN_FLEXHOOK']>0,'SUPPORT_POS','SUPPORT_NEG')}
    for method in METHODS:
        scores[method]=np.asarray([rows[method][s['sample_id']]['probability'] for s in samples])
        states[method]=np.asarray([rows[method][s['sample_id']]['state'] for s in samples])
    rules=json.loads((output/'source_calibration_rules_T0.json').read_text())
    scores['SOFTMATCH_WEIGHTED']=scores['YES_FULL'].copy();states['SOFTMATCH_WEIGHTED']=states['YES_FULL'].copy()
    soft=softmatch_weights(scores['YES_FULL'],rules['softmatch_source_confidence_mean'],rules['softmatch_source_confidence_variance'])
    grouped=defaultdict(list);episode_indices=defaultdict(list)
    for i,s in enumerate(samples):
        grouped[s['video_id'],s['query_id'],s['frame_id']].append(i)
        episode_indices[s['video_id'],s['query_id']].append(i)
    episode_keys=sorted({(s['video_id'],s['query_id']) for s in samples})
    first_GT_read_utc=datetime.now(timezone.utc).isoformat()
    truth={key:load_eval_gt(DATA,*key) for key in episode_keys}
    labels=np.zeros(n,dtype=bool);primary=np.zeros(n,dtype=bool);groups=[]
    for (video,query,frame),ix in sorted(grouped.items()):
        gt=truth[video,query].get(frame,[]);iou=overlaps([samples[i]['bbox_xyxy'] for i in ix],[r[1] for r in gt])
        labels[ix]=(iou>=0.5).any(axis=1);primary[ix]=bool(gt)
        groups.append(dict(video=video,query=query,frame=frame,indices=np.asarray(ix),gt=gt,iou=iou,
            track_ids=[samples[i]['track_id'] for i in ix]))
    positive_videos=sorted({s['video_id'] for s,y in zip(samples,labels) if y})
    audit=dict(protocol='OFFICIAL_REFER_DANCE_MARK_NONZERO_CLASS1_FRAME_OFFSET0_IOU05',
        sealed_before_GT=True,seal_sha256=file_sha(output/'config_hashes.json'),videos=25,video_queries=425,
        sealed_at_utc=seal['sealed_at_utc'],independent_GT_read_started_utc=first_GT_read_utc,
        query_present_episodes=sum(bool(v) for v in truth.values()),query_empty_episodes=sum(not v for v in truth.values()),
        sampled_frame_queries=len(groups),GT_present_frame_queries=sum(bool(g['gt']) for g in groups),
        positive_candidate_opportunities=int(labels.sum()),positive_videos=positive_videos,
        GT_observations_at_fixed_anchors=sum(len(g['gt']) for g in groups),
        warning='official-unmatched/empty is NOT human-exhaustive semantic absence; per-clause GT unavailable; sparse samples NOT full HOTA')
    atomic_json(output/'evaluation_protocol.json',audit)
    quality=[];per_video=[];per_query=[];by_type=[];presence=[];curves=[];swaps=[];visual=[];recoveries={}
    all_ix=np.arange(n);video_index=np.asarray([s['video_id'] for s in samples])
    for method,score in scores.items():
        for name,mask in [('all425_official_scored',np.ones(n,dtype=bool)),('GT_present_frame_queries',primary)]:
            metric=aggregate(method,np.flatnonzero(mask),score,states[method],labels,groups,soft if method=='SOFTMATCH_WEIGHTED' else None)
            metric['scope']=name;quality.append(metric)
            for selection,accepted in [('source_frozen_deployment_POS_only',states[method][mask]=='SUPPORT_POS'),
                ('POST_HOC_all_score_thresholds_not_online_rule',np.ones(int(mask.sum()),dtype=bool))]:
                for row in precision_coverage(score[mask],labels[mask],accepted):
                    curves.append(dict(method=method,scope=name,selection=selection,**row))
        for video in sorted(set(video_index)):
            for name,mask in [('all425_official_scored',video_index==video),('GT_present_frame_queries',(video_index==video)&primary)]:
                metric=aggregate(method,np.flatnonzero(mask),score,states[method],labels,groups)
                per_video.append(dict(video=video,scope=name,**metric))
        for video,query in episode_keys:
            ix=np.asarray(episode_indices[video,query])
            per_query.append(dict(video=video,query_id=query,scope='all425_official_scored',
                **aggregate(method,ix,score,states[method],labels,groups)))
        type_masks={'MIXED':np.ones(n,dtype=bool),
            'parser_typo_alias':np.array([s['query_text'].startswith('dacning ') for s in samples]),
            'without_parser_typo_alias':np.array([not s['query_text'].startswith('dacning ') for s in samples]),
            'contains_APPEARANCE':np.array([any(a['kind']=='APPEARANCE' for a in s['attributes']) for s in samples]),
            'contains_ACTION':np.array([any(a['kind']=='ACTION' for a in s['attributes']) for s in samples]),
            'contains_SPATIAL':np.array([any(a['kind']=='SPATIAL' for a in s['attributes']) for s in samples])}
        for kind,mask in type_masks.items():
            metric=aggregate(method,np.flatnonzero(mask&primary),score,states[method],labels,groups)
            by_type.append(dict(expression_type=kind,scope='GT_present_frame_queries_overlapping_subgroups_not_clause_accuracy',**metric))
        for label,present in [('Query-Present',True),('Query-Empty',False)]:
            keys=[key for key in episode_keys if bool(truth[key])==present]
            ix=np.asarray([i for key in keys for i in episode_indices[key]],dtype=int)
            accepted=states[method][ix]=='SUPPORT_POS'
            positive_episodes=sum(any(states[method][i]=='SUPPORT_POS' for i in episode_indices[key]) for key in keys)
            presence.append(dict(method=method,group=label,episodes=len(keys),sampled_pairs=len(ix),
                accepted_POS=int(accepted.sum()),official_unmatched_POS=int((accepted&~labels[ix]).sum()),
                episodes_with_any_POS=positive_episodes,nonempty_prediction_rate=divide(positive_episodes,len(keys)),
                note='Query-Empty official-unmatched is not proven semantic false'))
        ep_scores=[];ep_labels=[]
        for key in episode_keys:
            ix=episode_indices[key]
            ep_scores.append(float(score[ix].max()));ep_labels.append(bool(truth[key]))
        presence.append(dict(method=method,group='episode_max_sampled_score_presence',episodes=len(episode_keys),
            presence_AUROC=auroc(ep_scores,ep_labels),presence_AUPRC=average_precision(ep_scores,ep_labels),
            note='five anchors and sampled candidates only; GT grouping offline'))
        swaps.extend(query_swap_metrics(method,score,labels,samples,primary))
        recoveries[method]=comparison_counts(states['FROZEN_FLEXHOOK'],states[method],labels,groups)
    cache_meta=json.loads((output/'teacher_cache_index.json').read_text())
    with gzip.open(cache_meta['raw_response_index']['path'],'rt') as handle:
        raw_flags={r['request_id']:dict(variant=r['variant'],binary=r['top1_is_yes_no']) for r in map(json.loads,handle)}
    bottlenecks=defaultdict(Counter)
    with (Path(run)/'full/T0/STRUCTURED_EVIDENCE/teacher_outputs.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line);i=by_id[row['sample_id']];sample=samples[i]
            group=('official_complete_query_MATCHED' if labels[i] else
                'official_complete_query_UNMATCHED_at_GT_present_anchor' if primary[i] else 'no_official_GT_at_anchor')
            count=bottlenecks[group];count['candidate_pairs']+=1
            probs=row.get('attribute_probabilities',[]);attrs=sample['attributes']
            if len(probs)!=len(attrs) or not attrs or any(p is None for p in probs):
                count['attribute_evidence_missing']+=1;continue
            minimum=min(probs);kinds=sorted({a['kind'] for a,p in zip(attrs,probs) if p==minimum})
            count['min_score_kind_'+'_AND_'.join(kinds)]+=1
            below=sorted({a['kind'] for a,p in zip(attrs,probs) if p<rules['positive_threshold']})
            count['below_source_POS_gate_'+'_AND_'.join(below or ['NONE'])]+=1
            count['C_state_'+row['decision']['state']]+=1
            count['nonbinary_original_answer_pairs']+=int(any(ref and not raw_flags[ref]['binary'] for ref in row['request_ids']))
    atomic_json(output/'clause_bottleneck_summary.json',dict(groups={k:dict(v) for k,v in bottlenecks.items()},
        interpretation='score bottleneck of the complete expression on matched/unmatched candidate pairs; NOT independent clause accuracy or per-attribute GT; no threshold/rule change'))
    visual_groups=defaultdict(list);control_validity=defaultdict(Counter)
    for method in ['EVIDENCE_VERIFIED','YES_ENSEMBLE_B_PLUS']:
        with (Path(run)/'full/T0'/method/'teacher_outputs.jsonl').open() as handle:
            for line in handle:
                row=json.loads(line)
                i=by_id[row['sample_id']];state=row['decision']['state']
                validity=control_validity[method+'/'+state]
                validity['candidate_pairs']+=1
                invalid=[raw_flags[ref]['variant'] for ref in row['request_ids'] if ref and not raw_flags[ref]['binary']]
                validity['pairs_with_nonbinary_original']+=int('original' in invalid)
                validity['pairs_with_nonbinary_control']+=int(any(v!='original' for v in invalid))
                for variant in set(invalid):validity['pairs_nonbinary_'+variant]+=1
                for attr in row['decision']['features'].get('attributes',[]):
                    visual.append(dict(method=method,sample_id=row['sample_id'],video=row['video_id'],
                        query_id=row['query_id'],state=row['decision']['state'],**attr))
                    label='official_complete_query_MATCHED' if labels[i] else 'official_complete_query_UNMATCHED'
                    scope='GT_present_frame_queries' if primary[i] else 'no_official_GT_at_anchor'
                    visual_groups[method,state,label,scope].append((row['sample_id'],attr))
    visual_summary=[]
    fields=('visual_delta','target_probability_change','background_probability_change',
        'target_sensitivity_excess','entropy_delta','cross_observation_spread')
    for (method,state,label,scope),items in sorted(visual_groups.items()):
        record=dict(method=method,state=state,complete_query_IoU_group=label,scope=scope,
            candidate_pairs=len({sid for sid,_ in items}),attribute_evaluations=len(items),
            attribute_GT_available=False)
        for field in fields:record['mean_'+field]=float(np.mean([a[field] for _,a in items]))
        record['zero_target_sensitivity_excess_fraction']=float(np.mean([a['target_sensitivity_excess']==0 for _,a in items]))
        visual_summary.append(record)
    atomic_json(output/'visual_evidence_summary.json',dict(feature_groups=visual_summary,
        per_pair_nonbinary_controls={key:dict(value) for key,value in control_validity.items()},
        caution='no-image/occlusion conditional Yes-No ratios may be poorly calibrated or OOD even when original answer is valid; no control non-answer is relabelled as No; complete-query IoU groups are NOT per-attribute truth'))
    write_csv(output/'quality_by_method.csv',quality);write_csv(output/'quality_by_video.csv',per_video)
    write_csv(output/'quality_by_query.csv',per_query)
    write_csv(output/'quality_by_expression_type.csv',by_type);write_csv(output/'precision_coverage.csv',curves)
    write_csv(output/'query_presence.csv',presence);write_csv(output/'query_swap.csv',swaps)
    write_csv(output/'visual_evidence_diagnostics.csv',visual)
    atomic_json(output/'GT_recovery_and_FP.json',recoveries)
    config=yaml.safe_load((ROOT/'configs/teacher/probe_protocol.yaml').read_text())
    boot,draws,vi=bootstrap_metrics(scores,states,labels,primary,video_index,groups,config['bootstrap'])
    main_metrics={r['method']:r for r in quality if r['scope']=='GT_present_frame_queries'}
    for method in scores:
        if method=='FROZEN_FLEXHOOK':continue
        boot['paired_comparisons'][method+'_minus_FROZEN_FLEXHOOK_AUPRC']['point_delta']=main_metrics[method]['AUPRC']-main_metrics['FROZEN_FLEXHOOK']['AUPRC'] if main_metrics[method]['AUPRC'] is not None else None
    comparisons={}
    for left in ['STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED']:
        controls=list(config['gates']['gate_B']['required_comparators'])
        if left=='EVIDENCE_VERIFIED':controls.append('STRUCTURED_EVIDENCE')
        for right in controls:
            comparisons[left+'_vs_'+right]=matched_coverage_comparison(left,right,scores,states,labels,primary,draws,vi,groups)
    boot['matched_coverage_POST_HOC_comparisons']=comparisons
    atomic_json(output/'bootstrap.json',boot)
    gates=make_verdict(main_metrics,boot,comparisons,audit,rules,seal,config,output)
    atomic_json(output/'final_scientific_verdict.json',gates)
    costs=build_budget(seal,output,run)
    summary=dict(protocol=audit,quality=quality,gates=gates,source_rules=rules,
        evaluator_script_sha256=file_sha(Path(__file__)),
        target_teacher_training=False,student_training_started=False,
        computation_wall_seconds=time.perf_counter()-started,
        query_swap_common_candidate_rows=len(swaps),per_method_costs=costs,
        unknown_reasons={m:dict(Counter(reason for r in rows[m].values() if r['state']=='UNKNOWN' for reason in r['reasons'])) for m in METHODS},
        semantic_GT_limitation='no independent per-clause clothing/action labels; no spatial expressions; all17 actual queries MIXED',
        query_swap_limitation='intersection of sampled same-box queries only; no imputation of missing other-query predictions; unmatched queries may share valid semantics',
        recall_limitation='Recall@k uses one-to-one IoU matches/all eligible GT at anchors; <=4 sampled candidates, so Recall@5 saturates sampled universe',
        softmatch_limitation='source-frozen half-Gaussian loss weights only; same A scores/states, no arbitrary extra hard threshold; NOT full SoftMatch SSL training')
    atomic_json(output/'evaluation_summary.json',summary)
    print(json.dumps(dict(protocol=audit,gates=gates,wall_seconds=summary['computation_wall_seconds']),indent=2))
    return summary


def make_verdict(metrics,boot,comparisons,audit,rules,seal,config,output):
    gate_cfg=config['gates'];a_cfg=gate_cfg['gate_A'];b_cfg=gate_cfg['gate_B'];c_cfg=gate_cfg['gate_C']
    enough=audit['positive_candidate_opportunities']>=gate_cfg['min_scored_positive_opportunities'] and len(audit['positive_videos'])>=gate_cfg['min_positive_videos']
    baseline=metrics['FROZEN_FLEXHOOK'];gate_a={}
    for method,r in metrics.items():
        if method in ('FROZEN_FLEXHOOK','SOFTMATCH_WEIGHTED'):continue
        ci=boot['paired_comparisons'][method+'_minus_FROZEN_FLEXHOOK_AUPRC']['CI95']
        checks=dict(enough_positive_support=enough,
            AP_gain=(r['AUPRC'] is not None and baseline['AUPRC'] is not None and r['AUPRC']-baseline['AUPRC']>=a_cfg['min_AUPRC_gain_vs_matched_Frozen']),
            Recall1_gain=(r['Recall_at1'] is not None and baseline['Recall_at1'] is not None and r['Recall_at1']-baseline['Recall_at1']>=a_cfg['min_Recall_at1_gain_vs_matched_Frozen']),
            AP_video_CI_lower_positive=bool(ci and ci[0]>0),
            POS_precision=r['POS_precision'] is not None and r['POS_precision']>=a_cfg['min_POS_precision'],
            GT_coverage=r['GT_positive_coverage'] is not None and r['GT_positive_coverage']>=a_cfg['min_GT_positive_coverage'],
            accepted_distinct_GT_observations=r['accepted_matched_GT']>=a_cfg['min_distinct_accepted_positive'])
        gate_a[method]=dict(passed=all(checks.values()),checks=checks)
    gate_b={}
    for method in ['STRUCTURED_EVIDENCE','EVIDENCE_VERIFIED']:
        checks={}
        for control in b_cfg['required_comparators']:
            row=comparisons[method+'_vs_'+control];ci=row.get('CI95')
            checks[control]=bool(row.get('precision_gain',-1)>=b_cfg['min_precision_gain_at_matched_coverage'] and ci and ci[0]>0 and
                row.get('GT_coverage_retained_vs_full_right') is not None and row['GT_coverage_retained_vs_full_right']>=b_cfg['minimum_GT_coverage_retained_vs_matched_baseline'])
        gate_b[method]=dict(passed=all(checks.values()),checks=checks,
            warning='C has fewer views than B-plus; D/B-plus is exact visual/call matched decomposition test; no positive claim from an unmatched comparison')
    cache=json.loads((output/'teacher_cache_index.json').read_text())['by_method']
    numerator=cache['EVIDENCE_VERIFIED']['logical_request_slots'];denominator=cache['YES_ENSEMBLE_B_PLUS']['logical_request_slots']
    ratio=divide(numerator,denominator)
    source_ready=rules['status']=='SOURCE_RULES_FROZEN' and rules['source_verifier_validated']
    teacher_useful=any(g['passed'] for g in gate_a.values());structured_useful=gate_b['EVIDENCE_VERIFIED']['passed']
    evidence_ready=bool(teacher_useful and gate_a['EVIDENCE_VERIFIED']['passed'] and structured_useful and source_ready and ratio is not None and ratio<=c_cfg['maximum_forward_budget_ratio_vs_Bplus'])
    return dict(status='EVIDENCE_P2_PREPARATION_ELIGIBLE_NOT_TRAINING_AUTHORIZATION' if evidence_ready else
        ('SIMPLE_TEACHER_ONLY_NO_EVIDENCE_CONTRIBUTION' if teacher_useful else 'INSUFFICIENT_TEACHER_SUPERVISION'),
        Gate_A=gate_a,Gate_B=gate_b,Gate_C=dict(passed=evidence_ready,source_rules_and_verifier_validated=source_ready,
            exact_input_fairness=seal['fairness'],D_to_Bplus_logical_call_ratio=ratio,
            T1_missing_user_waived=True,provenance_legal_inputs=True,pretraining_overlap_unknown=True),
        any_teacher_Gate_A_passed=teacher_useful,evidence_P2_ready=evidence_ready,
        formal_P2_executable=False,formal_P2_blockers=['matching40Train predicted MOT missing','independent10video legal referringGT not yet verified',
            'source-only reliability or teacher/evidence quality gates may be insufficient'],
        target_GT_threshold_tuning=False,student_training_started=False,
        decision='STOP_AFTER_P1; no automatic new long GPU training; retain all negative/inconclusive results')


def build_budget(seal,output,run):
    cache=json.loads((output/'teacher_cache_index.json').read_text())['by_method'];rows=[];worker_rows=[]
    attempts=[];overhead=[]
    for path in sorted(Path(run).glob('*/T0/launcher_attempts/*/launcher_status.json')):
        attempt=json.loads(path.read_text());attempts.append(attempt)
        overhead.append(dict(phase=attempt['phase'],status=attempt['status'],path=str(path),
            actual_scheduler_wall_seconds=attempt['actual_wall_seconds'],
            allocated_worker_GPU_hours=sum(r.get('elapsed_seconds',0) for r in attempt['workers'])/3600,
            workers=attempt['workers']))
    for method in METHODS:
        receipt=seal['receipts'][method];budget=receipt['budget']
        method_attempts=[r for a in attempts if a['phase']=='full' for r in a['workers'] if r['method']==method and 'elapsed_seconds' in r]
        allocation_seconds=sum(r['elapsed_seconds'] for r in method_attempts)
        span=[(r['start_epoch_seconds'],r['start_epoch_seconds']+r['elapsed_seconds']) for r in method_attempts if 'start_epoch_seconds' in r]
        actual_wall=max(t[1] for t in span)-min(t[0] for t in span) if span else allocation_seconds
        for shard in receipt.get('source_worker_receipts',[]):
            worker=json.loads(Path(shard['receipt']).read_text())
            worker_rows.append(dict(method=method,physical_gpu=worker['physical_gpu'],samples=worker['samples'],
                worker_wall_seconds=worker['worker_elapsed_seconds'],candidate_pairs_per_second=worker['samples']/worker['worker_elapsed_seconds'],
                peak_forward_allocated_bytes=worker['budget']['peak_allocated_bytes'],actual_new_forwards=worker['budget']['actual_candidate_prompt_forwards'],
                receipt=shard['receipt']))
        rows.append(dict(method=method,scope='full_fixed_probe_incremental_worker_plus_required_raw_equivalents',
            candidate_query_pairs=receipt['samples'],worker_wall_seconds=receipt['worker_elapsed_seconds'],
            actual_method_wall_span_seconds=actual_wall,worker_seconds_interpretation='sum across whole-video shard workers',
            GPU_hours_allocated_worker=allocation_seconds/3600,
            actual_including_failed_resumed_worker_seconds=allocation_seconds,
            candidate_pairs_per_worker_second=receipt['samples']/receipt['worker_elapsed_seconds'],
            actual_forward_seconds=budget['actual_forward_seconds'],actual_new_prompt_forwards=budget['actual_candidate_prompt_forwards'],
            actual_new_forward_batches=budget['actual_model_forward_batches'],cache_hits=budget['cache_hits'],
            peak_allocated_bytes=budget['peak_allocated_bytes'],trainable_teacher_parameters=0,optimizer_updates=0,
            student_updates=0,physical_gpu=receipt['physical_gpu'],**cache[method]))
    parent=json.loads((Path(run)/'full/T0/phase_completion.json').read_text())
    for row in rows:
        row['full_phase_scheduler_wall_seconds']=parent['actual_wall_seconds']
        row['scheduler_candidate_method_pairs_per_second']=sum(r['candidate_query_pairs'] for r in rows)/parent['actual_wall_seconds']
    a=dict(next(r for r in rows if r['method']=='YES_FULL'));a.update(method='SOFTMATCH_WEIGHTED',
        scope='inherits_complete_A_teacher_cost_plus_CPU_weights_not_free_independent_MLLM')
    rows.append(a);write_csv(output/'budget.csv',rows)
    write_csv(output/'budget_by_worker.csv',worker_rows)
    atomic_json(output/'stage_costs_and_attempts.json',dict(attempts=overhead,
        all_pipeline_allocated_GPU_hours=sum(a['allocated_worker_GPU_hours'] for a in overhead),
        standalone_first2pair_functionality_worker_seconds=16.6985,
        cost_warning='shared GPUs; allocated worker time is not exclusive utilization/energy; raw equivalent seconds are amortized observed batches, not a new standalone timing run; SoftMatch inherits A; phase wall times are not summed as concurrent method times'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--run-root',type=Path,default=RUN);a=p.parse_args();evaluate(a.output,a.run_root)


if __name__=='__main__':main()
