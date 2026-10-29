#!/usr/bin/env python3
"""Independent offline PSR attribution, ONLY after full student/control seals."""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.pf_diag_common import DATA,TRACKS,FROZEN,load_source_video,candidate_labels,official_metrics,official_video_values,write_csv
from scripts.analyze_pf_full_offline import analyze as attribution,load_eval_gt,overlaps,matched_gt,auroc
from scripts.analyze_uda_full import presence_from_stream
from scripts.analyze_uda_text_signal_offline import scores_for
from scripts.analyze_pf_video_uncertainty import paired_clusters
from scripts.diag_pf_query_information import average_precision,frame_metrics,add,finalize
from flex_uda.teacher.psr_dataset import PSRPseudoDataset
from scripts.train_psr import METHODS


def binary(score,y):
    return dict(candidate_pairs=len(y),positive_candidates=int(y.sum()),AUROC=auroc(score,y),AP=average_precision(score,y),selected_rows=int((score>0).sum()))


def dense(root,methods,public):
    names=['FROZEN','SOURCE_REHEARSAL',*methods];parts={m:[] for m in names};labels_all=[];present_all=[];endpoint_all=[];middle_all=[]
    recall={m:Counter() for m in names};denominator=Counter();text={m:{} for m in names};text_by_video={m:[] for m in names}
    policies={policy:PSRPseudoDataset('/data2/user/TTA-RMOT-assets/psr/val_v1/'+policy,allow_val_diagnostic=True) for policy in ('NAIVE','PSR','UNIFORM','CONFIDENCE')}
    policy_records={p:[] for p in policies};coverage_records={p:Counter() for p in policies}
    weak_records={p:[] for p in policies};per_video=[]
    direct_at={p:defaultdict(list) for p in policies};weak_at={p:defaultdict(list) for p in policies}
    for policy,data in policies.items():
        for row in data.rows:
            if row['valid'] and row['state']=='POS':direct_at[policy][row['video_id'],row['query_id'],row['frame_id']].append(row)
        for row in data.expanded:weak_at[policy][row['video_id'],row['query_id'],row['frame_id']].append(row)
    from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes
    for video in sorted({r['video_id'] for r in policies['PSR'].runtime}):
        bank=load_source_video(video);y,_=candidate_labels(bank);qcol={q:j for j,(q,_) in enumerate(bank['expressions'])}
        positions={(int(t),int(i)):k for k,(t,i) in enumerate(zip(bank['frame'],bank['ids']))}
        scores={'FROZEN':bank['scores']};scores.update({m:scores_for(root,m,bank) for m in names[1:]})
        truth={q:load_eval_gt(DATA,video,q) for q in qcol};boxes=load_tracker_boxes(TRACKS,video)
        endpoint=np.zeros_like(y);middle=np.zeros_like(y);present=np.zeros_like(y)
        anchors={(r['query_id'],r['frame_id']) for r in policies['PSR'].windows if r['video_id']==video}
        for q,gt in truth.items():
            for t in gt:denominator['all']+=1;denominator['teacher_endpoint' if any((q,t+d) in anchors for d in (-1,1)) else 'non_teacher_endpoint']+=1
        for t,a,b in bank['slices']:
            ids=bank['ids'][a:b].tolist();n=b-a
            for q,j in qcol.items():
                gt=truth[q].get(t,[]);present[a:b,j]=bool(gt);middle[a:b,j]=(q,t) in anchors
                endpoint[a:b,j]=any((q,t+d) in anchors for d in (-1,1))
                if not gt:continue
                matrix=overlaps([boxes[t][i] for i in ids],[box for _,box in gt])
                scope='teacher_endpoint' if endpoint[a,j] else 'non_teacher_endpoint'
                for method in names:
                    order=sorted(range(n),key=lambda k:(-scores[method][a+k,j],ids[k]))
                    for k in (1,3,5):
                        found,_=matched_gt(matrix,order[:k]);value=len(found)/len(gt)
                        recall[method]['all_'+str(k)]+=value;recall[method][scope+'_'+str(k)]+=value
            for policy,data in policies.items():
                for q,j in qcol.items():
                    gt=truth[q].get(t,[])
                    if not gt:continue
                    matrix=overlaps([boxes[t][i] for i in ids],[box for _,box in gt])
                    direct=direct_at[policy].get((video,q,t),[])
                    weak=weak_at[policy].get((video,q,t),[])
                    # Count only distinct one-to-one GT observations, never duplicated people.
                    for tag,rows in [('direct',direct),('expanded',weak)]:
                        if rows:
                            chosen=[ids.index(r['track_id']) for r in rows]
                            found,_=matched_gt(matrix,chosen);coverage_records[policy][tag+'_GT_observations']+=len(found)
        for method in names:
            parts[method].append(scores[method].ravel());local={}
            for _,a,b in bank['slices']:
                rows,_=frame_metrics(scores[method][a:b],y[a:b])
                for row in rows:add(local,row);add(text[method],row)
            text_by_video[method].append(local)
            per_video.append(dict(video=video,method=method,**binary(scores[method].ravel(),y.ravel())))
        for policy,data in policies.items():
            for row in (r for r in data.rows if r['video_id']==video):
                i=positions[row['frame_id'],row['track_id']];j=qcol[row['query_id']]
                policy_records[policy].append(dict(video=video,query=row['query_id'],frame=row['frame_id'],track=row['track_id'],state=row['state'],
                    p=row['teacher_prob'],official_match=bool(y[i,j]),source_margin=float(bank['scores'][i,j]),sample_id=row['sample_id']))
            for row in (r for r in data.expanded if r['video_id']==video):
                i=positions[row['frame_id'],row['track_id']];j=qcol[row['query_id']]
                parent_y=[bool(y[positions[t,row['track_id']],j]) for t in row['observed_teacher_frames']]
                weak_records[policy].append(dict(video=video,query=row['query_id'],frame=row['frame_id'],track=row['track_id'],
                    official_match=bool(y[i,j]),actual_middle_frame_source_margin=float(bank['scores'][i,j]),
                    endpoint_official_matches=parent_y,possible_action_boundary_or_identity_problem=all(parent_y) and not bool(y[i,j]),
                    sample_id=row['sample_id'],parent_sample_ids=row['parent_sample_ids']))
        labels_all.append(y.ravel());present_all.append(present.ravel());endpoint_all.append(endpoint.ravel());middle_all.append(middle.ravel())
        print(json.dumps(dict(event='PSR_offline_dense_video',video=video)),flush=True)
    y=np.concatenate(labels_all);present=np.concatenate(present_all);endpoint=np.concatenate(endpoint_all);middle=np.concatenate(middle_all)
    masks={'all425_official':np.ones(len(y),bool),'GT_present_frame_queries':present,'direct_teacher_endpoint_frames':endpoint,
        'potential_expansion_middle_frames':middle,'non_teacher_endpoint_frames':~endpoint,
        'not_direct_or_middle_frames':~endpoint&~middle,'not_direct_or_middle_GT_present':~endpoint&~middle&present}
    result={};rows=[]
    for method,arrays in parts.items():
        score=np.concatenate(arrays);result[method]={}
        for scope,mask in masks.items():
            value=binary(score[mask],y[mask]);result[method][scope]=value;rows.append(dict(method=method,scope=scope,**value))
        result[method]['GT_recall_at_K']={scope:{str(k):recall[method][scope+'_'+str(k)]/denominator[scope] for k in (1,3,5)} for scope in denominator}
    teaching={}
    for policy,records in policy_records.items():
        p=np.asarray([r['p'] for r in records]);target=np.asarray([r['official_match'] for r in records]);positive=np.asarray([r['state']=='POS' for r in records]);negative=np.asarray([r['state']=='NEG' for r in records])
        weak=weak_records[policy];weak_correct=sum(r['official_match'] for r in weak)
        correct={(r['video'],r['query'],r['frame'],r['track']) for r in records if r['state']=='POS' and r['official_match']}
        naive={(r['video'],r['query'],r['frame'],r['track']) for r in policy_records['NAIVE'] if r['state']=='POS' and r['official_match']}
        teaching[policy]=dict(direct_samples=len(records),direct_teacher_AUROC=auroc(p,target),direct_teacher_AP=average_precision(p,target),
            accepted_POS=int(positive.sum()),accepted_POS_correct=int((positive&target).sum()),
            accepted_POS_precision=float(target[positive].mean()) if positive.any() else None,
            positive_discovery_recall_within_queried_candidates=float((positive&target).sum()/target.sum()) if target.any() else None,
            diagnostic_NEG_official_match_fraction=float(target[negative].mean()) if negative.any() else None,
            UNKNOWN_not_hard_negative=True,correct_POS_not_source_selected=sum(r['state']=='POS' and r['official_match'] and r['source_margin']<=0 for r in records),
            new_correct_candidate_POS_vs_Naive=len(correct-naive),lost_correct_candidate_POS_vs_Naive=len(naive-correct),
            expanded_weak=len(weak),expanded_weak_official_correct=weak_correct,
            expanded_weak_official_error_fraction=1-weak_correct/len(weak) if weak else None,
            endpoint_positive_but_middle_wrong=sum(r['possible_action_boundary_or_identity_problem'] for r in weak),
            one_to_one_GT_coverage_counts=dict(coverage_records[policy]),
            warning='official unmatched is not exhaustive semantic absence; sampled/weak counts are not unique people; soft NEG-like probability is not a hard negative label')
        write_csv(public/(policy+'_direct_teacher_offline.csv'),records)
        atomic_json(public/(policy+'_expanded_weak_offline.json'),weak)
    write_csv(public/'candidate_ranking_by_scope.csv',rows);write_csv(public/'candidate_quality_by_video.csv',per_video)
    atomic_json(public/'teacher_positive_support.json',teaching)
    atomic_json(public/'text_signal_offline.json',dict(methods={m:finalize(v) for m,v in text.items()},
        visual_control='candidate mean across17 real queries, not separate visual network',other_queries_not_assumed_negative=True))
    atomic_json(public/'dense_ranking_transfer.json',dict(methods=result,GT_frame_query_denominators=dict(denominator),
        scope_note='pooled AUROC/AP are not same-frame semantic ranking; one-to-oneGTRecall includes allGTframes, not candidate hit-rate',
        GT_read_only_after_all_registered_predictions_sealed=True))
    return result,teaching


def analyze(root,training,public):
    root,training,public=map(lambda p:Path(p).resolve(),(root,training,public));public.mkdir(parents=True,exist_ok=True)
    seal=json.loads((root/'controls_sealed.json').read_text());methods=seal['student_methods'];all_methods=seal['methods']
    metrics={};values={};official=[];presence=[]
    for method in all_methods:
        audit=json.loads((root/method/'full_audit.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425 or audit['unique_videos']!=25:raise ValueError('complete25/425 required')
        metrics[method]=official_metrics(root/method/'results');values[method]=official_video_values(root/method/'results/pedestrian_detailed.csv')
        dest=public/method;dest.mkdir(exist_ok=True)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(root/method/'results'/name,dest/name)
        official.append(dict(method=method,**{k:metrics[method][k] for k in ('HOTA','DetA','AssA','DetRe','DetPr','IDF1','IDSW','Dets','CLR_TP','CLR_FP')}))
    if abs(metrics['FROZEN']['HOTA']-8.2180)>.0001:raise ValueError('baseline protocol drift')
    pairs=[(m,'FROZEN') for m in methods]+[(m,'FROZEN_AT_'+m+'_COUNT') for m in methods]+[(m+'_AT_FROZEN_COUNT','FROZEN') for m in methods]
    pairs += [('PSR',m) for m in methods if m!='PSR']
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    for a,b in pairs:
        x,y=values[a][1],values[b][1];per=paired_clusters(x,y,np.eye(25));boot=paired_clusters(x,y,draws)
        comparisons[a+'_vs_'+b]=dict(delta_HOTA=float(paired_clusters(x,y,np.ones(25))),CI95=np.percentile(boot,[2.5,97.5]).tolist(),
            improved_videos=int((per>1e-9).sum()),declined_videos=int((per< -1e-9).sum()),per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(values[a][0],per)])
    for method in methods:
        dest=public/method
        with (dest/'offline_analysis_stdout.txt').open('w') as log,redirect_stdout(log):report=attribution(root,method,FROZEN,DATA,TRACKS,dest/'offline_attribution.json')
        episodes,quality=presence_from_stream(root/method,report);write_csv(dest/'episode_presence.csv',episodes);atomic_json(dest/'presence_quality.json',quality)
        for group,record in report['query_presence'].items():presence.append(dict(method=method,group=group,**record))
    ranking,teaching=dense(root,methods,public);costs=[];checkpoints={}
    for method in methods:
        train=json.loads((training/method/'training_receipt.json').read_text());run=json.loads((root/method/'run_metadata.json').read_text())
        if train['steps']!=8973 or train['target_GT_open_attempts'] or run['target_GT_open_attempts']:raise ValueError('invalid training/data permission')
        checkpoints[method]={k:run[k] for k in ('checkpoint','checkpoint_sha256','selected_step','source_val_CE','selection')}
        policy=METHODS[method][0];teacher=json.loads((ROOT/'results/psr/supervision'/policy/'supervision_audit.json').read_text())['teacher_budget']
        attempts=[json.loads(p.read_text()) for p in (training/method/'attempt_runtimes').glob('*.json')]
        all_attempt_seconds=sum(r['elapsed_seconds'] for r in attempts if r['scope']=='full') or train['wall_seconds']
        costs.append(dict(method=method,training_GPU_hours=all_attempt_seconds/3600,inference_GPU_hours=run['worker_seconds']/3600,
            actual_training_wall_seconds=all_attempt_seconds,successful_training_body_seconds=train['wall_seconds'],actual_inference_scheduler_wall_seconds=run['actual_scheduler_wall_seconds'],
            allocated_inference_worker_seconds=run['worker_seconds'],peak_training_allocated_bytes=train['peak_allocated_bytes'],peak_inference_allocated_bytes=run['peak_allocated_bytes'],
            scheduler_query_frame_FPS=433636/run['actual_scheduler_wall_seconds'],scheduler_active_video_frame_FPS=25508/run['actual_scheduler_wall_seconds'],
            trainable_parameters=train['trainable_parameters'],optimizer_updates=train['counters']['optimizer_updates'],unique_direct_labels=train['unique_teacher_samples_seen'],
            unique_expanded_weak=train.get('unique_expanded_samples_seen',0),required_teacher_judgments=8500,
            required_teacher_amortized_forward_seconds=teacher['equivalent_amortized_forward_seconds']))
    verdict='PROMISING_REQUIRES_BUDGET_AND_COUNT_CONTROLLED_3SEED_CONFIRMATION' if metrics['PSR']['HOTA']>max(metrics['FROZEN']['HOTA'],metrics['NAIVE_PAIRED']['HOTA']) else 'PSR_NOT_ABOVE_FROZEN_AND_MATCHED_NAIVE'
    summary=dict(status='COMPLETE_FOR_LISTED_METHODS',protocol='USER_AUTHORIZED_25VAL_SAME_SET_NOT_TRAIN_TO_VAL',videos=25,expressions=425,student_methods=methods,
        official_metrics=metrics,comparisons=comparisons,selected_checkpoints=checkpoints,cost=costs,scientific_verdict=verdict,
        formal_40Train_to_Val_complete=False,source_and_target_training_GT_permissions='sourceGT only; targetGT after prediction seal',
        counts_and_intervals_are_not_blind_test_proof=True,unrounded_video_cluster_bootstrap='25videos,2000draws,seed0',
        Qwen_pretraining_overlap='UNKNOWN',historical_Naive8_0327='not same paired sampling; historical reference only')
    write_csv(public/'official_comparison.csv',official);write_csv(public/'query_presence.csv',presence);write_csv(public/'compute_cost.csv',costs)
    atomic_json(public/'summary.json',summary);print(json.dumps(dict(status=summary['status'],metrics={m:metrics[m]['HOTA'] for m in ('FROZEN','SOURCE_REHEARSAL',*methods)},verdict=verdict),indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-root',type=Path,required=True);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results/psr/full');a=p.parse_args();analyze(a.evaluation_root,a.training_root,a.output)
