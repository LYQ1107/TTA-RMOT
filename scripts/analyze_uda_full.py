#!/usr/bin/env python3
"""Independent offline GT analysis of sealed25/425 Val-transductive outputs.

Never imported by adaptation. This reports source-selected checkpoints and
the dependent Count-Matched diagnostic, not GT-selected online thresholds.
"""
import argparse
from contextlib import redirect_stdout
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'vendor/FlexHook/TrackEval'))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.training import SOURCE_SHA
from scripts.pf_diag_common import DATA,TRACKS,FROZEN,official_metrics,official_video_values,write_csv
from scripts.analyze_pf_full_offline import analyze as attribution
from scripts.analyze_pf_video_uncertainty import paired_clusters,pooled_hota
from scripts.diag_pf_query_information import binary_summary


def presence_from_stream(folder,attribution_report):
    rows=[]
    for e in attribution_report['episodes']:
        video,expression=e['video'],e['expression'];all_scores=[];maxes=[];positive=0
        with gzip.open(folder/'results'/video/expression/'candidate_margins.jsonl.gz','rt') as stream:
            for line in stream:
                s=np.asarray(json.loads(line)['adapted']);all_scores.extend(s.tolist());maxes.append(float(s.max()));positive+=int((s>0).sum())
        rows.append(dict(video=video,expression=expression,query_present=bool(e['gt_observations']),
            mean_margin=float(np.mean(all_scores)),p90_margin=float(np.quantile(all_scores,.9)),
            mean_frame_max_margin=float(np.mean(maxes)),max_margin=max(maxes),positive_fraction=positive/len(all_scores)))
    y=[r['query_present'] for r in rows]
    return rows,{k:binary_summary([r[k] for r in rows],y) for k in ('mean_margin','p90_margin','mean_frame_max_margin','max_margin','positive_fraction')}


def analyze(root,training,public):
    root,training,public=map(lambda p:Path(p).resolve(),(root,training,public));public.mkdir(parents=True,exist_ok=True)
    if any(not(root/m/'full_audit.json').exists() for m in METHODS):raise ValueError('all four methods must be sealed')
    reports={};metrics={'FROZEN':official_metrics(root/'FROZEN/results')}
    values={'FROZEN':official_video_values(root/'FROZEN/results/pedestrian_detailed.csv')}
    for method in METHODS:
        folder=root/method;destination=public/method;destination.mkdir(parents=True,exist_ok=True)
        with (destination/'offline_analysis_stdout.txt').open('w') as stream,redirect_stdout(stream):
            report=attribution(root,method,FROZEN,DATA,TRACKS,destination/'offline_attribution.json')
        metrics[method]=official_metrics(folder/'results');values[method]=official_video_values(folder/'results/pedestrian_detailed.csv')
        cm=method+'_COUNT_MATCHED_FROZEN'
        metrics[cm]=official_metrics(root/cm/'results');values[cm]=official_video_values(root/cm/'results/pedestrian_detailed.csv')
        train=json.loads((training/method/'training_receipt.json').read_text());run=json.loads((folder/'run_metadata.json').read_text())
        records=[json.loads(s) for s in (training/method/'metrics_train.jsonl').read_text().splitlines()]
        bad=[r['step'] for r in records if any(not math.isfinite(r.get(k,0.)) for k in ('source_loss','target_loss','gradient_norm'))]
        if bad or len(records)!=8973 or train['target_GT_open_attempts'] or run['target_GT_open_attempts']:
            raise ValueError('invalid numerical/data-access record')
        presence_rows,presence=presence_from_stream(folder,report);write_csv(destination/'episode_presence_scores.csv',presence_rows)
        train_seconds=train['wall_seconds'];eval_seconds=run['wall_seconds']
        full_audit=json.loads((folder/'full_audit.json').read_text())
        cost=dict(training_GPU_hours=train_seconds/3600,inference_GPU_hours=eval_seconds/3600,
            total_GPU_hours=(train_seconds+eval_seconds)/3600,actual_worker_training_wall_seconds=train_seconds,
            actual_worker_inference_wall_seconds=eval_seconds,
            per_worker_query_frame_FPS=full_audit['frames']/eval_seconds,
            peak_training_allocated_bytes=train['peak_allocated_bytes'],peak_inference_allocated_bytes=run['peak_allocated_bytes'],
            trainable_parameters=24576,optimizer_updates=train['counters']['optimizer_updates'],
            target_candidate_exposures=train['counters'].get('target_candidate_rows',0),
            pseudo_POS_exposures=train['counters'].get('pseudo_positive',0),pseudo_NEG_exposures=train['counters'].get('pseudo_negative',0),
            target_presence_supported_positive=0,target_presence_supported_negative=0,
            target_presence_UNKNOWN=True,temporal_valid_candidates=train['counters'].get('temporal_valid_candidates',0),
            effective_relation_candidates=dict(equivalent=train['counters'].get('equivalent_query_candidates',0),
                exclusive=train['counters'].get('exclusive_query_candidates',0)),
            exposure_warning='repeated training exposures, not unique verified semantic labels',
            peak_scope='allocator peak in the successful/resumed process; initial failed-process peak was not recorded separately',
            timing_warning='shared-GPU process elapsed; not exclusive-device utilization or standalone source-asset preparation')
        summary=dict(method=method,protocol='TRANSDUCTIVE_VAL_DIAGNOSTIC',episodes=425,videos=25,
            metrics=metrics[method],delta_HOTA_vs_Frozen=metrics[method]['HOTA']-metrics['FROZEN']['HOTA'],
            countmatched_metrics=metrics[cm],count_conditioned_delta=metrics[method]['HOTA']-metrics[cm]['HOTA'],
            countmatched_warning='POST_HOC_NOT_DEPLOYABLE; total cost depends on full UDA run; not a causal percentage',
            checkpoint={k:run[k] for k in ('checkpoint','checkpoint_sha256','selected_step','source_val_CE','selection')},
            presence=presence,cost=cost,training=train,attribution={k:v for k,v in report.items() if k!='episodes'})
        atomic_json(destination/'metrics.json',summary);reports[method]=summary
        atomic_json(destination/'timing.json',cost)
        atomic_json(destination/'checkpoint_metadata.json',summary['checkpoint'])
        atomic_json(destination/'model_sha256.json',dict(source_checkpoint_sha256=SOURCE_SHA,
            adapted_checkpoint_sha256=run['checkpoint_sha256'],checkpoint_path=run['checkpoint']))
        prov=json.loads((training/method/'provenance.json').read_text())
        atomic_json(destination/'data_manifest_sha256.json',dict(plan_sha256=prov['plan_sha256'],
            target_runtime_manifest_sha256=file_sha(ROOT/'manifests/dance_val_runtime.jsonl'),
            source_train_manifest_sha256=file_sha(ROOT/'manifests/source_kitti_v2_train.jsonl')))
        (destination/'git_commit.txt').write_text(prov['experiment_commit']+'\n')
        atomic_json(destination/'episode_receipts.json',dict(valid_episodes=425,
            episodes=[{k:e[k] for k in ('video','expression_id','prediction_path','prediction_sha256','margins_path','margins_sha256','stats')} for e in run['episodes']]))
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(folder/'results'/name,destination/name)
        shutil.copy2(folder/'trackeval.log',destination/'trackeval_stdout.txt')
        shutil.copy2(folder/'full_audit.json',destination/'full_audit.json')
        for name in ('training_receipt.json','source_validation.jsonl','provenance.json','config.json'):
            shutil.copy2(training/method/name,destination/name)
        with (training/method/'metrics_train.jsonl').open('rb') as source,(destination/'metrics_train.jsonl.gz').open('wb') as dest:
            with gzip.GzipFile(filename='',fileobj=dest,mode='wb',mtime=0) as compressed:shutil.copyfileobj(source,compressed)
        with (training/(method+'_stdout.txt')).open('rb') as source,(destination/'training_stdout.txt.gz').open('wb') as dest:
            with gzip.GzipFile(filename='',fileobj=dest,mode='wb',mtime=0) as compressed:shutil.copyfileobj(source,compressed)
        atomic_json(destination/'inference_receipt.json',{k:v for k,v in run.items() if k!='episodes'})
        shutil.copytree(folder/'parity',destination/'parity',dirs_exist_ok=True)
        if (training/method/'numerical_retries.jsonl').exists():shutil.copy2(training/method/'numerical_retries.jsonl',destination/'numerical_retries.jsonl')
        if (training/method/'resume_trace_recovery.json').exists():shutil.copy2(training/method/'resume_trace_recovery.json',destination/'resume_trace_recovery.json')
        cmdest=destination/'countmatched';cmdest.mkdir(exist_ok=True)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):shutil.copy2(root/cm/'results'/name,cmdest/name)
        shutil.copy2(root/cm/'full_audit.json',cmdest/'full_audit.json')
    counts=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    pairs=[(m,'FROZEN') for m in METHODS]+[(m,'SOURCE_REHEARSAL') for m in METHODS[1:]]
    pairs += [('COMBINED_UDA',m) for m in ('MEAN_TEACHER_UDA','WEAK_MIL_UDA')]
    pairs += [(m,m+'_COUNT_MATCHED_FROZEN') for m in METHODS]
    for a,b in pairs:
        if values[a][0]!=values[b][0]:raise ValueError('video-cluster IDs differ')
        x,y=values[a][1],values[b][1];delta=float(paired_clusters(x,y,np.ones(25)))
        boots=paired_clusters(x,y,counts);individual=paired_clusters(x,y,np.eye(25))
        leave=paired_clusters(x,y,np.ones((25,25))-np.eye(25))
        comparisons[a+'_vs_'+b]=dict(delta_HOTA=delta,video_cluster_CI95=np.percentile(boots,[2.5,97.5]).tolist(),
            improved_videos=int((individual>1e-9).sum()),declined_videos=int((individual< -1e-9).sum()),
            tied_videos=int((np.abs(individual)<=1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(d),delta_without_video=float(l)) for v,d,l in zip(values[a][0],individual,leave)])
    scheduler=json.loads((training/'full_suite_completion.json').read_text());inference_scheduler=json.loads((root/'evaluation_suite_completion.json').read_text())
    result=dict(status='COMPLETE_TRANSDUCTIVE_VAL_DIAGNOSTIC',videos=25,episodes=425,metrics=metrics,
        comparisons=comparisons,cluster_unit='25 whole videos; expressions/frames not independent',
        uncertainty='descriptive same-Val paired bootstrap2000/seed0, not independent test or training-seed uncertainty',
        actual_scheduler_training_wall_seconds=scheduler['wall_seconds'],actual_scheduler_inference_wall_seconds=inference_scheduler['wall_seconds'],
        evaluator_active_union_wall_seconds=inference_scheduler['union_evaluator_active_wall_seconds'],
        pipeline_wall_overlap=True,actual_pipeline_wall_seconds=inference_scheduler['last_evaluator_return_unix']-(training/'gpu_admission.json').stat().st_mtime,
        scheduler_query_frame_FPS=4*433636/inference_scheduler['wall_seconds'],
        checkpoint_selection='fixed64 source-val windows CE only, including step0',target_GT_used_by_training=False,
        target_GT_used_for_thresholds=False,threshold=0,raw_training_root=str(training),raw_evaluation_root=str(root))
    atomic_json(public/'summary.json',result)
    write_csv(public/'official_comparison.csv',[dict(method=m,**{k:r[k] for k in ('HOTA','DetA','AssA','IDF1','DetRe','DetPr','IDSW','CLR_TP','CLR_FP','Dets')}) for m,r in metrics.items()])
    write_csv(public/'cost_comparison.csv',[dict(method=m,**{k:v for k,v in r['cost'].items() if not isinstance(v,(dict,list))}) for m,r in reports.items()])
    # Requested compact scientific tables live one level above the full logs.
    tables=public.parent
    shutil.copy2(public/'official_comparison.csv',tables/'summary.csv');shutil.copy2(public/'cost_comparison.csv',tables/'training_cost.csv')
    ranking=[];presence_table=[];pseudo=[];cm_rows=[];video_rows=[];expression_rows=[]
    gt={e['video']+'+'+e['expression']:e['gt_observations'] for e in json.loads((public/METHODS[0]/'offline_attribution.json').read_text())['episodes']}
    for method,report in reports.items():
        a=report['attribution'];c=report['cost'];cm=report['countmatched_metrics']
        ranking.append(dict(method=method,source_AUROC=a['source_auroc'],adapted_AUROC=a['adapted_auroc'],
            **{'source_Recall@'+k:v for k,v in a['source_Recall_at_1_3_5'].items()},
            **{'adapted_Recall@'+k:v for k,v in a['adapted_Recall_at_1_3_5'].items()},
            recovered_GT=a['recovered_gt'],lost_GT=a['lost_gt'],new_FP_rows=a['new_false_positive_rows'],
            removed_FP_rows=a['removed_false_positive_rows'],net_FP_change=a['net_false_positive_change']))
        for group,stats in a['query_presence'].items():
            presence_table.append(dict(method=method,group=group,**stats))
        pseudo.append(dict(method=method,pseudo_POS_exposures=c['pseudo_POS_exposures'],pseudo_NEG_exposures=c['pseudo_NEG_exposures'],
            pseudo_UNKNOWN_exposures=report['training']['counters'].get('pseudo_unknown',0),
            target_presence_positive=0,target_presence_negative=0,target_presence_UNKNOWN=True,
            effective_temporal_candidates=c['temporal_valid_candidates'],
            evidence_note='confidence proxies are not verified semantic labels; no GT-filtered training'))
        cm_rows.append(dict(method=method,Frozen_HOTA=metrics['FROZEN']['HOTA'],adapted_HOTA=metrics[method]['HOTA'],
            CountMatched_Frozen_HOTA=cm['HOTA'],count_conditioned_delta=report['count_conditioned_delta'],
            protocol='POST_HOC_DIAGNOSTIC_NOT_DEPLOYABLE',total_cost_includes_parent_UDA=True))
    for method in metrics:
        with (root/method/'results/pedestrian_detailed.csv').open() as stream:
            detailed=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
        for row in detailed:
            video,expression=row['seq'].split('+',1)
            expression_rows.append(dict(method=method,video=video,expression=expression,query_present=bool(gt[row['seq']]),
                HOTA=100*float(row['HOTA___AUC']),IDF1=100*float(row['IDF1']),CLR_FP=int(float(row['CLR_FP'])),CLR_TP=int(float(row['CLR_TP']))))
        for i,video in enumerate(values[method][0]):
            seqs=[r for r in detailed if r['seq'].startswith(video+'+')]
            video_rows.append(dict(method=method,video=video,HOTA=float(pooled_hota(values[method][1],np.eye(25))[i]),
                CLR_FP=sum(int(float(r['CLR_FP'])) for r in seqs),CLR_TP=sum(int(float(r['CLR_TP'])) for r in seqs),episodes=len(seqs)))
    write_csv(tables/'candidate_ranking.csv',ranking);write_csv(tables/'pseudo_label_statistics.csv',pseudo)
    write_csv(tables/'count_matched_diagnostics.csv',cm_rows);write_csv(tables/'per_video.csv',video_rows)
    write_csv(tables/'per_expression.csv',expression_rows)
    # Present/Empty groups have the same schema in the reused evaluator.
    columns=list(dict.fromkeys(k for r in presence_table for k in r))
    write_csv(tables/'query_presence.csv',[{k:r.get(k) for k in columns} for r in presence_table])
    atomic_json(tables/'benchmark_protocol.json',dict(name='TRANSDUCTIVE_VAL_DIAGNOSTIC',source_train_videos=17,
        target_adaptation_videos=25,target_evaluation_videos=25,target_queries=425,target_adaptation_split='same25Val',
        independent_target_generalization=False,source_free=False,online_TTA=False,target_GT_in_training=False,
        target_GT_threshold_selection=False,source_val_selection_windows=64,coverage_cycles=1,steps_per_method=8973,
        MOT_candidate_rows=211085,candidate_query_observations=3588445,threshold=0,rank=4,seed=17,lr=5e-5,
        original40Train_to25Val_task_complete=False,GT_group_labels_offline_only=True,raw_roots=dict(training=str(training),evaluation=str(root))))
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--public-root',type=Path,default=ROOT/'results/uda/full')
    a=p.parse_args();analyze(a.output_root,a.training_root,a.public_root)
