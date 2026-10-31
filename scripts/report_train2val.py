#!/usr/bin/env python3
"""Offline-only Train→Val Table C, fixed count attribution and video bootstrap."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts import diag_attrib_v2_common as common
from scripts.train2val_controls import verify_native_predictions
from scripts.run_train2val_training import RUN,PUBLIC,ARMS
from scripts.pf_diag_common import official_metrics,official_video_values
from scripts.analyze_pf_video_uncertainty import paired_clusters,pooled_hota
from scripts.analyze_pf_full_offline import load_eval_gt,overlaps,matched_gt,auroc
from scripts.diagnose_flexhook_v2_dance import load_tracker_boxes
from scripts.analyze_psr_matrix import official_query_groups


def literal_verdict(hotas,ci):
    if ci[0]>0 and all(h>13.848 for h in hotas):
        answer='MLLM 存在性知识可跨视频、跨表达泛化'
    elif ci[0]<=0<=ci[1] or any(h<=13.848 for h in hotas):
        answer='同集增益未能泛化到未见视频和表达'
    else:
        answer='预注册规则未覆盖的情况，如实报告，不新增规则'
    g=float(np.mean(hotas));retention=(g-8.218)/(29.702-8.218)
    level='大部分保留' if retention>=.7 else '部分保留' if retention>=.3 else '基本未保留'
    return dict(literal_user_rule_verdict=answer,G=g,retention=retention,retention_label=level,
        expression_split_limitation='Train/Val share8 expression filenames: not a strict all-unseen-language test',
        does_not_establish_strict_unseen_expression_generalization=True)


def markdown(headers,rows):
    def fmt(v):
        if v is None:return 'N/A'
        if isinstance(v,(int,np.integer)):return str(v)
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])


def report():
    started=time.perf_counter();verify_native_predictions();common.EVAL=RUN/'evaluation'
    planned=json.loads((RUN/'controls_seal.json').read_text());completion=json.loads((RUN/'official_evaluation_completion.json').read_text())
    if completion['status']!='COMPLETE_ALL_PREDICTION_SETS':raise ValueError('finish official evaluation first')
    meta,old_scores,groups,_=common.load_bank();methods=planned['methods'];native=list(planned['new_native_methods'])
    main=['FROZEN','SAME_SET_D1','SAME_SET_D3','B1_SRC_RATE_GLOBAL','OFFICIAL_DANCE_COMPLETE',*native]
    manifests={m:common.verify_seal(m) for m in methods}
    metrics={m:official_metrics(common.EVAL/m/'results') for m in methods};values={};videos=None
    for m in methods:
        v,stats=official_video_values(common.EVAL/m/'results/pedestrian_detailed.csv')
        if videos is not None and v!=videos:raise ValueError('25-video cluster alignment differs')
        videos=v;values[m]=stats
        with (common.EVAL/m/'results/pedestrian_detailed.csv').open() as h:
            combined=next(r for r in csv.DictReader(h) if r['seq']=='COMBINED')
        if not np.isclose(float(pooled_hota(stats,np.ones(25))),100*float(combined['HOTA___AUC']),rtol=0,atol=1e-8):
            raise ValueError('official TP-weighted video pooling changed')
        if manifests[m]['prediction_rows']!=metrics[m]['Dets']:raise ValueError('sealed count versus official Dets mismatch')
    scores={'FROZEN':old_scores['FROZEN'],'SAME_SET_D1':old_scores['D1'],'SAME_SET_D3':old_scores['D3'],
        'B1_SRC_RATE_GLOBAL':old_scores['FROZEN'],'OFFICIAL_DANCE_COMPLETE':old_scores['OFFICIAL_DANCE_COMPLETE']}
    scores.update({m:np.load(RUN/'score_bank'/(m+'.npy'),mmap_mode='r') for m in native})
    g=meta['query_frame_groups'];labels=np.zeros(meta['candidate_query_rows'],dtype=bool);gt_counts=np.zeros(g,dtype=np.int32)
    k={m:np.zeros(g,dtype=np.int32) for m in main};recall={m:0. for m in main};denom=0
    query_truth={};retrieval_per_episode=[];boxes={v:load_tracker_boxes(common.TRACKS,v) for v in videos}
    recovery={m:Counter() for m in main}
    # Fixed same-frame GT retrieval denominator includes every official GT frame,
    # including any frame with no retrievable candidate. No teacher endpoint mask:
    # all Val videos are outside Train by construction.
    for q in meta['episodes']:
        v,e=q['video_id'],q['expression_id'];truth=load_eval_gt(common.DATA,v,e)
        query_truth[v,e]=sum(map(len,truth.values()));denom+=len(truth)
        chosen={}
        for m in main:
            selected={}
            for line in (common.EVAL/m/'results'/v/e/'predict.txt').read_text().splitlines():
                if not line.strip():continue
                fields=line.split(',');selected.setdefault(int(float(fields[0])),set()).add(int(float(fields[1])))
            chosen[m]=selected
        episode_recall={m:0. for m in main}
        for t,a,b,group in q['groups']:
            gt=truth.get(t,[]);gt_counts[group]=len(gt)
            for m in main:k[m][group]=len(chosen[m].get(t,set()))
            if not gt:
                base=chosen['FROZEN'].get(t,set())
                for m in main:
                    current=chosen[m].get(t,set());recovery[m]['offline_FP']+=len(current)
                    recovery[m]['new_FP_rows']+=len(current-base);recovery[m]['removed_FP_rows']+=len(base-current)
                continue
            ids=sorted(boxes[v].get(t,{}))
            if len(ids)!=b-a:raise ValueError('offline candidate geometry alignment changed')
            iou=overlaps([boxes[v][t][i] for i in ids],[box for _,box in gt])
            if len(ids):labels[a:b]=(iou>=.5).any(axis=1)
            base_indices=[j for j,i in enumerate(ids) if i in chosen['FROZEN'].get(t,set())]
            bm,_=matched_gt(iou,base_indices)
            for m in main:
                order=np.lexsort((np.asarray(ids),-scores[m][a:b]))
                found,_=matched_gt(iou,order[:1].tolist())
                value=len(found)/len(gt);recall[m]+=value;episode_recall[m]+=value
                current=chosen[m].get(t,set());base=chosen['FROZEN'].get(t,set())
                selected_indices=[j for j,i in enumerate(ids) if i in current]
                am,matched_indices=matched_gt(iou,selected_indices)
                matched_ids={ids[j] for j in matched_indices}
                recovery[m]['Recovered_GT']+=len(am-bm);recovery[m]['Lost_GT']+=len(bm-am)
                recovery[m]['matched_GT']+=len(am);recovery[m]['offline_FP']+=len(current)-len(am)
                recovery[m]['new_FP_rows']+=len((current-base)-matched_ids)
                # A removed row is FP in the baseline's one-to-one assignment,
                # not merely a difference in official CLEAR FP after reassignment.
                _,base_matched_indices=matched_gt(iou,base_indices)
                base_matched_ids={ids[j] for j in base_matched_indices}
                recovery[m]['removed_FP_rows']+=len((base-current)-base_matched_ids)
        retrieval_per_episode.append(dict(video=v,expression=e,GT_frames=len(truth),
            GT_observations=query_truth[v,e],R1={m:episode_recall[m]/len(truth) if truth else None for m in main}))
    if sum(query_truth.values())!=212541 or sum(n>0 for n in query_truth.values())!=40:
        raise ValueError('fixed official GT protocol changed')
    alignment={};gt_report=dict(official_GT_detections=int(gt_counts.sum()),query_present_episodes=40,query_empty_episodes=385,
        query_frames=g,GT_present_frames=int((gt_counts>0).sum()),GT_empty_frames=int((gt_counts==0).sum()),
        R1_denominator_GT_frames=denom,GT_only_opened_in_independent_post_seal_evaluator=True)
    for m in main:
        x=k[m];nonempty=x[x>0]
        present=sum(e['selected_rows'] for e in manifests[m]['episodes'] if query_truth[e['video'],e['expression']])
        empty=manifests[m]['prediction_rows']-present
        grouped=official_query_groups(common.EVAL/m,dict(episodes=[dict(video=v,expression=e,gt_observations=n) for (v,e),n in query_truth.items()]))
        alignment[m]=dict(method=m,**metrics[m],prediction_rows=int(x.sum()),relative_GT=float(x.sum()/212541),
            relative_OFFICIAL_COMPLETE=float(x.sum()/254910),selection_rate=float(x.sum()/3588445),
            nonempty_query_frames=int(len(nonempty)),nonempty_query_frame_fraction=float(len(nonempty)/g),
            mean_k_nonempty=float(nonempty.mean()) if len(nonempty) else 0.,
            p50_k_nonempty=float(np.quantile(nonempty,.5)) if len(nonempty) else None,
            p90_k_nonempty=float(np.quantile(nonempty,.9)) if len(nonempty) else None,
            mean_k_GT_present_frame=float(x[gt_counts>0].mean()),mean_k_GT_empty_frame=float(x[gt_counts==0].mean()),
            query_present_selected_rows=present,query_empty_selected_rows=empty,
            Present_HOTA=grouped['query_present']['HOTA'],Empty_FP=grouped['query_empty']['official_CLEAR_FP'],
            Present_CLEAR_FP=grouped['query_present']['official_CLEAR_FP'],GT_R1_all=recall[m]/denom,
            AUROC=auroc(scores[m],labels),target_GT_open_attempts_in_generator=0,
            **{key:recovery[m][key] for key in ('Recovered_GT','Lost_GT','matched_GT','offline_FP','new_FP_rows','removed_FP_rows')},
            information='official25Val; no target training labels/Val checkpoint selection; references inherited same-Val development')
        if alignment[m]['prediction_rows']!=metrics[m]['Dets']:raise ValueError('Table C row-count disagreement')
    common.write_csv(PUBLIC/'TABLE_C.csv',list(alignment.values()));atomic_json(PUBLIC/'alignment.json',dict(GT=gt_report,methods=alignment))
    atomic_json(PUBLIC/'retrieval_per_episode.json',retrieval_per_episode)
    atomic_json(PUBLIC/'GT_recovery.json',dict(methods={m:dict(r) for m,r in recovery.items()},
        definition='independent IoU>=.5 max-cardinality frame assignment; new/removed rows distinguished from official CLEAR matching reassignment',
        threshold_or_training_feedback=False))
    draws=np.random.default_rng(0).multinomial(25,np.ones(25)/25,size=2000);comparisons={}
    def compare(a,b):
        va,vb=values[a],values[b];samples=paired_clusters(va,vb,draws);per=paired_clusters(va,vb,np.eye(25))
        comparisons[a+'_minus_'+b]=dict(first=a,second=b,delta_HOTA=float(paired_clusters(va,vb,np.ones(25))),
            CI95=np.percentile(samples,[2.5,97.5]).tolist(),positive_videos=int((per>1e-9).sum()),
            negative_videos=int((per < -1e-9).sum()),tied_videos=int((np.abs(per)<=1e-9).sum()),
            per_video=[dict(video=v,delta_HOTA=float(d)) for v,d in zip(videos,per)])
        return samples
    seeds=[m for m in native if m.startswith('T2V_D1_')]
    seed_summary={}
    for baseline in ('FROZEN','B1_SRC_RATE_GLOBAL','SAME_SET_D1','OFFICIAL_DANCE_COMPLETE'):
        boot=[compare(m,baseline) for m in seeds]
        delta=[comparisons[m+'_minus_'+baseline]['delta_HOTA'] for m in seeds]
        seed_summary[baseline]=dict(seeds=[ARMS[m][1] for m in seeds],mean_delta_HOTA=float(np.mean(delta)),
            std_delta_HOTA=float(np.std(delta,ddof=1)) if len(delta)>1 else None,
            paired_video_CI95_of_seed_mean=np.percentile(np.mean(boot,axis=0),[2.5,97.5]).tolist())
    if 'T2V_D3_s17' in native:compare('T2V_D3_s17','FROZEN')
    for m in native:
        compare(m,'FROZEN_AT_'+m+'_COUNT')
        compare(m+'_AT_OFFICIAL_COUNT','FROZEN_AT_OFFICIAL_COUNT')
    if len(seeds)==3:
        hotas=[metrics[m]['HOTA'] for m in seeds]
        verdict=literal_verdict(hotas,comparisons['T2V_D1_s17_minus_B1_SRC_RATE_GLOBAL']['CI95'])
        verdict.update(three_seed_HOTA=hotas,HOTA_seed_std=float(np.std(hotas,ddof=1)),
            unrounded_G=float(np.mean([float(pooled_hota(values[m],np.ones(25))) for m in seeds])))
    else:verdict=dict(literal_user_rule_verdict='INCOMPLETE_REGISTERED_SEEDS_NO_SCIENTIFIC_VERDICT')
    bootstrap=dict(clusters=25,samples=2000,seed=0,pooling='same v2 official TP-weighted HOTA sufficient statistics',
        comparisons=comparisons,three_seed_summary=seed_summary,verdict=verdict,
        caveat='video uncertainty conditional on fixed trained seeds; expressions/frames are not independent; Val was historical development data')
    atomic_json(PUBLIC/'bootstrap.json',bootstrap)
    controls=[dict(method=m,**metrics[m],protocol='POST_HOC_DIAGNOSTIC_INHERITS_PARENT_COST') for m in methods if m not in main]
    common.write_csv(PUBLIC/'count_ranking.csv',controls)
    per_video=[]
    for m in methods:
        for v,h in zip(videos,pooled_hota(values[m],np.eye(25))):per_video.append(dict(method=m,video=v,HOTA=float(h)))
    common.write_csv(PUBLIC/'per_video.csv',per_video)
    table=markdown(['方法','HOTA','DetA','AssA','IDF1','Present HOTA','Empty FP','k(GT有目标帧)','k(GT空帧)','GT R@1','AUROC'],
        [[m,*[alignment[m][k] for k in ('HOTA','DetA','AssA','IDF1','Present_HOTA','Empty_FP',
            'mean_k_GT_present_frame','mean_k_GT_empty_frame','GT_R1_all','AUROC')]] for m in main])
    (PUBLIC/'TABLE_C.md').write_text(table+'\n\n完整v2表C字段见TABLE_C.csv。Empty FP/Present CLEAR FP用官方CLEAR口径；数量×排序为事后诊断，包含父模型训练/推理成本。\n')
    (PUBLIC/'COUNT_RANKING.md').write_text(markdown(['诊断','HOTA','DetA','AssA','IDF1','输出行数'],
        [[r['method'],*[r[k] for k in ('HOTA','DetA','AssA','IDF1','Dets')]] for r in controls])+'\n')
    (PUBLIC/'BOOTSTRAP.md').write_text(markdown(['配对比较','ΔHOTA','视频95% CI','上升/下降'],
        [[name,r['delta_HOTA'],str(r['CI95']),f"{r['positive_videos']}/{r['negative_videos']}"] for name,r in comparisons.items()])+'\n')
    for m in methods:
        dest=PUBLIC/'official'/m;dest.mkdir(parents=True,exist_ok=True)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
            shutil.copy2(common.EVAL/m/'results'/name,dest/name)
    with gzip.open(PUBLIC/'prediction_files.csv.gz','wt') as handle:
        writer=csv.DictWriter(handle,fieldnames=['method','video','expression','prediction_path','prediction_sha256','selected_rows','margins_path','margins_sha256'])
        writer.writeheader()
        for m,seal in manifests.items():
            for e in seal['episodes']:writer.writerow(dict(method=m,**{k:e.get(k,'') for k in writer.fieldnames if k!='method'}))
    # Full stage logs remain on disk; compact stdout/receipts are publishable.
    with tarfile.open(PUBLIC/'stage_logs.tar.gz','w:gz') as archive:
        for path in sorted(RUN.rglob('*')):
            if path.is_file() and path.suffix in ('.log','.txt','.json') and any(x in path.parts for x in
                ('prep_launchers','training_launchers','inference_launchers','official_jobs','logs')):
                archive.add(path,arcname=str(path.relative_to(RUN)))
    report=(PUBLIC/'REPORT.md').read_text();cut=report.index('## 4. 表 B / C、')
    report=report[:cut]+'## 4. 表 B / C、全量 Val 结果\n\n'+(PUBLIC/'TABLE_C.md').read_text()+\
        '\n'+(PUBLIC/'COUNT_RANKING.md').read_text()+'\n## 5. 25视频bootstrap与预注册判定\n\n'+\
        (PUBLIC/'BOOTSTRAP.md').read_text()+'\n```json\n'+json.dumps(verdict,ensure_ascii=False,indent=2)+'\n```\n'+\
        '\n三seed比较的均值/样本标准差与条件视频区间见bootstrap.json。D3仅注册seed17，不杜撰三个seed。\n'+\
        '\n固定划分仍有8个表达文件名重合；字面判定不能证明全部语言未见。Val是历史反复开发过的25视频，不称全项目盲测。\n'+\
        '\n## 6. GT隔离、成本、失败、恢复与停止\n\n训练/教师输入审计与kernel白名单见各provenance/input_access_audit；真实禁止输入/目标GT尝试均须0。\n'+\
        '原候选、Teacher原始logits、四final-step checkpoint、完整预测与长metrics日志保留服务器，路径/SHA见server_artifacts.json与prediction_files.csv.gz。\n'+\
        'Train使用原始GT几何/ID、Val使用预测NeuralSORT；每窗口曝光约4.60对旧约16.89，均为固定协议差异，不改超参补偿。\n'+\
        '完整官方summary/detailed和阶段stdout/失败/恢复记录已归档；所有成本见costs.json，count-controls不作为免费独立系统。\n'+\
        '完成本轮后停止，不开展新方法、密集教师标注、超参或新seed搜索。\n'
    report=report.replace('全量教师待执行，P3–P5 未运行','P2–P5 完成，全部结果上报，停止')
    (PUBLIC/'REPORT.md').write_text(report)
    atomic_json(PUBLIC/'completion.json',dict(status='COMPLETE_TRAIN_TO_VAL_STOP_NO_NEW_METHOD',
        methods=main,official_prediction_sets=len(methods),GT=gt_report,verdict=verdict,
        independent_offline_report_seconds=time.perf_counter()-started,report_sha256=file_sha(PUBLIC/'REPORT.md')))
    print(json.dumps(dict(event='T2V_FINAL_REPORT_COMPLETE',verdict=verdict),ensure_ascii=False),flush=True)


if __name__=='__main__':report()
