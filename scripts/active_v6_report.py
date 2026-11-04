"""Independent post-seal v6/v5 comparison; never imported by GPU workers."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import target_metadata
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS
from scripts.report_active import (write_table,saved_presence_decisions,final_confidence_summary,
    behavior_and_reliability,calibration_calls,committed_teacher_equivalent_hours)


def report(execution):
    from scripts.active_eval import summarize,offline_labels,bootstrap
    from scripts.active_attribution import attribute
    from scripts.presence_finish_a import references
    from scripts.presence_redesign_report import table
    public=execution.public;old=Path(execution.manifest['prior_v5_active_root'])
    plan=json.loads((public/'registered_jobs.json').read_text());jobs=list(plan['jobs'])
    decision_path=public/'baseline_budget_decisions.json'
    decisions=json.loads(decision_path.read_text()) if decision_path.exists() else None
    if decisions:jobs+=decisions['jobs']
    def folder(pair,method,new=True):
        return (execution.active if new else old)/pair/method/'rho_0.3'
    # No GT loads until this stage's entire fixed prediction set is sealed.
    for j in jobs:verify_predictions(folder(j['pair'],j['method']))
    rows=[];costs=[];presence=[];rounds=[];videos=[];reliability=[];calibration=[];artifacts=[];attributions={}
    truth_cache={};meta_cache={};intervals={};summaries={}
    for j in jobs:
        pair,method=j['pair'],j['method'];target=PAIRS[pair][1];path=folder(pair,method)
        seal=verify_predictions(path);r=summarize(path,target);summaries[pair,method]=r
        r=dict(r,pair=pair,version='v6',method=method+'_v6' if method in ('FULL','MINUS_SLOW') else method+'_REMATCHED')
        rows.append(r);artifacts.append(dict(pair=pair,method=r['method'],path=str(path),
            seal_sha256=file_sha(path/'predictions_seal.json'),episodes=seal['episodes']))
        receipts=[json.loads(p.read_text()) for p in sorted(path.glob('videos/*/receipt.json'))]
        teacher=seal['teacher']
        attempts=[json.loads(p.read_text()) for p in sorted(path.glob('worker_attempts/*.json'))]
        recorded_wall=sum(a['wall_seconds'] for a in attempts if a.get('wall_seconds') is not None)
        costs.append(dict(pair=pair,method=r['method'],rho=.3,
            logical_calls=teacher['logical_calls'],new_calls=teacher['new_calls'],cache_hits=teacher['cache_hits'],
            teacher_forward_GPU_hours=teacher['forward_seconds']/3600,
            committed_teacher_equivalent_forward_GPU_hours=committed_teacher_equivalent_hours(receipts,teacher,method),
            LoRA_GPU_hours=seal['LoRA_GPU_seconds']/3600,rescore_GPU_hours=seal['rescore_GPU_seconds']/3600,
            seal_process_wall_hours=seal['wall_seconds']/3600,
            assigned_worker_wall_hours_including_failures=recorded_wall/3600,
            failed_worker_attempts=sum(a.get('status')=='FAILED_RETAINED_FOR_RECOVERY' for a in attempts),
            unfinished_attempt_wall_unknown=sum(a.get('wall_seconds') is None for a in attempts),
            wall_not_GPU_utilization_integral=True,
            incomplete_failed_score_phase_not_fabricated_as_zero=True,
            peak_teacher_allocated_bytes=teacher['peak_allocated_bytes'],
            optimizer_updates=sum(rr['slow']['optimizer_updates'] for v in receipts for rr in v.get('rounds',[])),
            frame_prior_CPU_hours=sum(v.get('decode_CPU_seconds',0) for v in receipts)/3600))
        if method not in ('FULL','MINUS_SLOW'):continue
        if target not in truth_cache:
            meta,truth,_=offline_labels(target);meta_cache[target]=meta;truth_cache[target]=truth
        meta,truth=meta_cache[target],truth_cache[target]
        presence+=saved_presence_decisions(path,meta,truth,pair,r['method'],.3)
        a,b,c=behavior_and_reliability(path,target,pair,r['method'],.3,truth)
        rounds+=a;videos+=b;reliability+=c;calibration+=calibration_calls(receipts,pair,r['method'],.3)
        attributions[pair,method]=attribute(path,target)
    confidence=final_confidence_summary(presence)
    old_public=Path(execution.manifest['prior_v5_public_root'])
    with (old_public/'final_confidence.csv').open() as f:old_conf=list(csv.DictReader(f))
    with (old_public/'direction_summary.csv').open() as f:old_directions={x['pair']:x for x in csv.DictReader(f)}
    with (old_public/'reliability_vs_GT.csv').open() as f:old_reliability=list(csv.DictReader(f))
    comparisons=[];main=[];mechanism=[];recall=[];baseline_refs=[]
    for pair in sorted({j['pair'] for j in jobs if j['method']=='FULL'}):
        target=PAIRS[pair][1];model=PAIRS[pair][0];current=folder(pair,'FULL')
        ds=next((d for d in decisions['decisions'] if d['pair']==pair),None) if decisions else None
        yt_new=bool(ds and ds['rerun_YT_PRIOR'])
        refs={'NATIVE':dict(references(target,model))['NATIVE'],
            'PRES_TOPK':execution.stage_a/target/model/'PRES_TOPK',
            'FULL_v5':folder(pair,'FULL',False),'UNIFORM_K':folder(pair,'UNIFORM_K',False),
            'YT_PRIOR':folder(pair,'YESTRACK_BUDGET_PRIOR',yt_new)}
        values={}
        for label,path in refs.items():
            key={'FULL_v5':'FULL','YT_PRIOR':'YESTRACK_BUDGET_PRIOR'}.get(label,label)
            values[label]=(summaries[pair,'YESTRACK_BUDGET_PRIOR']['HOTA_unrounded']
                if label=='YT_PRIOR' and yt_new else float(old_directions[pair][key]))
            baseline_refs.append(dict(pair=pair,label=label,path=str(path),
                seal_sha256=file_sha(Path(path)/'predictions_seal.json'),reused_not_GPU_rerun=not(label=='YT_PRIOR' and yt_new)))
            intervals[pair+'__FULL_v6_minus_'+label]=bootstrap(current,path,target)
        r=summaries[pair,'FULL'];best=max(values['YT_PRIOR'],values['UNIFORM_K'])
        main.append(dict(pair=pair,source=model,target=target,**{k:values[k] for k in ('NATIVE','PRES_TOPK')},
            FULL_v5=values['FULL_v5'],FULL_v6=r['HOTA_unrounded'],
            FULL_v6_minus_v5=r['HOTA_unrounded']-values['FULL_v5'],
            MINUS_SLOW_v6=summaries.get((pair,'MINUS_SLOW'),{}).get('HOTA_unrounded'),
            YT_PRIOR=values['YT_PRIOR'],UNIFORM_K=values['UNIFORM_K'],above_both_controls=r['HOTA_unrounded']>best,
            teacher_calls=r['teacher_calls'],YT_PRIOR_rematched=yt_new))
        c=next(x for x in confidence if x['pair']==pair and x['method']=='FULL_v6')
        b=next(x for x in old_conf if x['pair']==pair and x['method']=='FULL')
        for label,item in [('v5',b),('v6',c)]:
            denom=int(item['matched_candidate_positive_tracks_majority']);wrong=int(item['confident_negative_GT_FN'])
            unasked=int(item['confident_negative_GT_FN_unqueried'])
            mechanism.append(dict(pair=pair,version=label,GT_positive_long_track_pairs=denom,
                high_confident_negative_GT_positive_pairs=wrong,high_negative_rate_among_GT_positive=wrong/denom if denom else None,
                high_negative_unasked_GT_positive_pairs=unasked,
                unasked_fraction_within_high_negative_GT_positive=unasked/wrong if wrong else None,
                note='posterior diagnosis, not masked optimizer pseudo labels'))
            ap=folder(pair,'FULL',label=='v6')/'offline_attribution.json'
            ar=attributions[pair,'FULL'] if label=='v6' else json.loads(ap.read_text())
            recall.append(dict(pair=pair,version=label,GT_recall=ar['adapted_matched_gt']/ar['gt_observations'],
                GT_observations=ar['gt_observations'],matched_GT_observations=ar['adapted_matched_gt'],
                output_rows=ar['adapted_selected_rows'],new_FP=ar['new_false_positive_rows'],
                removed_FP=ar['removed_false_positive_rows'],net_GT_recovery=ar['net_gt_recovery'],
                note='offline IoU>=.5 max-cardinality, not official CLEAR'))
        if (pair,'MINUS_SLOW') in summaries:
            intervals[pair+'__FULL_v6_minus_MINUS_SLOW_v6']=bootstrap(current,folder(pair,'MINUS_SLOW'),target)
    for name,records in [('main',rows),('direction_summary',main),('costs',costs),('presence_decisions',presence),
        ('final_confidence',confidence),('mechanism_compare',mechanism),('recall_compare',recall),
        ('behavior_round',rounds),('behavior_video',videos),('reliability_vs_GT',reliability),('calibration_calls',calibration)]:
        write_table(public/(name+'.csv'),records)
    atomic_json(public/'bootstrap.json',intervals);atomic_json(public/'artifacts.json',dict(new=artifacts,reused=baseline_refs))
    ds_compare=[]
    for pair in sorted({x['pair'] for x in main}):
        for version,source,method in [('v5',old_reliability,'FULL'),('v6',reliability,'FULL_v6')]:
            for worker in ('CROP','CLIP','FRAME_BOX'):
                values=[x for x in source if x['pair']==pair and x['method']==method and x['worker']==worker
                        and x['GT_scope']=='TRACK_MAJORITY_50_PERCENT']
                if not values:continue
                row=max(values,key=lambda x:x['through_video'])
                ds_compare.append(dict(pair=pair,version=version,worker=worker,
                    specificity_DS=row['specificity_DS'],specificity_GT=row['specificity_GT'],
                    sensitivity_DS=row['sensitivity_DS'],sensitivity_GT=row['sensitivity_GT'],
                    observations=row['observations'],GT_scope='TRACK_MAJORITY_50_PERCENT'))
    write_table(public/'reliability_compare.csv',ds_compare)
    six=len(main)==6;wins=sum(x['above_both_controls'] for x in main)
    verdict=dict(status='SIX_DIRECTION_GATE_EVALUATED' if six else 'FIVE_JOB_COST_GATE_STAGE_ONLY',
        evaluated_FULL_directions=len(main),directions_above_both_controls=wins,
        may_request_full_matrix=six and wins>=4,full_matrix_automatically_authorized=False,
        pending_original_v6_directions=sorted(set(PAIRS)-{r['pair'] for r in main}),
        remaining_v5_80_denied=True,all_predictions_sealed_before_GT=True)
    atomic_json(public/'verdict.json',verdict)
    text='# v6关键集合结果\n\n'+verdict['status']+'。只报告实际完成方向，不以3方向替代6方向验收。\n\n'
    text+='原10项保守估算84.058 GPU小时>20，故按用户指令只运行C1/C2/C3 FULL和C1/C3 MINUS_SLOW。\n'
    text+='v5剩余80项及本轮未授权方向未启动；不根据目标GT选择参数、checkpoint或扩展范围。\n\n'
    text+=table(list(main[0]),main)+'\n\n## 完整指标\n\n'+table(
        ['pair','method','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','Present_HOTA','teacher_calls'],rows)
    text+='\n\n## 高置信错负（最终posterior，非伪标签）\n\n'+table(list(mechanism[0]),mechanism)
    text+='\n\n## GT召回与预测行数\n\n'+table(list(recall[0]),recall)
    text+='\n\n## 教师与适应成本\n\n'+table(list(costs[0]),costs)
    if ds_compare:text+='\n\n## DS估计与事后GT（实际提问集合，轨迹多数口径）\n\n'+table(list(ds_compare[0]),ds_compare)
    text+='\n\nDS完整估计/实测特异度与敏感度、两种GT口径见reliability_vs_GT.csv；'
    text+='v5相同表保留于../redesign_v5/active_map_v1_critical28/reliability_vs_GT.csv。\n'
    text+='视频聚类bootstrap2000次seed0，KITTI仅4视频点估计；条件于既定有序流，不覆盖重新适应/训练seed不确定性。\n'
    text+='本设定冻结教师辅助、video-transductive，Val历史开发复用，非盲测/严格帧因果。\n'
    text+='缓存含视频起始全帧与LoRA后固定32帧版本，非所有帧最后模型；PCD为FP32，但原两标量存在性头只读margin。\n'
    text+='GT仅在完整预测封存后独立CPU评测；未问伪标签UNKNOWN是规则，不是语义正确率。\n'
    text+='调用变化>5%才按v6逐视频预算补跑YT_PRIOR，未触发者复用v5且明确非新逐视频精确匹配。\n'
    (public/'REPORT.md').write_text(text)
    print(json.dumps(verdict),flush=True)


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');p.parse_args()
    execution,_=authorize_active();report(execution)


if __name__=='__main__':main()
