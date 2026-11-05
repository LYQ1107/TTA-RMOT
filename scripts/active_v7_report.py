"""Post-seal official metrics, EB diagnostics and fixed-evidence slow effects."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.predictions import verify_predictions
from scripts.active_run import PAIRS
from scripts.report_active import write_table,saved_presence_decisions,final_confidence_summary,behavior_and_reliability


def verdict_for(method, directions, control_pending):
    by={r['pair']:r for r in directions};values={p:r.get(method) for p,r in by.items()}
    complete=all(values.get(p) is not None for p in PAIRS)
    wins=[p for p,r in by.items() if r.get(method) is not None and r[method]>max(r['YT_PRIOR'],r['UNIFORM_K'])]
    dance_ok=all(values.get(p) is not None and values[p]>=floor for p,floor in [('C1',35.724),('C6',34.469)])
    passed=complete and len(wins)>=4 and dance_ok and not control_pending
    return dict(method=method,six_directions_complete=complete,point_estimate_wins=wins,
        directions_above_both_controls=len(wins),Dance_both_within_two_of_v5=dance_ok,
        result='PENDING_CONTROLS' if control_pending else ('PASS' if passed else 'NOT_EVALUATED' if not complete else 'FAIL'),
        may_request_curves_and_ablations=passed,automatically_authorized=False)


def plot_stream(rows, public):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field in ('rho_hat','pi_hat'):
        fig,axes=plt.subplots(2,3,figsize=(15,7),squeeze=False)
        for ax,pair in zip(axes.flat,PAIRS):
            for method in sorted({x['method'] for x in rows if x['pair']==pair}):
                seq=[x for x in rows if x['pair']==pair and x['method']==method]
                ax.plot([0]+[x['video_index']+1 for x in seq],[.5]+[x['after_'+field] for x in seq],label=method,lw=1.4)
            ax.set(title=pair+': '+PAIRS[pair][1],xlabel='Completed videos (ID order)',ylabel=field,ylim=(-.02,1.02))
            ax.grid(alpha=.2)
            if ax.lines:ax.legend(fontsize=7)
        fig.suptitle('GT-free online EB; each direction/arm starts Beta(1,1)')
        fig.tight_layout();fig.savefig(public/(field+'_stream.png'),dpi=150);plt.close(fig)


def report(execution):
    from scripts.active_eval import summarize,bootstrap,offline_labels
    from scripts.active_attribution import attribute
    from scripts.presence_finish_a import references
    from scripts.presence_redesign_report import table
    public=execution.public;plan=json.loads((public/'registered_jobs.json').read_text())
    decisions_path=public/'baseline_budget_decisions.json'
    decisions=json.loads(decisions_path.read_text()) if decisions_path.exists() else dict(jobs=[],decisions=[])
    jobs=plan['jobs']+decisions['jobs']
    def folder(pair,method):return execution.active/pair/method/'rho_0.3'
    jobs=[j for j in jobs if (folder(j['pair'],j['method'])/'predictions_seal.json').exists()]
    for j in jobs:verify_predictions(folder(j['pair'],j['method']))
    old_public=Path(execution.manifest['prior_v5_public_root']);old_active=Path(execution.manifest['prior_v5_active_root'])
    with (old_public/'direction_summary.csv').open() as handle:old={r['pair']:r for r in csv.DictReader(handle)}
    with (Path(execution.manifest['prior_v6_public_root'])/'direction_summary.csv').open() as handle:
        old6={r['pair']:r for r in csv.DictReader(handle)}
    rows=[];costs=[];presence=[];stream=[];llrows=[];slowrows=[];recall=[];reliability=[];artifacts=[];intervals={};truths={};sports_sensitivity=[]
    summaries={}
    for job in jobs:
        pair,method=job['pair'],job['method'];target=PAIRS[pair][1];path=folder(pair,method)
        seal=verify_predictions(path);r=summarize(path,target);r.update(pair=pair,method=method+'_v7')
        rows.append(r);summaries[pair,method]=r
        if target=='SportsMOT':
            from scripts.active_eval import official_stats
            from scripts.pf_diag_common import official_metrics
            subset=path/'sensitivity_418';subset_seal=verify_predictions(subset)
            assert subset_seal['episode_count']==418
            assert subset_seal['original_full466_seal_sha256']==file_sha(path/'predictions_seal.json')
            sports_sensitivity.append(dict(pair=pair,method=r['method'],episodes=418,
                **official_metrics(subset/'results'),HOTA_unrounded=official_stats(subset)['HOTA_unrounded'],
                full466_HOTA=r['HOTA_unrounded'],prediction_rows=subset_seal['prediction_rows'],
                GT_protocol=subset_seal['sensitivity_protocol'],path=str(subset),
                prediction_seal_sha256=file_sha(subset/'predictions_seal.json')))
        artifacts.append(dict(pair=pair,method=method,path=str(path),seal_sha256=file_sha(path/'predictions_seal.json'),
            episodes=seal['episodes']))
        teacher=seal['teacher'];attempts=[json.loads(p.read_text()) for p in path.glob('worker_attempts/*.json')]
        receipts=[json.loads(p.read_text()) for p in sorted(path.glob('videos/*/receipt.json'))]
        sessions=[json.loads(p.read_text()) for p in sorted(path.glob('teacher_sessions/*.json'))]
        costs.append(dict(pair=pair,method=method,rho=.3,logical_questions=teacher['logical_calls'],
            new_teacher_calls=teacher.get('new_calls',0),cache_hits=teacher.get('cache_hits',0),
            replayed_answers=teacher.get('replayed_answers',0),
            teacher_forward_GPU_hours=teacher.get('forward_seconds',0)/3600,
            teacher_equivalent_GPU_hours=teacher.get('equivalent_forward_seconds',0)/3600,
            replay_teacher_cost_in_MINUS_dependency=method=='SLOW_REPLAY',
            LoRA_GPU_hours=seal.get('LoRA_GPU_seconds',0)/3600,rescore_GPU_hours=seal.get('rescore_GPU_seconds',0)/3600,
            assigned_worker_wall_hours=sum(x.get('wall_seconds',0) for x in attempts)/3600,
            failed_worker_attempts=sum(x['status']=='FAILED_RETAINED_FOR_RECOVERY' for x in attempts),
            peak_parent_allocated_bytes=max((x['parent_peak_allocated_bytes'] for x in attempts if x.get('parent_peak_allocated_bytes') is not None),default=None),
            peak_teacher_allocated_bytes=max((x['peak_allocated_bytes'] for x in sessions),default=None),
            trainable_parameters_observed_in_updates=sorted({rr['slow']['trainable_parameters'] for v in receipts for rr in v.get('rounds',[]) if 'trainable_parameters' in rr['slow']}),
            optimizer_updates=sum(rr['slow']['optimizer_updates'] for v in receipts for rr in v.get('rounds',[])),
            videos_with_LoRA_updates=sum(any(rr['slow']['optimizer_updates'] for rr in v.get('rounds',[])) for v in receipts),
            wall_is_not_GPU_utilization_integral=True))
        if method.startswith('YESTRACK'):continue
        if target not in truths:truths[target]=offline_labels(target)
        meta,truth,_=truths[target]
        presence+=saved_presence_decisions(path,meta,truth,pair,method+'_v7',.3)
        _,_,ds=behavior_and_reliability(path,target,pair,method+'_v7',.3,truth);reliability+=ds
        for vi,v in enumerate(receipts):
            eb=v['EB'];stream.append(dict(pair=pair,method=method,video=v['video'],video_index=vi,
                **{'before_'+k:eb['before'][k] for k in ('rho_hat','pi_hat')},
                **{'after_'+k:eb['after'][k] for k in ('rho_hat','pi_hat')},
                present=eb['confirmed_present'],absent=eb['confirmed_absent'],unknown=eb['unknown'],
                pi_distinct_pairs=len(eb['pi_pair_soft_counts'])))
            p=json.loads((path/'videos'/v['video']/'presence.json').read_text())
            llrows.extend(dict(pair=pair,method=method,video=v['video'],**x) for x in p['queries'])
            for rr in v['rounds']:
                s=rr['slow'];slowrows.append(dict(pair=pair,method=method,video=v['video'],round=rr['round'],
                    status=s['status'],weights_used_for_optimizer=s['optimizer_updates']>0,
                    training_pool_positive=s.get('training_pool_positive',0),training_pool_negative=s.get('training_pool_negative',0),
                    actual_positive_samples=s['actual_optimized_positive_samples'],actual_negative_samples=s['actual_optimized_negative_samples'],
                    optimizer_updates=s['optimizer_updates'],binary_positive_weight=s['binary_positive_weight'],none_class_weight=s['none_class_weight'],
                    positive_rechecks=rr['positive_rechecks'],round_end_positive_pairs=rr['positive_pairs'],
                    round_end_counts_not_optimizer_inputs=True))
        ar=attribute(path,target)
        recall.append(dict(pair=pair,method=method,GT_observations=ar['gt_observations'],
            matched_GT=ar['adapted_matched_gt'],GT_recall=ar['adapted_matched_gt']/ar['gt_observations'],
            output_rows=ar['adapted_selected_rows'],recovered_GT=ar['recovered_gt'],lost_GT=ar['lost_gt'],
            net_GT_vs_Native=ar['net_gt_recovery'],new_FP_rows=ar['new_false_positive_rows'],removed_FP_rows=ar['removed_false_positive_rows'],
            unchanged_rows_new_FP=ar['unchanged_rows_new_FP'],unchanged_rows_removed_FP=ar['unchanged_rows_removed_FP'],
            net_FP_vs_Native=ar['net_FP_change'],definition=ar['definition']))
    directions=[];control_pending=False
    for pair,(model,target) in PAIRS.items():
        decision=next((d for d in decisions['decisions'] if d['pair']==pair),None)
        use_new=bool(decision and decision['rerun_YT_PRIOR'])
        yt=folder(pair,'YESTRACK_BUDGET_PRIOR') if use_new else old_active/pair/'YESTRACK_BUDGET_PRIOR/rho_0.3'
        pending=use_new and not (yt/'predictions_seal.json').exists();control_pending|=pending
        row=dict(pair=pair,source=model,target=target,NATIVE=float(old[pair]['NATIVE']),PRES_TOPK=float(old[pair]['PRES_TOPK']),
            FULL_v5=float(old[pair]['FULL']),FULL_v6=float(old6[pair]['FULL_v6']) if pair in old6 else None,
            FULL_v7=summaries.get((pair,'FULL'),{}).get('HOTA_unrounded'),
            MINUS_SLOW_v7=summaries.get((pair,'MINUS_SLOW'),{}).get('HOTA_unrounded'),
            SLOW_REPLAY_v7=summaries.get((pair,'SLOW_REPLAY'),{}).get('HOTA_unrounded'),
            YT_PRIOR=summaries[pair,'YESTRACK_BUDGET_PRIOR']['HOTA_unrounded'] if (pair,'YESTRACK_BUDGET_PRIOR') in summaries else float(old[pair]['YESTRACK_BUDGET_PRIOR']),
            UNIFORM_K=float(old[pair]['UNIFORM_K']),YT_PRIOR_rematch_pending=pending)
        directions.append(row)
        refs={'NATIVE':dict(references(target,model))['NATIVE'],
            'PRES_TOPK':execution.stage_a/target/model/'PRES_TOPK',
            'FULL_v5':old_active/pair/'FULL/rho_0.3',
            'YT_PRIOR':yt if not pending else old_active/pair/'YESTRACK_BUDGET_PRIOR/rho_0.3',
            'UNIFORM_K':old_active/pair/'UNIFORM_K/rho_0.3'}
        if pair in old6:refs['FULL_v6']=Path(execution.manifest['prior_v6_active_root'])/pair/'FULL/rho_0.3'
        for method in ('FULL','MINUS_SLOW','SLOW_REPLAY'):
            if (pair,method) not in summaries:continue
            for name,ref in refs.items():
                if name=='YT_PRIOR' and pending:continue
                intervals[pair+'__'+method+'_v7_minus_'+name]=bootstrap(folder(pair,method),ref,target)
            if method!='MINUS_SLOW' and (pair,'MINUS_SLOW') in summaries:
                intervals[pair+'__'+method+'_v7_minus_MINUS_SLOW_v7']=bootstrap(folder(pair,method),folder(pair,'MINUS_SLOW'),target)
    confidence=final_confidence_summary(presence)
    likelihood_by_query={(r['pair'],r['method']+'_v7',r['video'],r['query']):r for r in llrows}
    existence=[]
    for pair,method in sorted({(r['pair'],r['method']) for r in presence}):
        values=[r for r in presence if r['pair']==pair and r['method']==method]
        existence.append(dict(pair=pair,method=method,query_present_episodes=sum(r['query_present_GT'] for r in values),
            query_empty_episodes=sum(not r['query_present_GT'] for r in values),
            correctly_inferred_empty=sum(not r['query_present_GT'] and r['teacher_absent'] for r in values),
            present_incorrectly_inferred_empty=sum(r['query_present_GT'] and r['teacher_absent'] for r in values),
            empty_rejected_without_presence_answers=sum(not r['query_present_GT'] and r['teacher_absent']
                and likelihood_by_query[pair,method,r['video'],r['expression']]['presence_answers']==0 for r in values),
            present_rejected_without_presence_answers=sum(r['query_present_GT'] and r['teacher_absent']
                and likelihood_by_query[pair,method,r['video'],r['expression']]['presence_answers']==0 for r in values),
            query_empty_prediction_rows=sum(r['selected_prediction_rows'] for r in values if not r['query_present_GT'])))
    for name,values in [('main',rows),('direction_summary',directions),('costs',costs),('eb_stream',stream),
        ('presence_likelihood',llrows),('slow_rounds',slowrows),('presence_decisions',presence),
        ('existence_summary',existence),('final_confidence',confidence),('recall_compare',recall),('reliability_vs_GT',reliability),('sports_sensitivity',sports_sensitivity)]:
        write_table(public/(name+'.csv'),values)
    atomic_json(public/'bootstrap.json',intervals);atomic_json(public/'artifacts.json',dict(new=artifacts))
    gates=[verdict_for(m,directions,control_pending) for m in ('FULL_v7','MINUS_SLOW_v7')]
    verdict=dict(status='REGISTERED_TIER_REPORTED',arms=gates,completed_jobs=len(rows),
        expected_primary_jobs=len(plan['jobs']),conditional_controls_pending=control_pending,
        may_request_curves_and_ablations=any(g['may_request_curves_and_ablations'] for g in gates),
        automatically_authorized=False,remaining_v5_80_denied=True,v6_C4_C6_denied=True)
    atomic_json(public/'verdict.json',verdict);plot_stream(stream,public)
    primary_done=sum(not r['method'].startswith('YESTRACK') for r in rows)
    text='# v7：在线EB存在性与慢速分支公平检验\n\n'
    text+=f'当前已封存并完成官方评测 {primary_done}/{len(plan["jobs"])} 个主项；只对实际完成集合统计，不能把阶段结果当最终验收。\n\n'
    text+=table(['method','six_directions_complete','directions_above_both_controls','Dance_both_within_two_of_v5','result'],gates)
    text+='\n## 1. 六方向主表（未运行格为NA，不是0）\n\n'+table(list(directions[0]),
        [{k:'NA' if v is None else v for k,v in row.items()} for row in directions])
    text+='\n## 2. 完整官方指标\n\n'+table(['pair','method','HOTA_unrounded','DetA','AssA','IDF1','IDSW',
        'prediction_rows','query_empty_prediction_rows','Present_HOTA','teacher_calls','GT_open_attempts'],rows)
    text+='\n## 3. 存在性与在线比例\n\n'+table(list(existence[0]),existence)
    text+='\n![rho stream](rho_hat_stream.png)\n\n![pi stream](pi_hat_stream.png)\n\n'
    text+='逐视频先验与软计数见eb_stream.csv/原presence.json；逐句未截断/截断LL和触发次数见presence_likelihood.csv。停止概率与按最后DS重算概率分别记录，不冒充同一个时点。\n'
    text+='presence_decisions.csv中的teacher_absent/teacher_confirmed_present是继承的字段名，v7实际表示EB的终止判空/判存在，不是教师直接审阅完整视频的标注；无分层回答而仅由先验触发的拒绝另行列出。\n'
    text+='\n## 4. 高置信错负与真实恢复\n\n'+table(['pair','method','matched_candidate_positive_tracks_majority',
        'confident_negative_GT_FN','confident_negative_GT_FN_unqueried','GTpositive_high_negative_rate'],confidence)
    text+='\n此表高置信错负仍用v6的p<.05以可比；实际伪监督阈值为.15/.85，且未问始终UNKNOWN。最终posterior不是实际优化输入。\n\n'
    text+='v7组合prior截断到[-2,2]且未问对evidence=0，因此未判空句子的未问对posterior至少为sigmoid(-2)≈.1192；本表未问p<.05来自整句判空置零，不是未问对被赋了训练负标签。\n\n'
    text+=table(list(recall[0]),recall)
    text+='\nGT召回是独立IoU≥.5最大基数匹配，不替代官方TrackEval；新增/删除预测行的FP与不变行匹配重分配分列。\n'
    text+='\n## 5. 慢速更新、成本与成对差值\n\n'+table(list(costs[0]),costs)
    effects=[dict(comparison=k,**v) for k,v in intervals.items() if k.endswith('minus_MINUS_SLOW_v7')]
    if effects:text+='\n'+table(['comparison','delta_HOTA','CI95','clusters'],effects)
    text+='\n逐轮真实优化输入正负暴露、训练池计数、两种权重、更新步数见slow_rounds.csv与原逐步records；不由轮末标签倒推。SLOW_REPLAY复用MINUS实际问题/帧/看法/原始回答，新增教师调用为0；独立部署总成本仍包含MINUS教师依赖，不能说完全不用教师。\n'
    text+='\nFULL−MINUS包括选题反馈；REPLAY−MINUS固定教师内容但包含LoRA及模型引起的f/快校准/EB下游效应，并非将派生状态也固定。零adapter输入已用相同冻结bank锚定，后续FP32打分路径及32帧近似的限制仍需披露。\n'
    text+='\n## 6. 不确定性与边界\n\n完整视频bootstrap2000次seed0见bootstrap.json；KITTI四视频仅点估计，条件于当前固定流，未覆盖重新适应后的顺序/seed不确定性。\n'
    text+='\n视频转导式/冻结教师辅助，Val历史开发复用，非严格帧因果或盲测。所有GT仅封存后独立CPU读取。DS事后值仅实际提问集合，不代表全候选教师准确率。worker/phase elapsed非GPU利用率积分；历史候选、冻结推理、源头成本不是零。\n'
    text+='\n成本关卡见budget_preflight.json；本批结束后停止，不自动开预算曲线/消融、未选v7 FULL、v6扩展或旧80项。\n'
    if sports_sensitivity:text+='\n## SportsMOT 418条敏感性（v4内容不变，仅限制序列ID）\n\n'+table(
        ['pair','method','full466_HOTA','HOTA_unrounded','DetA','AssA','IDF1','IDSW','episodes','prediction_rows'],sports_sensitivity)
    (public/'REPORT.md').write_text(text)
    print(json.dumps(verdict,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',action='store_true');p.parse_args()
    execution,_=authorize_active();report(execution)
