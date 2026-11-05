"""Close the fixed v7 tier using sealed evaluations and logs only, without GT."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from scripts.report_active import write_table
from scripts.presence_redesign_report import table


def close(execution):
    public=execution.public
    read=lambda name:json.loads((public/name).read_text())
    csv_rows=lambda name:list(csv.DictReader((public/name).open()))
    queue,validation,verdict,logs=map(read,('queue_completion.json','validation.json','verdict.json','logs.json'))
    plan=read('registered_jobs.json');controls=read('baseline_budget_decisions.json')
    assert queue['status']=='V7_AUTHORIZED_TIER_COMPLETE'
    assert validation['status']=='V7_AUTHORIZED_TIER_VALIDATED'
    assert len(plan['jobs'])==queue['primary_jobs']==validation['primary_jobs']==12
    assert len(controls['jobs'])==queue['conditional_YT_PRIOR_jobs']==validation['conditional_jobs']
    main=csv_rows('main.csv');recall=csv_rows('recall_compare.csv');costs=csv_rows('costs.csv')
    assert len(main)==12+len(controls['jobs'])==logs['completed_active_methods']
    assert not logs['unpackaged_launcher_logs']
    assert all(int(r['GT_open_attempts'])==0 for r in main)
    for item in main:assert file_sha(Path(item['path'])/'predictions_seal.json')==item['prediction_seal_sha256']
    for name in ('archive','raw_teacher_responses'):
        item=logs[name];assert file_sha(item['path'])==item['sha256']
    from scripts.active_v6_close import verify_packaged_records
    archive_audit=verify_packaged_records(public,logs)
    intervals=read('bootstrap.json');m={(r['pair'],r['method']):r for r in main}
    a={(r['pair'],r['method']):r for r in recall};effects=[]
    for pair in ('C1','C3','C4'):
        baseline=m[pair,'MINUS_SLOW_v7'];ar=a[pair,'MINUS_SLOW']
        for method in ('FULL','SLOW_REPLAY'):
            current=m[pair,method+'_v7'];ac=a[pair,method]
            key=pair+'__'+method+'_v7_minus_MINUS_SLOW_v7'
            effects.append(dict(pair=pair,comparison=method+'_v7_minus_MINUS_SLOW_v7',**intervals[key],
                delta_GT_recall=float(ac['GT_recall'])-float(ar['GT_recall']),
                delta_prediction_rows=int(current['prediction_rows'])-int(baseline['prediction_rows']),
                delta_Query_Empty_rows=int(current['query_empty_prediction_rows'])-int(baseline['query_empty_prediction_rows']),
                delta_Present_HOTA=float(current['Present_HOTA'])-float(baseline['Present_HOTA']),
                teacher_content_fixed=method=='SLOW_REPLAY',derived_fast_EB_and_f_not_numerically_frozen=True))
    write_table(public/'slow_effects.csv',effects)
    started=read('execution_started.json');durable=Path(started['durable_dir'])
    launch=json.loads((durable/'launch.json').read_text());finished=json.loads((durable/'completion.json').read_text())
    assert finished['exit_code']==0
    total=lambda name:sum(float(r[name]) for r in costs)
    cost=dict(primary_jobs=12,conditional_YT_PRIOR_jobs=len(controls['jobs']),
        logical_questions_including_fixed_replay=int(total('logical_questions')),
        new_teacher_calls=int(total('new_teacher_calls')),cache_hits=int(total('cache_hits')),
        replayed_answers=int(total('replayed_answers')),
        teacher_forward_GPU_phase_hours=total('teacher_forward_GPU_hours'),
        LoRA_GPU_phase_hours=total('LoRA_GPU_hours'),rescore_GPU_phase_hours=total('rescore_GPU_hours'),
        measured_new_GPU_phase_hours=total('teacher_forward_GPU_hours')+total('LoRA_GPU_hours')+total('rescore_GPU_hours'),
        assigned_worker_wall_hours=total('assigned_worker_wall_hours'),
        scheduler_wall_hours=(finished['finished_time_ns']-launch['launched_time_ns'])/3.6e12,
        failed_worker_attempts=int(total('failed_worker_attempts')),
        optimizer_updates=int(total('optimizer_updates')),
        GPU_phase_and_worker_times_are_not_utilization_integrals=True,
        scheduler_wall_includes_waiting_loading_CPU_evaluation_and_reports=True,
        teacher_subprocess_wall_not_double_added=True,
        replay_teacher_dependency_cost_in_MINUS_not_added_twice=True,
        inherited_official_model_and_Stage_A_cost_not_zero_or_remeasured=True)
    assert cost['logical_questions_including_fixed_replay']==cost['new_teacher_calls']+cost['cache_hits']+cost['replayed_answers']
    atomic_json(public/'cost_totals.json',cost)
    atomic_json(public/'final_official_results.json',dict(status='V7_FIXED_AUTHORIZED_TIER_OFFICIALLY_EVALUATED',
        completed=main,validation_sha256=file_sha(public/'validation.json')))
    final=dict(status='V7_FIXED_AUTHORIZED_TIER_COMPLETE_AND_STOPPED',
        primary_jobs=12,conditional_YT_PRIOR_jobs=len(controls['jobs']),
        total_episodes=validation['total_episodes'],prediction_rows=validation['prediction_rows'],GT_open_attempts=0,
        arms=verdict['arms'],may_request_curves_and_ablations=verdict['may_request_curves_and_ablations'],
        new_experiment_automatically_authorized=False,
        v6_C4_C6_started=False,remaining_v5_80_started=False,unselected_v7_FULL_C2_C5_C6_started=False,
        original_108_job_goal_not_claimed_complete=True,
        registered_plan_sha256=file_sha(public/'registered_jobs.json'),
        numerical_code_frozen_since='ea08a284a80bcdf7c2b556fff266c1dcf59aa8af',
        original_runtime_root=str(execution.active.parent),durable_recovery_dir=str(durable),
        exact_log_and_teacher_archive_validation=archive_audit['status'])
    atomic_json(public/'final_stage_status.json',final)
    report=public/'REPORT.md';text=report.read_text()
    text=text.replace('只对实际完成集合统计，不能把阶段结果当最终验收。',
        '原授权成本回退集合全部完成；未运行的FULL C2/C5/C6仍为NA，不能当作六方向FULL验收。')
    text+='\n## 固定证据回放与主慢速分支的最终差值\n\n'
    text+=table(['pair','comparison','delta_HOTA','CI95','delta_GT_recall','delta_prediction_rows','delta_Query_Empty_rows','teacher_content_fixed'],effects)
    text+='\n固定回放保持MINUS的问题、帧、视图和原始回答不变；复核资格按原MINUS选题状态审计，而不是依据回放改变后的后验重选。LoRA引起的f/快速校准/EB变化仍是下游效应。\n'
    text+='\n## 最终成本、日志与恢复路径\n\n'+table(['metric','value'],[dict(metric=k,value=v) for k,v in cost.items()])
    text+='\n主集合为六MINUS、三FULL、三REPLAY；额外YT仅按FULL调用变化严格>5%规则触发。MINUS与FULL调用量不同，复用对照不伪称逐臂精确等预算。\n'
    text+='\nGT审计及完整候选/EB更新/实际优化输入核验见validation.json；逐句截断前后似然和生效次数见presence_likelihood.csv。\n'
    text+='完整关闭日志与实际用到的唯一教师原响应见logs.json、archive_validation.json及其列出的归档；checkpoint仅发布路径/大小/SHA，预测与PCD留服务器。\n'
    text+='\n运行根目录：`'+str(execution.active.parent)+'`。原durable恢复记录：`'+str(durable)+'`。封存方法不会重跑已完成episode。\n'
    text+='\n## 最终研究边界与停止\n\n'
    text+='不以loss下降、梯度非零、代码通过或某个单方向提升代替六方向验收；不跨FULL/MINUS逐格挑最优。v7复合改动包含EB、伪标签、复核和损失平衡，未做单因素消融，不能把全部差值仅归因EB。\n'
    text+='DS各视图来自同一MLLM且相关；分层内按价值选题会影响π估计，未证明无偏。数据流先验可以导致无新教师证据的拒绝，存在/空句误拒及未问错负均完整报告。固定32帧不是末模型全视频重打分。\n'
    text+='视频聚类CI条件于本次有序适应流，不覆盖流顺序或重新训练的不确定性；历史开发Val不是盲测集。本设置是MLLM辅助、视频转导式，不是严格帧因果或完全不调用教师的Source-Free在线系统。\n'
    text+='\n本批结束并停止。不运行v6 C4–C6、旧80项、未选v7 FULL、预算曲线、消融或新方法；即便满足申请门槛，也必须等待用户下一条明确授权。\n'
    report.write_text(text)
    print(json.dumps(dict(final,costs=cost),ensure_ascii=False),flush=True)


if __name__=='__main__':
    execution,_=authorize_active();close(execution)
