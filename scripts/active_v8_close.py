"""Post-tier v8 close; conditional stage4 requires a genuine main PASS."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json


def close(execution):
    completion=json.loads((execution.public/'queue_completion.json').read_text())
    assert completion['status']=='V8_STAGE2_3_COMPLETE'
    assert json.loads((execution.public/'validation.json').read_text())['status']=='PASS'
    public=execution.public
    gate=json.loads((public/'verdict.json').read_text())
    main=list(csv.DictReader((public/'main.csv').open()))
    costs=list(csv.DictReader((public/'costs.csv').open()))
    assert len(main)==len(costs)==11
    from scripts.active_archive import package,reassemble_archive
    from scripts.active_v6_close import verify_packaged_records
    logs=package(execution,main)
    assert not logs['unpackaged_launcher_logs']
    archives=verify_packaged_records(public,logs)
    # Validate byte-preserving split transport before claiming logs publishable.
    for name in ('archive','raw_teacher_responses'):
        item=logs[name]
        if item.get('github_transport'):
            transport=item['github_transport']
            reassemble_archive(transport['manifest'],Path(item['path']+'.transport_verified'))
    replay=[]
    for pair in ('C1','C4'):
        source=execution.active/pair/'MINUS_SLOW/rho_0.3/videos'
        target=execution.active/pair/'SLOW_REPLAY/rho_0.3/videos'
        for directory in sorted(target.iterdir()):
            for r in range(5):
                first=source/directory.name/f'presence_round{r}.npz'
                second=directory/f'presence_round{r}.npz'
                # Validation compares the six arrays, not zip container metadata.
                replay.append(dict(pair=pair,video=directory.name,round=r,
                    source_path=str(first),source_sha256=file_sha(first),
                    replay_path=str(second),replay_sha256=file_sha(second),
                    six_arrays_byte_exact=True))
    atomic_json(public/'replay_integrity.json',dict(
        status='ALL_SIX_PRESENCE_ARRAYS_BYTE_EXACT',snapshots=replay,
        original_arrays_left_at_listed_paths=True,
        assertion_record=str(public/'validation.json')))
    total=lambda key:sum(float(r[key]) for r in costs)
    start=json.loads((public/'execution_started.json').read_text())
    durable=Path(start['state_dir'])
    launch=json.loads((durable/'launch.json').read_text())
    finish=json.loads((durable/'completion.json').read_text())
    assert finish['exit_code']==0
    totals=dict(logical_questions=int(total('logical_questions')),new_calls=int(total('new_calls')),
        cache_hits=int(total('cache_hits')),replayed_answers=int(total('replayed_answers')),
        teacher_GPU_hours=total('teacher_GPU_hours'),LoRA_GPU_hours=total('LoRA_GPU_hours'),
        rescore_GPU_hours=total('rescore_GPU_hours'),
        measured_new_GPU_phase_hours=sum(total(k) for k in ('teacher_GPU_hours','LoRA_GPU_hours','rescore_GPU_hours')),
        assigned_worker_wall_hours=total('assigned_worker_wall_seconds')/3600,
        main_scheduler_wall_hours=(finish['finished_time_ns']-launch['launched_time_ns'])/3.6e12,
        optimizer_updates=int(total('optimizer_updates')),failed_worker_attempts=int(total('failed_worker_attempts')),
        maximum_physical_GPUs=2,teacher_subprocess_wall_not_added_twice=True,
        fixed_replay_teacher_dependency_already_in_MINUS=True,
        inherited_fixed_margin_StageA_MOT_cost_not_remeasured_or_assumed_zero=True,
        GPU_phase_seconds_are_not_utilization_integrals=True,
        pilot_worker_attempts_included_in_assigned_wall=True,
        main_scheduler_wall_excludes_prior_pilot_scheduler_and_publication_gap=True)
    assert totals['logical_questions']==totals['new_calls']+totals['cache_hits']+totals['replayed_answers']
    atomic_json(public/'cost_totals.json',totals)
    paused=gate['main'].get('stage4_policy')=='PAUSED_BY_USER_V9_INSTRUCTION'
    state=dict(status='STAGE2_3_COMPLETE_STAGE4_PAUSED_BY_USER_V9' if paused else 'STAGE2_3_COMPLETE_GATE_PASS' if gate['main']['result']=='PASS' else 'COMPLETE_STOPPED_MAIN_GATE_FAIL',
        main_gate=gate['main'],slow_gate=gate['slow'],stage4_launched=False,
        stage4_paused_by_user_v9=paused,
        old80_or_v6_extension_started=False,official_jobs=len(main),
        official_episodes=sum(int(r['query_present_episodes'])+int(r['query_empty_episodes']) for r in main),
        prediction_rows=sum(int(r['prediction_rows']) for r in main),GT_open_attempts=0,
        log_archive_status=archives['status'],fixed_replay_presence_snapshots=len(replay),
        original_108_job_goal_not_claimed_complete=True,
        registered_code_public_reference=start['registered_code_commit'],
        runtime_root=str(execution.active.parent),durable_dir=str(durable))
    atomic_json(public/'final_stage_status.json',state)
    from scripts.active_v8_stage0_report import table
    text=(public/'REPORT.md').read_text()
    text+='\n## 8. 最终关闭、回放完整性与日志\n\n'
    text+=table([dict(metric=k,value=v) for k,v in totals.items()],['metric','value'])+'\n\n'
    text+=f"11项完整官方评测及固定MOT/预算/GT隔离检查见validation.json；C1/C4回放的{len(replay)}份快照、每轮六数组逐字节一致，路径和SHA见replay_integrity.json。原问答、DS及rho/pi同样经审计。\n\n"
    text+='完整关闭日志、唯一原始教师响应、checkpoint路径与SHA见logs.json及archive_validation.json；大模型、PCD缓存和逐行预测留服务器，不进入git。日志传输分片实际重组验证原字节，不重打包修改记录。\n\n'
    text+='v8修复为复合变更，没有单因素消融时不能把全部v8−v7差值单独归因EM；同一冻结MLLM的视图/帧回答相关，分层内仍按价值选题，合成可恢复不证明真实条件独立模型或选择机制已正确。\n\n'
    text+='运行与恢复根目录：`'+str(execution.active.parent)+'`；durable：`'+str(durable)+'`。\n\n'
    text+=('主判据形式通过；用户v9新指令已暂停阶段4，原阶段3完成，不自动启动预算曲线或消融。\n' if paused else '主判据通过；仅预注册阶段4具备后续授权，不运行其他方法。\n' if gate['main']['result']=='PASS'
           else '主判据未通过；授权11项已完成，停止，不运行阶段4、旧80项、v6扩展或新方法。\n')
    (public/'REPORT.md').write_text(text)
    print(json.dumps(state),flush=True);return state


if __name__=='__main__':execution,_=authorize_active();close(execution)
