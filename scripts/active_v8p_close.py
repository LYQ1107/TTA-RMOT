"""Close completed v8P records; no inference, training or new GT evaluation."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.execution import authorize_active
from flex_uda.presence.predictions import verify_predictions


def close(execution):
    public=execution.public;start=json.loads((public/'execution_started.json').read_text())
    durable=Path(start['state_dir']);finished=json.loads((durable/'completion.json').read_text())
    assert finished['exit_code']==0
    gate=json.loads((public/'stage_A_gate.json').read_text())
    if gate['result']=='FAIL':
        assert not (public/'registered_jobs_B1.json').exists()
        assert not (public/'registered_jobs_B2.json').exists()
    else:
        assert json.loads((public/'stage_B_completion.json').read_text())['B1']
        assert json.loads((public/'stage_B_completion.json').read_text())['B2']
    rows=[];totals=dict(logical_calls=0,new_calls=0,cache_hits=0,replayed_answers=0,
        teacher_GPU_hours=0.,LoRA_GPU_hours=0.,rescore_GPU_hours=0.)
    artifact_files=[]
    for path in sorted(execution.active.glob('*/*/rho_*/predictions_seal.json')):
        folder=path.parent;seal=verify_predictions(folder)
        summary=json.loads((folder/'offline_summary.json').read_text())
        assert summary['prediction_seal_sha256']==file_sha(path)
        row=dict(pair=seal['pair'],method=seal['method'],rho=seal['rho'],path=str(folder),
            HOTA=summary['HOTA_unrounded'],prediction_rows=seal['prediction_rows'],
            prediction_seal_sha256=file_sha(path),GT_open_attempts=seal['GT_open_attempts'])
        rows.append(row)
        teacher=seal['teacher']
        for k in ('logical_calls','new_calls','cache_hits','replayed_answers'):
            totals[k]+=teacher.get(k,0)
        totals['teacher_GPU_hours']+=teacher.get('forward_seconds',0)/3600
        totals['LoRA_GPU_hours']+=seal.get('LoRA_GPU_seconds',0)/3600
        totals['rescore_GPU_hours']+=seal.get('rescore_GPU_seconds',0)/3600
        for candidate in folder.glob('videos/*/*'):
            if candidate.is_file() and candidate.suffix in ('.npz','.npy'):
                artifact_files.append(dict(path=str(candidate),bytes=candidate.stat().st_size,sha256=file_sha(candidate)))
    assert totals['logical_calls']==totals['new_calls']+totals['cache_hits']+totals['replayed_answers']
    launch=json.loads((durable/'launch.json').read_text())
    totals.update(scheduler_wall_hours=(finished['finished_time_ns']-launch['launched_time_ns'])/3.6e12,
        maximum_total_project_physical_GPUs=2,shared_with_unmodified_original_v8=True,
        GPU_phase_durations_not_exclusive_utilization_integrals=True,
        reused_C1_C6_original_GPU_cost_not_counted_as_new=True,
        fixed_MOT_margin_StageA_inherited_cost_not_assumed_zero=True)
    atomic_json(public/'cost_totals.json',totals)
    atomic_json(public/'numerical_artifact_paths.json',dict(files=artifact_files,
        large_predictions_and_arrays_not_in_git=True))
    from scripts.active_archive import package,reassemble_archive
    from scripts.active_v6_close import verify_packaged_records
    logs=package(execution,rows)
    assert not logs['unpackaged_launcher_logs']
    archived=verify_packaged_records(public,logs)
    for name in ('archive','raw_teacher_responses'):
        item=logs[name]
        if item.get('github_transport'):
            reassemble_archive(item['github_transport']['manifest'],Path(item['path']+'.transport_verified'))
    final=dict(status='STOPPED_STAGE_A_FAIL' if gate['result']=='FAIL' else
        'CONDITIONAL_B_CLOSED_B3_DEFERRED' if (public/'B3_deferred.json').exists() else 'REGISTERED_V8P_COMPLETE',
        main_gate=gate,official_new_jobs=len(rows),official_new_episodes=sum(
            json.loads((Path(r['path'])/'predictions_seal.json').read_text())['episode_count'] for r in rows),
        GT_open_attempts=0,archive_status=archived['status'],old80_started=False,
        original_v8_stage3_unmodified=True,runtime_root=str(execution.active.parent),durable_dir=str(durable),
        next_method_started=False)
    atomic_json(public/'final_status.json',final)
    text=(public/'REPORT.md').read_text()
    text+='\n## 关闭与可恢复记录\n\n完整新方法日志及原始教师输出见 logs.json，原字节/冻结revision/BF16概率验核见 archive_validation.json；大数组、模型与逐行预测留在 numerical_artifact_paths.json 和 prediction_paths.json 所列服务器路径，不进git。\n\n'
    text+='GPU相位时长不是独占显卡利用率积分；本轮与原v8共享同两张物理卡，不能把两个队列的相位时长相加后冒充独占卡时。新增调用、缓存、继承C1/C6与回放分别记账。\n\n'
    text+='关闭状态：'+final['status']+'。原v8阶段3仍按其原队列独立运行/关闭，不由本关闭器启动旧阶段4。\n'
    (public/'REPORT.md').write_text(text)
    print(json.dumps(final),flush=True);return final


if __name__=='__main__':close(authorize_active()[0])
