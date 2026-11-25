"""Completed two-video MAP pilot accounting; deliberately no GT/HOTA evaluation."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_uda.presence.runtime import PUBLIC
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.presence_redesign_report import table
from scripts.diag_attrib_v2_common import write_csv


def calls(snapshot,pair,variant):
    rows=[]
    for video in snapshot['video_receipts']:
        for r in video['rounds']:
            entries=[('BEFORE_SLOW',r['fast'])]
            if 'after_slow' in r['fast']:entries.append(('AFTER_SLOW',r['fast']['after_slow']))
            for position,x in entries:
                identity=x.get('identity',{});presence=x.get('presence',{})
                if variant=='MAPv1':
                    if not np.isfinite([x['slope'],x['temperature'],x['bias']]).all() or min(x['slope'],x['temperature'])<=0:
                        raise ValueError('MAP parameter positivity/finiteness failed')
                    for item in (identity,presence):
                        if item['status']=='MAP_UPDATED':
                            if not item['positive'] or not item['negative']:raise ValueError('Unsupported one-class update')
                            if item['objective_after']>item['objective_before']+1e-7:raise ValueError('MAP objective increased')
                rows.append(dict(variant=variant,pair=pair,video=video['video'],round=r['round'],position=position,
                    status=x['status'],slope=x['slope'],temperature=x['temperature'],bias=x['bias'],
                    identity_status=identity.get('status','LEGACY_UNRESTRICTED'),
                    supported_positive=identity.get('positive'),supported_negative=identity.get('negative'),
                    unqueried_excluded=identity.get('unqueried_confident_excluded'),
                    presence_status=presence.get('status','LEGACY_UNRESTRICTED'),
                    confirmed_positive=presence.get('positive'),confirmed_negative=presence.get('negative'),
                    identity_objective_before=identity.get('objective_before'),identity_objective_after=identity.get('objective_after'),
                    presence_objective_before=presence.get('objective_before'),presence_objective_after=presence.get('objective_after')))
    return rows


def main():
    execution,publication=authorize_active();record=execution.public
    audit_path=record / 'pilot_map_first_video.json';audit=json.loads(audit_path.read_text())
    baseline_path=PUBLIC / 'redesign_v5/active/pilot_first_video.json';old=json.loads(baseline_path.read_text())
    rows=[];calibration=[];artifacts=[]
    for pair in ('C1','C3'):
        new=audit['pilot_snapshots'][pair];baseline=old['pilot_snapshots'][pair]
        if new['videos']!=baseline['videos']:raise ValueError('Comparisons must use the same registered first video')
        for variant,r in [('LEGACY',baseline),('MAPv1',new)]:
            cc=calls(r,pair,variant);calibration+=cc;rounds=[z for v in r['video_receipts'] for z in v['rounds']]
            teacher=r['teacher'];last=cc[-1]
            rows.append(dict(pair=pair,variant=variant,video=r['videos'][0],logical_teacher_calls=teacher['logical_calls'],
                new_forward_calls=teacher['new_calls'],cache_hits=teacher['cache_hits'],
                teacher_forward_seconds=teacher['forward_seconds'],teacher_equivalent_forward_seconds=teacher['equivalent_forward_seconds'],
                teacher_peak_allocated_bytes=teacher['peak_allocated_bytes'],
                optimizer_updates=sum(z['slow']['optimizer_updates'] for z in rounds),
                LoRA_seconds=r['LoRA_GPU_seconds'],rescore_seconds=r['rescore_GPU_seconds'],
                final_slope=last['slope'],final_temperature=last['temperature'],final_bias=last['bias'],
                max_slope=max(z['slope'] for z in cc),max_temperature=max(z['temperature'] for z in cc),
                max_abs_bias=max(abs(z['bias']) for z in cc),
                MAP_identity_updates=sum(z['identity_status']=='MAP_UPDATED' for z in cc),
                MAP_presence_updates=sum(z['presence_status']=='MAP_UPDATED' for z in cc),
                final_confident_positive=rounds[-1]['positive_pairs'],final_confident_never_asked=rounds[-1]['confident_pairs_never_asked'],
                prediction_rows=sum(v['selected_rows'] for v in r['video_receipts']),GT_open_attempts=r['GT_open_attempts'],
                full_RMOT_HOTA='NOT_EVALUATED'))
        root=execution.active / pair / 'FULL/rho_0.3'
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.suffix in ('.json','.jsonl','.npz','.pt') or p.name=='mask.npy'):
                artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    write_csv(record / 'pilot_comparison.csv',rows);write_csv(record / 'calibration_rounds.csv',calibration)
    protected=['results/presence_active/REPORT.md','vendor/FlexHook',
        'flex_uda/teacher/yestrack_backend.py','flex_uda/teacher/visual_input_builder.py',
        'results/presence_active/redesign_v3','results/presence_active/redesign_v4']
    changed=subprocess.check_output(['git','diff','--name-only','f0fc286893fc97821d502f59c5b9d75620776b01','--',*protected],cwd=ROOT,text=True)
    if changed.strip():raise ValueError('Historical results/official backend changed')
    running=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p / 'cmdline').read_bytes().split(b'\0');cwd=(p / 'cwd').resolve()
        except OSError:continue
        if cwd==ROOT and len(argv)>1 and Path(argv[1].decode(errors='replace')).name in ('active_run.py','active_teacher_worker.py','active_launch.py'):
            running.append(dict(pid=int(p.name),command=argv[1].decode(errors='replace')))
    if running:raise RuntimeError('Wait for healthy own workers; do not kill: '+str(running))
    new_rows=[r for r in rows if r['variant']=='MAPv1']
    updates=sum(r['MAP_identity_updates']+r['MAP_presence_updates'] for r in new_rows)
    status='NUMERICS_AND_EVIDENCE_GATE_VERIFIED_WITH_REAL_MAP_UPDATES' if updates else 'ONLY_SAFE_ABSTENTION_OBSERVED'
    costs=dict(new_teacher_calls=sum(r['new_forward_calls'] for r in new_rows),
        logical_calls=sum(r['logical_teacher_calls'] for r in new_rows),cache_hits=sum(r['cache_hits'] for r in new_rows),
        new_teacher_forward_seconds=sum(r['teacher_forward_seconds'] for r in new_rows),
        teacher_equivalent_forward_seconds=sum(r['teacher_equivalent_forward_seconds'] for r in new_rows),
        LoRA_seconds=sum(r['LoRA_seconds'] for r in new_rows),rescore_seconds=sum(r['rescore_seconds'] for r in new_rows),
        workers=[dict(path=str(p),**json.loads(p.read_text())) for p in sorted(execution.active.glob('*/*/*/worker_attempts/*.json'))],
        meaning='shared-device elapsed phase timing, not utilization integral; teacher cache is not a zero-cost independent system')
    costs['measured_GPU_phase_hours']=(costs['new_teacher_forward_seconds']+costs['LoRA_seconds']+costs['rescore_seconds'])/3600
    costs['assigned_worker_wall_hours']=sum(r['wall_seconds'] for r in costs['workers'])/3600
    launcher_files=sorted(p for p in execution.launchers.rglob('*') if p.is_file() and p.suffix in ('.json','.log'))
    launcher_archive=record / 'launcher_logs.tar.gz'
    with tarfile.open(launcher_archive,'w:gz') as handle:
        for p in launcher_files:handle.add(p,arcname=str(p.relative_to(execution.launchers)),recursive=False)
    for p in launcher_files:
        artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)))
    atomic_json(record / 'launcher_logs.json',dict(archive=str(launcher_archive),sha256=file_sha(launcher_archive),
        admissions=[dict(path=str(p),**json.loads(p.read_text())) for p in execution.launchers.glob('*/admission.json')]))
    atomic_json(record / 'costs.json',costs)
    atomic_json(record / 'artifacts.json',dict(files=artifacts,baseline_snapshot_path=str(baseline_path),baseline_sha256=file_sha(baseline_path),
        execution_manifest=str(execution.manifest_path),execution_manifest_sha256=file_sha(execution.manifest_path),
        stage_A_publication=publication))
    validation=dict(status=status,GT_open_attempts=0,official_vendor_teacher_and_historical_results_unchanged=True,
        active_scientific_workers_remaining=running,tests_passed=37,source_Stage_A_unchanged=True,
        full_matrix_started=False,new_full_HOTA_evaluations=0,semantic_or_RMOT_gain_not_established=True)
    atomic_json(record / 'validation.json',validation)
    text='# Source-Anchored MAP：两段真实pilot结果\n\n'+status+'。\n\n'
    text+=('37项定向与既有回归、compileall和GT-free静态证据重放通过。原版全部结果保留；'
        '新版本从官方checkpoint及fresh LoRA/DS开始，在完全相同的两个首视频、rho=.3和四轮预算上运行。'
        '不启动完整矩阵、不读目标GT、不做新的HOTA评测；数值稳定不是语义恢复证据。\n\n')
    text+=table(list(rows[0]),rows)
    text+='\n## 1. 每次校准调用\n\n'+table(['variant','pair','round','position','status','slope','temperature','bias',
        'identity_status','supported_positive','supported_negative','presence_status','confirmed_positive','confirmed_negative'],calibration)
    text+=('\n两类教师支持不足时明确SKIP，不复制未问伪标签作为fast监督；presence矛盾记UNKNOWN。'
        '原源头、冻结教师/三视图、DS、预算、关键帧、LoRA和slow伪标签逻辑不变。固定单位高斯源锚以及正b/T参数化见PREREG.md，未按GT搜参。\n\n'
        '默认单类/分离数据下的极端参数与[Heinze & Schemper](https://onlinelibrary.wiley.com/doi/10.1002/sim.1047)'
        '讨论的非有限MLE问题一致，是机制解释而非唯一因果证明。固定源先验借鉴'
        '[Gelman et al.](https://doi.org/10.1214/08-AOAS191)的正则化原则，不复现其Cauchy默认先验或声称RMOT保证。\n')
    text+='\n## 2. 成本与覆盖\n\n```json\n'+json.dumps(costs,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+=('每个目标只有1视频，不是25/45视频全量。教师实际新增forward与缓存命中/独立等价forward分开，'
        '前端、源训练和v5 Stage A既有成本不假定为零。原默认矩阵330.345 GPU小时是旧方法小pilot保守外推，不是本轮已花费，也不是新MAP版已验证成本。\n')
    text+='\n## 3. 尚未解决的问题\n\n'
    text+=('- 没有目标GT质量评估，不能报告伪标签准确率、语义恢复或HOTA提升；输出行数不是正确目标数。\n'
        '- SlowAdapter原规则仍可使用未问先验伪标签、非互斥场景也可只有负例；这里只隔离修正fast，未声称消除自确认偏差。\n'
        '- 两次候选No推断整句不存在、视图相关性和DS可辨识性属于原机制风险，未用本轮GT调解。\n'
        '- 保留完整视频统计/轨迹均值，是video-transductive pilot，不冒充严格逐帧因果TTA。\n'
        '- 完成此小pilot后停止，不启动完整方法矩阵或额外源训练；后续正式验证需用户新指令。\n')
    text+='\n原始逐问题输出/轮记录、预测mask、可恢复stream checkpoint与SHA见artifacts.json及pilot_map_first_video_logs.tar.gz；大文件留data1，通过原data2链接访问。\n'
    (record / 'REPORT.md').write_text(text)
    atomic_json(record / 'report_receipt.json',dict(status=status,report_sha256=file_sha(record / 'REPORT.md'),
        comparison_sha256=file_sha(record / 'pilot_comparison.csv'),validation_sha256=file_sha(record / 'validation.json')))
    print(json.dumps(dict(status=status,rows=new_rows,full_matrix_started=False)),flush=True)


if __name__=='__main__':main()
