#!/usr/bin/env python3
"""Package a mandatory protocol-stop, NOT an incomplete result as success."""
import csv
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_audit import RUN, PUBLIC, OLD, OFFICIAL, MODELS, deny_gt, require_prereg
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def write_csv(path, rows):
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = time.perf_counter()
    require_prereg()
    attempts = deny_gt()
    audit = json.loads((RUN / 'input_audit.json').read_text())
    official = json.loads((RUN / 'official_reproduction.json').read_text())
    if official['status'] != 'STOP_OFFICIAL_COMPLETEBANK_CANNOT_REPRODUCE_HISTORICAL_NATIVE':
        raise ValueError('this packager only handles the actual mandatory-stop outcome')
    # Revalidate all files touched read-only; this is output integrity, not a
    # new evaluation or another model inference.
    files = json.loads((RUN / 'protected_inputs.json').read_text())
    for record in files:
        for field in ('prediction', 'margins'):
            if file_sha(record[field + '_path']) != record[field + '_sha256']:
                raise ValueError('original sealed file changed')
    old_manifest = json.loads((ROOT / 'results/diag_fit/server_artifacts.json').read_text())
    weights = [r for r in old_manifest['checkpoints'] if r['phase'] == 'students' and
               r['path'].endswith('/checkpoint_final_step.pt')]
    if len(weights) != 3:
        raise ValueError('three old final checkpoints required')
    for record in weights:
        if file_sha(record['path']) != record['sha256']:
            raise ValueError('old final checkpoint changed')
    if file_sha(OFFICIAL / 'result.json') != official['original_score_sha256']:
        raise ValueError('old official logits changed')
    seal = json.loads((OLD / 'predictions_seal.json').read_text())
    if file_sha(OLD / 'predictions_seal.json') != file_sha(ROOT / 'results/diag_fit/prediction_manifest.json'):
        raise ValueError('old aggregate seal differs from published immutable manifest')
    for record in seal['methods'].values():
        if file_sha(record['run_metadata']) != record['run_metadata_sha256']:
            raise ValueError('old run metadata changed')
    native_folders = dict(OFFICIAL_DANCE_HISTORICAL=OFFICIAL, **MODELS,
                         D0_NAIVE=OLD / 'evaluation/merged/NAIVE_PAIRED')
    alignment = {}
    for name, folder in native_folders.items():
        summary = folder / 'results/pedestrian_summary.txt'
        headers, values = summary.read_text().splitlines()[:2]
        metrics = dict(zip(headers.split(), map(float, values.split())))
        alignment[name] = dict(existing_official_metrics=metrics, summary_path=str(summary),
            summary_sha256=file_sha(summary), result_reused_not_reevaluated=True,
            native_candidate_coverage='INCOMPLETE_210393_MOT' if name.endswith('HISTORICAL') else 'COMPLETE_211085_MOT',
            prediction_rows=int(metrics['Dets']), historical_official_count_ratio=metrics['Dets']/254118,
            completebank_OFFICIAL_not_evaluated=True)
    atomic_json(PUBLIC / 'alignment.json', dict(status='PARTIAL_EXISTING_RESULTS_ONLY_PROTOCOL_STOP',
        caution='historical official has different score coverage; NOT a new matched comparison or final Table C',
        methods=alignment, GT_frame_count_metrics=None, target_GT_open_attempts=attempts))
    names = ['FROZEN', 'D1', 'D2', 'D3', 'OFFICIAL_DANCE']
    write_csv(PUBLIC / 'attrib_matrix.csv', [dict(ranking=rank, count=count,
        status='NOT_RUN_MANDATORY_OFFICIAL_REPRODUCTION_STOP', HOTA='', DetA='', AssA='', IDF1='', IDSW='', rows='')
        for rank in names for count in names+['ORACLE_GT']])
    baselines = ['B1_SRC_RATE_GLOBAL', 'B1_SRC_RATE_QUERY', 'B2_Z_OFFLINE', 'B2_Z_CAUSAL',
                 'B3_MEAN_GLOBAL', 'B3_MEAN_QUERY', 'B4_TEACHER_RATE', 'B4b_TEACHER_GATE']
    write_csv(PUBLIC / 'baselines.csv', [dict(baseline=name,
        status='NOT_RUN_MANDATORY_OFFICIAL_REPRODUCTION_STOP', HOTA='', rows='', target_GT_open_attempts='N/A_NOT_STARTED')
        for name in baselines])
    atomic_json(PUBLIC / 'bootstrap.json', dict(status='NOT_RUN_MANDATORY_PROTOCOL_STOP',
        planned_video_clusters=25, planned_samples=2000, planned_seed=0, comparisons={},
        rule_A='NOT_EVALUATED', rule_B='NOT_EVALUATED', rule_C='NOT_FINALIZED_UNMATCHED_OFFICIAL_PROTOCOL'))
    impact = []
    for row in audit['official']['per_video']:
        subset = [e for e in official['added_rows_per_episode'] if e['video'] == row['video']]
        impact.append(dict(**row, added_positive_rows=sum(e['added_rows'] for e in subset), changed_episodes=len(subset)))
    write_csv(PUBLIC / 'per_video_protocol_impact.csv', impact)
    atomic_json(PUBLIC / 'prediction_manifest.json', dict(status='STOP_BEFORE_NEW_PREDICTION_GENERATION',
        new_predictions=[], new_full_bank_predictions_sealed=False,
        new_raw_logit_supplements=official['supplements'],
        protected_sealed_inputs_manifest=str(RUN / 'protected_inputs.json'),
        protected_sealed_inputs_manifest_sha256=file_sha(RUN / 'protected_inputs.json'),
        old_final_checkpoints=weights, old_checkpoints_and_predictions_unchanged=True,
        supplements_are_NOT_accepted_OFFICIAL_margin_input_to_matrix=True))
    atomic_json(PUBLIC / 'gt_audit.json', dict(status='PASS_ZERO_TARGET_GT_OPENS',
        input_audit=dict(started=True, target_GT_open_attempts=audit['target_GT_open_attempts']),
        official_recovery=dict(started=True, target_GT_open_attempts=official['target_GT_open_attempts']),
        packaging=dict(started=True, target_GT_open_attempts=attempts),
        matrix=dict(started=False), calibration_baselines=dict(started=False),
        ORACLE_GT=dict(started=False), B0=dict(started=False), TrackEval=dict(started=False),
        GT_open_attempts_for_not_started_processes='N/A, not falsely reported as executed audits',
        optimizer_steps=0, new_teacher_calls=0))
    costs = dict(CPU_only_input_audit_wall_seconds=audit['CPU_wall_seconds'],
        official_recovery_total_wall_seconds=official['total_wall_seconds'],
        GPU_worker_seconds=official['GPU_worker_seconds'], GPU_worker_hours=official['GPU_worker_hours'],
        peak_allocated_GPU_bytes=official['peak_allocated_bytes'], physical_GPUs=[official['physical_GPU']],
        GPU_hours_definition='process wall hours, not exclusive GPU utilization hours',
        CPU_core_hours=None, CPU_core_hours_note='not instrumented; do not substitute wall time',
        optimizer_steps=0, full_GPU_inference_reruns=0, official_missing_logit_calls=11764,
        new_TrackEval_jobs=0, bootstrap_resamples_executed=0)
    atomic_json(PUBLIC / 'costs.json', costs)
    destination = PUBLIC / 'logs'
    destination.mkdir(exist_ok=True)
    for path in (RUN / 'logs').glob('*.txt'):
        shutil.copy2(path, destination / path.name)
    shutil.copy2(RUN / 'official_gpu_admission.json', PUBLIC / 'official_gpu_admission.json')
    shutil.copy2(RUN / 'official_missing_observations.json', PUBLIC / 'official_missing_observations.json')
    shutil.copy2(RUN / 'prereg_publication.json', PUBLIC / 'prereg_publication.json')
    preserved = dict(status='PASS', original_prediction_margin_pairs=len(files),
        original_final_checkpoints_verified=len(weights), old_aggregate_prediction_seal_unchanged=True,
        official_original_score_SHA_unchanged=True, target_GT_open_attempts=attempts,
        packaging_wall_seconds=time.perf_counter()-start)
    atomic_json(PUBLIC / 'post_stop_integrity.json', preserved)
    table = ['| 历史/已封存模型 | HOTA（旧结果） | 原生预测行数 | /历史官方254,118 | 覆盖状态 |',
             '|---|---:|---:|---:|---|']
    for name, item in alignment.items():
        table.append(f"| {name} | {item['existing_official_metrics']['HOTA']:.4f} | {item['prediction_rows']:,} | {item['historical_official_count_ratio']:.4f} | {item['native_candidate_coverage']} |")
    report = '''# D-FIT 事后归因诊断：协议自检失败，按指令停止

状态：`STOP_OFFICIAL_COMPLETEBANK_CANNOT_REPRODUCE_HISTORICAL_NATIVE`。
这不是任务全部完成，不是新的方法结果；矩阵、校准基线与bootstrap均未运行。

## 1. 结论摘要（Q1–Q4）

1. Q1：已有行数中D1约为历史官方1.3588倍，D2/D3分别约3.9207/2.7355倍，但官方旧分数不完整，不能把这张历史描述表当作本轮已对齐的正式表C或判定C验收。
2. Q2：完整的排序×数量归因未开始，不能新增数量/存在性与排序的归因结论；上一轮封存结论和数值不改。
3. Q3：B1–B4未开始，不能判断无学习校准能否达到D1；判定A未执行。
4. Q4：D3在合理数量下的可转化HOTA与视频CI未计算，判定B未执行。
5. 停止原因是同域官方遗漏692条MOT末观测；真实补分数新增792条正预测、影响56条序列，使254,118变为254,910行，与必须逐行复现历史官方输出的硬条件不相容。

## 2. 表C：仅保留已有结果，不冒充新对齐

下列HOTA直接读取旧官方summary；没有新GT评价。OFFICIAL_DANCE_HISTORICAL的分数覆盖仅210,393条MOT，其余模型采用完整211,085条候选。**新补齐官方HOTA未知，绝不沿用32.134。**

''' + '\n'.join(table) + '''

完整原有DetA/AssA/DetRe/DetPr/AssRe/AssPr/IDF1/IDSW/CLEAR FP/FN/Dets/IDs等在alignment.json中；有输出帧/GT分组k未在停止后补做。历史行数比只供解释影响，不是已通过的匹配评测。

## 3. 5×6矩阵

HOTA矩阵、行数矩阵和同列差值矩阵均未启动；attrib_matrix.csv为30个明确NOT_RUN的格子，空值不是0。未把792条新增正例删去、压成负分或只对旧覆盖集合评分以强行通过。

## 4. 校准基线

B0和B1–B4全部未启动。baselines.csv列明8个GT-free版本的NOT_RUN状态，不把未启动进程的GT审计伪报成已经执行。

## 5. 视频bootstrap

未运行。bootstrap.json保留预注册25视频/2000次/seed0计划，实际comparisons为空；不使用旧区间冒充新比较，不执行判定A/B/C。

## 6. B0曲线

未生成：停止后不继续新的阈值评测或GT-informed曲线，亦不绘制虚构图。

## 7. 自检与失败影响

- FROZEN、D1、D2、D3各425表达、3,588,445行分数全部通过旧封存SHA核对；margin>0逐候选ID/坐标复现原生预测99,704 / 345,302 / 996,330 / 695,148行。官方旧输出按数值MOT字段规范化比较，原文件不改。
- 官方旧result.json SHA为de8ff46428818cbd1ee0dd2b6ea1a7692eae2b7e0133c521c1470b67b58da110；其中3,576,681条已覆盖分数能复现全部425条历史预测，但比完整要求少11,764条。
- 原因：vendor/FlexHook/data/mydataloader.py:354将sample_end设为obj_last−sample_frame_num，359行range的右端不包含，跳过每条轨迹末观测；没有修改vendor或历史结果。
- 仅使用官方Dance checkpoint，SHA fbb5df7339a1201f34300c105df81e893a5edbb8ef939b7f6eaf8780348eac6c；FP32，参数冻结，tokenizer person=[5970]。
- 8个真实已覆盖帧对照：缓存解码vs原始forward最大margin误差1.0491e−5；vs历史margin最大误差1.3257e−4；选择翻转0。新增正例有明确大正margin，不能用浮点误差解释。
- 25视频692条漏观测的11,764组双类logits完整补在新目录；其中792组margin>0，影响56条表达，完整模型按同规则必然254,910行。例：dancetrack0004 / dancing-person-dressed-all-in-white / frame138 / track4，margin=7.9885921478，历史没有该候选。
- 因行集合已经不等，完整官方HOTA不再运行；32.134只保留历史，不赋给254,910行版本。新官方margin也未供给矩阵。
- 每视频漏项与新增正例见per_video_protocol_impact.csv；完整候选明细见official_missing_observations.json；25个补分数文件路径和SHA见prediction_manifest.json及official_reproduction.json。
- 旧6个count_controls的数值不改，但本轮未启动相应矩阵等价性运行，不能标记为本轮已完成自检。
- 补分数进程退出码3是刻意触发的协议停止，不是OOM、loss发散或训练失败。3项必要单元检查通过；不以此代替科研验收。

## 8. GT隔离与旧结果完整性

CPU输入核验、单GPU补分数、停止归档均启用目标GT打开审计，实际尝试0。没有打开Dance表达原始注释、gt.txt或labels文件；只用已有白名单runtime文本manifest。未运行ORACLE或TrackEval，无新教师调用、无优化步。
停止后复核全部1,700对原生预测/margin、3个final checkpoint、原aggregate seal对应metadata和历史官方logits，均未改变。详见gt_audit.json、post_stop_integrity.json；大文件未放入Git。

## 9. 成本

''' + f"CPU-only输入审计wall={audit['CPU_wall_seconds']:.3f}s；GPU 4补分数worker wall={official['GPU_worker_seconds']:.3f}s（{official['GPU_worker_hours']:.6f} GPU-process h），含准备总wall={official['total_wall_seconds']:.3f}s；峰值allocated={official['peak_allocated_bytes']/2**30:.3f}GiB。" + '''
入场时GPU4无计算进程、81,154MiB空闲，最多1张卡，没有停止他人任务。GPU-process小时不是独占GPU利用率小时；CPU-core小时未测，不伪造。另有少量归档/测试CPU成本，归档wall在post_stop_integrity.json。

## 10. 失败与修复记录

- 真实协议失败：历史官方分数遗漏候选，而任务同时要求完整库与逐行复现254,118；补分数证明二者不相容。**未通过改定义、删除正例、换权重或改阈值绕过。**
- 一次compileall命令误用不存在的裸python，shell报/bin/bash: python: command not found；随后用既定dfot绝对解释器重新compileall并通过。该调用不执行模型，后续补分数本来已用正确解释器；不是科研改进。原报错单独保留于logs/compile_attempt_1.txt。
- pytest三项通过、compileall通过、代码diff检查通过；生成CSV默认CRLF曾触发空白检查，已仅转LF且不改字段数值；原始日志尾随空格原样保留，不为格式检查改写证据。没有训练、没有完整GPU重推理、没有新的官方评测或参数更新。

## 建议（只建议，不执行）

请用户确认是否允许预注册v2：保留32.134/254,118为历史，接受同一checkpoint补齐后的254,910行官方原生输出，单独封存并重新计算其官方HOTA，再以该匹配官方数量列继续原定CPU矩阵与校准基线。需更新OFFICIAL_DANCE数量参照及复现门槛并先推送v2，不覆盖v1和历史结果；不能把新HOTA预先假定为32.134。
在明确确认前不运行任何剩余矩阵、阈值扫描、bootstrap、新学生或Train→Val实验。
'''
    (PUBLIC / 'REPORT.md').write_text(report)
    atomic_json(RUN / 'stop_package_completion.json', dict(status='STOP_REPORTED_NOT_RESEARCH_COMPLETE',
        protocol_stop=True, all_new_experiment_processes_finished=True, official_exit_code=3,
        remaining_work_requires_user_protocol_revision=True, target_GT_open_attempts=attempts,
        report_path=str(PUBLIC / 'REPORT.md'), report_sha256=file_sha(PUBLIC / 'REPORT.md')))
    print(json.dumps(dict(status='STOP_PACKAGE_COMPLETE', report=str(PUBLIC / 'REPORT.md'),
                         post_stop_integrity=preserved)), flush=True)


if __name__ == '__main__':
    main()
