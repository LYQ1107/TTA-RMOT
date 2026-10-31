#!/usr/bin/env python3
"""Publishable Train teacher-fit table; no Val inputs, metrics or GT."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.train2val import ASSETS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.run_train2val_training import ARMS,RUN,PUBLIC


def build():
    completion=json.loads((RUN/'full_completion.json').read_text());rows={}
    labels=[json.loads(s) for s in (ASSETS/'pseudo/NAIVE/soft_labels.jsonl').read_text().splitlines()]
    pos=sum(r['valid'] and r['teacher_prob']>=.95 for r in labels)
    for arm in ARMS:
        folder=RUN/'students'/arm;status=completion['arms'][arm]
        if status['status']!='COMPLETE':
            rows[arm]=dict(status=status['status']);continue
        receipt=json.loads((folder/'training_receipt.json').read_text());config=json.loads((folder/'config.json').read_text())
        if receipt['steps']!=8973 or receipt['target_GT_open_attempts'] or receipt['forbidden_input_open_attempts']:
            raise ValueError('invalid Train-only training receipt')
        checkpoint=folder/'checkpoint_final_step.pt'
        if file_sha(checkpoint)!=receipt['checkpoints'][checkpoint.name]['sha256']:raise ValueError('changed final checkpoint')
        validation=[json.loads(s) for s in (folder/'source_validation.jsonl').read_text().splitlines()]
        fit=json.loads((folder/'teacher_fit_summary.json').read_text())
        rows[arm]=dict(status='COMPLETE',seed=ARMS[arm][1],steps=receipt['steps'],
            trainable_parameters=receipt['trainable_parameters'],lr=config['optimizer']['lr'],
            pos_weight=json.loads((folder/'pos_weight.json').read_text())['pos_weight'],
            **{k:fit[k] for k in ('teacher_POS_selected_rate','teacher_NEG_selected_rate','pair_agreement',
                'trained_teacher_BCE','source_teacher_BCE','teacher_POS','teacher_NEG')},
            source_val_CE_step0=validation[0]['source_val_CE'],source_val_CE_final=validation[-1]['source_val_CE'],
            target_windows=receipt['target_registered_windows'],mean_window_exposures=receipt['mean_target_window_exposures'],
            target_GT_open_attempts=receipt['target_GT_open_attempts'],forbidden_input_open_attempts=receipt['forbidden_input_open_attempts'],
            GPU_worker_hours=receipt['wall_seconds']/3600,peak_allocated_GiB=receipt['peak_allocated_bytes']/(1<<30),
            checkpoint_path=str(checkpoint.resolve()),checkpoint_sha256=file_sha(checkpoint))
        dest=PUBLIC/'training'/arm;dest.mkdir(parents=True,exist_ok=True)
        for name in ('training_receipt.json','config.json','provenance.json','teacher_fit_summary.json',
                     'teacher_fit_curve.jsonl','source_validation.jsonl','pos_weight.json','final_step_selection.json',
                     'initial_frozen_equivalence.json','source_only_lambda_validation.json','fit_subset.json','input_access_audit.json'):
            shutil.copy2(folder/name,dest/name)
    table=['# 表 A：Train 标签拟合，尚未进行学生 Val 推理','',
        f'Train教师判断 {len(labels):,}；POS(p≥0.95) {pos:,} ({pos/len(labels):.6%})。',
        '历史Naive为656/8500；用户给出的657/8500来自PSR自身缓存，分母与manifest不可混用。','',
        '| Arm | POS选中率 | NEG选中率 | 帧内pair agreement | KD BCE | source-dev CE(final) | 窗口平均曝光 | GPUh(进程body) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in rows.items():
        if r['status']!='COMPLETE':table.append('| '+arm+' | '+r['status']+' | N/A | N/A | N/A | N/A | N/A | N/A |');continue
        fields=('teacher_POS_selected_rate','teacher_NEG_selected_rate','pair_agreement','trained_teacher_BCE',
            'source_val_CE_final','mean_window_exposures','GPU_worker_hours')
        table.append('| '+arm+' | '+' | '.join('N/A' if r[k] is None else f'{r[k]:.6f}' for k in fields)+' |')
    table+=['','固定final-step，source-best仅日志、不用于Val模型选择。各seed目标标签完全相同。',
        '所有值仅Train教师拟合/源dev诊断，不是目标GT准确率或Val HOTA；不因表A好坏更换方法、阈值或checkpoint。',
        '8973×4/7800≈4.60次窗口曝光（实际窗口数与平均值见JSON）；原Val2125窗口约16.89次，训练预算不随窗口数增加。']
    path=PUBLIC/'TABLE_A.md';path.write_text('\n'.join(table)+'\n')
    result=dict(status='ALL_REGISTERED_TRAIN_ARMS_REPORTED_VAL_INFERENCE_NOT_STARTED',methods=rows,
        teacher_samples=len(labels),teacher_POS=pos,teacher_POS_fraction=pos/len(labels),TABLE_A_sha256=file_sha(path),
        target_GT_open_attempts=0,Val_data_read_by_training=False)
    atomic_json(PUBLIC/'table_a.json',result);atomic_json(RUN/'table_a.json',result)
    report=(PUBLIC/'REPORT.md').read_text();a=report.index('## 3. 表 A：');b=report.index('## 4. 表 B / C、')
    report=report[:a]+'## 3. 表 A：Train 教师拟合\n\n'+path.read_text()+'\n'+report[b:]
    report='\n'.join('状态：P0–P3 完成，P4/P5 等待表A推送门；尚无本轮 Val 结果。' if line.startswith('状态：') else line for line in report.splitlines())+'\n'
    (PUBLIC/'REPORT.md').write_text(report)
    print(json.dumps(result,indent=2))


if __name__=='__main__':build()
