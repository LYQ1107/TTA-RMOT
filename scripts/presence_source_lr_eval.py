"""Evaluate the single preregistered source-LR trial after loss-only selection.

This CPU coordinator never selects a checkpoint by HOTA. Default Stage-A
artifacts/gate are preserved. Each trial decode is GT-free and sealed before
its separate official evaluator opens GT. No teacher or B/C worker is started.
"""
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def collect(selection, selection_sha):
    from scripts.diag_attrib_v2_common import write_csv
    from scripts.pf_diag_common import official_metrics
    from scripts.presence_finish_a import references
    rows, selected_rows, artifacts, costs = [], [], [], []
    for target, model in IN_DOMAIN.items():
        choice = selection['models'][model]
        native = official_metrics(dict(references(target, model))['NATIVE'] / 'results')['HOTA']
        for variant, head_root in [('default', RUN / 'heads'), ('source_lr1e5', RUN / 'heads_source_lr1e5')]:
            receipt_path = head_root / model / 'training_receipt.json'
            receipt = json.loads(receipt_path.read_text())
            costs.append(dict(source_domain=target, model=model, variant=variant,
                source_learning_rate=receipt.get('learning_rate', 3e-4),
                epochs=receipt['epochs'], optimizer_updates=receipt['optimizer_updates'],
                wall_seconds=receipt['wall_seconds'], GPU_hours=receipt['wall_seconds']/3600,
                peak_allocated_bytes=receipt['peak_allocated_bytes'], physical_gpu=receipt['physical_gpu'],
                parameters=receipt['parameters'], path=str(receipt_path), sha256=file_sha(receipt_path)))
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            for variant, base in [('default', RUN / 'stage_A'),
                                  ('source_lr1e5', RUN / 'stage_A_source_lr1e5')]:
                folder = base / target / model / method
                summary = json.loads((folder / 'offline_summary.json').read_text())
                seal_path = folder / 'predictions_seal.json'
                seal = json.loads(seal_path.read_text())
                if seal['GT_open_attempts']:
                    raise ValueError('Trial/default prediction GT isolation failed')
                chosen = variant == choice['selected']
                if chosen and seal['head_checkpoint_sha256'] != choice['checkpoint_sha256']:
                    raise ValueError('Evaluated head differs from pre-HOTA source-loss selection')
                loss = next(c['loss'] for c in choice['candidates'] if c['variant'] == variant)
                row = dict(summary, source_variant=variant, selected_by_source_loss=chosen,
                           source_validation_loss=loss, native_HOTA=native,
                           delta_HOTA=summary['HOTA']-native)
                rows.append(row)
                if chosen and method == 'PRES_TOPK': selected_rows.append(row)
                artifacts.append(dict(target=target, model=model, variant=variant, method=method,
                    path=str(folder), checkpoint=seal['head_checkpoint'],
                    checkpoint_sha256=seal['head_checkpoint_sha256'],
                    prediction_seal_sha256=file_sha(seal_path),
                    official_completion_sha256=file_sha(folder / 'official_completion.json')))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(PUBLIC / 'source_lr_trial.csv', [{k: row.get(k) for k in fields} for row in rows])
    write_csv(PUBLIC / 'source_lr_trial_costs.csv', costs)
    failures = [row['target'] for row in selected_rows if row['delta_HOTA'] < -1]
    gate = dict(status='FAIL_SELECTED_SOURCE_HEADS_STOP_BEFORE_BC' if failures else 'PASS_SELECTED_SOURCE_HEADS',
        failed_targets=failures, criterion='Each selected own-source PRES_TOPK HOTA >= native HOTA -1',
        selected_by_source_validation_loss_only=True, selection_sha256=selection_sha,
        selected_heads=selection['models'], published_commit=None,
        default_gate_preserved=str(PUBLIC / 'stage_A_gate.json'),
        additional_teacher_calls=0, additional_target_LoRA_updates=0)
    atomic_json(PUBLIC / 'source_lr_trial_gate.json', gate)
    atomic_json(PUBLIC / 'source_lr_trial_artifacts.json', dict(selection_sha256=selection_sha, files=artifacts))
    table = ['|源/同域测试|变体|源验证loss|Native|PRES_TOPK|Δ|Smooth|源loss选择|',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        if row['method'] != 'PRES_TOPK': continue
        smooth = next(r['HOTA'] for r in rows if r['target'] == row['target'] and
                      r['source_variant'] == row['source_variant'] and r['method'] == 'PRES_TOPK_SMOOTH')
        table.append(f"|{row['target']}|{row['source_variant']}|{row['source_validation_loss']:.6f}|"
            f"{row['native_HOTA']:.3f}|{row['HOTA']:.3f}|{row['delta_HOTA']:+.3f}|{smooth:.3f}|"
            f"{'是' if row['selected_by_source_loss'] else '否'}|")
    text = '# 单次源域LR缓解对照：真实结果\n\n'
    if (PUBLIC / 'source_label_full_audit.json').exists():
        audit = json.loads((PUBLIC / 'source_label_full_audit.json').read_text())
        if audit['cached_mismatching_observations']:
            text += ('**历史v1口径：Sports头受源标签按相同文本合并的bug影响，不作为主结果；'
                'KITTI/Dance不受影响。修复定义及独立v2结果见SOURCE_LABEL_FIX.md。旧数值完整保留。**\n\n')
    text += ('默认3e-4与固定替代1e-5，其他结构/损失/增强/seed/早停规则不变。'
        '选择文件在任何替代头HOTA计算前封存，只按各自源验证loss选，不按HOTA选择。'
        '源官方测试集用于任务明确允许的早停；不是盲测。全部默认结果保留。\n\n')
    text += '\n'.join(table)+'\n\n'
    text += f"选择文件SHA256：`{selection_sha}`。最终门槛：`{gate['status']}`。失败：{failures}。\n\n"
    text += ('不能把较低源loss或工程检查通过当成HOTA成功。若失败，不增加学习率搜索，'
        '不启动B/C；保留全部默认A结果与一次缓解对照，报告设计/泛化问题。'
        '若通过，仍须完成选中头的九套A解码和发布，再进入原任务的真实pilot与40GPU小时成本关卡。\n\n')
    text += ('本对照只训练1,521,676参数的PresenceHead；没有新教师调用、目标适应或FlexHook源重训。'
        '完整质量/存在性/数量/见过与未见句子分组见`source_lr_trial.csv`；大文件路径和SHA见'
        '`source_lr_trial_artifacts.json`。源训练成本从独立训练receipt逐项报告，不与默认训练混算。\n')
    (PUBLIC / 'SOURCE_LR_REPORT.md').write_text(text)
    return gate


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    selection_path = PUBLIC / 'source_lr_trial_selection.json'
    # One blocking coordinator owns this wait, not agent-level process polling.
    while not selection_path.exists(): time.sleep(20)
    selection_sha = file_sha(selection_path)
    selection = json.loads(selection_path.read_text())
    if not selection['selection_uses_source_validation_loss_only'] or not selection['trial_HOTA_not_computed_at_selection']:
        raise ValueError('Source-only pre-HOTA selection was not sealed')
    for choice in selection['models'].values():
        if file_sha(choice['checkpoint']) != choice['checkpoint_sha256']:
            raise ValueError('Selected source checkpoint changed after selection')
    if args.collect_only:
        print(json.dumps(collect(selection, selection_sha)), flush=True)
        return
    log_root = RUN / 'source_lr_eval' / str(time.time_ns()); log_root.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
    def cell(job):
        target, model = job
        destination = RUN / 'stage_A_source_lr1e5' / target / model
        checkpoint = RUN / 'heads_source_lr1e5' / model / 'checkpoint_best_source.pt'
        commands = [['scripts/presence_eval_offline.py', '--target', target, '--model', model,
            '--phase', 'decode', '--head-checkpoint', str(checkpoint), '--output-dir', str(destination)]]
        # Refresh independent default summaries to include the same diagnostic fields.
        for base in (RUN / 'stage_A', RUN / 'stage_A_source_lr1e5'):
            for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
                command = ['scripts/active_eval.py', '--target', target, '--folder', str(base / target / model / method)]
                if target == 'SportsMOT': command.append('--sports418')
                commands.append(command)
        with (log_root / (target+'.log')).open('a') as handle:
            for command in commands:
                code = subprocess.call([sys.executable, *command], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if code: raise RuntimeError('Source-LR CPU evaluation failed: '+str(handle.name))
        if file_sha(selection_path) != selection_sha: raise ValueError('Selection changed during evaluation')
        print(json.dumps(dict(event='SOURCE_LR_TRIAL_OWN_DOMAIN_EVALUATED', target=target)), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(cell, IN_DOMAIN.items()))
    gate = collect(selection, selection_sha)
    atomic_json(log_root / 'completion.json', gate)
    print(json.dumps(gate), flush=True)


if __name__ == '__main__': main()
