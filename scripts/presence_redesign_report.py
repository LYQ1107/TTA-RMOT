"""Summarize the fixed v3 head trial without replacing historical reports."""
import csv
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.presence_redesign import OUT, RECORD


def table(fields, rows):
    def val(x):
        if isinstance(x, float): return f'{x:.6f}'
        return str(x)
    return '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+''.join(
        '| '+' | '.join(val(r.get(k, '—')) for k in fields)+' |\n' for r in rows)


def main():
    gate = json.loads((RECORD / 'gate.json').read_text())
    selection = json.loads((RECORD / 'selection.json').read_text())
    path = RECORD / ('stage_A.csv' if gate['completed_full_A'] else 'same_domain.csv')
    with path.open() as handle: rows = list(csv.DictReader(handle))
    costs = []
    for target, name in IN_DOMAIN.items():
        folder = OUT / 'heads' / name
        receipt = json.loads((folder / 'training_receipt.json').read_text())
        curves = [json.loads(s) for s in (folder / 'curve.jsonl').read_text().splitlines() if s]
        selected = min(curves, key=lambda r: (r['validation']['loss'], r['epoch']))
        initial = json.loads((folder / 'epoch0_native_prior.json').read_text())
        costs.append(dict(source=target, model=name, epochs=receipt['epochs'], selected_epoch=selected['epoch'],
            native_prior_loss=initial['metrics']['loss'], source_loss=receipt['best_source_loss'],
            changed_source_units=selected['validation']['native_MAP_changed_units'],
            validation_units=receipt['validation_units'], parameters=receipt['parameters'],
            updates=receipt['optimizer_updates'], GPU_hours=receipt['wall_seconds']/3600,
            peak_allocated_GiB=receipt['peak_allocated_bytes']/(1<<30), physical_gpu=receipt['physical_gpu'],
            target_GT_open_attempts=receipt['target_GT_open_attempts'],
            first_gradient_norm=selected['train']['first_real_batch_gradient_norm']))
    own = [r for r in rows if r['own_domain'] == 'True' and r['method'] == 'PRES_TOPK']
    with (PUBLIC / 'source_lr_trial_v2.csv').open() as handle:
        historical = list(csv.DictReader(handle))
    comparison = []
    for row in own:
        old = {r['source_variant']: r for r in historical if r['target'] == row['target'] and r['method'] == 'PRES_TOPK'}
        comparison.append(dict(source=row['target'], native_HOTA=row['native_HOTA'],
            original_default_HOTA=old['default']['HOTA'], original_source_lr1e5_HOTA=old['source_lr1e5']['HOTA'],
            residual_v3_HOTA=row['HOTA'], v3_delta_vs_native=row['delta_HOTA']))
    metric_fields = ['target','model','method','native_HOTA','HOTA','delta_HOTA','DetA','AssA','IDF1','IDSW',
                     'prediction_rows','query_empty_prediction_rows','Present_HOTA','count_MAE_head',
                     'decoded_presence_recall','changed_count_units_vs_native','added_prediction_rows_vs_native',
                     'removed_prediction_rows_vs_native','GT_open_attempts']
    status = '同域门槛失败，B/C 未启动' if gate['failed_targets'] else (
        '同域门槛通过，九格 A 完成；须发布并显式接入新头，再做真实 pilot/成本关卡' if gate['completed_full_A']
        else '同域门槛通过，继续完整九格 A；尚未授权绕过成本关卡')
    text = '# PresenceHead v3：授权结构/损失改造的真实结果\n\n'+status+'。\n\n'
    text += '## 1. 验收\n\n原门槛：三个自身源 PRES_TOPK HOTA 均不得低于完整 Native 超过 1。只按自身源验证新目标函数选择 checkpoint，不按 HOTA 回选旧头或 epoch。\n\n'
    text += table(['target','model','native_HOTA','HOTA','delta_HOTA','passed'], gate['same_domain'])
    text += '\n与完整保留的旧设计对照（仅比较，不按 HOTA 重新选择）：\n\n'+table(
        ['source','native_HOTA','original_default_HOTA','original_source_lr1e5_HOTA','residual_v3_HOTA','v3_delta_vs_native'], comparison)
    text += '\n## 2. 结构、监督与选择\n\n'
    text += ('本轮只执行 [PREREG.md](PREREG.md) 固定的一种结构：小型共享 PCD/几何集合网络为冻结排序的前缀能量学习残差，'
        '联合数量 softmax 覆盖 0..n；输出用联合 MAP，不改身份排序。原生 margin 提供零初始化；CE(count)+0.1 KL(native||head)，'
        '固定 lr1e-4、seed0、batch256、max20epoch/patience3。未训练 FlexHook、未加入教师。独立句子向量/ID 捷径移除，但 PCD 本身仍是查询条件特征。\n\n'
        '三个全量源验证集的 epoch0 原生数量等价检查均必须 0 差异；这只是初始化对照，不是增益。短前缀能量用 FP64 累加以避免近零正增量被大前缀舍入掉，'
        '网络与缓存输入仍 FP32，旧 backbone/官方分数不改。Sports 源监督使用已核对全部表达身份的修正标签 v2。'
        '按原任务允许，自身源官方验证/测试用于早停，因此同域验收不是独立盲测；跨域 GT 不选训练 checkpoint。\n\n')
    text += table(['source','model','epochs','selected_epoch','native_prior_loss','source_loss','changed_source_units','validation_units'], costs)
    text += '\n存在性概率是联合分布的 P(k>0)，阈值0.5与联合MAP的 k>0 可能不一致；两者混淆会夸大召回。本表 decoded_* 指实际输出，head_* 为概率分支诊断。无固定10上限，但不保证召回或排序被修复。\n\n'
    text += '## 3. 完整已计算结果\n\n'+table(metric_fields, rows)
    text += '\n### 3.1 数量决策与概率分支的区别\n\n'
    behavior = []
    for r in own:
        decoded = Path(r['path']).parent
        k = np.load(decoded / 'raw_k.npy')
        p = np.load(decoded / 'presence_logits.npy')
        behavior.append(dict(source=r['target'], MAP_presence_recall=float(r['decoded_presence_recall']),
            native_presence_recall=float(r['native_presence_recall']), MAP_presence_FPR=float(r['decoded_presence_false_positive_rate']),
            native_presence_FPR=float(r['native_presence_false_positive_rate']), MAP_count_MAE=float(r['decoded_count_MAE']),
            native_count_MAE=float(r['native_count_MAE']), max_MAP_k=int(k.max()),
            MAP_vs_probability_disagreements=int(((k>0) != (p>=0)).sum()),
            above_ten_MAP_units=int((k>10).sum()), units=len(k)))
    text += table(['source','MAP_presence_recall','native_presence_recall','MAP_presence_FPR','native_presence_FPR',
        'MAP_count_MAE','native_count_MAE','max_MAP_k','above_ten_MAP_units','MAP_vs_probability_disagreements'], behavior)
    text += ('\n身份排序完全不变，所以所有 HOTA 变化来自数量选择及其造成的关联/匹配变化，不能称为跨模态排序恢复。'
        '若 MAE 下降但 HOTA 不保留，说明该平均误差不足以代表关键真目标的召回与关联；概率阈值准确率同样不能替代实际 MAP 输出。'
        '大于10的输出说明容量上限确已去除，不意味着集合外漏检候选被补回。源GT候选→验证tracker候选的支持偏移仍在，'
        '这里只能作为与结果一致的机制解释，不据此断言唯一失败原因；不再按这些验证结果改本轮阈值或损失。\n')
    if not gate['completed_full_A']:
        text += '\n本轮只完成三套同域和各自平滑对照；六个跨域新头结果未计算，不把旧头成绩当新头成绩，也不把 NOT_RUN 写为零。\n'
    text += '\n## 4. 成本与 GT 审计\n\n'+table(['source','model','GPU_hours','peak_allocated_GiB','physical_gpu','updates','parameters','first_gradient_norm','target_GT_open_attempts'], costs)
    text += (f'\n新增源头计时 GPU 进程小时合计 {sum(r["GPU_hours"] for r in costs):.6f}；共享卡 elapsed，不是利用率积分。'
        '独立 CPU 解码/官方 TrackEval 时间见 costs.json；backbone 特征导出未重做。新增教师调用=0、目标 LoRA 更新=0（本头阶段）。'
        '训练允许自身源数值标签；预测生成用 runtime_guard 禁止 raw GT/labels.npy/GT_counts.npy，预测先 SHA 封存，再由独立进程读取官方 GT。'
        'Sports 主评测 v4 全466序列，另存相同 v4 内容限制原418ID的敏感性。\n\n')
    decode_costs = []
    for parent in sorted((OUT / 'stage_A').glob('*/*')):
        decoded = json.loads((parent / 'decode_receipt.json').read_text())
        evaluations = []
        for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):
            c = json.loads((parent / method / 'official_completion.json').read_text())
            evaluations.append(dict(method=method, wall_seconds=c['wall_seconds']))
        decode_costs.append(dict(target=parent.parent.name,model=parent.name,
            decode_CPU_wall_seconds=decoded['wall_seconds'],GT_open_attempts=decoded['GT_open_attempts'],evaluations=evaluations))
    atomic_json(RECORD / 'costs.json', dict(source_training=costs, CPU_decode_evaluate=decode_costs,
        new_teacher_calls=0, target_LoRA_updates=0, frozen_feature_exports_reused=True))
    text += ('## 5. 保存、局限与后续条件\n\n新结果物理目录 /data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v3，'
        '通过原 results/runs/presence_active_20261102 链接访问。checkpoint/预测路径 SHA 见 artifacts.json、selection.json；大文件不入 Git。'
        '原默认 A、lr1e-5、Sports 标签修复结果及主 REPORT.md 均不覆写。该头保留冻结模型先验是明确设计，不能把接近原生称为新语义能力。'
        '即使通过同域，也须单独检查跨域是否有效；不能以 source loss、有限梯度或测试通过宣称全方法成功。\n\n')
    if gate['failed_targets']:
        text += '本次固定改造仍有同域失败：'+', '.join(gate['failed_targets'])+'。遵守预注册，未启动 B/C、未追加第二个新结构或目标阈值搜索；完整方法仍未完成。\n'
    else:
        text += 'A 通过不等于 B/C 完成。后续仍需 v3 manifest/动态数量接口集成、真实 C1/C3 pilot、40 GPU小时估算；不得直接沿用旧失败 gate 或默认头路径。\n'
    text += '\n完整新阶段日志、源选择曲线、官方详细表在 logs.tar.gz；清单及 SHA 见 logs.json。历史 experiment_logs.tar.gz 不改。模型/缓存与预测均不打包。\n'
    (RECORD / 'REPORT.md').write_text(text)
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and (p.suffix in ('.log','.json','.jsonl') or
        p.name in ('pedestrian_detailed.csv','pedestrian_summary.txt')))
    files = [dict(path=str(p),archive_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in paths]
    archive_path = RECORD / 'logs.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for p, r in zip(paths, files): archive.add(p, arcname=r['archive_path'], recursive=False)
    for r in files:
        if file_sha(r['path']) != r['sha256']: raise RuntimeError('Wait for log owner to finish before packaging')
    atomic_json(RECORD / 'logs.json',dict(path=str(archive_path),bytes=archive_path.stat().st_size,
        sha256=file_sha(archive_path),files=files,scope='new v3 only; historical archive not rewritten'))
    atomic_json(RECORD / 'report_receipt.json', dict(status=status, gate_sha256=file_sha(RECORD / 'gate.json'),
        report_sha256=file_sha(RECORD / 'REPORT.md'), selection_sha256=file_sha(RECORD / 'selection.json'),
        completed_rows=len(rows), target_GT_open_attempts=sum(int(r['GT_open_attempts']) for r in rows),
        new_teacher_calls=0,target_LoRA_updates=0))
    print(json.dumps(dict(status=status, report=str(RECORD / 'REPORT.md'), completed_rows=len(rows))),flush=True)


if __name__ == '__main__': main()
