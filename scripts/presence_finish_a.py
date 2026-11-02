"""Blocking CPU-only completion of all nine Stage-A cells and its source gate.

The original exporters and source-head pipeline own their healthy GPU work.
This command waits internally, reuses their finished cells, and never starts B/C.
"""
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN, target_metadata, sealed_margins
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def references(target, model):
    from scripts.multidomain_diagnostics import reusable, base, DEFAULT_SOURCE
    own = IN_DOMAIN[target] == model
    source = DEFAULT_SOURCE[target] if own else model
    names = [('NATIVE', 'IN_DOMAIN' if own else 'NATIVE'),
             ('R_SxK_ORACLE', 'R_IxK_ORACLE' if own else 'R_SxK_ORACLE')]
    return [(label, reusable(target, name, source) or base(target, source) / name) for label, name in names]


def verify_reference(seal, meta, margins, *, native):
    if native and seal.get('GT_open_attempts', 0): raise ValueError('Historical native GT isolation failed')
    query_index = {(q['video_id'], q['expression_id']): q for q in meta['episodes']}
    for e in seal['episodes']:
        if file_sha(e['prediction_path']) != e['prediction_sha256']:
            raise ValueError('Changed historical prediction; do not reuse')
        if native:
            # Old v2 seals predate a per-method GT counter. Do not invent it:
            # check their native rows against complete-bank margin>0 instead.
            q = query_index[e['video'], e['expression']]
            lines = meta['videos'][e['video']]['lines']
            selected = np.flatnonzero(margins[q['start']:q['stop']] > 0)
            content = ''.join(lines[i]+'\n' for i in selected)
            if hashlib.sha256(content.encode()).hexdigest() != e['prediction_sha256']:
                raise ValueError('Historical native is not complete-bank margin>0')


def collect():
    from scripts.diag_attrib_v2_common import write_csv
    from scripts.pf_diag_common import official_metrics
    from scripts.active_eval import offline_labels
    from scripts.report_presence_active import stage_a_gate
    rows = []; artifacts = []; costs = []
    for target in IN_DOMAIN:
        _, truth, _ = offline_labels(target)
        meta = target_metadata(target)
        for model in ('V2-best', 'Dance-best', 'LaMOT-best'):
            margins, _ = sealed_margins(target, model)
            for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
                folder = RUN / 'stage_A' / target / model / method
                row = json.loads((folder / 'offline_summary.json').read_text())
                rows.append(dict(row, own_domain=model == IN_DOMAIN[target], oracle=False))
                artifacts.append(dict(kind=method, target=target, model=model, path=row['path'],
                    prediction_seal_sha256=row['prediction_seal_sha256']))
            for method, folder in references(target, model):
                seal_path = folder / 'predictions_seal.json'
                seal = json.loads(seal_path.read_text())
                verify_reference(seal, meta, margins, native=method == 'NATIVE')
                empty = sum(e['selected_rows'] for e in seal['episodes']
                            if not truth[e['video'], e['expression']]['counts'].sum())
                metrics = official_metrics(folder / 'results')
                rows.append(dict(target=target, model=model, method=method, own_domain=model == IN_DOMAIN[target],
                    oracle=method.endswith('ORACLE'), **{k:metrics[k] for k in ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW')},
                    prediction_rows=seal['prediction_rows'], query_empty_prediction_rows=empty,
                    teacher_calls=0, teacher_forward_GPU_hours=0., LoRA_GPU_hours=0., path=str(folder),
                    historical_GT_counter_recorded='GT_open_attempts' in seal,
                    native_rows_verified_against_sealed_scores=method == 'NATIVE'))
            feature = RUN / 'features' / target / model / 'export_receipt.json'
            info = json.loads(feature.read_text())
            artifacts.append(dict(kind='complete_target_features', target=target, model=model,
                path=str(feature), sha256=file_sha(feature), rows=info['rows']))
            costs.append(dict(stage='target_features', target=target, model=model,
                wall_seconds_latest_worker=info['wall_seconds'],
                successful_video_stage_seconds=sum(r['wall_seconds'] for r in info['videos']),
                GPU_hours=sum(r['wall_seconds'] for r in info['videos'])/3600,
                physical_gpu=info.get('physical_gpu'), peak_allocated_bytes=info.get('peak_allocated_bytes'),
                GPU_hours_definition='successful immutable video stages across all resumed workers; shared-GPU elapsed, not utilization integral',
                failed_and_initial_model_load_overhead='separate launcher/attempt logs; not assumed zero'))
    for target, model in IN_DOMAIN.items():
        for stage, path in [
            ('source_features', RUN / 'features/source_train' / target / model / 'export_receipt.json'),
            ('source_head_training', RUN / 'heads' / model / 'training_receipt.json')]:
            receipt = json.loads(path.read_text())
            costs.append(dict(stage=stage, target=target, model=model, wall_seconds=receipt['wall_seconds'],
                GPU_hours=receipt['wall_seconds']/3600, physical_gpu=receipt.get('physical_gpu'),
                peak_allocated_bytes=receipt.get('peak_allocated_bytes')))
            artifacts.append(dict(kind=stage, target=target, model=model, path=str(path), sha256=file_sha(path)))
        checkpoint = RUN / 'heads' / model / 'checkpoint_best_source.pt'
        artifacts.append(dict(kind='source_head', model=model, path=str(checkpoint), sha256=file_sha(checkpoint)))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(PUBLIC / 'stage_A.csv', [{k: row.get(k) for k in fields} for row in rows])
    cost_fields = list(dict.fromkeys(k for row in costs for k in row))
    write_csv(PUBLIC / 'stage_A_costs.csv', [{k: row.get(k) for k in cost_fields} for row in costs])
    atomic_json(PUBLIC / 'artifacts_stage_A.json', dict(files=artifacts))
    stage_a_gate()
    gate = json.loads((PUBLIC / 'stage_A_gate.json').read_text())
    table = ['|目标|模型|Native HOTA|PRES_TOPK|Smooth|同域Δ|', '|---|---|---:|---:|---:|---:|']
    for target in IN_DOMAIN:
        for model in ('V2-best', 'Dance-best', 'LaMOT-best'):
            cell = {r['method']: r for r in rows if r['target'] == target and r['model'] == model}
            native, pres, smooth = [cell[k]['HOTA'] for k in ('NATIVE', 'PRES_TOPK', 'PRES_TOPK_SMOOTH')]
            delta = f'{pres-native:+.3f}' if model == IN_DOMAIN[target] else '跨域'
            table.append(f'|{target}|{model}|{native:.3f}|{pres:.3f}|{smooth:.3f}|{delta}|')
    text = (PUBLIC / 'REPORT.md').read_text()
    start = text.find('\n## 阶段A完整结果')
    if start >= 0: text = text[:start]
    text = text.replace('阶段A尚未完成训练/官方评测；阶段B/C尚未调用教师或启动LoRA适应。',
        '阶段A已完成源头训练及九套完整候选的官方评测；阶段B/C尚未调用教师或启动LoRA适应。')
    text = text.replace('九套目标和三套源特征正在最多两张经准入GPU上全量导出；尚无新的完整HOTA。',
        '九套目标和三套源特征已完成导出；三个PresenceHead及全部阶段A完整HOTA见下方结果。')
    text = text.replace('其余缓存继续中，尚未报告新的HOTA。', '缓存和阶段A评测已完成，完整结果如下。')
    text += '\n## 阶段A完整结果\n\n'+'\n'.join(table)+'\n\n'
    text += f'固定同域门槛：`{gate["status"]}`。失败目标：{gate["failed_targets"]}。'
    text += ('必须先定位实现/泛化原因；当前不启动B/C。\n\n' if gate['failed_targets'] else
             '本阶段推送并记录发布提交后，才允许进入B/C的C1/C3实测与成本估算。\n\n')
    text += ('全部指标（含DetA/AssA/IDF1/IDSW、输出、空表达误检、场景及可观测存在性/数量误差）见`stage_A.csv`；'
        '源特征、训练及目标特征成本见`stage_A_costs.csv`；大文件路径和SHA见`artifacts_stage_A.json`。'
        'ORACLE只引用既有独立GT诊断，不是可部署方法。Sports主评测为v4全466序列，418敏感性保留相同v4内容。\n')
    (PUBLIC / 'REPORT.md').write_text(text)
    return gate


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--check-references-only', action='store_true'); p.add_argument('--collect-only', action='store_true'); a = p.parse_args()
    if a.collect_only:
        print(json.dumps(collect(), indent=2), flush=True); return
    if a.check_references_only:
        from flex_uda.presence.runtime import runtime_guard
        attempts = runtime_guard()
        for target in IN_DOMAIN:
            meta = target_metadata(target)
            for model in ('V2-best', 'Dance-best', 'LaMOT-best'):
                margins, _ = sealed_margins(target, model)
                for name, folder in references(target, model):
                    seal = json.loads((folder / 'predictions_seal.json').read_text())
                    verify_reference(seal, meta, margins, native=name == 'NATIVE')
                print(json.dumps(dict(status='HISTORICAL_REFERENCES_VERIFIED', target=target, model=model)), flush=True)
        if attempts: raise ValueError('GT opened during reference preflight')
        return
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
    log_root = RUN / 'finish_A' / str(time.time_ns()); log_root.mkdir(parents=True)
    ready = [RUN / 'features' / t / m / 'export_receipt.json'
             for t in IN_DOMAIN for m in ('V2-best', 'Dance-best', 'LaMOT-best')]
    ready += [RUN / 'stage_A' / t / m / 'PRES_TOPK/official_completion.json' for t, m in IN_DOMAIN.items()]
    # All waiting stays inside this one shell command. Source/GPU jobs keep
    # their original parent; this worker neither restarts nor cancels them.
    while not all(path.exists() for path in ready): time.sleep(20)
    jobs = [(t, m) for t in IN_DOMAIN for m in ('V2-best', 'Dance-best', 'LaMOT-best')]
    def cell(job):
        target, model = job
        commands = [['scripts/presence_eval_offline.py', '--target', target, '--model', model, '--phase', 'decode']]
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            command = ['scripts/active_eval.py', '--target', target,
                       '--folder', str(RUN / 'stage_A' / target / model / method)]
            if target == 'SportsMOT': command.append('--sports418')
            commands.append(command)
        with (log_root / (target+'_'+model+'.log')).open('a') as log:
            for command in commands:
                code = subprocess.call([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if code: raise RuntimeError('Stage A CPU cell failed: '+str(log.name))
        print(json.dumps(dict(event='A_COMPLETE_CELL', target=target, model=model)), flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(cell, jobs))
    result = collect()
    atomic_json(log_root / 'completion.json', result)


if __name__ == '__main__': main()
