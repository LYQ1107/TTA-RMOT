"""Post-seal scientific accounting. Never imported by a GT-free runtime worker."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, target_metadata
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.active_run import PAIRS
from scripts.active_plan import jobs


def write_table(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def folder_for(job):
    suffix = 'original_delta10' if job['method'] == 'YESTRACK_FULL_D10' else 'rho_'+str(job['rho'])
    return RUN / 'active' / job['pair'] / job['method'] / suffix


def behavior_and_reliability(folder, target, pair, method, rho, truth):
    """Keep anchor-observation and majority-track GT diagnostic labels separate."""
    meta = target_metadata(target); rounds = []; videos = []; reliability = []
    cumulative = defaultdict(list); identity_seen = set()
    for vi, video in enumerate(sorted(meta['videos'])):
        root = folder / 'videos' / video
        if not (root / 'receipt.json').exists(): continue
        receipt = json.loads((root / 'receipt.json').read_text())
        bank = meta['videos'][video]
        positions = {tuple(o): j for j, o in enumerate(bank['observations'])}
        track_indices = defaultdict(list)
        for j, (t, track) in enumerate(bank['observations']): track_indices[track].append(j)
        question_path = root / 'questions.json'
        answers = json.loads(question_path.read_text()) if question_path.exists() else []
        for answer in answers:
            query, track, frame = answer['query_id'], answer['track_id'], answer['frame_id']
            y = truth[video, query]['labels']; at = positions[frame, track]
            track_y = bool(y[track_indices[track]].mean() >= .5)
            cumulative[answer['effective_view']].append((answer['yes_probability'] >= .5, bool(y[at]), track_y))
            key = video, query, track
            if key not in identity_seen:
                identity_seen.add(key)
                cumulative['IDENTITY'].append((bool(answer['identity_vote']), bool(y[at]), track_y))
        used = 0; thresholds = {}; previous = 0
        for row in receipt.get('rounds', []):
            used += row['actual_questions']; total = row['actual_questions']
            rounds.append(dict(pair=pair, method=method, rho=rho, video=video, video_index=vi, round=row['round'],
                actual_questions=total, presence_question_fraction=row['presence_questions']/total if total else None,
                identity_question_fraction=row['identity_questions']/total if total else None,
                new_confirmed_teacher=row['new_confirmed_pairs_teacher'], new_confirmed_final=row['new_confirmed_pairs_final'],
                new_distinct_confirmed=row['new_distinct_confirmed_pairs'], confirmed_pairs=row['confirmed_pairs'],
                additional_distinct_confirmed=row['additional_distinct_confirmed_pairs'],
                confident_pairs_never_asked=row['confident_pairs_never_asked'],
                optimizer_updates=row['slow']['optimizer_updates'],
                LoRA_GPU_seconds=row['slow']['GPU_seconds']))
            added = row['additional_distinct_confirmed_pairs']
            for k in (1, 5, 10, 25, 50):
                if k not in thresholds and added >= k: thresholds[k] = used
            previous = added
        videos.append(dict(pair=pair, method=method, rho=rho, video=video, video_index=vi,
            actual_questions=receipt.get('questions', receipt.get('actual_calls', 0)),
            initial_confident_pairs=receipt.get('initial_confident_pairs'), additional_distinct_confirmed=previous,
            calls_per_additional_confirmation=used/previous if previous else None,
            confirmation_quota_calls_at_round_boundary=json.dumps(thresholds, sort_keys=True),
            quota_caveat='round boundary upper bound, confidence is not correctness',
            coverage_fraction=receipt.get('coverage_fraction'), delta=receipt.get('delta')))
        final_ds = receipt.get('rounds', [{}])[-1].get('ds', {}).get('reliability', {})
        for worker, values in cumulative.items():
            for scope, column in [('ANCHOR_OBSERVATION', 1), ('TRACK_MAJORITY_50_PERCENT', 2)]:
                # IDENTITY is a track prior, so its anchor frame is only the
                # first queried observation; majority is the primary label.
                p = np.asarray([v[0] for v in values], bool); y = np.asarray([v[column] for v in values], bool)
                pos, neg = int(y.sum()), int((~y).sum())
                estimated = final_ds.get(worker, {})
                reliability.append(dict(pair=pair, method=method, rho=rho, through_video=video, worker=worker,
                    GT_scope=scope, observations=len(y), positives=pos, negatives=neg,
                    sensitivity_GT=float((p & y).sum()/pos) if pos else None,
                    specificity_GT=float((~p & ~y).sum()/neg) if neg else None,
                    sensitivity_DS=estimated.get('sensitivity'), specificity_DS=estimated.get('specificity'),
                    ORACLE_GT_DIAGNOSTIC_ONLY=True,
                    correlation_caveat='repeated views/track frames are correlated; no independent-frame significance claim'))
    return rounds, videos, reliability


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--partial', action='store_true'); a = p.parse_args()
    from scripts.active_eval import summarize, bootstrap, sports418, offline_labels
    from scripts.presence_finish_a import references
    rows = []; status = []; artifacts = []; round_rows = []; video_rows = []; reliability_rows = []; costs = []
    cached_truth = {}
    specifications = jobs()+[dict(pair=pair, method='GATE_REASSIGN_REF', rho=None) for pair in ('C1', 'C6')]
    for job in specifications:
        pair, method, rho = job['pair'], job['method'], job['rho']; target = PAIRS[pair][1]
        folder = RUN / 'active' / pair / method if method == 'GATE_REASSIGN_REF' else folder_for(job)
        if not (folder / 'predictions_seal.json').exists():
            status.append(dict(job, status='PILOT_ONLY' if (folder / 'pilot_receipt.json').exists() else 'NOT_RUN'))
            continue
        seal = verify_predictions(folder)
        result = summarize(folder, target)
        result.update(pair=pair, rho=rho); rows.append(result)
        status.append(dict(job, status='SEALED_AND_OFFICIALLY_EVALUATED'))
        if target == 'SportsMOT': sports418(folder)
        attempts = [json.loads(path.read_text()) for path in (folder / 'worker_attempts').glob('*.json')]
        receipts = [json.loads(path.read_text()) for path in (folder / 'videos').glob('*/receipt.json')]
        updates = sum(r['slow']['optimizer_updates'] for v in receipts for r in v.get('rounds', []))
        costs.append(dict(pair=pair, method=method, rho=rho, teacher_calls=result['teacher_calls'],
            teacher_new_forward_calls=seal.get('teacher', {}).get('new_calls', 0),
            teacher_attempted_requests=seal.get('teacher', {}).get('attempted_logical_calls'),
            teacher_cache_hits=seal.get('teacher', {}).get('cache_hits', 0),
            teacher_forward_GPU_hours=result['teacher_forward_GPU_hours'],
            teacher_equivalent_forward_GPU_hours=result['teacher_equivalent_forward_GPU_hours'],
            LoRA_GPU_hours=result['LoRA_GPU_hours'], rescore_GPU_hours=result['rescore_GPU_hours'],
            assigned_worker_wall_hours=sum(r.get('wall_seconds', 0) for r in attempts)/3600,
            failed_worker_attempts=sum(r['status'] != 'RETURNED_SUCCESSFULLY' for r in attempts),
            wall_is_not_utilization_integral=True, optimizer_updates=updates,
            LoRA_trainable_parameters=24576 if method in ('FULL', 'RANDOM_ACQ', 'MINUS_PRES', 'MINUS_DS', 'MINUS_ROUTE', 'MINUS_FAST', 'RESET_LORA') else 0,
            peak_teacher_allocated_bytes=seal.get('teacher', {}).get('peak_allocated_bytes'),
            peak_student_allocated_bytes=max((r.get('parent_peak_allocated_bytes', 0) for r in attempts), default=0),
            covered_pairs=sum(r.get('covered_pairs', 0) for r in receipts) if method.startswith('YESTRACK') else None,
            all_pairs=sum(r.get('all_pairs', 0) for r in receipts) if method.startswith('YESTRACK') else None,
            delta_by_video=json.dumps({r['video']:r['delta'] for r in receipts if 'delta' in r}, sort_keys=True)))
        if target not in cached_truth: cached_truth[target] = offline_labels(target)[1]
        rr, vv, dd = behavior_and_reliability(folder, target, pair, method, rho, cached_truth[target])
        round_rows += rr; video_rows += vv; reliability_rows += dd
        for path in [folder / 'predictions_seal.json', folder / 'results/pedestrian_detailed.csv',
                     folder / 'results/pedestrian_summary.txt', *sorted((folder / 'videos').glob('*/questions.json'))]:
            artifacts.append(dict(pair=pair, method=method, rho=rho, path=str(path), bytes=path.stat().st_size, sha256=file_sha(path)))
    incomplete = [r for r in status if r['status'] != 'SEALED_AND_OFFICIALLY_EVALUATED' and r['method'] != 'GATE_REASSIGN_REF']
    if incomplete and not a.partial: raise RuntimeError('Requested matrix incomplete; use --partial to report honestly without claiming completion')
    comparisons = {}
    for row in rows:
        if row['method'] != 'FULL': continue
        pair, rho, target = row['pair'], row['rho'], row['target']
        native = dict(references(target, PAIRS[pair][0]))['NATIVE']
        targets = [('NATIVE', native), ('PRES_TOPK', RUN / 'stage_A' / target / PAIRS[pair][0] / 'PRES_TOPK')]
        targets += [(r['method'], Path(r['path'])) for r in rows if r['pair'] == pair and r['method'] != 'FULL' and r['rho'] in (rho, None)]
        for method, folder in targets:
            comparisons[f'{pair}/rho{rho}/FULL_minus_{method}'] = bootstrap(row['path'], folder, target)
    write_table(PUBLIC / 'main.csv', rows); write_table(PUBLIC / 'matrix_status.csv', status)
    write_table(PUBLIC / 'costs.csv', costs); write_table(PUBLIC / 'behavior_round.csv', round_rows)
    write_table(PUBLIC / 'behavior_video.csv', video_rows); write_table(PUBLIC / 'reliability_vs_GT.csv', reliability_rows)
    atomic_json(PUBLIC / 'bootstrap.json', dict(comparisons=comparisons, samples=2000, seed=0,
        cluster_unit='video; correlated expressions/frames kept together', KITTI='four videos, points only',
        multiplicity='descriptive comparisons, no target-GT selection of variants'))
    atomic_json(PUBLIC / 'active_artifacts.json', dict(large_files_not_in_git=True, files=artifacts))
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        curves = ('FULL', 'YESTRACK_BUDGET_RANDOM', 'YESTRACK_BUDGET_PRIOR', 'UNIFORM_K')
        for ax, pair in zip(axes.flat, PAIRS):
            for method in curves:
                rr = sorted([r for r in rows if r['pair'] == pair and r['method'] == method], key=lambda r:r['rho'])
                if rr:
                    ax.plot([r['teacher_calls'] for r in rr], [r['HOTA'] for r in rr], 'o-', label=method)
                    for r in rr: ax.annotate(str(r['rho']), (r['teacher_calls'], r['HOTA']), fontsize=7)
            for method, marker in [('YESTRACK_FULL_D10', 'x'), ('GATE_REASSIGN_REF', 'D')]:
                for r in rows:
                    if r['pair'] == pair and r['method'] == method: ax.scatter(r['teacher_calls'], r['HOTA'], marker=marker, label=method)
            ax.set(title=pair+': '+PAIRS[pair][0]+' -> '+PAIRS[pair][1], xlabel='Teacher logical judgements', ylabel='Official HOTA')
            ax.grid(alpha=.2)
        handles, labels = [], []
        for ax in axes.flat:
            hh, ll = ax.get_legend_handles_labels()
            for h, label in zip(hh, ll):
                if label not in labels: handles.append(h); labels.append(label)
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
        fig.suptitle('PARTIAL: missing points are NOT run' if incomplete else 'All fixed budgets; no best-method selection')
        fig.tight_layout(rect=(0, .09, 1, .96)); fig.savefig(PUBLIC / 'teacher_calls_vs_HOTA.png', dpi=160); plt.close(fig)
    text = (PUBLIC / 'REPORT.md').read_text(); marker = '\n## 阶段B/C已封存结果'
    if marker in text: text = text[:text.index(marker)]
    table = ['|组合|方法|ρ|HOTA|DetA|AssA|IDF1|IDSW|行数|Empty行|教师次数|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append('|'+ '|'.join(str(r[k]) for k in ('pair','method','rho','HOTA','DetA','AssA','IDF1','IDSW','prediction_rows','query_empty_prediction_rows','teacher_calls'))+'|')
    text += marker+'\n\n状态：'+('完整108项运行矩阵完成。' if not incomplete else f'部分完成：{len(incomplete)}项尚未完成，不宣称完整方法通过。')+'\n\n'
    text += '\n'.join(table)+'\n\n'
    text += ('固定候选/原ID/原框不变；GT仅在封存后的独立评测中使用。Dance和Sports按视频聚类2000次seed0；KITTI仅点估计。'
        'C1 Gate-Reassign逐行复现旧参照，C6是同门控的新CPU派生参照；二者沿用同一批29410次历史判断，不将其成本解释成零成本独立系统。\n\n'
        '成本见costs.csv：真实新增forward、缓存复用、等价独立forward及LoRA/重打分分别报告；worker wall包含恢复失败尝试，'
        '不是GPU利用率积分，不能与同驻教师进程wall重复相加。源Head训练、前端与特征预计算另见阶段A。\n\n'
        '行为见behavior_round/video.csv。置信不等于正确；分别标记先验即高置信、从未问过但高置信，以及教师/最终新增确认。'
        '达到固定确认数的调用是轮边界上界，不据此单独宣称越问越省。DS估计与GT后验诊断见reliability_vs_GT.csv，'
        '同时给出提问帧与50%轨迹多数标签；身份先验以轨迹标签为主要参照，重复视图/帧不是独立样本。\n\n'
        '完整机器表、bootstrap及大文件路径/SHA均在本目录；Sports使用v4全466和相同v4内容的旧418 ID敏感性。'
        '没有根据目标GT调整阈值、种子或选择最优变体。\n')
    (PUBLIC / 'REPORT.md').write_text(text)
    print(json.dumps(dict(status='FULL_MATRIX_COMPLETE' if not incomplete else 'PARTIAL_REPORTED_NOT_COMPLETE',
        evaluated=len(rows), pending=len(incomplete), comparisons=len(comparisons))), flush=True)


if __name__ == '__main__': main()
