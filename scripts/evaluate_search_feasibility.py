"""Independent post-seal GT labels, ORACLE simulation, then fixed probe metrics."""
from functools import lru_cache
from fractions import Fraction
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.search_feasibility_common import *


def selection_metrics(selected, labels, g):
    tp = sum(bool(labels[i]) for i in selected); k = len(selected); positives = sum(labels)
    precision = tp / k if k else 0.
    recall = tp / positives if positives else 0.
    f1 = 2 * tp / (k + positives) if k + positives else 0.
    if not g: f1 = float(not k)
    return dict(selected_count=k, TP=tp, FP=k-tp, FN=positives-tp,
                precision=precision, recall=recall, F1=f1,
                count_MAE=abs(k-g), presence_accuracy=int(bool(k) == bool(g)))


def exact_f1(metrics, positives, g):
    k = metrics['selected_count']
    if not g: return Fraction(int(not k), 1)
    return Fraction(2*metrics['TP'], k+positives) if k+positives else Fraction(0, 1)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require_prereg(); sealed = verify_search(); start = time.perf_counter()
    # No GT routine or target-root resolution occurs before the complete prediction seal passes.
    from scripts.analyze_pf_full_offline import load_eval_gt, overlaps, matched_gt, auroc
    from scripts.multidomain_diagnostics import target_root
    runtime = list(json_rows(sealed['runtime']['path'])); by_id = {r['list_id']: r for r in runtime}
    dest = RUN / 'offline'; dest.mkdir(parents=True, exist_ok=True)
    gt_roots = {t: target_root(t) for t in TARGETS}; gt_inputs = {}
    wanted = {}
    for r in runtime: wanted.setdefault((r['target'], r['video'], r['query']), set()).add(r['frame'])

    @lru_cache(maxsize=None)
    def truth(target, video, query):
        key = target, video, query
        path = gt_roots[target] / 'gt_template' / video / query / 'gt.txt'
        gt_inputs[key] = record(path)
        return {f: boxes for f, boxes in load_eval_gt(gt_roots[target], video, query).items() if f in wanted[key]}

    labels = {}; list_rows = []
    labels_path = dest / 'labels.jsonl.gz'
    with gzip.open(labels_path, 'wt') as stream:
        for r in runtime:
            target, video, query, frame = r['target'], r['video'], r['query'], r['frame']
            gt = truth(target, video, query).get(frame, [])
            boxes = [[x, y, x+w, y+h] for x, y, w, h in r['boxes']]
            iou = overlaps(boxes, [box for _, box in gt])
            _, selected = matched_gt(iou, list(range(r['N'])))
            y = [i in selected for i in range(r['N'])]
            row = dict(list_id=r['list_id'], target=target, video=video, frame=frame, query=query,
                N=r['N'], g=len(gt), matched_candidates=sum(y), labels=y,
                A_GT={m: auroc(r['margins'][m], y) for m in MODELS})
            labels[r['list_id']] = row; stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            list_rows.append({k: v for k, v in row.items() if k not in ('A_GT', 'labels')} |
                dict(A_GT_V2=row['A_GT']['V2-best'], A_GT_Dance=row['A_GT']['Dance-best'],
                     A_GT_LaMOT=row['A_GT']['LaMOT-best'], GT_positive_track_ids=json.dumps([r['track_ids'][i] for i in selected])))
    csv_rows(PUBLIC / 'lists.csv', list_rows)
    dump(dest / 'labels_seal.json', dict(status='POST_SEARCH_GT_LABELS_SEALED', labels=record(labels_path),
        search_predictions=sealed['predictions'], GT_inputs=list(gt_inputs.values()),
        GT_protocol='unchanged KITTI/Dance; authorized SportsMOT v4 original annotations',
        candidate_matching='score-independent maximum-cardinality then IoU; >=0.5, mark!=0, class=1',
        GT_read_only_by_offline_evaluator=True, GT_feedback_to_search=False))
    # ORACLE is a separate diagnostic artifact, never fed to the GT-free search process.
    oracle_path = dest / 'oracle_predictions.jsonl.gz'; oracle_count = 0
    with gzip.open(oracle_path, 'wt') as stream:
        for r in runtime:
            y = labels[r['list_id']]['labels']
            for model in MODELS:
                order = ranking(r['margins'][model], r['track_ids'])
                answers = [y[i] for i in order]
                runs = suite(answers, [r['margins'][model][i] for i in order], r['subseed'])
                pba = next(x for x in runs if x['method'] == 'PBA_0.2')
                out = dict(list_id=r['list_id'], target=r['target'], video=r['video'], model=model,
                    verifier='GT_ORACLE', track_order=[r['track_ids'][i] for i in order], runs=runs,
                    C_dense=dense_conflict(answers), C_online=online_conflict(pba['asked_positions'], pba['answers']))
                stream.write(json.dumps(out, separators=(',', ':')) + '\n'); oracle_count += len(runs)
    dump(dest / 'oracle_seal.json', dict(status='ORACLE_DIAGNOSTIC_SEALED_BEFORE_METRICS',
        role='ORACLE_ONLY_NOT_A_METHOD', predictions=record(oracle_path), labels=record(labels_path),
        GT_free_search_predictions=sealed['predictions'], runs=oracle_count, teacher_calls=0, GPU_hours=0))
    conflict_rows = []; total = 0; combinations = set()
    def metrics_rows():
        nonlocal total
        for path in (sealed['predictions']['path'], oracle_path):
            for group in json_rows(path):
                key = group['list_id'], group['model'], group['verifier']
                if key in combinations: raise ValueError('Duplicate list/model/verifier results')
                combinations.add(key)
                r = by_id[group['list_id']]; lab = labels[r['list_id']]; model = group['model']; verifier = group['verifier']
                expected_order = ranking(r['margins'][model], r['track_ids'])
                if group['track_order'] != [r['track_ids'][i] for i in expected_order]: raise ValueError('Changed ranking')
                order = expected_order
                answers = [lab['labels'][i] if verifier == 'GT_ORACLE' else r['probabilities'][verifier][i] >= .5 for i in order]
                if tuple(x['method'] for x in group['runs']) != METHODS: raise ValueError('Missing fixed method')
                f1 = {}; f1_exact = {}; questions = {}
                for run in group['runs']:
                    positions = run['selected_positions']; asked = run['asked_positions']; k = run['b_hat']
                    if len(set(positions)) != len(positions) or len(set(asked)) != len(asked): raise ValueError('Repeated candidate')
                    if not all(1 <= i <= r['N'] for i in positions + asked): raise ValueError('Invalid one-indexed position')
                    if k != len(positions) or run['m'] != len(asked): raise ValueError('Count/trace mismatch')
                    if run['answers'] != [answers[i-1] for i in asked]: raise ValueError('Answer trace changed')
                    if run['method'] not in ('DENSE', 'NATIVE') and positions != list(range(1, k+1)):
                        raise ValueError('Search output is not prescribed prefix')
                    selected = [order[i-1] for i in positions]
                    metrics = selection_metrics(selected, lab['labels'], lab['g']); f1[run['method']] = metrics['F1']
                    f1_exact[run['method']] = exact_f1(metrics, lab['matched_candidates'], lab['g'])
                    questions[run['method']] = run['m']; total += 1
                    yield dict(list_id=r['list_id'], target=r['target'], video=r['video'], frame=r['frame'], query=r['query'],
                        model=model, cross_domain=int(model != IN_DOMAIN[r['target']]), verifier=verifier, method=run['method'],
                        role='ORACLE_ONLY_DIAGNOSTIC' if verifier == 'GT_ORACLE' else 'SEALED_TEACHER_CACHE_SIMULATION',
                        N=r['N'], g=lab['g'], matched_candidates=lab['matched_candidates'], m=run['m'], b_hat=k,
                        questions_over_log=run['m'] / math.ceil(math.log2(r['N']+1)), **metrics,
                        selected_track_ids=json.dumps([r['track_ids'][i] for i in selected], separators=(',', ':')),
                        asked_positions=json.dumps(asked, separators=(',', ':')),
                        answers=json.dumps(run['answers'], separators=(',', ':')), posterior_max=run['posterior_max'])
                if questions['RANDOM_SAME_BUDGET'] != questions['PBA_0.2']: raise ValueError('Unmatched random budget')
                pba = next(x for x in group['runs'] if x['method'] == 'PBA_0.2')
                c_dense = dense_conflict(answers); c_online = online_conflict(pba['asked_positions'], pba['answers'])
                if c_dense != group['C_dense'] or c_online != group['C_online']: raise ValueError('Changed conflict trace')
                conflict_rows.append(dict(list_id=r['list_id'], target=r['target'], video=r['video'], model=model,
                    cross_domain=int(model != IN_DOMAIN[r['target']]), verifier=verifier, N=r['N'], g=lab['g'],
                    verifier_yes=sum(answers), C_dense=c_dense, C_online=c_online,
                    A_GT=lab['A_GT'][model], dense_F1=f1['DENSE'], pba_F1=f1['PBA_0.2'],
                    search_failure=int(f1_exact['DENSE'] - f1_exact['PBA_0.2'] > Fraction(1, 10)),
                    simulated_pba_questions=questions['PBA_0.2']))
    metrics_path = dest / 'search_runs.csv.gz'; csv_rows(metrics_path, metrics_rows())
    if len(combinations) != len(runtime) * 3 * 4 or total != len(runtime) * 3 * 4 * len(METHODS):
        raise ValueError('Incomplete result cross product')
    conflict_path = dest / 'conflict.csv'; csv_rows(conflict_path, conflict_rows)
    result = dict(status='ALL_SEARCH_AND_ORACLE_RESULTS_OFFLINE_SCORED', teacher_search=sealed,
        oracle_seal=record(dest / 'oracle_seal.json'), metrics=record(metrics_path), lists=record(PUBLIC / 'lists.csv'),
        conflicts=record(conflict_path), runs=total, all_combinations=len(combinations),
        GT_feedback_to_search=False, teacher_calls=0, model_calls=0, optimizer_updates=0, GPU_hours=0,
        CPU_wall_seconds=time.perf_counter()-start)
    dump(dest / 'evaluation.json', result)
    print(json.dumps(dict(event='SEARCH_OFFLINE_EVALUATED', runs=total, combinations=len(combinations),
                         oracle_runs=oracle_count, wall_seconds=result['CPU_wall_seconds'])), flush=True)


if __name__ == '__main__': main()
