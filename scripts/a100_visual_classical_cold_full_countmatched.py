"""Post-seal fixed-count diagnostics for the physically fresh whole-C4 run.

This program does not request teacher answers, fit a head, or use GT counts.
Its generation and official-evaluation steps run in separate processes.
Counterfactual output and evaluation time are not deployment-method costs.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT


def comparisons():
    pairs = []
    for label in ('RAW', 'CLIP'):
        full = label + '_COHORT'
        ranks = [full, label + '_KD', label + '_KD_400', label + '_NO_GRAPH',
                 label + '_STRONG_PRIOR', label + '_MEAN',
                 label + '_MEAN_COHORT', 'HARD_KD', 'HARD_KD_400']
        pairs.extend((rank, full) for rank in ranks)
        for count in (label + '_KD', label + '_MEAN', 'HARD_KD_400'):
            pairs.extend(((full, count), (count, count)))
    return list(dict.fromkeys(pairs))


def generate(name):
    from flex_uda.a100.mechanism import generation_guard, metadata
    from flex_uda.presence.predictions import seal_mask
    guard = generation_guard()
    source = OUT / 'cold_full_c4' / name
    receipt = json.loads((source / 'completion.json').read_text())
    assert receipt['GT_open_attempts'] == receipt['prepaid_artifact_reads'] == 0
    destination = source / 'countmatched'
    if (destination / 'completion.json').is_file():
        return
    start = time.perf_counter()
    meta = metadata('SportsMOT')
    pairs = comparisons()
    masks = {pair: np.zeros(meta['candidate_query_rows'], dtype=bool) for pair in pairs}
    ranks = sorted({rank for rank, _ in pairs})
    count_names = sorted({count for _, count in pairs})
    for video, bank in sorted(meta['videos'].items()):
        queries = [q for q in meta['episodes'] if q['video_id'] == video]
        observations = np.asarray(bank['observations'])
        scores = {rank: np.load(source / 'methods' / rank / 'videos' / video / 'scores.npy', mmap_mode='r')
                  for rank in ranks}
        counts = {count: np.load(source / 'methods' / count / 'videos' / video / 'mask.npy', mmap_mode='r')
                  for count in count_names}
        margin_path = source / 'source' / video / 'output_margin.npy'
        if not margin_path.is_file():
            margin_path = source / 'source' / video / 'margin.npy'
        margin = np.load(margin_path, mmap_mode='r')
        for frame in np.unique(observations[:, 0]):
            ids = np.flatnonzero(observations[:, 0] == frame)
            for qi, query in enumerate(queries):
                order = {rank: ids[np.lexsort((observations[ids, 1], -margin[qi, ids], -scores[rank][qi, ids]))]
                         for rank in ranks}
                for (rank, count), mask in masks.items():
                    cardinality = int(counts[count][qi, ids].sum())
                    selected = order[rank][:cardinality]
                    assert len(selected) == cardinality
                    mask[query['start'] + selected] = True
        print(json.dumps(dict(event='FRESH_C4_COUNTMATCHED_VIDEO', video=video)), flush=True)
    methods = []
    for (rank, count), mask in masks.items():
        method = rank + '__AT_' + count + '_COUNT'
        seal = seal_mask(destination / method, meta, mask, target='SportsMOT', model='Dance-best',
                         method=method, GT_open_attempts=len(guard), extra=dict(
                             source_fresh_whole_C4=name, rank_method=rank, count_method=count,
                             counts_from_GT=False, counts_equal_per_query_frame=True,
                             same_source_and_teacher_stream=True,
                             common_projection='descending method score, descending fresh source margin, ascending original ID',
                             deployment_method=False))
        methods.append(dict(method=method, rows=seal['prediction_rows']))
    assert not guard
    # A self projection must preserve the actual full mask in this C4 task.
    count_self_changes = {}
    for count in count_names:
        method = count + '__AT_' + count + '_COUNT'
        if (count, count) not in pairs:
            continue
        original = json.loads((source / 'methods' / count / 'predictions_seal.json').read_text())
        projected = json.loads((destination / method / 'predictions_seal.json').read_text())
        changes = sum(a['prediction_sha256'] != b['prediction_sha256']
                      for a, b in zip(original['episodes'], projected['episodes']))
        count_self_changes[count] = changes
        assert changes == 0, (count, 'CountSelf changed actual C4 predictions', changes)
    atomic_json(destination / 'completion.json', dict(status='SEALED_FRESH_C4_COUNTMATCHED',
                source_run=name, methods=methods, GT_open_attempts=0, counts_from_GT=False,
                count_self_changed_query_files=count_self_changes,
                generation_wall_seconds=time.perf_counter() - start))


def evaluate_one(folder):
    from flex_uda.a100.evaluation_reuse import evaluate
    return Path(folder).name, evaluate(folder, 'C4')


def main(name,state_prefix='cold_full_countmatched',analysis_name='fresh_whole_C4_countmatched'):
    root = OUT / 'cold_full_c4' / name
    state = OUT / 'supervisor' / (state_prefix + '_state.json')
    atomic_json(state, dict(status='WAITING_FOR_FRESH_C4_SEAL'))
    while not (root / 'completion.json').is_file():
        parent = OUT / 'supervisor' / (state_prefix.removesuffix('_countmatched') + '_state.json')
        if parent.is_file() and json.loads(parent.read_text()).get('status', '').startswith('FAILED'):
            raise RuntimeError('Fresh C4 source worker failed; preserve and diagnose its logs before resuming')
        time.sleep(15)
    atomic_json(state, dict(status='GENERATING_FRESH_C4_COUNTMATCHED'))
    subprocess.run([sys.executable, __file__, '--name', name, '--generate-only'], cwd=ROOT, check=True)
    sealed = json.loads((root / 'countmatched' / 'completion.json').read_text())
    atomic_json(state, dict(status='EVALUATING_FRESH_C4_COUNTMATCHED'))
    from flex_uda.a100.evaluation_reuse import bootstrap
    bootstrap()
    folders = [root / 'countmatched' / row['method'] for row in sealed['methods']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        evaluations = dict(pool.map(evaluate_one, folders))
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_mechanism_analyze import compare_official
    stats = {folder.name: read_stats(folder) for folder in folders}
    results = {}
    for label in ('RAW', 'CLIP'):
        full = label + '_COHORT'
        self_key = full + '__AT_' + full + '_COUNT'
        for rank, count in comparisons():
            key = rank + '__AT_' + count + '_COUNT'
            if count == full and rank != full:
                results[full + '_vs_' + rank + '_AT_FULL_COUNTS'] = compare_official(stats[self_key], stats[key])
        for count in (label + '_KD', label + '_MEAN', 'HARD_KD_400'):
            results[full + '_vs_' + count + '_AT_BASELINE_COUNTS'] = compare_official(
                stats[full + '__AT_' + count + '_COUNT'], stats[count + '__AT_' + count + '_COUNT'])
    report = dict(status='PASS_FRESH_C4_COUNTMATCHED', source_run=name,
                  HOTA={k: v['HOTA_unrounded'] for k, v in evaluations.items()}, comparisons=results,
                  count_self_changed_query_files=sealed['count_self_changed_query_files'],
                  count_source_is_method_not_GT=True, adaptation_GT_open_attempts=0,
                  diagnostic_not_deployment_cost=True, all_development_data=True)
    atomic_json(root / 'countmatched' / 'official_evaluation.json', report)
    atomic_json(OUT / 'analysis' / (analysis_name + '.json'), report)
    atomic_json(state, dict(status='PASS_FRESH_C4_COUNTMATCHED'))
    print(json.dumps(dict(event='FRESH_C4_COUNTMATCHED_COMPLETE', comparisons=results)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='C4_fresh_full_stream_seed0')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--state-prefix', default='cold_full_countmatched')
    parser.add_argument('--analysis-name', default='fresh_whole_C4_countmatched')
    args = parser.parse_args()
    if args.generate_only:
        generate(args.name)
    else:
        main(args.name,args.state_prefix,args.analysis_name)
