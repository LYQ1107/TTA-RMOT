"""Post-seal official evaluation and two-direction quantity diagnostics."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_sparse_observation_launch import conditions

STATE = OUT / 'supervisor/sparse_observation_evaluation_state.json'


def projection_pairs(kind):
    pairs = []
    if kind == 'OBSERVATION':
        donors = {'ANCHOR_FULL': ['POOLED_SOFT', 'POOLED_MEAN', 'POOLED_HARD400', 'ANCHOR_KD',
            'ANCHOR_NO_GRAPH', 'ANCHOR_VISUAL_ONLY', 'ANCHOR_TEMPORAL_ONLY',
            'CONSTANT_TEMPORAL', 'MASS_MATCHED_CONSTANT_TEMPORAL'],
            'ANCHOR_MEAN_FULL': ['POOLED_MEAN', 'ANCHOR_MEAN']}
        reverse = {'ANCHOR_FULL': ['POOLED_SOFT', 'POOLED_MEAN', 'POOLED_HARD400', 'ANCHOR_KD'],
            'ANCHOR_MEAN_FULL': ['POOLED_MEAN', 'ANCHOR_MEAN']}
    else:
        donors = {'GRAPH': ['SOFT_FINAL', 'SOFT_MEAN_400_FINAL', 'HARD_400', 'NO_GRAPH'],
            'MEAN_GRAPH': ['SOFT_MEAN_400_FINAL']}
        reverse = {'GRAPH': ['SOFT_FINAL', 'SOFT_MEAN_400_FINAL', 'HARD_400'],
            'MEAN_GRAPH': ['SOFT_MEAN_400_FINAL']}
    for full, ranks in donors.items():
        pairs.extend((rank, full) for rank in [full]+ranks)
    for full, counts in reverse.items():
        for count in counts:
            pairs.extend(((full, count), (count, count)))
    return list(dict.fromkeys(pairs))


def generate_counts(record):
    from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
    from flex_uda.active.acquisition import exclusive_group
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_benchmark_scope import FRESH
    guard = generation_guard()
    begin = time.perf_counter()
    pair, source = record['pair'], Path(record['folder'])
    model, target = PAIRS[pair]
    original = json.loads((source / 'completion.json').read_text())
    assert original['GT_open_attempts'] == 0
    root = source / 'countmatched'
    if (root / 'completion.json').is_file():
        return str(root)
    m = metadata(target)
    pairs = [tuple(p) for p in record['projection_pairs']] if 'projection_pairs' in record else projection_pairs(record['kind'])
    ranks = sorted({r for r, _ in pairs})
    count_names = sorted({c for _, c in pairs})
    sources = {name: Path(record.get('method_sources', {}).get(name, source / name))
               for name in set(ranks) | set(count_names)}
    masks = {key: np.zeros(m['candidate_query_rows'], dtype=bool) for key in pairs}
    for video, bank in sorted(m['videos'].items()):
        obs = np.asarray(bank['observations'])
        queries = [q for q in m['episodes'] if q['video_id'] == video]
        exclusive = exclusive_group([q['query_text'] for q in queries], target)
        counts = {c: np.load(sources[c] / 'videos' / video / 'mask.npy', mmap_mode='r') for c in count_names}
        prior_path = (Path(record['fresh_prior_root']) / video / 'sparse_inputs.npz'
            if 'fresh_prior_root' in record else
            FRESH / 'source' / video / 'sparse_inputs.npz' if pair == 'C4' else OUT / 'inputs' / pair / video / 'inputs.npz')
        with np.load(prior_path) as a:
            tracks, prior = a['tracks'], a['prior']
        lookup = {int(t): i for i, t in enumerate(tracks)}
        columns = np.asarray([lookup[int(t)] for _, t in obs])
        if record.get('coarse_scores'):
            with np.load(source / 'videos' / video / 'coarse_scores.npz') as bank_scores:
                assert np.array_equal(tracks, bank_scores['tracks'])
                scores = {r: bank_scores[r][:, columns] for r in ranks}
        else:
            scores = {r: np.load(sources[r] / 'videos' / video / 'scores.npy', mmap_mode='r') for r in ranks}
        for frame in np.unique(obs[:, 0]):
            ids = np.flatnonzero(obs[:, 0] == frame)
            if exclusive:
                cardinalities = {c: counts[c][:, ids].sum(1).astype(int) for c in count_names}
                for c in count_names:
                    assert np.all(counts[c][:, ids].sum(0) <= 1)
                for (rank, count), mask in masks.items():
                    cardinality = cardinalities[count]
                    slots = np.repeat(np.arange(len(queries)), cardinality)
                    assert len(slots) <= len(ids)
                    if not len(slots):
                        continue
                    slot_ids, candidates = linear_sum_assignment(-scores[rank][slots][:, ids].astype(float))
                    chosen_query = slots[slot_ids]
                    for qi, query in enumerate(queries):
                        chosen = ids[candidates[chosen_query == qi]]
                        assert len(chosen) == cardinality[qi]
                        mask[query['start'] + chosen] = True
            else:
                for qi, query in enumerate(queries):
                    orders = {rank: ids[np.lexsort((obs[ids, 1], -prior[qi, columns[ids]], -scores[rank][qi, ids]))]
                              for rank in ranks}
                    for (rank, count), mask in masks.items():
                        k = int(counts[count][qi, ids].sum())
                        chosen = orders[rank][:k]
                        assert len(chosen) == k
                        mask[query['start'] + chosen] = True
        print(json.dumps(dict(event='SPARSE_OBSERVATION_COUNT_VIDEO', pair=pair,
            source=str(source), video=video)), flush=True)
    methods = []
    for (rank, count), mask in masks.items():
        name = rank+'__AT_'+count+'_COUNT'
        seal = seal_mask(root / name, m, mask, target=target, model=model, method=name,
            GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=count,
                source=str(source), counts_from_GT=False, counts_equal_per_query_frame=True,
                same_frozen_evidence_and_official_candidates=True,
                common_projection='Joint maximum score assignment with fixed per-query slots for exclusive tasks; otherwise descending score, source fraction, original ID.',
                exclusive_task_constraint_preserved=True, diagnostic_not_deployment_cost=True))
        methods.append(dict(method=name, rows=seal['prediction_rows']))
    self_changes = {}
    for count in count_names:
        if (count, count) not in pairs:
            continue
        a = json.loads((sources[count] / 'predictions_seal.json').read_text())
        b = json.loads((root / (count+'__AT_'+count+'_COUNT') / 'predictions_seal.json').read_text())
        before = {(r['video'], r['expression']): r['prediction_sha256'] for r in a['episodes']}
        after = {(r['video'], r['expression']): r['prediction_sha256'] for r in b['episodes']}
        self_changes[count] = sum(before[k] != after[k] for k in before)
    atomic_json(root / 'countself_check.json', self_changes)
    assert not any(self_changes.values()), ('CountSelf regression', source, self_changes)
    assert not guard
    atomic_json(root / 'completion.json', dict(status='SEALED_SPARSE_OBSERVATION_COUNTS',
        pair=pair, source=str(source), methods=methods, GT_open_attempts=0,
        CountSelf_changed_query_files=self_changes, generation_wall_seconds=time.perf_counter()-begin))
    return str(root)


def evaluate_one(task):
    folder, pair = task
    from flex_uda.a100.evaluation_reuse import evaluate
    return folder, evaluate(folder, pair)


def main():
    generation = json.loads((OUT / 'supervisor/sparse_observation_state.json').read_text())
    assert generation['status'] == 'ALL_SPARSE_OBSERVATION_PREDICTIONS_SEALED_NO_NEW_GT_YET'
    records = conditions()
    for record in records:
        assert json.loads((Path(record['folder']) / 'completion.json').read_text())['GT_open_attempts'] == 0
    primary = [r for r in records if r['kind'] == 'OBSERVATION' or r['fraction'] == .1]
    atomic_json(STATE, dict(status='GENERATING_FIXED_QUANTITY_CONTROLS_NO_NEW_GT', conditions=len(primary), pid=__import__('os').getpid()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        count_roots = list(pool.map(generate_counts, primary))
    # Every raw prediction AND registered counterfactual is now sealed.
    tasks = []
    for record in records:
        source = Path(record['folder'])
        receipt = json.loads((source / 'completion.json').read_text())
        tasks.extend((str(source / r['method']), record['pair']) for r in receipt['methods'])
    for record, root in zip(primary, count_roots):
        root = Path(root)
        receipt = json.loads((root / 'completion.json').read_text())
        tasks.extend((str(root / r['method']), record['pair']) for r in receipt['methods'])
    atomic_json(STATE, dict(status='ALL_COUNTERFACTUALS_SEALED_OFFICIAL_EVALUATING', methods=len(tasks), pid=__import__('os').getpid()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(evaluate_one, tasks))
    atomic_json(OUT / 'analysis/sparse_observation_official.json', dict(status='ALL_SPARSE_OBSERVATION_OFFICIAL_COMPLETE',
        results=results, methods=len(tasks), goal_status='ACTIVE', all_development_data=True,
        no_adaptation_GT=True, all_predictions_and_counterfactuals_sealed_before_this_GT=True))
    atomic_json(STATE, dict(status='PASS_SPARSE_OBSERVATION_OFFICIAL_GOAL_STILL_ACTIVE', methods=len(tasks), pid=__import__('os').getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--counts-only', action='store_true')
    p.add_argument('--source')
    p.add_argument('--pair', choices=['C4', 'C1'])
    p.add_argument('--kind', choices=['BUDGET', 'OBSERVATION'])
    args = p.parse_args()
    if args.counts_only:
        generate_counts(dict(pair=args.pair, folder=args.source, kind=args.kind))
    else:
        try:
            main()
        except Exception as exc:
            atomic_json(STATE, dict(status='FAILED_SPARSE_OBSERVATION_EVALUATION_REQUIRES_ROOT_CAUSE',
                error=repr(exc), goal_status='ACTIVE'))
            raise
