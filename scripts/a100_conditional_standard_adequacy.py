"""Check ordinary raw/unit-weight KD against independently rebuilt targets.

Reconstruct the frozen own-policy raw view mixtures and coarse means, then
compare their prefix cardinalities/entropies with the actual fit receipts.
No images, model weights, GT, HOTA or new neural forwards are consumed.
"""
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.observation_learning_inputs import ANSWERS
from scripts.a100_conditional_standard_kd import OUT, ARMS, METHODS, COARSE_ARMS
from scripts.a100_six_direction import PUBLIC, SEEDS


def entropy(values):
    p = np.asarray(values, dtype='float64')
    return -p*np.log(np.maximum(p, 1e-300))-(1-p)*np.log(np.maximum(1-p, 1e-300))


def own_targets(pair, video):
    doc = json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
    assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
    seen, grouped = {}, defaultdict(list)
    for item in doc['answers']:
        rid, p = item['request_id'], float(item['yes_probability'])
        if rid in seen:
            assert seen[rid] == p
            continue
        seen[rid] = p
        key = int(item['local_query']), int(item['local_track']), int(item['frame_id'])
        grouped[key].append(p)
    keys = np.asarray(sorted(grouped), dtype=int)
    dense = np.asarray([np.mean(grouped[tuple(key)]) for key in keys], dtype='float32')
    _, bag = np.unique(keys[:, :2], axis=0, return_inverse=True)
    coarse = (np.bincount(bag, weights=dense)/np.bincount(bag)).astype('float32')
    assert np.isfinite(dense).all() and ((dense >= 0) & (dense <= 1)).all()
    return dense, coarse


def main():
    guard = generation_guard()
    start = time.perf_counter()
    prepared = {}
    for pair, (_, target) in PAIRS.items():
        dense, coarse, rows = [], [], {}
        for video in sorted(metadata(target)['videos']):
            a, b = own_targets(pair, video)
            dense.extend(a.tolist())
            coarse.extend(b.tolist())
            rows[video] = dict(dense_examples=len(dense), coarse_examples=len(coarse),
                dense_entropy=float(entropy(dense).mean()), coarse_entropy=float(entropy(coarse).mean()))
        prepared[pair] = rows
    conditions = {}
    widths = Counter()
    total_fits = prefixes = continuations = 0
    maximum_entropy_error = maximum_risk = 0.
    for seed in SEEDS:
        for pair in PAIRS:
            root = OUT/pair/f'seed{seed}'
            done = json.loads((root/'completion.json').read_text())
            assert done['GT_open_attempts'] == 0 and not done['target_fit_failures']
            assert {r['method'] for r in done['methods']} == set(METHODS)
            assert [r['video'] for r in done['receipts']] == list(prepared[pair])
            fit_seconds = Counter()
            own_widths = Counter()
            for i, receipt in enumerate(done['receipts']):
                assert receipt['GT_open_attempts'] == 0 and receipt['current_and_past_only']
                assert receipt['all_confidence_weights_one'] and not receipt['soft_target_clipping']
                assert set(receipt['fits']) == set(ARMS)
                prefixes += 1
                target = prepared[pair][receipt['video']]
                for arm, fit in receipt['fits'].items():
                    coarse = arm in COARSE_ARMS
                    label = 'coarse' if coarse else 'dense'
                    hard = arm.endswith('_HARD')
                    recipe = 'FRAME_'+arm.rsplit('_', 1)[1] if coarse else arm.removeprefix('STD_')
                    assert fit['recipe'] == recipe and fit['kind'] == ('flat' if 'FLAT' in arm else 'local')
                    assert fit['confidence_weights_all_one'] and fit['soft_targets_not_clipped']
                    assert not fit['balanced_exposure'] and fit['profile_weight'] == 0 and fit['hard_targets'] == hard
                    assert fit['minimum_steps'] == (4000 if i == 0 else 400)
                    assert fit['steps'] >= fit['minimum_steps']
                    assert fit['teacher_examples'] == target[label+'_examples']
                    expected = 0. if hard else target[label+'_entropy']
                    error = abs(fit['metrics']['target_entropy']-expected)
                    assert error < 1e-10, (pair, seed, receipt['video'], arm, error)
                    maximum_entropy_error = max(maximum_entropy_error, error)
                    risk = fit['metrics']['objective_excess']
                    assert fit['target_fit_gate_pass'] and risk <= .005 and fit['metrics']['target_KL'] <= .005
                    assert fit['final_CE_recomputed_from_actual_float32_deployed_head']
                    assert fit['target_GT_or_HOTA_used'] is False
                    maximum_risk = max(maximum_risk, risk)
                    total_fits += 1
                    widths[str(fit['width'])] += 1
                    own_widths[str(fit['width'])] += 1
                    continuations += int(bool(fit['teacher_risk_continuations']))
                    fit_seconds[arm] += fit['wall_seconds']
            conditions[pair+f'_seed{seed}'] = dict(video_prefixes=len(done['receipts']),
                fits=len(done['receipts'])*len(ARMS), widths=dict(own_widths), fit_wall_seconds=dict(fit_seconds))
    assert not guard and prefixes == 444 and total_fits == 2664
    report = dict(status='PASS_ALL18_STANDARD_RAW_UNIT_KD_ADEQUATE_AND_INDEPENDENT_TARGET_ENTROPIES',
        conditions=conditions, video_prefixes=prefixes, fit_receipts=total_fits, widths=dict(widths),
        all_original_teacher_risk_gates_passed=True, maximum_objective_excess=maximum_risk,
        original_risk_rule_not_weakened=True, independent_prefix_target_cardinalities_all_match=True,
        maximum_independent_target_entropy_error=maximum_entropy_error, entropy_tolerance=1e-10,
        raw_answers_rebuilt_once_per_pair_reused_across_neural_seeds=True,
        fits_with_teacher_risk_continuations=continuations, no_current_quality_or_GT_read=True,
        GT_open_attempts=0, new_fits_or_teacher_backbone_calls=0, not_a_cold_deployment_cost=True,
        CPU_wall_seconds=time.perf_counter()-start, goal_status='ACTIVE')
    atomic_json(OUT/'fit_adequacy_audit.json', report)
    public = PUBLIC/'conditional_standard_kd_fit_adequacy.json'
    public.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps({k:report[k] for k in ('status', 'video_prefixes', 'fit_receipts', 'widths',
        'maximum_objective_excess', 'maximum_independent_target_entropy_error',
        'fits_with_teacher_risk_continuations', 'CPU_wall_seconds')}))


if __name__ == '__main__':
    main()
