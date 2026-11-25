"""GT-free exact-input feasibility audit; no fitting or quality evaluation.

Different teacher views can share the deployed student's input. The weighted
conditional mean is the optimum of their common Bernoulli cross entropy.
The resulting Jensen gap is irreducible for any deterministic student with
that input, even one with unlimited capacity. This audit does not choose a
model, checkpoint, or target-domain hyperparameter.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.residual_rsi_learning import data, paired_training_table
from scripts.a100_residual_rsi_primary import OUT, PLAN, PUBLIC


def entropy(y):
    y = np.asarray(y, dtype=np.float64)
    return -y * np.log(np.clip(y, 1e-15, 1)) - (1-y) * np.log(np.clip(1-y, 1e-15, 1))


def conditional_floor(x, y, weight):
    """Exact bytes only; no similarity clustering, GT, or label editing."""
    x = np.ascontiguousarray(x)
    # Hash buckets are verified against complete rows before declaring equality.
    buckets, indices, exemplars = {}, [], []
    for row in x:
        digest = hashlib.sha256(row.tobytes()).digest()
        group = buckets.get(digest)
        if group is None:
            group = len(exemplars)
            buckets[digest] = group
            exemplars.append(row)
        else:
            assert np.array_equal(row, exemplars[group])
        indices.append(group)
    group = np.asarray(indices, dtype=np.int64)
    y, weight = np.asarray(y, dtype=np.float64), np.asarray(weight, dtype=np.float64)
    mass = np.bincount(group, weights=weight)
    mean = np.bincount(group, weights=weight*y) / mass
    h = entropy(y)
    floor = float(np.sum(weight * (entropy(mean[group])-h)) / weight.sum())
    contradiction_groups = sum(np.ptp(y[group == k]) > 1e-7 for k in range(len(mass)))
    return dict(rows=len(x), unique_exact_inputs=len(mass),
                contradictory_exact_input_groups=int(contradiction_groups),
                irreducible_individual_teacher_KL=floor,
                maximum_target_span_on_identical_input=float(max(np.ptp(y[group == k]) for k in range(len(mass)))))


def main():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    plan = json.loads(PLAN.read_text())
    records = []
    for case in plan['cases']:
        features, query, doc, quotas, auxiliary, ay, _, _ = data(case)
        folder = OUT / case['pair']
        draws = json.loads((folder/'primary/draws.json').read_text())['draws']
        for mode in ('SOFT_KD', 'HARD_KD', 'IPW_KD', 'UNIFORM_SOFT_KD'):
            used = draws if mode != 'UNIFORM_SOFT_KD' else json.loads((folder/'uniform_draws.json').read_text())['draws']
            targets = (ay >= .5).astype(float) if mode == 'HARD_KD' else ay
            x, y, w = paired_training_table(features, query, used, quotas, auxiliary, targets, mode)
            row = dict(pair=case['pair'], mode=mode, **conditional_floor(x, y, w))
            records.append(row)
            print(json.dumps(row), flush=True)
        with np.load(folder/'primary/risk_stage_3.npz') as saved:
            x = np.r_[features, auxiliary]
            y = np.r_[saved['prequential_proxy_mean'], ay]
            w = np.r_[saved['population_weights'], np.full(len(ay), .25/len(ay))]/1.25
        records.append(dict(pair=case['pair'], mode='IMPUTED_KD', **conditional_floor(x, y, w)))
    assert not guard
    result = dict(status='COMPLETE_GT_FREE_EXACT_INPUT_FEASIBILITY_AUDIT',
                  original_registration_sha256=file_sha(PLAN), records=records,
                  GT_open_attempts=0, optimizer_steps=0, teacher_calls=0,
                  no_quality_read_or_hyperparameter_selection=True,
                  scope='An input-conditioned Bayes lower bound, not an achieved neural training error or semantic correctness.',
                  goal_status='ACTIVE')
    atomic_json(OUT/'exact_input_feasibility.json', result)
    atomic_json(PUBLIC/'residual_distillation_feasibility_20261125.json', result)


if __name__ == '__main__':
    main()
