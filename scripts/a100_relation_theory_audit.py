"""Check which identification claims apply to the frozen teacher relation fit.

This is a teacher-only diagnostic, not a new method or parameter search. It
reads the already frozen prefix rows, never quality predictions or manual GT.
The quadratic formula is reported only with its actual regime of validity.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard
from flex_uda.active.relational_transport import center_groups, fit
from scripts.a100_relation_transport_prepare import OUT as PREPARED
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX / 'relational_transport/identification_theory_audit'
REGISTRATION = PUBLIC / 'relation_identification_theory_registration.json'


def quadratic_reference(x, y, w, mu):
    gram = float(w @ (x * x))
    numerator = float(w @ (x * y))
    return max(0., numerator / (mu + gram)), gram


def describe(rows, expected):
    beta, receipt = fit(rows, mu=expected['mu'], delta=expected['Huber_delta'])
    assert abs(beta - expected['beta']) < 1e-10
    assert receipt['centered_examples'] == expected['centered_examples']
    x, y, w, counts = center_groups(rows)
    quadratic_beta, gram = quadratic_reference(x, y, w, receipt['mu'])
    assert abs(gram - expected['weighted_centered_Gram']) < 1e-10
    residual = beta * x - y
    delta = receipt['Huber_delta']
    inlier = np.abs(residual) < delta
    # Away from the Huber kinks, this is optimization curvature, not Fisher
    # information or an assertion that independent teacher errors exist.
    data_curvature = float(w @ (x * x * inlier))
    all_quadratic = bool(np.all(np.abs(residual) <= delta))
    if all_quadratic:
        assert abs(beta - quadratic_beta) < 1e-8
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['group'])].append(row)
    multi = sum(len(group) >= 2 for group in groups.values())
    return dict(**counts, beta=beta, mu=receipt['mu'], Huber_delta=delta,
        groups_with_multiple_unique_requests=multi,
        unexcited_groups=counts['total_groups'] - counts['excited_groups'],
        excited_group_fraction=(counts['excited_groups'] / counts['total_groups']
                                if counts['total_groups'] else None),
        weighted_centered_Gram=gram,
        quadratic_reference_beta=quadratic_beta,
        exact_quadratic_formula_applies=all_quadratic,
        Huber_vs_quadratic_beta_difference=beta - quadratic_beta,
        Huber_tail_rows=int((~inlier).sum()),
        Huber_tail_weight=float(w @ (~inlier)),
        total_excited_group_weight=float(w.sum()),
        local_data_curvature_away_from_kinks=data_curvature,
        local_total_curvature_away_from_kinks=receipt['mu'] + data_curvature,
        curvature_is_not_statistical_information=True,
        KKT_residual=receipt['KKT_residual'])


def main():
    started = time.perf_counter()
    registration = json.loads(REGISTRATION.read_text())
    assert registration['source_sha256'] == file_sha(Path(__file__))
    assert registration['relation_module_sha256'] == file_sha(
        ROOT / 'flex_uda/active/relational_transport.py')
    guard = generation_guard()
    prepared = json.loads((PREPARED / 'completion.json').read_text())
    assert prepared['status'] == 'TEACHER_ONLY_RELATION_PREFIX_FITS_PREPARED'
    results = {}
    for pair in PAIRS:
        rows = []
        prefixes = []
        for expected in prepared['pairs'][pair]['prefix_fits']:
            video = expected['video']
            rows.extend(json.loads((PREPARED / pair / video / 'teacher_rows.json').read_text())['rows'])
            assert len(rows) == expected['prefix_rows']
            prefixes.append(dict(video=video, prefix_rows=len(rows), **describe(rows, expected)))
        # Prefixes overlap; their counts must never be added as independent
        # datasets. The final prefix gives the nonduplicated setting summary.
        results[pair] = dict(source=PAIRS[pair][0], target=PAIRS[pair][1],
            prefix_count=len(prefixes), final_prefix=prefixes[-1],
            prefixes=prefixes if prepared['pairs'][pair]['eligible_queries'] else [],
            no_future_video_rows=True)
    assert not guard
    report = dict(status='TEACHER_ONLY_IDENTIFICATION_THEORY_REGIME_AUDITED', pairs=results,
        target_manual_GT_open_attempts=0, quality_predictions_read=0,
        new_teacher_encoder_GPU_or_neural_fit_calls=0,
        existing_scalar_fits_reproduced=True, existing_method_parameters_changed=False,
        prefix_counts_not_independent=True,
        interpretation='Nonzero within-group state variation is necessary to separate a slope from arbitrary stable group intercepts. Ridge gives uniqueness even without variation; it does not create identification. The quadratic closed form applies only when all Huber residuals are in its quadratic region.',
        limitation='No semantic correctness, HOTA gain, Fisher information, independent-error guarantee, new-method acceptance or isolated deployment cost follows from this audit.',
        theory_document='docs/TTA_RMOT_从经典思想到对象关系机制_理论与实验登记_20261122.txt',
        all_data_development=True, goal_status='ACTIVE',
        wall_seconds=time.perf_counter() - started)
    atomic_json(OUT / 'completion.json', report)
    (PUBLIC / 'relation_identification_theory_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps(dict(status=report['status'], pairs={p:r['final_prefix']
        for p,r in results.items()}, target_manual_GT_open_attempts=0,
        existing_method_parameters_changed=False), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
