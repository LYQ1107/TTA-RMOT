"""Audit all registered unrounded-logit count controls, without GT/HOTA.

Exact cutoff ties and CountSelf are execution diagnostics. Their frequencies
are neither semantic error rates nor quality gains; replayed FULL counts
also differ from the old probability cohort because its refit drifted.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_conditional_logit_precision import OUT, METHODS, BASELINES, PRIMARY, conditions
from scripts.a100_six_direction import PUBLIC


def main():
    guard = generation_guard()
    started = time.perf_counter()
    marker = json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())
    assert marker['methods'] == 1116 and marker['GT_open_attempts'] == 0
    assert marker['required_parent_studies'] == [1692, 378, 162]
    pairs = [(PRIMARY, PRIMARY)]+[p for b in BASELINES for p in ((b, PRIMARY), (PRIMARY, b), (b, b))]
    methods = [rank+'__AT_'+donor+'_COUNT' for rank, donor in pairs]
    assert len(methods) == len(set(methods)) == 46
    rows, tie_matrix, cutoff_matrix = [], [], []
    aggregate = {name:dict(ties=0, nontrivial_cutoffs=0) for name in methods}
    checked_raw = checked_counts = checked_self = 0
    largest = 0.
    nonzero_cross = []
    for record in conditions():
        root = Path(record['folder'])
        raw = json.loads((root/'completion.json').read_text())
        count = json.loads((root/'countmatched/completion.json').read_text())
        assert raw['GT_open_attempts'] == count['GT_open_attempts'] == 0
        assert {x['method'] for x in raw['methods']} == set(METHODS)
        assert {x['method'] for x in count['methods']} == set(methods)
        changed = count['CountSelf_changed_query_files']
        assert set(changed) == set(METHODS) and not any(changed.values()), changed
        ties = count['actual_exact_rank_cutoff_ties']
        cutoffs = count['nontrivial_cutoffs']
        assert set(ties) <= set(methods) and set(cutoffs) <= set(methods)
        slug = record['pair']+f"_seed{record['seed']}"
        tv, cv = [], []
        for (rank, donor), name in zip(pairs, methods):
            t, n = int(ties.get(name, 0)), int(cutoffs.get(name, 0))
            assert 0 <= t <= n, (slug, name, t, n)
            if rank == donor:
                assert t == 0, (slug, name, 'Own threshold should not split an equal-score group')
            elif t:
                rate = t/n
                largest = max(largest, rate)
                nonzero_cross.append(dict(condition=slug, projection=name, tied_cutoffs=t,
                    nontrivial_cutoffs=n, fraction=rate))
            aggregate[name]['ties'] += t
            aggregate[name]['nontrivial_cutoffs'] += n
            tv.append(t)
            cv.append(n)
        rows.append(slug)
        tie_matrix.append(tv)
        cutoff_matrix.append(cv)
        checked_raw += len(raw['methods'])
        checked_counts += len(count['methods'])
        checked_self += len(changed)
    assert len(rows) == 18 and checked_raw == checked_self == 288 and checked_counts == 828
    assert not guard
    report = dict(status='PASS_ALL1116_UNQUANTIZED_LOGIT_COUNT_RECEIPTS_AUDITED_NO_GT',
        condition_order=rows, projection_order=methods,
        exact_cutoff_tie_counts=tie_matrix, nontrivial_cutoff_counts=cutoff_matrix,
        checked_raw_outputs=checked_raw, checked_CountMatched_outputs=checked_counts,
        checked_CountSelf_projections=checked_self, all_CountSelf_changed_query_files_zero=True,
        all_own_count_cutoff_ties_zero=True, aggregate_projection_counts=aggregate,
        nonzero_cross_projection_conditions=nonzero_cross,
        maximum_cross_projection_tie_fraction=largest,
        audit_target_GT_or_current_HOTA_read=False, audit_GT_open_attempts=0,
        complete_study_is_already_in_isolated_GT_evaluation=True,
        correlated_operations_not_independent_samples=True,
        old_new_FULL_operating_counts_not_assumed_equal=True,
        not_a_pure_fixed_operating_point_tie_reduction_or_HOTA_gain=True,
        no_model_fits_teacher_or_backbone_calls=True,
        prior_probability_tie_reference='conditional_rank_precision.json',
        CPU_wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(OUT/'count_receipt_audit.json', report)
    public = PUBLIC/'conditional_logit_count_audit.json'
    public.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=report['status'], conditions=len(rows), raw=checked_raw,
        CountMatched=checked_counts, CountSelf=checked_self,
        nonzero_cross_projection_conditions=len(nonzero_cross), maximum_cross_tie_fraction=largest,
        total_cross_ties=sum(x['tied_cutoffs'] for x in nonzero_cross),
        public=str(public), CPU_wall_seconds=report['CPU_wall_seconds'])))


if __name__ == '__main__':
    main()
