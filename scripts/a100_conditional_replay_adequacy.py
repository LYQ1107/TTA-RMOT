"""Audit all completed precision replicas without reading GT or HOTA.

Target-risk sufficiency, original target counts/entropy, normalization-row
counts and actual output drift are separate checks. None certifies semantic
accuracy, and old-to-new FULL changes are not pure quantization effects.
"""
import collections
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, generation_guard
from scripts.a100_six_direction import SEEDS, PUBLIC
from scripts.a100_conditional_logit_precision import OUT, PARENT, COARSE, COARSE_ARMS


def main():
    start = time.perf_counter()
    guard = generation_guard()
    conditions = {}
    widths = collections.Counter()
    gates = prefixes = 0
    maxrisk = 0.
    for seed in SEEDS:
        for pair in PAIRS:
            new = json.loads((OUT/pair/f'seed{seed}'/'completion.json').read_text())
            old = json.loads((PARENT/pair/f'seed{seed}'/'completion.json').read_text())
            coarse = json.loads((COARSE/pair/f'seed{seed}'/'completion.json').read_text())
            assert new['GT_open_attempts'] == old['GT_open_attempts'] == coarse['GT_open_attempts'] == 0
            assert not new['target_fit_failures'] and not new['reused_generic_target_fit_failures']
            assert [r['video'] for r in new['receipts']] == [r['video'] for r in old['receipts']] == [r['video'] for r in coarse['receipts']]
            dense, mean = {}, {}
            entropymax, example_mismatch, row_mismatch = 0., 0, 0
            for current, original, mean_original in zip(new['receipts'], old['receipts'], coarse['receipts']):
                assert current['GT_open_attempts'] == 0 and current['current_and_past_only']
                row_mismatch += int(current['state_normalization_rows'] != original['normalization_rows'])
                prefixes += 1
                for arm, fit in current['fits'].items():
                    previous = mean_original['fits'][arm.rsplit('_', 1)[1]] if arm in COARSE_ARMS else original['fits'][arm]
                    assert fit['target_fit_gate_pass'] and fit['target_GT_or_HOTA_used'] is False
                    assert fit['steps'] >= fit['minimum_steps']
                    gates += 1
                    widths[str(fit['width'])] += 1
                    maxrisk = max(maxrisk, fit['metrics']['objective_excess'])
                    entropymax = max(entropymax, abs(fit['metrics']['target_entropy']-previous['metrics']['target_entropy']))
                    example_mismatch += int(fit['teacher_examples'] != previous['teacher_examples'])
                for key, dest in (('original_probability_replica_audit', dense), ('coarse_probability_replica_audit', mean)):
                    for arm, values in current[key].items():
                        row = dest.setdefault(arm, dict(max_probability_difference_from_original=0., changed_original_raw_decisions=0,
                            replay_rounded_vs_original_rounded_decisions=0, logit_vs_own_rounded_probability_decisions=0))
                        row['max_probability_difference_from_original'] = max(row['max_probability_difference_from_original'], values['max_probability_difference_from_original'])
                        for field in ('changed_original_raw_decisions', 'replay_rounded_vs_original_rounded_decisions', 'logit_vs_own_rounded_probability_decisions'):
                            row[field] += values[field]
            assert entropymax == 0 and example_mismatch == 0 and row_mismatch == 0, (pair, seed, entropymax, example_mismatch, row_mismatch)
            conditions[pair+f'_seed{seed}'] = dict(video_prefixes=len(new['receipts']), all_teacher_target_entropies_exact=True,
                all_teacher_example_counts_equal=True, normalization_prefix_row_counts_equal=True,
                dense_candidate_query_rows=dense, coarse_query_track_scores=mean)
    assert not guard and gates == 3108 and prefixes == 444, (len(guard), gates, prefixes)
    report = dict(status='PASS_ALL18_PRECISION_REPLICA_FIT_GATES_AND_GT_FREE_DRIFT_AUDIT', conditions=conditions,
        video_prefixes=prefixes, fit_receipts=gates, all_original_teacher_risk_gates_passed=True, widths=dict(widths),
        maximum_objective_excess=maxrisk, no_GT_or_current_HOTA_read=True, new_fits_or_teacher_backbone_calls=0,
        all_coarse_and_dense_teacher_entropy_example_counts_and_normalization_row_counts_equal=True,
        drift_and_precision_counts_not_additive=True, not_a_quality_or_cold_cost_result=True,
        CPU_wall_seconds=time.perf_counter()-start, goal_status='ACTIVE')
    private = ROOT/'a100_runs/visual_classical_20261121/supervisor/completed_all18_logit_fit_drift_audit_20261123.json'
    atomic_json(private, report)
    public = PUBLIC/'conditional_logit_precision_fit_drift_allseeds.json'
    public.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=report['status'], video_prefixes=prefixes, fit_receipts=gates, widths=dict(widths),
        maximum_objective_excess=maxrisk,
        ordinary_exact_conditions=sum(all(v['max_probability_difference_from_original'] == 0
            for n, v in r['dense_candidate_query_rows'].items() if not n.startswith('FULL_')) for r in conditions.values()),
        FULL_refit_changes={p:r['dense_candidate_query_rows']['FULL_FINAL']['replay_rounded_vs_original_rounded_decisions'] for p, r in conditions.items()},
        public=str(public), CPU_wall_seconds=report['CPU_wall_seconds'])))


if __name__ == '__main__':
    main()
