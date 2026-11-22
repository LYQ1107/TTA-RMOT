"""Export sealed cold-run evidence without predictions, teacher answers or weights.

This is post-seal analysis: GT-derived official sufficient statistics are copied
for reproducibility, never passed back into adaptation or checkpoint selection.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_analyze import read_stats


RUNS = {
    'full': 'C4_fresh_full_stream_seed0',
    'key': 'C4_fresh_track_three_seed0',
    'fixed_QA': 'C4_sparse_fixedQA_seed0',
    'single_KD': 'C4_key_single_CLIP_KD_seed0',
    'single_cohort': 'C4_key_single_CLIP_COHORT_seed0',
}
RESULTS = (
    'fresh_whole_C4', 'fresh_whole_C4_sparse',
    'fresh_whole_C4_countmatched', 'fresh_whole_C4_sparse_countmatched',
    'fresh_sparse_fixedQA', 'independent_single_cold_C4', 'physical_full_vs_key_C4',
)


def main():
    destination = ROOT / 'docs/a100_visual_classical_20261121'
    sufficient = {}
    for name in RESULTS:
        value = json.loads((OUT / 'analysis' / (name + '.json')).read_text())
        atomic_json(destination / (name + '.json'), value)
    costs = {}
    for label, name in RUNS.items():
        root = OUT / 'cold_full_c4' / name
        receipt = json.loads((root / 'completion.json').read_text())
        assert receipt['GT_open_attempts'] == 0
        methods = [root / 'methods' / row['method'] for row in receipt['methods']]
        if label in ('full', 'key'):
            count_receipt = json.loads((root / 'countmatched/completion.json').read_text())
            assert count_receipt['GT_open_attempts'] == 0
            methods += [root / 'countmatched' / row['method'] for row in count_receipt['methods']]
        sufficient[label] = {}
        for folder in methods:
            stats = read_stats(folder)
            key = ('counts/' if folder.parent.name == 'countmatched' else '') + folder.name
            sufficient[label][key] = dict(
                parents=stats['parent_videos'],
                numerator=stats['parent_stats'][0].tolist(),
                denominator=stats['parent_stats'][1].tolist(), HOTA=stats['HOTA'])
        if label == 'fixed_QA':
            continue
        phases = json.loads((root / 'phase_ledger.json').read_text())
        totals = {}
        for row in phases['common_phases']:
            totals[row['phase']] = totals.get(row['phase'], 0.) + row['seconds']
        costs[label] = dict(
            protocol=receipt['protocol'], source=receipt['source'], teacher=receipt['teacher'],
            phase_totals_seconds=totals, cost_by_method=receipt['cost_by_method'],
            GT_open_attempts=receipt['GT_open_attempts'],
            prepaid_artifact_reads=receipt['prepaid_artifact_reads'],
            single_uninterrupted_attempt=receipt['single_uninterrupted_attempt'],
            single_method_execution=receipt.get('single_method_execution', False),
            external_process=json.loads((root / 'worker_process_receipt.json').read_text()),
            actual_internal_worker_seconds=receipt.get('actual_worker_wall_seconds', receipt.get('actual_multiarm_worker_wall_seconds')),
            source_run=name)
    atomic_json(destination / 'fresh_cold_C4_sufficient_statistics.json', sufficient)
    atomic_json(destination / 'fresh_cold_C4_cost_receipts.json', dict(
        runs=costs, frozen_candidates_provided=True,
        upstream_detection_and_tracking_generation_not_remeasured=True,
        GT_evaluation_excluded_from_deployment_cost=True,
        common_prefix_component_estimates_distinct_from_independent_process_walls=True,
        all_development_data=True))
    print(json.dumps(dict(status='SEALED_COLD_EVIDENCE_EXPORTED', runs=list(RUNS),
                          results=list(RESULTS), directory=str(destination))))


if __name__ == '__main__':
    main()
