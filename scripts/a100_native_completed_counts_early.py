"""Generate existing native-study counts early for sealed C1/C6 inputs.

This changes scheduling only. The original controller later reuses the same
completion files through its existing counts() early return. No new method,
teacher, fit, prediction ranking, source patch, or GT evaluation is introduced.
"""
import concurrent.futures
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as native

PAIRS = ('C1', 'C6')
SUP = native.OUT.parent.parent / 'supervisor'
PLAN = SUP / 'native_completed_counts_early_plan_20261125.json'
RECEIPT = SUP / 'native_completed_counts_early_20261125.json'


def generate(pair):
    native.registration()
    root = native.OUT / pair / 'seed0'
    raw_path = root / 'completion.json'
    raw = json.loads(raw_path.read_text())
    assert raw['GT_open_attempts'] == 0 and len(raw['methods']) == 41
    before = file_sha(raw_path)
    existed = (root / 'countmatched/completion.json').is_file()
    begin = time.perf_counter()
    native.counts(pair)
    path = root / 'countmatched/completion.json'
    result = json.loads(path.read_text())
    assert file_sha(raw_path) == before
    assert len(result['methods']) == 2 * len(native.COMPARATORS) == 80
    assert result['GT_open_attempts'] == 0
    assert result['CountSelf_identical_masks_for_all_methods']
    assert len(result['CountSelf_aliases']) == len(native.METHODS) == 41
    return dict(pair=pair, completion=str(path), completion_sha256=file_sha(path),
                raw_completion_sha256=before, raw_completion_unchanged=True,
                CountMatched_methods=80, CountSelf_aliases=41,
                GT_open_attempts=0, existing_completion_reused=existed,
                wall_seconds=time.perf_counter()-begin)


def main():
    native.registration()
    for pair in PAIRS:
        root = native.OUT / pair / 'seed0'
        assert (root / 'completion.json').is_file()
    # Launch only while the original whole-study controller is still fitting;
    # it owns the final all-output barrier, official evaluation and analysis.
    remaining = [p for p in native.PAIRS if not (native.OUT/p/'seed0/completion.json').is_file()]
    assert remaining, 'Original all-raw barrier is ready; leave counts to its controller.'
    assert not (native.OUT / 'all_predictions_sealed_before_GT.json').exists()
    atomic_json(PLAN, dict(scope='existing native726 counts; scheduling only', pairs=list(PAIRS),
        original_registration=str(native.PLAN), registration_sha256=file_sha(native.PLAN),
        pinned_sources_unchanged=True, remaining_raw_settings=remaining,
        original_count_function='scripts.a100_native_binary_six_setting.counts',
        CPU_workers=2, new_neural_seeds=0, new_models_teacher_fits_or_rankings=0,
        official_evaluation_started_by_this_entry=False, expected_CountMatched_methods=160,
        expected_CountSelf_aliases=82, goal_status='ACTIVE'))
    print(json.dumps(dict(status='GENERATING_EXISTING_SEALED_NATIVE_COUNTS_EARLY_NO_GT',
                         pairs=PAIRS, CPU_workers=2)), flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(generate, pair): pair for pair in PAIRS}
        for future in concurrent.futures.as_completed(pending):
            result = future.result(); results.append(result)
            print(json.dumps(dict(status='SEALED_NATIVE_COUNT_PAIR_NO_GT', **result)), flush=True)
    native.registration()
    receipt = dict(status='PASS_EXISTING_NATIVE_C1_C6_COUNTS_READY_FOR_PARENT_REUSE',
        results=sorted(results, key=lambda r:r['pair']), CountMatched_methods=160,
        CountSelf_aliases=82, GT_open_attempts=0, new_models_teacher_fits_or_rankings=0,
        original_all_output_seal_and_official_controller_unchanged=True,
        execution_cost_is_quantity_diagnostic_not_method_deployment=True, goal_status='ACTIVE')
    atomic_json(RECEIPT, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
