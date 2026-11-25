"""Schedule original native counts for newly completed settings, without GT."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_native_completed_counts_early import generate
from scripts import a100_native_binary_six_setting as native


def count_and_check_reuse(pair):
    result = generate(pair)
    path = Path(result['completion'])
    before_sha, before_mtime = file_sha(path), path.stat().st_mtime_ns
    begin = time.perf_counter()
    native.counts(pair)
    assert file_sha(path) == before_sha and path.stat().st_mtime_ns == before_mtime
    result.update(original_count_function_reuse_verified=True,
                  reuse_wall_seconds=time.perf_counter()-begin,
                  completion_hash_and_mtime_unchanged_on_reuse=True)
    return result


def main(pairs):
    native.registration()
    assert len(pairs) == len(set(pairs))
    remaining = [p for p in native.PAIRS if not (native.OUT/p/'seed0/completion.json').is_file()]
    assert remaining and not (native.OUT/'all_predictions_sealed_before_GT.json').exists()
    for pair in pairs:
        root = native.OUT/pair/'seed0'
        assert (root/'completion.json').is_file()
        assert not (root/'countmatched/completion.json').exists(), 'Do not rerun existing counts'
    tag = '_'.join(pairs)
    supervisor = native.OUT.parent.parent/'supervisor'
    plan = dict(scope='Only original registered counts, scheduled ahead of whole-study barrier',
                pairs=pairs, remaining_raw_settings=remaining, CPU_workers=min(2, len(pairs)),
                original_registration=str(native.PLAN), registration_sha256=file_sha(native.PLAN),
                entry_sha256=file_sha(Path(__file__)), original_count_function=native.counts.__module__+'.counts',
                helper_sha256=file_sha(ROOT/'scripts/a100_native_completed_counts_early.py'),
                new_models_fits_teacher_rankings_neural_seeds_or_GT=0, goal_status='ACTIVE')
    atomic_json(supervisor/('native_ready_counts_'+tag+'_plan_20261125.json'), plan)
    print(json.dumps(dict(status='GENERATING_EXISTING_REGISTERED_COUNTS_NO_GT', **plan)), flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(2, len(pairs))) as pool:
        for result in pool.map(count_and_check_reuse, pairs):
            results.append(result)
            print(json.dumps(dict(status='ORIGINAL_COUNTS_READY_AND_REUSE_VERIFIED', **result)), flush=True)
    native.registration()
    receipt = dict(status='PASS_EXISTING_NATIVE_READY_COUNTS_AND_ORIGINAL_REUSE',
                   results=results, CountMatched_methods=80*len(pairs), CountSelf_aliases=41*len(pairs),
                   GT_open_attempts=0, new_models_fits_teacher_rankings_or_neural_seeds=0,
                   original_all_output_seal_official_controller_and_sources_unchanged=True,
                   diagnostic_scheduling_cost_not_single_method_deployment_cost=True,
                   original_registration_sha256=file_sha(native.PLAN), entry_sha256=file_sha(Path(__file__)),
                   goal_status='ACTIVE')
    atomic_json(supervisor/('native_ready_counts_'+tag+'_receipt_20261125.json'), receipt)
    atomic_json(native.PUBLIC/('native_ready_counts_'+tag+'_receipt_20261125.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', choices=tuple(native.PAIRS), required=True)
    main(parser.parse_args().pairs)
