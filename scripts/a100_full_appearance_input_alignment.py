"""Verify full descriptor coverage and exact historical augmentation reuse.

This dependency task waits for the real collector's completion. It does
not monitor GPU jobs or repeat extraction, training, teacher or evaluation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_dino_fullsix_collect import OUT as FULL, PLAN as FULL_PLAN

OUT = FULL.parent/'full_appearance_input_alignment_20261125'
PLAN = OUT/'registration.json'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
SOURCES = ('scripts/a100_full_appearance_input_alignment.py',
           'flex_uda/active/full_appearance_learning_inputs.py',
           'flex_uda/active/appearance_learning_inputs.py',
           'scripts/a100_dino_fullsix_collect.py', 'scripts/a100_dino_appearance_collect.py',
           'flex_uda/a100/mechanism.py', 'flex_uda/a100/runtime.py')


def prepare():
    assert not PLAN.exists()
    report = dict(status='REGISTERED_FULL_INPUT_ALIGNMENT_AFTER_COLLECTOR',
        source_sha256={n: file_sha(ROOT/n) for n in SOURCES},
        input_plan_sha256=file_sha(FULL_PLAN),
        coverage='All74 physical candidate videos, not only the old44 subset videos.',
        exact_reuse='For all44 old units, every old3844D anchor/current/coarse descriptor is compared byte-for-byte through the new full bank.',
        no_query_expanded_full_feature_store=True, new_teacher_or_training_RMOT_predictions_GT_reads=0,
        preserves_live_collector_and_all_old_numerical_helpers=True, goal_status='ACTIVE')
    atomic_json(PLAN, report)
    atomic_json(PUBLIC/'full_appearance_input_alignment_registration_20261125.json', report)
    print(json.dumps(report), flush=True)


def main():
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard, metadata
    from flex_uda.active.full_appearance_learning_inputs import bank as full_bank
    from flex_uda.active.appearance_learning_inputs import bank as old_bank, OUT as OLD, PLAN as OLD_PLAN
    plan = json.loads(PLAN.read_text()); guard = generation_guard()
    for n, sha in plan['source_sha256'].items(): assert file_sha(ROOT/n) == sha, n
    assert file_sha(FULL_PLAN) == plan['input_plan_sha256']
    print('WAITING_REAL_COLLECTOR_COMPLETION_FOR_INPUT_ALIGNMENT_NO_GT', flush=True)
    waited = time.perf_counter()
    while not (FULL/'completion.json').exists(): time.sleep(15)
    wait_wall = time.perf_counter()-waited; started = time.perf_counter()
    completed = json.loads((FULL/'completion.json').read_text())
    assert completed['status'] == 'COMPLETE_SHARED_APPEARANCE_ALL74_PHYSICAL_VIDEOS'
    old_units = {(u['target'], u['video']): u for u in json.loads(OLD_PLAN.read_text())['units']}
    inputs = json.loads(FULL_PLAN.read_text()); reports = []; reuse_rows = 0
    for unit in inputs['units']:
        target, video = unit['target'], unit['video']; folder = FULL/target/video
        full = full_bank(target, video)
        obs = np.load(folder/'observations.npy')
        assert np.array_equal(obs, np.asarray(metadata(target)['videos'][video]['observations']))
        # Exercise every descriptor, including new observations, in bounded
        # batches. This avoids a many-terabyte query-expanded dense tensor.
        for begin in range(0, len(obs), 512):
            rows = obs[begin:begin+512]
            value = full.rows(rows[:, 0], rows[:, 1])
            assert value.shape == (len(rows), 3844) and np.isfinite(value).all()
        reused = 0
        if (target, video) in old_units:
            old = old_bank(target, video); old_obs = np.load(OLD/target/video/'observations.npy')
            for begin in range(0, len(old_obs), 512):
                rows = old_obs[begin:begin+512]
                assert np.array_equal(old.rows(rows[:, 0], rows[:, 1]), full.rows(rows[:, 0], rows[:, 1]))
            reused = len(old_obs); reuse_rows += reused
            # Historical lru_cache is unbounded. Release it after this unit;
            # no cached data in another running process is affected.
            old_bank.cache_clear()
        reports.append(dict(target=target, video=video, full_rows=len(obs),
                            exact_historical_3844D_rows=reused))
        print(json.dumps(dict(status='FULL_VISUAL_INPUT_ALIGNMENT_UNIT_COMPLETE', **reports[-1])), flush=True)
    assert len(reports) == 74 and reuse_rows == 17399 and not guard
    result = dict(status='PASS_FULL74_VISUAL_INPUT_COVERAGE_AND_EXACT_OLD_AUGMENTATION',
        units=reports, full_rows=sum(r['full_rows'] for r in reports),
        exact_old_3844D_rows=reuse_rows, all_descriptor_coordinates_equal_on_reused_rows=True,
        collection_wait_wall_seconds=wait_wall, alignment_wall_seconds=time.perf_counter()-started,
        registration_sha256=file_sha(PLAN), completion_sha256=file_sha(FULL/'completion.json'),
        GT_open_attempts=0, new_models_teacher_calls_RMOT_predictions=0,
        same_bank_required_for_all_future_ordinary=True, no_main_method_quality_claim=True,
        goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', result)
    atomic_json(PUBLIC/'full_appearance_input_alignment_completion_20261125.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'units'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else: main()
