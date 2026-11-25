"""Deterministic video-influence analysis after the full study completes.

No retraining, new seed, module ablation, or new teacher call. Recompute
the official sufficient-count aggregation with each video omitted in turn.
This is finite-cohort sensitivity, not a confidence interval or blind test.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_visual_six import OUT, PLAN, PUBLIC, STEM, PRIMARY, COMPARATORS, registration
from scripts.a100_semantic_pattern_teacher_statistics import value

OWN = OUT / 'leave_video_out_registration.json'


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    parent = registration()
    assert not OWN.exists() and not (OUT / 'quality_results.json').exists()
    plan = dict(status='PREREGISTERED_DETERMINISTIC_LEAVE_VIDEO_OUT_ANALYSIS',
        source_sha256=file_sha(__file__), parent_registration_sha256=file_sha(PLAN),
        video_units=len(parent['cases']), settings=6, controls=list(COMPARATORS),
        scope='After all4588 seal and186/180 reproduction, remove each whole video with all its query sequences. Recompute TPweightedAssA and TP/FP/FN totals, never average videoHOTAs.',
        contrasts='Same fixed primary versus all10controls, raw and both original CountMatched effects; no primary reselection or changed acceptance threshold.',
        outputs='Public full/min/max/positive counts for all60 comparisons; complete per-omitted-video values private.',
        interpretation='Exact finite-cohort influence; no probability interval, no IID assumption or heldoutgeneralization claim. Four KITTIvideos remain a small cohort.',
        new_training_seeds_GPU_teacher_fits_or_predictions=0, GT_reads=0,
        no_prepositive_module_ablation_matrix=True, all_data_DEV=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(OWN, plan)
    atomic_json(PUBLIC / (STEM + '_leave_video_out_registration.json'), plan)
    print(json.dumps(dict(status=plan['status'], video_units=len(parent['cases']), GT_reads=0)), flush=True)


def read_counts(folder):
    with (Path(folder) / 'results/pedestrian_detailed.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['seq'] != 'COMBINED']
    numerator, denominator = np.zeros(19), np.zeros(19)
    for row in rows:
        for i, threshold in enumerate(range(5, 100, 5)):
            tp = float(row['HOTA_TP___' + str(threshold)])
            numerator[i] += float(row['AssA___' + str(threshold)]) * tp
            denominator[i] += tp + float(row['HOTA_FP___' + str(threshold)]) + float(row['HOTA_FN___' + str(threshold)])
    return dict(numerator=numerator, denominator=denominator, episodes=len(rows))


def differences(values, other):
    return dict(raw=values[PRIMARY] - values[other],
        primary_at_ordinary_count=values[PRIMARY + '__AT_' + other + '_COUNT'] - values[other],
        ordinary_at_primary_count=values[PRIMARY] - values[other + '__AT_' + PRIMARY + '_COUNT'])


def main():
    parent = registration(); own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__)
    assert own['parent_registration_sha256'] == file_sha(PLAN)
    reproduced = json.loads((PUBLIC / (STEM + '_reproduction.json')).read_text())
    assert reproduced['status'] == 'PASS_ALL186_HOTA_AND180_EFFECTS'
    seal = json.loads((OUT / 'all4588_sealed_before_GT.json').read_text())
    assert seal['GT_open_attempts'] == 0 and len(seal['tasks']) == 4588
    assert seal['teacher_final_weight_seals'] == 4
    published = json.loads((PUBLIC / (STEM + '_statistics.json')).read_text())['records']
    full_analysis = json.loads((PUBLIC / (STEM + '_analysis.json')).read_text())
    methods = (PRIMARY, *COMPARATORS,
        *[PRIMARY + '__AT_' + c + '_COUNT' for c in COMPARATORS],
        *[c + '__AT_' + PRIMARY + '_COUNT' for c in COMPARATORS])
    counts = {}
    for task in seal['tasks']:
        key = task['pair'], task['method'], task['video']
        assert key not in counts
        counts[key] = read_counts(task['folder'])
    assert len(counts) == 4588
    detailed, summaries = [], []
    for pair in sorted({c['pair'] for c in parent['cases']}):
        videos = sorted(c['evaluated_video'] for c in parent['cases'] if c['pair'] == pair)
        assert len(videos) >= 2
        totals = {}
        for method in methods:
            n = sum((counts[pair, method, v]['numerator'] for v in videos), np.zeros(19))
            d = sum((counts[pair, method, v]['denominator'] for v in videos), np.zeros(19))
            totals[method] = dict(numerator=n, denominator=d)
            ref = published[pair + '/' + method]
            assert np.allclose(n, ref['numerator'], rtol=1e-12, atol=1e-6)
            assert np.array_equal(d, ref['denominator'])
            assert abs(value(totals[method]) - ref['HOTA']) < 1e-8
        omitted = []
        for video in videos:
            values = {m: value(dict(
                numerator=totals[m]['numerator'] - counts[pair, m, video]['numerator'],
                denominator=totals[m]['denominator'] - counts[pair, m, video]['denominator'])) for m in methods}
            omitted.append(dict(video=video, effects={c: differences(values, c) for c in COMPARATORS}))
        detailed.append(dict(pair=pair, omitted_videos=omitted))
        full_values = {m: value(totals[m]) for m in methods}
        for other in COMPARATORS:
            full = differences(full_values, other)
            summaries.append(dict(pair=pair, comparator=other, videos=len(videos),
                contrasts={key: dict(full_delta=delta,
                    minimum_leave_one_video_out_delta=min(r['effects'][other][key] for r in omitted),
                    maximum_leave_one_video_out_delta=max(r['effects'][other][key] for r in omitted),
                    positive_leave_one_video_out_count=sum(r['effects'][other][key] > 0 for r in omitted),
                    zero_leave_one_video_out_count=sum(r['effects'][other][key] == 0 for r in omitted))
                    for key, delta in full.items()}))
        print(json.dumps(dict(status='DETERMINISTIC_VIDEO_INFLUENCE_COMPLETE', pair=pair, videos=len(videos))), flush=True)
    assert len(summaries) == 60
    result = dict(status='COMPLETE_ALL6_DETERMINISTIC_LEAVE_VIDEO_OUT', summaries=summaries,
        parent_scientific_screen_unchanged=full_analysis['candidate_eligible_for_larger_screen'],
        source_sha256=file_sha(__file__), new_training_seeds_fits_teacher_GPU_or_predictions=0,
        finite_cohort_sensitivity_not_confidence_interval=True,
        no_mean_of_video_HOTA_no_changed_primary_or_criterion=True,
        all_data_DEV=True, goal_status='ACTIVE')
    atomic_json(OUT / 'leave_video_out_full_private.json', dict(details=detailed, **result))
    atomic_json(PUBLIC / (STEM + '_leave_video_out.json'), result)
    print(json.dumps(dict(status=result['status'], comparisons=60)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
