"""Locate FULL refit drift relative to the actual paid observation support.

Compare each video's own saved original/replay probabilities. No final head
is applied to an earlier video, and no GT or current HOTA is consumed.
An asked TrackID is only a supplied candidate bag, not a verified identity.
"""
import argparse
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
from scripts.a100_conditional_six_setting import OUT as ORIGINAL
from scripts.a100_conditional_logit_precision import OUT as REPLAY
from scripts.a100_six_direction import PUBLIC

CATEGORIES = ('PAID_NOMINAL_ANCHOR', 'OTHER_ROW_IN_ASKED_BAG', 'UNASKED_BAG')


def empty():
    return dict(rows=0, original_yes=0, replay_yes=0, flipped_to_yes=0, flipped_to_no=0,
        probability_absolute_difference_sum=0., maximum_absolute_probability_difference=0.)


def analyze(seeds):
    guard = generation_guard()
    started = time.perf_counter()
    conditions = {}
    for seed in seeds:
        for pair, (_, target) in PAIRS.items():
            old_root, new_root = ORIGINAL/pair/f'seed{seed}', REPLAY/pair/f'seed{seed}'
            old_done = json.loads((old_root/'completion.json').read_text())
            new_done = json.loads((new_root/'completion.json').read_text())
            assert old_done['GT_open_attempts'] == new_done['GT_open_attempts'] == 0
            md = metadata(target)
            aggregate = {name:empty() for name in CATEGORIES}
            verified_changes = 0
            for receipt in new_done['receipts']:
                video = receipt['video']
                obs = np.asarray(md['videos'][video]['observations'])
                queries = [q for q in md['episodes'] if q['video_id'] == video]
                positions = {tuple(map(int, row)):i for i, row in enumerate(obs)}
                doc = json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
                assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
                asked = {}
                anchors = set()
                for item in doc['answers']:
                    qi, track, frame = int(item['local_query']), int(item['track_id']), int(item['frame_id'])
                    assert queries[qi]['expression_id'] == item['query_id']
                    asked.setdefault(qi, set()).add(track)
                    anchors.add((qi, positions[frame, track]))
                category = np.full((len(queries), len(obs)), 2, dtype='uint8')
                for qi, tracks in asked.items():
                    category[qi, np.isin(obs[:, 1], list(tracks))] = 1
                for qi, row in anchors:
                    category[qi, row] = 0
                old = np.load(old_root/'videos'/video/'FULL_FINAL_scores.npy', mmap_mode='r')
                new = np.load(new_root/'videos'/video/'FULL_FINAL_scores.npy', mmap_mode='r')
                assert old.shape == new.shape == category.shape
                video_changes = 0
                for index, name in enumerate(CATEGORIES):
                    x, y = np.asarray(old[category == index]), np.asarray(new[category == index])
                    if not len(x):
                        continue
                    difference = np.abs(x.astype(float)-y)
                    a, b = x > .5, y > .5
                    row = aggregate[name]
                    row['rows'] += len(x)
                    row['original_yes'] += int(a.sum())
                    row['replay_yes'] += int(b.sum())
                    row['flipped_to_yes'] += int((~a & b).sum())
                    row['flipped_to_no'] += int((a & ~b).sum())
                    row['probability_absolute_difference_sum'] += float(difference.sum())
                    row['maximum_absolute_probability_difference'] = max(row['maximum_absolute_probability_difference'], float(difference.max()))
                    video_changes += int((a != b).sum())
                expected = receipt['original_probability_replica_audit']['FULL_FINAL']['replay_rounded_vs_original_rounded_decisions']
                assert video_changes == expected, (pair, seed, video, video_changes, expected)
                verified_changes += video_changes
            assert sum(r['rows'] for r in aggregate.values()) == md['candidate_query_rows']
            for row in aggregate.values():
                row['changed_decisions'] = row['flipped_to_yes']+row['flipped_to_no']
                row['changed_fraction'] = row['changed_decisions']/row['rows'] if row['rows'] else None
                row['mean_absolute_probability_difference'] = row['probability_absolute_difference_sum']/row['rows'] if row['rows'] else None
            conditions[pair+f'_seed{seed}'] = dict(categories=aggregate, verified_total_changed_decisions=verified_changes)
    assert not guard
    report = dict(status='FULL_REFIT_DRIFT_BY_ACTUAL_EVIDENCE_SUPPORT_NO_GT', conditions=conditions, seeds=seeds,
        same_actual_per_video_prefix_outputs=True, no_future_head_replay=True,
        rows_are_correlated_no_significance_or_CI_claim=True,
        nominal_anchor_CLIP_assignment_is_shared_approximation=True,
        no_HOTA_error_rate_or_teacher_accuracy_inference=True, new_fits_or_teacher_backbone_calls=0,
        current_study_quality_or_GT_read=False, GT_open_attempts=0,
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    name = 'conditional_evidence_stability_seed'+'_'.join(map(str, seeds))+'.json'
    atomic_json(REPLAY/name, report)
    (PUBLIC/name).write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=report['status'], conditions=conditions,
        wall_seconds=report['wall_seconds'], public=str(PUBLIC/name)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    assert len(set(args.seeds)) == len(args.seeds) and set(args.seeds) <= {0, 17, 42}
    analyze(args.seeds)
