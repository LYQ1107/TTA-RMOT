"""Check clip/anchor annotation disagreement after the complete main seal.

Do not choose a training target from the best-matching GT interpretation.
These are canonical candidate-matching labels, not pure teacher semantics.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_visual_constraint_primary import OUT, DATA, PUBLIC, STEM, registration


def confusion(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    return dict(TP=int((truth & prediction).sum()), FP=int((~truth & prediction).sum()),
                FN=int((truth & ~prediction).sum()), TN=int((~truth & ~prediction).sum()))


def main():
    plan = registration()
    assert len(json.loads((OUT/'all84_sealed_before_GT.json').read_text())['tasks']) == 84
    assert json.loads((PUBLIC/(STEM+'_reproduction.json')).read_text())['records'] == 84
    from flex_uda.a100.mechanism import metadata, history, OUT as BANK
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    auxiliary = [json.loads(line) for line in (BANK/'evidence/responses.jsonl').read_text().splitlines()]
    reports = []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video']
        md = metadata(case['target']); obs = np.asarray(md['videos'][video]['observations'], dtype=int)
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        qindex = {q['expression_id']: i for i, q in enumerate(queries)}
        position = {tuple(map(int, r)): i for i, r in enumerate(obs)}
        truth = np.load(BANK/'offline_GT'/CANONICAL[pair]/video/'labels.npy') >= 0
        assert truth.shape == (len(queries), len(obs))
        keys = json.loads((DATA/pair/'row_keys.json').read_text())['keys']
        with np.load(DATA/pair/'teacher_inputs.npz') as data: probability = data['probability']
        rows = json.loads((history(pair)/'videos'/video/'questions.json').read_text())
        rows += [r for r in auxiliary if r['target'] == case['target'] and r['video_id'] == video]
        by_event = defaultdict(list); unique = set(); raw_count = len(rows)
        for row in rows:
            uid = row['request_id']
            if uid in unique: continue
            unique.add(uid)
            key = (row['video_id'], row['query_id'], int(row['track_id']), int(row['frame_id']))
            by_event[key].append(row)
        counts = Counter(); labels = defaultdict(list); teacher = []
        for i, key in enumerate(keys):
            if key[0] != video or not np.isfinite(probability[i, 1]): continue
            counts['CLIP_events'] += 1
            event = by_event[tuple(key)]
            clips = [r for r in event if r['effective_view'] == 'CLIP']
            assert clips and all(r['query_text'] == queries[qindex[key[1]]]['query_text'] for r in clips)
            lists = [tuple(r.get('clip', {}).get('frames', ())) if r.get('clip') else () for r in clips]
            if not all(lists): counts['missing_frame_manifest'] += 1; continue
            if len(set(lists)) != 1: counts['different_clip_frame_manifests_same_event'] += 1; continue
            frames = lists[0]; counts['repeated_clip_frame_slots'] += len(frames)-len(set(frames))
            assert max(frames) <= int(key[3])
            if frames[-1] != int(key[3]): counts['last_frame_differs_from_requested_anchor'] += 1
            qi = qindex[key[1]]; track = int(key[2]); known = [f for f in frames if (int(f), track) in position]
            if len(known) != len(frames): counts['clip_frame_not_in_canonical_candidates'] += 1; continue
            history_labels = np.asarray([truth[qi, position[int(f), track]] for f in frames])
            current = bool(truth[qi, position[int(key[3]), track]])
            values = dict(anchor=current, any=bool(history_labels.any()), all=bool(history_labels.all()),
                          majority=bool(history_labels.mean() >= .5))
            for name, value in values.items(): labels[name].append(value)
            answer = bool(probability[i, 1] > .5); teacher.append(answer)
            counts['fully_mapped_events'] += 1
            counts['canonical_membership_varies_inside_clip'] += int(len(set(history_labels.tolist())) > 1)
            counts['anchor_vs_any_disagree'] += int(values['anchor'] != values['any'])
            counts['anchor_vs_all_disagree'] += int(values['anchor'] != values['all'])
            counts['anchor_vs_majority_disagree'] += int(values['anchor'] != values['majority'])
            counts['teacher_anchor_disagree'] += int(answer != current)
            counts['teacher_anchor_disagree_but_any_agrees'] += int(answer != current and answer == values['any'])
            counts['teacher_anchor_disagree_but_all_agrees'] += int(answer != current and answer == values['all'])
            counts['teacher_anchor_disagree_but_majority_agrees'] += int(answer != current and answer == values['majority'])
        main = [i for i, key in enumerate(keys) if key[0] == video]
        view_reports = {}
        for j, view in enumerate(('CROP', 'CLIP', 'FRAME_BOX')):
            selected = [i for i in main if np.isfinite(probability[i, j])]
            labels_now = [truth[qindex[keys[i][1]], position[int(keys[i][3]), int(keys[i][2])]] for i in selected]
            view_reports[view] = dict(events=len(selected), canonical_anchor_confusion=confusion(labels_now, probability[selected, j] > .5))
        report = dict(pair=pair, video=video, input_raw_answer_rows=raw_count, distinct_raw_requests=len(unique),
            clip_scope_counts=dict(counts), alternative_descriptive_confusions={name: confusion(value, teacher) for name, value in labels.items()},
            raw_views_against_nominal_anchor=view_reports)
        reports.append(report); print(json.dumps(report), flush=True)
    output = dict(status='COMPLETE_POSTSEAL_TEACHER_MEASUREMENT_SCOPE_DIAGNOSIS', cases=reports,
        exact_current_question_does_not_specify_a_unique_temporal_operator=True,
        alternative_scopes_not_true_teacher_labels_or_selected_training_targets=True,
        source_scope_variation_can_include_GT_visibility_matching_and_ID_mix=True,
        raw_views_cover_different_events_do_not_compare_their_rates_as_paired_accuracy=True,
        canonical_matching_not_pure_teacher_semantic_accuracy=True,
        no_new_teacher_fits_predictions_or_parameter_selection=True, all_data_DEV=True,
        source_sha256=file_sha(Path(__file__)), goal_status='ACTIVE')
    atomic_json(OUT/'teacher_measurement_scope_postseal.json', output)
    atomic_json(PUBLIC/'teacher_measurement_scope_postseal_20261125.json', output)


if __name__ == '__main__': main()
