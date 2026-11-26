"""Build matched, same-event teacher-only inputs for a new joint-noise main.

Uses each case's first-video legacy answers and100 three-view calibration
events from the SAME physical target collection. Every future comparator
must receive this identical transductive auxiliary information. No GT.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import yaml
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard, history, OUT as BANK
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.teacher.decompose_inputs import CLIP_PROMPT, FRAME_PROMPT
    from scripts.a100_reciprocal_memory_pilot import OUT as PARENT

    guard = generation_guard()
    use_verified_local_features()
    plan = json.loads((PARENT/'registration.json').read_text())
    out = PARENT.parent/'joint_teacher_noise_primary_20261125'
    prompts = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    bank_path = BANK/'evidence/responses.jsonl'
    bank_receipt = json.loads((BANK/'evidence/completion.json').read_text())
    assert file_sha(bank_path) == bank_receipt['responses_sha256']
    bank = [json.loads(s) for s in bank_path.read_text().splitlines()]
    teacher_sha = bank_receipt['teacher']['teacher_weight_sha256']
    views = ('CROP', 'CLIP', 'FRAME_BOX')
    summary = []
    for case in plan['cases']:
        folder = out/case['pair']
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'completion.json').exists():
            summary.append(json.loads((folder/'completion.json').read_text()))
            continue
        path = history(case['pair'])/'videos'/case['video']/'questions.json'
        legacy = json.loads(path.read_text())
        auxiliary = [r for r in bank if r['target'] == case['target']]
        assert len(auxiliary) == 300
        groups = defaultdict(list)
        for row in legacy+auxiliary:
            assert row['frozen_weight_sha256'] == teacher_sha
            view = row['effective_view']
            text = row['query_text']
            prompt = row['input']['prompt']
            if view == 'CROP':
                # Verify the full prompt without loading images or the teacher.
                pattern = re.escape(prompts['official_single'])
                pattern = pattern.replace(re.escape('{coord}'), r'-?\d+(?:\.\d+)? -?\d+(?:\.\d+)?')
                pattern = pattern.replace(re.escape('{sentence}'), lambda_placeholder := '__LITERAL_QUERY__')
                before, after = pattern.split(lambda_placeholder)
                assert re.fullmatch(before+re.escape(text)+after, prompt), ('Unexpected CROP task', case['pair'], row['query_id'])
            else:
                expected = (CLIP_PROMPT if view == 'CLIP' else FRAME_PROMPT).format(expression=text)
                assert prompt == expected, ('Unexpected multiview task', case['pair'], view)
            assert row['input']['method'] == view
            key = row['video_id'], row['query_id'], int(row['track_id']), int(row['frame_id'])
            groups[key].append(row)
        keys = sorted(groups)
        probability = np.full((len(keys), 3), np.nan, dtype='float64')
        response_counts = np.zeros((len(keys), 3), dtype='int32')
        for i, key in enumerate(keys):
            for j, view in enumerate(views):
                own = [r for r in groups[key] if r['effective_view'] == view]
                if own:
                    values = np.asarray([r['yes_probability'] for r in own])
                    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                    probability[i, j] = values.mean()
                    response_counts[i, j] = len(values)
        features = np.empty((len(keys), 5508), dtype='float32')
        video_counts = {}
        for video in sorted({key[0] for key in keys}):
            observation = JointObservationVideo(case['pair'], video)
            inputs = Inputs(observation)
            queries = {r['expression_id']: i for i, r in enumerate(observation.queries)}
            tracks = {int(track): i for i, track in enumerate(observation.tracks)}
            indices = [i for i, key in enumerate(keys) if key[0] == video]
            records = []
            for i in indices:
                _, query, track, frame = keys[i]
                qi = queries[query]
                assert all(r['query_text'] == observation.queries[qi]['query_text'] for r in groups[keys[i]])
                assert (frame, track) in observation.positions
                records.append(dict(local_query=qi, local_track=tracks[track], frame_id=frame, track_id=track))
            arrays = inputs.arrays(records)
            features[indices] = np.column_stack((arrays['bags'].reshape(len(indices), -1), arrays['present'],
                                                arrays['current'], arrays['coarse']))
            video_counts[video] = len(indices)
        assert np.isfinite(features).all() and np.isfinite(probability).any(1).all()
        target_video = np.asarray([key[0] == case['video'] for key in keys])
        np.savez_compressed(folder/'teacher_inputs.npz', features=features, probability=probability,
                            response_counts=response_counts, target_video=target_video)
        atomic_json(folder/'row_keys.json', dict(keys=keys, views=views,
            repeated_same_event_view_responses_averaged_not_independent_votes=True))
        assert not guard
        receipt = dict(status='PREPARED_MATCHED_SAME_EVENT_JOINT_NOISE_INPUTS_NO_GT',
                       pair=case['pair'], target=case['target'], evaluated_video=case['video'],
                       unique_events=len(keys), main_video_events=int(target_video.sum()),
                       auxiliary_events=100, logical_source_responses=len(legacy)+len(auxiliary),
                       full_three_effective_view_events=int(np.sum(np.isfinite(probability).sum(1) == 3)),
                       input_dimension=5508, contributing_videos=video_counts,
                       legacy_sha256=file_sha(path), paired_bank_sha256=file_sha(bank_path),
                       teacher_weight_sha256=teacher_sha, data_sha256=file_sha(folder/'teacher_inputs.npz'),
                       row_keys_sha256=file_sha(folder/'row_keys.json'),
                       per_video_unlabeled_normalization=True, source_features_prepaid=True,
                       all_frozen_prompts_checked_against_current_exact_templates=True,
                       auxiliary_target_collection_transductive=True,
                       same_auxiliary_labels_and_features_must_be_given_to_all_baselines=True,
                       new_teacher_calls=0, GT_open_attempts=0, new_models_fits_or_predictions=0)
        atomic_json(folder/'completion.json', receipt)
        summary.append(receipt)
        print(json.dumps(receipt), flush=True)
    result = dict(status='COMPLETE_JOINT_NOISE_COMMON_INPUT_PREPARATION', cases=summary,
                  no_joint_model_or_HOTA_claim=True, no_new_teacher_GPU_calls=True,
                  all_target_data_DEV=True, goal_status='ACTIVE', source_sha256=file_sha(Path(__file__)))
    atomic_json(out/'data_preparation.json', result)
    print(json.dumps(dict(status=result['status'], cases=len(summary))), flush=True)


if __name__ == '__main__':
    main()
