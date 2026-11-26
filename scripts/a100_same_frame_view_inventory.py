"""Exact-frame multiview support inventory for the next joint-noise design.

This checks request identities, not answers, images or annotations. Different
frames cannot be silently pooled into one latent semantic observation.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def key(row):
    return (row['target'], row['video_id'], row['query_id'], int(row['track_id']), int(row['frame_id']))


def summary(records):
    groups = defaultdict(set)
    calls = Counter()
    kinds = Counter()
    for row in records:
        groups[key(row)].add(row['effective_view'])
        calls[row['effective_view']] += 1
        kinds[row.get('kind', 'PAIRED_BANK')] += 1
    return dict(logical_responses=len(records), exact_query_track_frame_groups=len(groups),
                groups_by_number_of_effective_views=dict(Counter(len(v) for v in groups.values())),
                effective_view_responses=dict(calls), question_kinds=dict(kinds),
                groups_with_all_three_effective_views=sum({'CROP', 'CLIP', 'FRAME_BOX'} <= v for v in groups.values()))


def main():
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard, metadata, history, PAIRS, OUT as BANK
    from scripts.a100_reciprocal_memory_pilot import OUT as PARENT

    guard = generation_guard()
    source = BANK/'evidence/responses.jsonl'
    completed = json.loads((BANK/'evidence/completion.json').read_text())
    assert file_sha(source) == completed['responses_sha256']
    bank = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(bank) == completed['responses']
    assert {r['frozen_weight_sha256'] for r in bank} == {completed['teacher']['teacher_weight_sha256']}
    cases = {c['pair']: c for c in json.loads((PARENT/'registration.json').read_text())['cases']}
    settings = {}
    input_hashes = {}
    for pair, (model, target) in PAIRS.items():
        legacy = []
        for video in sorted(metadata(target)['videos']):
            path = history(pair)/'videos'/video/'questions.json'
            rows = json.loads(path.read_text())
            input_hashes[str(path)] = file_sha(path)
            assert {r['frozen_weight_sha256'] for r in rows} == {completed['teacher']['teacher_weight_sha256']}
            legacy.extend(rows)
        auxiliary = [r for r in bank if r['target'] == target]
        row = dict(target=target, legacy=summary(legacy), paired_bank=summary(auxiliary),
                   co_location_union_only=summary(legacy+auxiliary))
        if pair in cases:
            video = cases[pair]['video']
            row['current_first_video'] = dict(video=video,
                legacy=summary([r for r in legacy if r['video_id'] == video]),
                paired_bank=summary([r for r in auxiliary if r['video_id'] == video]),
                co_location_union_only=summary([r for r in legacy+auxiliary if r['video_id'] == video]))
        settings[pair] = row
        print(json.dumps(dict(pair=pair, counts=row)), flush=True)
    assert not guard
    result = dict(status='COMPLETE_EXACT_FRAME_MULTIVIEW_METADATA_INVENTORY', settings=settings,
                   historical_inputs_sha256=input_hashes, paired_bank_sha256=file_sha(source),
                   no_teacher_probabilities_manual_GT_images_models_fits_or_predictions_used=True,
                   new_teacher_calls=0, new_neural_seeds=0,
                   target_only_bank_by_physical_target_not_by_source_checkpoint=True,
                   co_location_is_not_a_proof_of_matching_question_semantics_or_independent_errors=True,
                   prompt_task_equivalence_and_same_point_features_required_before_new_training=True,
                   no_parent_historical_sampling_seed_rerun=True, goal_status='ACTIVE',
                   source_sha256=file_sha(Path(__file__)))
    out = PARENT.parent/'same_frame_multiview_support_20261125.json'
    atomic_json(out, result)
    print(json.dumps(dict(status=result['status'], output=str(out))), flush=True)


if __name__ == '__main__':
    main()
