"""Prepare object/reference pseudo-targets from fixed paid teacher evidence.

No quality prediction, evaluator, target GT, teacher/encoder call or neural
fit is used. Earlier answers are re-expressed at each current-prefix slope;
future video data are excluded. Large arrays remain in local a100_runs.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, history, generation_guard
from flex_uda.active.object_reference import grouped_targets
from flex_uda.active.relational_features import frame_supervision
from flex_uda.active.relational_transport import horizontal_direction
from scripts.a100_relation_transport_prepare import OUT as RELATION_PREPARED
from scripts.a100_six_direction import OUT as SIX, PUBLIC, source

OUT = SIX / 'object_reference/teacher_prepare'
REGISTRATION = PUBLIC / 'object_reference_prepare_registration.json'


def main():
    registration = json.loads(REGISTRATION.read_text())
    for name, sha in registration['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    started = time.perf_counter()
    guard = generation_guard()
    results = {}
    for pair, (_, target) in PAIRS.items():
        m = metadata(target)
        past = []
        receipts = []
        for video, bank in sorted(m['videos'].items()):
            queries = [q for q in m['episodes'] if q['video_id'] == video]
            directions = np.asarray([horizontal_direction(q['query_text']) for q in queries], int)
            if not directions.any():
                assert not past, 'mixed eligibility across videos needs an explicit prefix record'
                continue
            with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:
                tracks = a['tracks']
            prepared = json.loads((RELATION_PREPARED/pair/video/'teacher_rows.json').read_text())
            relation = json.loads((RELATION_PREPARED/pair/video/'prefix_fit.json').read_text())
            assert prepared['query_directions'] == directions.tolist()
            records = json.loads((history(pair)/'videos'/video/'questions.json').read_text())
            cells, raw, coords, evidence = frame_supervision(records, queries, tracks,
                prepared['width'], prepared['height'])
            query_index = {q['expression_id']:i for i,q in enumerate(queries)}
            eligible_views = Counter(row['effective_view'] for row in records
                if directions[query_index[row['query_id']]])
            past.append(dict(video=video, cells=cells, probability=raw, coords=coords,
                directions=directions, evidence=evidence, eligible_views=dict(eligible_views)))
            blocks = [grouped_targets(item['cells'], item['probability'], item['coords'],
                item['directions'], relation['beta']) for item in past]
            loss_error = 0.
            for block in blocks:
                logits = np.random.default_rng(0).normal(size=len(block['cells']))
                frame_logits = logits[block['frame_to_group']]
                frame_loss = np.logaddexp(0.,frame_logits)-block['reference_frame_target']*frame_logits
                grouped_loss = np.logaddexp(0.,logits)-block['reference_target']*logits
                loss_error = max(loss_error, abs(float(frame_loss@block['raw_frame_weight']-
                    grouped_loss@block['sample_weight'])))
            assert loss_error < 1e-8, loss_error
            destination = OUT/pair/video
            destination.mkdir(parents=True, exist_ok=True)
            sizes = [len(b['cells']) for b in blocks]
            arrays = dict(reference_target=np.concatenate([b['reference_target'] for b in blocks]),
                unshifted_target=np.concatenate([b['unshifted_target'] for b in blocks]),
                sample_weight=np.concatenate([b['sample_weight'] for b in blocks]),
                cell_query=np.concatenate([b['cells'][:,0] for b in blocks]),
                cell_track=np.concatenate([b['cells'][:,1] for b in blocks]),
                cell_video=np.repeat(np.arange(len(blocks)), sizes))
            np.savez_compressed(destination/'prefix_targets.npz', **arrays)
            eligible_groups = np.concatenate([item['directions'][b['cells'][:,0]] != 0
                for item,b in zip(past,blocks)])
            difference = arrays['reference_target']-arrays['unshifted_target']
            receipt = dict(status='SEALED_CAUSAL_PREFIX_REFERENCE_PSEUDO_TARGETS', video=video,
                ordered_videos=[item['video'] for item in past], group_counts=sizes,
                prefix_reference_groups=sum(sizes), prefix_frame_cells=sum(len(item['cells']) for item in past),
                eligible_reference_groups=int(eligible_groups.sum()), beta=relation['beta'],
                reference_position=.5, common_confidence_from_raw_frame_target=True,
                target_min=float(arrays['reference_target'].min()), target_max=float(arrays['reference_target'].max()),
                unshifted_target_min=float(arrays['unshifted_target'].min()),
                unshifted_target_max=float(arrays['unshifted_target'].max()),
                mean_absolute_reference_target_change=float(np.abs(difference[eligible_groups]).mean()),
                exact_weighted_BCE_aggregation_max_error=loss_error,
                current_actual_frame_evidence=evidence, current_eligible_paid_views=dict(eligible_views),
                all_prefixes_use_only_past_and_current_teacher_answers=True,
                no_original_target_reclipping_after_reference_transport=True,
                target_arrays_not_to_Git=True, manual_target_GT_open_attempts=len(guard))
            atomic_json(destination/'completion.json', receipt)
            receipts.append(receipt)
        results[pair] = dict(source=PAIRS[pair][0], target=target,
            actual_reference_prefixes=len(receipts), exact_fallback_if_not_eligible=not bool(receipts),
            final_prefix=receipts[-1] if receipts else None)
    assert not guard
    result = dict(status='OBJECT_REFERENCE_TEACHER_TARGET_PREPARATION_COMPLETE', pairs=results,
        target_manual_GT_open_attempts=0, quality_predictions_read=0,
        new_teacher_encoder_GPU_calls=0, neural_fits=0, new_quality_predictions=0,
        existing_method_parameters_or_helpers_changed=False,
        interpretation='Training-target decomposition prepared for a new reference representation. No HOTA, semantic correctness or final method acceptance follows.',
        reference_head_not_yet_trained=True, all_data_development=True,
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', result)
    (PUBLIC/'object_reference_teacher_prepare.json').write_text(
        json.dumps(result, ensure_ascii=False, separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'], pairs={p:dict(prefixes=r['actual_reference_prefixes'],
        groups=r['final_prefix']['prefix_reference_groups'] if r['final_prefix'] else 0)
        for p,r in results.items()}, manual_target_GT_open_attempts=0),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
