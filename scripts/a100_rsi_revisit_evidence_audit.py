"""Reuse completed physical teacher pilots to describe possible revisit evidence.

No requests, fits, feature extraction, task predictions or GT. The temporal
and cross-candidate cases were fixed by their original pilot registrations;
their difference is NOT a matched intervention or semantic accuracy measure.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard

BASE = ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
TEACHER_SHA = '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'


def entropy(p):
    return -sum(x*math.log(x) for x in (p, 1-p) if x > 0)


def summarize(rows):
    assert rows
    return dict(cases=len(rows), binary_teacher_examples=2*len(rows),
        teacher_threshold_flips=sum(r['threshold_flip'] for r in rows),
        both_below_half=sum(r['both_below_half'] for r in rows),
        both_above_or_equal_half=sum(r['both_above_or_equal_half'] for r in rows),
        effective_view_mismatches=sum(not r['same_effective_view'] for r in rows),
        absolute_probability_difference_mean=float(np.mean([r['absolute_probability_difference'] for r in rows])),
        absolute_logit_difference_median_p90=np.percentile([r['absolute_logit_difference'] for r in rows], [50, 90]).tolist(),
        binary_teacher_JS_mean=float(np.mean([r['binary_teacher_JS'] for r in rows])),
        binary_teacher_JS_max=max(r['binary_teacher_JS'] for r in rows))


def main():
    guard = generation_guard(); details = []; cohorts = {}; signatures = []
    for kind, source, expected in (
            ('SAME_TRACK_TWO_MOMENTS', 'joint_observation_teacher_pilot_20261123', 24),
            ('TWO_TRACKS_SAME_MOMENT', 'cross_candidate_teacher_pilot_20261123', 72)):
        path = BASE/source/'responses.json'; doc = json.loads(path.read_text())
        assert doc['GT_open_attempts'] == 0 and len(doc['cases']) == expected
        responses = {(r['case'], r['mode']): r for r in doc['responses']}
        assert len(responses) == len(doc['responses'])
        rows = []
        for index, case in enumerate(doc['cases']):
            sides = [responses[index, 'INDEPENDENT_'+side] for side in ('A', 'B')]
            assert all(r['frozen_weight_sha256'] == TEACHER_SHA for r in sides)
            logits = [float(r['yes_logit']-r['no_logit']) for r in sides]
            probabilities = [float(r['yes_probability_fp32']) for r in sides]
            for z, p in zip(logits, probabilities):
                sigmoid = math.exp(-max(-z, 0))/(1+math.exp(-abs(z)))
                assert abs(sigmoid-p) < 1e-6
            if kind == 'SAME_TRACK_TWO_MOMENTS':
                assert case['frame_A'] != case['frame_B']
                metadata = sides[0]['input_metadata']
                assert metadata == sides[1]['input_metadata']
                for side, response in zip(('A', 'B'), sides):
                    recorded = metadata[side]
                    assert recorded['track_id'] == case['track_id']
                    assert recorded['frame_id'] == case['frame_'+side]
                    assert response['prompt'] == recorded['input']['prompt']
                views = [metadata[side]['effective_view'] for side in ('A', 'B')]
            else:
                assert case['track_A'] != case['track_B']
                assert all(r[side]['moments'][-1]['frame'] == case['frame']
                           for r in sides for side in ('A', 'B'))
                assert all(case['query_text'] in r['prompt'] for r in sides)
                views = [sides[0][side]['effective_view'] for side in ('A', 'B')]
            p, q = probabilities
            js = max(entropy((p+q)/2)-(entropy(p)+entropy(q))/2, 0.)
            row = dict(kind=kind, original_case=index, setting=case['pair'],
                requested_view=case['view'], effective_views=views,
                same_effective_view=views[0] == views[1],
                teacher_p_A=p, teacher_p_B=q,
                threshold_flip=(p >= .5) != (q >= .5),
                both_below_half=p < .5 and q < .5,
                both_above_or_equal_half=p >= .5 and q >= .5,
                absolute_probability_difference=abs(p-q),
                absolute_logit_difference=abs(logits[0]-logits[1]),
                binary_teacher_JS=js)
            rows.append(row); details.append(row)
        by_setting = defaultdict(list); by_view = defaultdict(list)
        for row in rows:
            by_setting[row['setting']].append(row)
            by_view[row['requested_view']].append(row)
        cohorts[kind] = dict(all=summarize(rows),
            settings={k:summarize(v) for k,v in by_setting.items()},
            requested_views={k:summarize(v) for k,v in by_view.items()})
        signatures.append(dict(path=str(path), sha256=file_sha(path), original_registered_cases=expected))
    assert not guard
    local = BASE/'rsi_revisit_evidence_audit_20261124/per_case.json'
    atomic_json(local, dict(rows=details, source_signatures=signatures))
    result = dict(status='COMPLETED_EXISTING_PHYSICAL_BINARY_REVISIT_EVIDENCE_AUDIT_NO_GT',
        cohorts=cohorts,
        scope='All 24 registered temporal and 72 cross-candidate cases from the completed pilots; only normal INDEPENDENT_A/B channels.',
        source_signatures=signatures, teacher_frozen_sha256=TEACHER_SHA,
        shared_or_joint_image_context_outputs_excluded=True,
        metric='JS between two Bernoulli teacher opinions, equal-weight: minimum teacher CE penalty of forcing one common probability rather than two independent probabilities.',
        lower_bound_is_for_teacher_targets_not_true_semantic_state=True,
        thresholds_are_original_binary_half_not_a_sweep=True,
        temporal_vs_cross_cases_are_not_matched_or_independent_samples=True,
        different_images_and_times_can_change_evidence_without_true_membership_change=True,
        effective_view_equality_is_not_identical_visual_content=True,
        no_teacher_correctness_or_RMOT_gain_or_need_for_unconditional_temporal_smoothing_claim=True,
        new_teacher_calls=0, new_GPU_forwards=0, new_fits=0, new_predictions=0,
        target_manual_GT_open_attempts=0, no_live_study_sources_or_predictions_changed=True,
        local_details=str(local), local_details_sha256=file_sha(local),
        source_sha256=file_sha(Path(__file__)), goal_status='ACTIVE')
    atomic_json(PUBLIC/'rsi_revisit_evidence_audit_20261124.json', result)
    print(json.dumps(dict(status=result['status'], summaries={k:v['all'] for k,v in cohorts.items()})), flush=True)


if __name__ == '__main__':
    main()
