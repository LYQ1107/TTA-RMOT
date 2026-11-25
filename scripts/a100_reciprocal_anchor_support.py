"""Audit whether the completed mutual curriculum touches paid anchor images.

Uses only request identities and the selected unlabeled image edges. Neither
manual annotations, teacher probabilities nor task scores are needed.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def coverage(inputs, paid, selection):
    import numpy as np
    video = inputs.video
    edges = np.asarray(selection['selected_edge_ids'])
    queries, indices = edges//len(inputs.left), edges % len(inputs.left)
    touched = {(int(q), int(row)) for q, i in zip(queries, indices)
               for row in (inputs.left[i], inputs.right[i])}
    training = {(int(r['local_query']), video.positions[int(r['frame_id']), int(r['track_id'])])
                for r in paid}
    whole = {(int(r['local_query']), video.positions[int(r['frame_id']), int(r['track_id'])])
             for r in paid if r['factor_mode'] == 'WHOLE'}
    by_query = Counter(q for q, row in training)
    by_query_hit = Counter(q for q, row in training & touched)
    weights = inputs.edge_data(edges)[2]
    return dict(selected_edges=len(edges), unique_paid_image_groups=len(training),
                touched_paid_image_groups=len(training & touched),
                unique_paid_WHOLE_groups=len(whole), touched_paid_WHOLE_groups=len(whole & touched),
                paid_questions=len(paid),
                paid_questions_with_selected_incident_edge=sum(
                    (int(r['local_query']), video.positions[int(r['frame_id']), int(r['track_id'])]) in touched
                    for r in paid),
                queries_with_paid_anchor=len(by_query),
                queries_with_touched_paid_anchor=len(by_query_hit),
                mean_selected_role_weights=weights.mean(0).tolist())


def main():
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.reciprocal_memory import MemoryInputs
    from scripts.a100_reciprocal_memory_pilot import OUT, registration

    guard = generation_guard()
    plan = registration()
    assert json.loads((OUT/'quality_results.json').read_text())['status'].startswith('COMPLETE_')
    use_verified_local_features()
    cases = {}
    for case in plan['cases']:
        folder = OUT/case['pair']
        paid = json.loads((folder/'paid_answers.json').read_text())['paid']
        inputs = MemoryInputs(JointObservationVideo(case['pair'], case['video']))
        rows = []
        for stage in range(3):
            record = json.loads((folder/'fits/PRIMARY'/('round'+str(stage)+'.json')).read_text())
            training = [paid[i] for i in record['training_indices']]
            rows.append(dict(stage=stage, **coverage(inputs, training, record['curriculum'])))
        # final_selection is a numerical execution artifact. Load it only to
        # read integer selected-edge identities, not model weights or scores.
        import torch
        selected = torch.load(folder/'fits/PRIMARY/final_selection.pt', map_location='cpu', weights_only=False)
        rows.append(dict(stage='final', **coverage(inputs, paid, selected['selection'])))
        cases[case['pair']] = rows
        print(json.dumps(dict(pair=case['pair'], rows=rows)), flush=True)
    assert not guard
    result = dict(status='COMPLETE_PAID_ANCHOR_TO_CURRICULUM_SUPPORT_AUDIT', cases=cases,
                  no_manual_GT_teacher_probabilities_or_task_scores_used=True,
                  no_new_fit_teacher_call_or_prediction=True,
                  descriptive_not_identified_HOTA_causation=True,
                  source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'paid_anchor_support_audit.json', result)


if __name__ == '__main__':
    main()
