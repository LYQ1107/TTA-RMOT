"""Post-seal audit of actual anchored weights, teacher risk and all counts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.reciprocal_memory import MemoryInputs, models
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_anchored_reciprocal_pilot import OUT, PUBLIC, METHODS, COMPARATORS, previous, registration
    from scripts.a100_reciprocal_memory_sealed_audit import cpu_risk, frame_counts

    torch.set_num_threads(2)
    guard = generation_guard()
    plan = registration()
    barrier = json.loads((OUT/'all57_sealed_before_GT.json').read_text())
    assert len(barrier['tasks']) == 57 and barrier['GT_open_attempts'] == 0
    assert barrier['registration_sha256'] == file_sha(OUT/'registration.json')
    seals = {}
    for task in barrier['tasks']:
        key = task['pair'], task['method']
        assert key not in seals
        seals[key] = verify_predictions(task['folder'])
    parent_audit = json.loads((previous.OUT/'independent_sealed_prediction_audit.json').read_text())
    assert parent_audit['ordinary_prefix_student_gates'] == 36
    parent_cases = {row['pair']: row for row in parent_audit['cases']}
    use_verified_local_features()
    final_students = exact_raw_refs = self_checks = projected_files = supported_stages = 0
    reports = []
    for case in plan['cases']:
        pair = case['pair']
        folder = OUT/pair
        paid = json.loads((folder/'paid_answers.json').read_text())['paid']
        assert file_sha(folder/'paid_answers.json') == case['paid_answers_sha256']
        inputs = MemoryInputs(JointObservationVideo(pair, case['video']))
        state = torch.load(folder/'fits/PRIMARY/final.pt', map_location='cpu', weights_only=False)
        assert state['seed'] == 0 and state['no_GT_selection']
        assert state['paid_answers_sha256'] == file_sha(folder/'paid_answers.json')
        assert np.array_equal(state['mean'], inputs.mean) and np.array_equal(state['std'], inputs.std)
        assert len(state['rounds']) == 3 and state['final_fit']['steps'] == 4000
        stage_reports = []
        for stage, row in enumerate([*state['rounds'], dict(fit=state['final_fit'], curriculum=state['final_curriculum'])]):
            if stage < 3:
                forbidden = {i for fold in case['heldout_rows'][stage:] for i in fold}
                indices = [i for i in range(len(paid)) if i not in forbidden]
                assert row['training_indices'] == indices
                assert row['withheld_indices'] == case['heldout_rows'][stage]
                assert row['fit']['steps'] == 500
                trained = {(paid[i]['local_query'], paid[i]['track_id'], paid[i]['frame_id']) for i in indices}
                withheld = {(paid[i]['local_query'], paid[i]['track_id'], paid[i]['frame_id']) for i in forbidden}
                assert not trained & withheld
            else:
                indices = range(len(paid))
            assert row['fit']['primary_teacher_interpolation_gate_not_imposed']
            assert row['fit']['original_query_role_mass_preserved']
            assert all(r['query_role_supervision_mass_maximum_error'] < .001 for r in row['fit']['optimization_history'])
            selected = row['curriculum']['selected_edge_ids']
            touched = {(int(i//len(inputs.left)), int(v)) for i in selected
                       for v in (inputs.left[i % len(inputs.left)], inputs.right[i % len(inputs.left)])}
            anchors = {(int(paid[i]['local_query']), inputs.video.positions[int(paid[i]['frame_id']), int(paid[i]['track_id'])]) for i in indices}
            # Independently enumerate which requested observations have any
            # available adjacent physical observation, without calling cover().
            neighbors = set(map(int, inputs.left)) | set(map(int, inputs.right))
            eligible = {(q, v) for q, v in anchors if v in neighbors}
            assert eligible <= touched
            stage_reports.append(dict(stage=stage, paid_groups=len(anchors), covered=len(anchors & touched),
                                      isolated=len(anchors-eligible), selected_edges=len(selected)))
            supported_stages += 1
        actual_risks = []
        for index, (model, saved) in enumerate(zip(models('cpu'), state['models'])):
            model.load_state_dict(saved)
            model.eval()
            risks = cpu_risk(model, inputs.arrays(paid), paid, hard=False)
            error = max(abs(v[k]-state['final_fit']['teacher_risks'][index][mode][k])
                        for mode, v in risks.items() for k in ('CE', 'floor', 'excess_KL'))
            assert error < 1e-5
            actual_risks.append(dict(memory=model.memory, risks=risks, maximum_report_error=error))
            final_students += 1
        for recipe in ('SOFT_DUAL', 'HARD_DUAL'):
            path = previous.OUT/pair/'fits'/recipe/'final.pt'
            assert file_sha(path) == parent_cases[pair]['models'][recipe]['checkpoint_sha256']
        for name in METHODS:
            scores = np.load(folder/'predictions'/name/'scores.npy')
            mask = np.load(folder/'predictions'/name/'mask.npy')
            assert np.isfinite(scores).all() and np.array_equal(mask, scores > 0)
            if name in COMPARATORS:
                old = previous.OUT/pair/'predictions'/name/'scores.npy'
                assert file_sha(old) == case['ordinary_score_sha256'][name]
                assert np.array_equal(scores, np.load(old))
                exact_raw_refs += 1
            for frame in np.unique(inputs.video.obs[:, 0]):
                ids = np.flatnonzero(inputs.video.obs[:, 0] == frame)
                for qi in range(len(inputs.video.queries)):
                    k = int(mask[qi, ids].sum())
                    assert set(ids[np.lexsort((ids, -scores[qi, ids]))][:k]) == set(ids[mask[qi, ids]])
            self_checks += 1
        for (p, name), seal in seals.items():
            if p != pair or 'count_method' not in seal:
                continue
            donor = {(r['video'], r['expression']): r for r in seals[p, seal['count_method']]['episodes']}
            assert len(donor) == len(seal['episodes']) == case['query_count']
            for row in seal['episodes']:
                assert frame_counts(row['prediction_path']) == frame_counts(donor[row['video'], row['expression']]['prediction_path'])
                projected_files += 1
        reports.append(dict(pair=pair, supported_stages=stage_reports, final_CPU_risks=actual_risks))
    assert final_students == 6 and exact_raw_refs == 18 and self_checks == 21 and supported_stages == 12 and not guard
    result = dict(status='PASS_ANCHORED_ALL57_SEALS_6_FINAL_CPU_STUDENTS_18_EXACT_REFERENCES_AND_COUNTS',
                  records=57, final_primary_students_recomputed=final_students,
                  exactly_reused_raw_controls=exact_raw_refs, CountSelf_exact=self_checks,
                  supported_prefix_or_final_stages=supported_stages, projected_query_files=projected_files,
                  original_ordinary_prefix_fidelity_gates=36, new_primary_fidelity_not_interpolation_constrained=True,
                  cases=reports, no_manual_GT_reads_or_new_teacher_fit_or_RMOT_predictions=True,
                  no_task_quality_or_HOTA_causal_claim=True, source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json', result)
    (PUBLIC/'anchored_reciprocal_primary_sealed_audit_20261125.json').write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
