"""Independent post-seal quantity and final CPU teacher-risk audit; no GT.

This executable accepts only the complete registered cohort. It does not
change weights, predictions, training gates, or the acceptance criterion.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def frame_counts(path):
    return Counter(int(float(line.split(',')[0]))
                   for line in Path(path).read_text().splitlines() if line.strip())


def cpu_risk(model, arrays, paid, hard):
    """Recompute the exact-input attainable floor, independently of trainer."""
    import numpy as np
    import torch
    from flex_uda.active.factor_evidence import MODES

    roles = np.array([MODES.index(r['factor_mode']) for r in paid])
    target = np.array([r['yes_probability'] for r in paid], dtype='float64')
    if hard:
        target = (target >= .5).astype('float64')
    features = [arrays['bags'].reshape(len(paid), -1), arrays['present'],
                arrays['current'], arrays['coarse']]
    if model.memory:
        features.append(arrays['memory'])
    packed = np.ascontiguousarray(np.column_stack((*features, roles)), dtype='float32')
    groups = {}
    for index, row in enumerate(packed):
        groups.setdefault(row.tobytes(), []).append(index)
    floors = np.zeros(len(paid), dtype='float64')
    for ids in groups.values():
        mean = float(target[ids].mean())
        p = np.clip(mean, 1e-12, 1-1e-12)
        floors[ids] = -p*np.log(p)-(1-p)*np.log1p(-p)
    prediction = []
    with torch.no_grad():
        for begin in range(0, len(paid), 256):
            data = {k: torch.as_tensor(v[begin:begin+256]) for k, v in arrays.items()}
            logits = model(**data).double().numpy()
            selection = roles[begin:begin+256]
            prediction.extend(logits[np.arange(len(selection)), selection])
    z = np.asarray(prediction)
    loss = np.logaddexp(0, z)-target*z
    return {mode: dict(CE=float(loss[roles == r].mean()),
                       floor=float(floors[roles == r].mean()),
                       excess_KL=float((loss-floors)[roles == r].mean()))
            for r, mode in enumerate(MODES) if np.any(roles == r)}


def main():
    import numpy as np
    import torch
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.reciprocal_memory import MemoryInputs, models
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_reciprocal_memory_pilot import OUT, PRIMARY, METHODS, RECIPES, registration

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
    use_verified_local_features()
    prefix_gates = final_gates = projected_files = self_checks = 0
    reports = []
    for case in plan['cases']:
        pair = case['pair']
        folder = OUT/pair
        done = json.loads((folder/'completion.json').read_text())
        assert done['GT_open_attempts'] == 0 and len(done['tasks']) == 19
        assert file_sha(folder/'paid_answers.json') == case['paid_answers_sha256']
        paid = json.loads((folder/'paid_answers.json').read_text())['paid']
        video = JointObservationVideo(pair, case['video'])
        inputs = MemoryInputs(video)
        arrays = inputs.arrays(paid)
        case_reports = {}
        for recipe, options in RECIPES.items():
            path = folder/'fits'/recipe/'final.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['neural_seed'] == 0 and state['no_GT_selection']
            assert state['paid_answers_sha256'] == file_sha(folder/'paid_answers.json')
            assert state['recipe'] == options and len(state['rounds']) == 3
            assert np.array_equal(state['mean'], inputs.mean)
            assert np.array_equal(state['std'], inputs.std)
            for stage, row in enumerate(state['rounds']):
                forbidden = {i for fold in case['heldout_rows'][stage:] for i in fold}
                expected = [i for i in range(len(paid)) if i not in forbidden]
                assert row['training_indices'] == expected
                assert row['withheld_indices'] == case['heldout_rows'][stage]
                train_groups = {(paid[i]['local_query'], paid[i]['track_id'], paid[i]['frame_id']) for i in expected}
                withheld_groups = {(paid[i]['local_query'], paid[i]['track_id'], paid[i]['frame_id']) for i in forbidden}
                assert not train_groups & withheld_groups
                if not options['mutual']:
                    assert row['fit']['teacher_fidelity_gate_passed']
                    for risk in row['fit']['teacher_risks']:
                        assert max(v['excess_KL'] for v in risk.values()) <= .005
                        prefix_gates += 1
            fit = state['final_fit']
            assert 4000 <= fit['steps'] <= 16000 and fit['teacher_fidelity_gate_passed']
            pair_models = models('cpu')
            independently_computed = []
            for index, (model, saved) in enumerate(zip(pair_models, state['models'])):
                model.load_state_dict(saved)
                model.eval()
                risk = cpu_risk(model, arrays, paid, options['hard'])
                maximum = max(v['excess_KL'] for v in risk.values())
                error = max(abs(v[k]-fit['teacher_risks'][index][mode][k])
                            for mode, v in risk.items() for k in ('CE', 'floor', 'excess_KL'))
                assert maximum <= .005+1e-6 and error < 1e-5, (pair, recipe, index, maximum, error)
                final_gates += 1
                independently_computed.append(dict(memory=model.memory, risks=risk,
                                                   maximum_report_error=error))
            case_reports[recipe] = dict(checkpoint_sha256=file_sha(path),
                                         final_CPU_risks=independently_computed)
        for name in METHODS:
            scores = np.load(folder/'predictions'/name/'scores.npy')
            mask = np.load(folder/'predictions'/name/'mask.npy')
            assert np.isfinite(scores).all() and np.array_equal(mask, scores > 0)
            for frame in np.unique(video.obs[:, 0]):
                ids = np.flatnonzero(video.obs[:, 0] == frame)
                for qi in range(len(video.queries)):
                    k = int(mask[qi, ids].sum())
                    ordered = ids[np.lexsort((ids, -scores[qi, ids]))][:k]
                    assert set(ordered) == set(ids[mask[qi, ids]])
            self_checks += 1
        for (p, name), seal in seals.items():
            if p != pair or 'count_method' not in seal:
                continue
            donor = seals[p, seal['count_method']]
            by = {(r['video'], r['expression']): r for r in donor['episodes']}
            assert len(by) == len(seal['episodes']) == case['query_count']
            for row in seal['episodes']:
                original = by[row['video'], row['expression']]
                assert frame_counts(row['prediction_path']) == frame_counts(original['prediction_path'])
                projected_files += 1
        reports.append(dict(pair=pair, models=case_reports))
    assert prefix_gates == 36 and final_gates == 18 and self_checks == 21 and not guard
    result = dict(status='PASS_ALL57_SEALS_36_ORDINARY_PREFIX_GATES_18_FINAL_CPU_STUDENTS_AND_COUNTS',
                  records=57, ordinary_prefix_student_gates=prefix_gates,
                  final_student_models_independently_recomputed=final_gates,
                  raw_CountSelf_exact_checks=self_checks, projected_query_files=projected_files,
                  cases=reports, no_manual_GT_reads=True, no_new_fit_teacher_or_prediction=True,
                  no_task_quality_claim=True, audit_source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
