"""Explain the completed primary using saved decisions; never fit on GT.

Canonical all-candidate IoU.5 membership is a diagnostic proxy and differs
from method-dependent official matching. Both are reported separately.
"""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import expit
import torch

from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_visual_constraint_primary import OUT, DATA, PUBLIC, STEM, PRIMARY, METHODS, registration


def confusion(truth, prediction):
    return dict(TP=int((truth & prediction).sum()), FP=int((~truth & prediction).sum()),
                FN=int((truth & ~prediction).sum()), TN=int((~truth & ~prediction).sum()))


def official(folder):
    path = folder/'results/pedestrian_detailed.csv'
    with path.open() as stream: rows = [r for r in csv.DictReader(stream) if r['seq'] != 'COMBINED']
    counts = {name: sum(float(r['HOTA_'+name+'___50']) for r in rows) for name in ('TP', 'FP', 'FN')}
    counts['TP_weighted_AssA'] = sum(float(r['AssA___50'])*float(r['HOTA_TP___50']) for r in rows)/max(counts['TP'], 1)
    return counts


def main():
    plan = registration()
    barrier = json.loads((OUT/'all84_sealed_before_GT.json').read_text())
    reproduction = json.loads((PUBLIC/(STEM+'_reproduction.json')).read_text())
    assert len(barrier['tasks']) == 84 and barrier['GT_open_attempts'] == 0
    assert reproduction['records'] == 84 and reproduction['effects'] == 81
    assert reproduction['status'].startswith('PASS_')
    # Only after the full barrier and official/statistical completion.
    from flex_uda.a100.mechanism import OUT as DIAGNOSTIC, metadata
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    from flex_uda.active.visual_constraint_learning import Observer
    torch.set_num_threads(2)
    cases = []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video']; folder = OUT/pair
        truth = np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy') >= 0
        md = metadata(case['target']); obs = np.asarray(md['videos'][video]['observations'], dtype=int)
        queries = [q for q in md['episodes'] if q['video_id'] == video]
        assert truth.shape == (len(queries), len(obs))
        paid = np.load(DATA/pair/'teacher_inputs.npz')
        keys = json.loads((DATA/pair/'row_keys.json').read_text())['keys']
        qlookup = {q['expression_id']: i for i, q in enumerate(queries)}
        lookup = {tuple(map(int, r)): i for i, r in enumerate(obs)}
        selected = [i for i, k in enumerate(keys) if k[0] == video]
        paid_nodes = np.asarray([qlookup[keys[i][1]]*len(obs)+lookup[int(keys[i][3]), int(keys[i][2])] for i in selected])
        original_teacher = np.nanmean(paid['probability'][selected], axis=1) > .5
        anchor_truth = truth.flatten()[paid_nodes]
        masks = {name: np.load(folder/'predictions'/name/'mask.npy') for name in METHODS}
        method_records = {}
        for name, decision in masks.items():
            assert decision.shape == truth.shape
            anchored = decision.flatten()[paid_nodes]
            correct_teacher, correct_student = original_teacher == anchor_truth, anchored == anchor_truth
            method_records[name] = dict(
                official_IoU50_counts=official(folder/'predictions'/name),
                canonical_membership_counts=confusion(truth, decision),
                canonical_positive_queries_zero_output=int((truth.any(1) & ~decision.any(1)).sum()),
                paid_nominal_anchor_transition=dict(
                    both_correct=int((correct_teacher & correct_student).sum()),
                    teacher_correct_student_wrong=int((correct_teacher & ~correct_student).sum()),
                    teacher_wrong_student_correct=int((~correct_teacher & correct_student).sum()),
                    both_wrong=int((~correct_teacher & ~correct_student).sum())))
        graph = dict(np.load(folder/'graph/edges.npz')); left, right = graph['left'], graph['right']
        flat = truth.flatten(); a, b = flat[left], flat[right]
        category = np.where(a & b, 0, np.where(a | b, 1, 2))
        checkpoint = torch.load(folder/'primary/final.pt', map_location='cpu', weights_only=False)
        observer = Observer().eval(); observer.load_state_dict(checkpoint['observer'])
        logits = np.load(folder/'predictions'/PRIMARY/'scores.npy').flatten()
        weighted = np.empty(len(left), dtype='float64')
        multipliers = np.empty(len(left), dtype='float64')
        with torch.no_grad():
            for begin in range(0, len(left), 8192):
                sl = slice(begin, begin+8192)
                p, q = expit(logits[left[sl]]), expit(logits[right[sl]])
                dynamic = np.c_[p, q, np.abs(p-q), 4*((p+q)/2)*(1-(p+q)/2)]
                context = np.c_[graph['context'][sl], dynamic].astype('float32')
                multipliers[sl] = observer(torch.from_numpy(context)).numpy()
                weighted[sl] = multipliers[sl]*graph['weight'][sl]
        graph_summary = {}
        for name, weight in (('unweighted', np.ones(len(left))), ('image_state_weight', graph['weight']), ('learned_weight', weighted)):
            mass = np.bincount(category, weights=weight, minlength=3)
            graph_summary[name] = dict(positive_positive=float(mass[0]), positive_negative=float(mass[1]), negative_negative=float(mass[2]),
                cross_class_mass_fraction=float(mass[1]/max(mass.sum(), 1e-30)),
                cross_class_fraction_conditioned_on_a_positive_endpoint=float(mass[1]/max(mass[0]+mass[1], 1e-30)))
        record = dict(pair=pair, video=video, methods=method_records,
            teacher_nominal_anchor_canonical_membership_counts=confusion(anchor_truth, original_teacher),
            graph_edges=graph_summary, graph_edge_classes_are_canonical_membership_not_semantic_GT=True,
            mean_observer_multiplier=float(multipliers.mean()),
            observer_parameter_change=checkpoint['observer_parameter_change'],
            teacher_targets_unchanged=checkpoint['teacher_targets_unchanged'],
            mean_virtual_probe_CE_change=float(np.mean([x['virtual_held_teacher_CE_after']-x['held_teacher_CE_before'] for x in checkpoint['feedback']])),
            virtual_progress_includes_support_gradient_not_pure_observer_contribution=True)
        cases.append(record)
        print(json.dumps(dict(status='POSTSEAL_VISUAL_CONSTRAINT_CASE_DIAGNOSED', pair=pair,
            primary_official=method_records[PRIMARY]['official_IoU50_counts'],
            graph_cross_class_positive_mass=graph_summary['learned_weight']['cross_class_fraction_conditioned_on_a_positive_endpoint'])), flush=True)
    report = dict(status='COMPLETE_POSTSEAL_VISUAL_CONSTRAINT_ERROR_DIAGNOSIS', cases=cases,
        scope='Three first videos, allDEV. Descriptive postseal analysis, no label-based adaptation/checkpoint selection.',
        official_counts_single_IoU50_not_additive_decomposition_of_19threshold_HOTA=True,
        canonical_teacher_transition_not_pure_teacher_semantic_accuracy=True,
        graph_weight_CPU_sigmoid_reconstruction_not_a_new_model_or_RMOT_output=True,
        graph_homophily_not_a_module_causal_ablation=True,
        new_fits_teacher_or_backbone_calls_predictions_and_neural_seeds=0,
        current_study_outputs_sealed_before_label_access=True,
        source_sha256=file_sha(Path(__file__)), goal_status='ACTIVE')
    atomic_json(OUT/'postseal_error_diagnosis.json', report)
    atomic_json(PUBLIC/(STEM+'_postseal_diagnosis.json'), report)


if __name__ == '__main__': main()
