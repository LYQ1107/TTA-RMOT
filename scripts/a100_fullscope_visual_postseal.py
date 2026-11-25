"""Diagnose the complete full-scope primary, without fitting or promotion.

This program reads manual labels only after all 4588 outputs, the unchanged
external teachers, and the complete 186/180 official analysis are sealed.
Pixel agreement and agreement with teacher answers are proxy diagnostics,
not independent semantic labels or an additive explanation of HOTA.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_visual_six import (
    OUT, PLAN, PUBLIC, STEM, PRIMARY, COMPARATORS, registration,
)


def confusion(truth, prediction):
    return dict(TP=int((truth & prediction).sum()),
                FP=int((~truth & prediction).sum()),
                FN=int((truth & ~prediction).sum()),
                TN=int((~truth & ~prediction).sum()))


def official(folder):
    with (folder / 'results/pedestrian_detailed.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['seq'] != 'COMBINED']
    return {key: sum(float(r['HOTA_' + key + '___50']) for r in rows)
            for key in ('TP', 'FP', 'FN')}


def prepare():
    plan = registration()
    target = OUT / 'postseal_diagnosis_registration.json'
    assert not target.exists()
    doc = dict(
        status='REGISTERED_FULL_SCOPE_POSTSEAL_FAILURE_DIAGNOSIS',
        numerical_parent_registration_sha256=file_sha(PLAN),
        source_sha256=file_sha(__file__), video_units=len(plan['cases']),
        parent_outputs=4588, parent_statistics=186, parent_effects=180,
        diagnostics=[
            'Official IoU50 TP/FP/FN for the fixed main and all ordinary controls.',
            'Unique teacher questions, track/frame coverage, teacher-confirmed positives and pre-answer R/S errors.',
            'After whole-study completion only: canonical nominal-anchor coverage and false-empty-query diagnosis.',
            'Pixel-pair agreement of bootstrap, closed-round, and final content representations; projection may undo content learning.',
        ],
        same_frame_CLIP_anchor_canonical_labels_not_pure_teacher_semantic_accuracy=True,
        acquired_sample_risk_not_unbiased_population_calibration=True,
        pixel_MSE_not_semantic_quality_or_module_ablation=True,
        no_new_GPU_fits_teacher_predictions_seeds_or_parameter_search=True,
        no_GT_selected_masks_queries_or_checkpoints=True,
        fixed_primary_and_all_comparators_preserved=True,
        all_data_DEV=True, goal_status='ACTIVE',
    )
    atomic_json(target, doc)
    atomic_json(PUBLIC / (STEM + '_postseal_registration.json'), doc)
    print(json.dumps(dict(status=doc['status'], video_units=len(plan['cases']),
                          GT_reads=0, numerical_parent_sources_unchanged=True)), flush=True)


def content_diagnostics(folder, case):
    import flex_uda.active.evidence_visual_learning as core
    with np.load(case['pixel_pairs_path']) as a:
        pixel = {k: torch.from_numpy(a[k].copy()) for k in ('left', 'right', 'weight')}
    if len(pixel['left']) < 2:
        return dict(weak_pairs=len(pixel['left']), pretext_available=False)
    with np.load(folder / 'unlabeled_normalizer.npz') as a:
        mean, std = a['mean'].copy(), a['std'].copy()
    checkpoints = {
        stage: torch.load(folder / 'primary' / name, map_location='cpu', weights_only=False)
        for stage, name in [('bootstrap', 'bootstrap.pt'), ('closed', 'closed.pt'), ('final', 'final.pt')]
    }
    result = dict(weak_pairs=len(pixel['left']), pretext_available=True)
    with torch.no_grad():
        for stage, saved in checkpoints.items():
            for name, key, outputs in [('S', 'solver', 1), ('R', 'observer', 2)]:
                if stage == 'closed' and name == 'R':
                    continue  # The final projection updates S only.
                model = core.make(mean, std, 'cpu', outputs=outputs)
                model.load_state_dict(saved[key])
                a, b = model.app(pixel['left']), model.app(pixel['right'])
                result[stage + '_' + name] = dict(
                    positive_pair_MSE=float((a - b).square().mean()),
                    mean_content_std=float(a.std(0).mean()),
                    image_pretext=float(core.content_loss(model, pixel)),
                )
    return result


def main():
    torch.set_num_threads(2)
    plan = registration()
    preregistered = json.loads((OUT / 'postseal_diagnosis_registration.json').read_text())
    assert preregistered['source_sha256'] == file_sha(__file__)
    assert preregistered['numerical_parent_registration_sha256'] == file_sha(PLAN)
    seal = json.loads((OUT / 'all4588_sealed_before_GT.json').read_text())
    assert seal['GT_open_attempts'] == 0 and len(seal['tasks']) == 4588
    assert seal['teacher_final_weight_seals'] == 4
    reproduced = json.loads((PUBLIC / (STEM + '_reproduction.json')).read_text())
    assert reproduced['status'] == 'PASS_ALL186_HOTA_AND180_EFFECTS'
    assert reproduced['records'] == 186 and reproduced['effects'] == 180
    for gpu in range(4):
        teacher = json.loads((OUT / 'supervisor' / f'GPU{gpu}_teacher_final.json').read_text())
        assert teacher['teacher'] is not None
        assert teacher['teacher']['backend_unmodified']
        assert teacher['teacher']['teacher_weight_sha256'] == '9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956'
    # All manual-label access and completed-output inspection starts below.
    from flex_uda.a100.mechanism import OUT as DIAGNOSTIC, metadata
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    reports = []
    for case in plan['cases']:
        pair, video = case['pair'], case['evaluated_video']
        folder = OUT / 'units' / pair / video
        done = json.loads((folder / 'generation_complete.json').read_text())
        assert done['GT_open_attempts'] == 0 and done['registration_sha256'] == file_sha(PLAN)
        truth = np.load(DIAGNOSTIC / 'offline_GT' / CANONICAL[pair] / video / 'labels.npy') >= 0
        observations = np.load(case['observations'])
        queries = json.loads(Path(case['temporal_queries']).read_text())['queries']
        actual_queries = [q for q in metadata(case['target'])['episodes'] if q['video_id'] == video]
        assert [q['query_id'] for q in queries] == [q['expression_id'] for q in actual_queries]
        assert truth.shape == (len(queries), len(observations))
        n = len(observations)
        names = (PRIMARY, *COMPARATORS)
        masks = {name: np.load(folder / 'predictions' / name / 'mask.npy') for name in names}
        metrics = {
            name: dict(canonical=confusion(truth, mask),
                       official_IoU50=official(folder / 'predictions' / name),
                       positive_queries_zero_output=int((truth.any(1) & ~mask.any(1)).sum()))
            for name, mask in masks.items()
        }
        active = json.loads((folder / 'primary/draws.json').read_text())['draws']
        uniform = json.loads((folder / 'uniform_draws.json').read_text())['draws']
        coverage = {}
        perquery = []
        for policy, draws in [('primary', active), ('uniform', uniform)]:
            indices = np.asarray([d['index'] for d in draws], dtype=int)
            assert len(indices) == len(set(indices.tolist())) == case['logical_budget']
            qindex, positions = np.divmod(indices, n)
            probabilities = np.asarray([d['probability'] for d in draws])
            labels = truth.flatten()[indices]
            missed = false_empty = 0
            unique_tracks = []
            unique_frames = []
            for q, query in enumerate(queries):
                selected = qindex == q
                chosen = positions[selected]
                assert len(chosen) == query['logical_budget']
                has_positive = bool(truth[q].any())
                asked_correct = bool(labels[selected].any())
                missed += has_positive and not asked_correct
                empty = has_positive and not masks[PRIMARY][q].any()
                false_empty += empty and not asked_correct
                unique_tracks.append(len(np.unique(observations[chosen, 1])))
                unique_frames.append(len(np.unique(observations[chosen, 0])))
                perquery.append(dict(
                    policy=policy, query_id=query['query_id'], query_text=query['query_text'],
                    distinct_questions=len(chosen), distinct_tracks=unique_tracks[-1],
                    distinct_frames=unique_frames[-1], canonical_positive_observations=int(truth[q].sum()),
                    canonical_positive_asked=asked_correct,
                    teacher_Yes_count=int((probabilities[selected] > .5).sum()),
                    main_zero_output=not bool(masks[PRIMARY][q].any()),
                ))
            coverage[policy] = dict(
                unique_questions=len(indices), repeated_questions=0,
                mean_distinct_tracks_per_query=float(np.mean(unique_tracks)),
                mean_distinct_frames_per_query=float(np.mean(unique_frames)),
                positive_queries_never_asked_correct=int(missed),
                main_false_empty_without_asked_correct=int(false_empty),
                teacher_nominal_anchor=confusion(labels, probabilities > .5),
                main_known_teacher_class_disagreements=int(
                    (masks[PRIMARY].flatten()[indices] != (probabilities > .5)).sum()
                ) if policy == 'primary' else None,
            )
        stages = []
        for stage in range(4):
            selected = [d for d in active if d['stage'] == stage]
            if not selected:
                stages.append(dict(stage=stage, questions=0))
                continue
            means = np.asarray([d['observer_mean_BEFORE_answer'] for d in selected])
            current_s = np.asarray([d['student_probability_BEFORE_answer'] for d in selected])
            residual = np.asarray([d['observer_residual_proxy_BEFORE_answer'] for d in selected])
            y = np.asarray([d['probability'] for d in selected])
            s_error, r_error = (current_s - y) ** 2, (means - y) ** 2
            proxy = residual + (means - current_s) ** 2
            stages.append(dict(
                stage=stage, questions=len(selected),
                pre_answer_student_MSE=float(s_error.mean()),
                pre_answer_observer_MSE=float(r_error.mean()),
                observer_residual_proxy_mean=float(residual.mean()),
                combined_current_student_risk_proxy_mean=float(proxy.mean()),
                proxy_squared_error_on_acquired_sample=float(((proxy - s_error) ** 2).mean()),
            ))
        record = dict(pair=pair, video=video, methods=metrics, coverage=coverage,
                      acquired_sample_risk=stages, content=content_diagnostics(folder, case),
                      physical_images=done['physical_images'], conceptual_actions=done['conceptual_actions'],
                      primary_teacher_KL=done['primary_teacher_KL'],
                      active_broker=done['active_broker'], uniform_broker=done['uniform_broker'],
                      shared_execution_video_wall_seconds=done['wall_seconds'])
        reports.append(record)
        atomic_json(folder / 'postseal_query_diagnosis.json', dict(queries=perquery, **record))
        print(json.dumps(dict(status='FULL_SCOPE_POSTSEAL_VIDEO_DIAGNOSED', pair=pair, video=video,
                              coverage=coverage)), flush=True)
    aggregate = {}
    for pair in sorted({c['pair'] for c in plan['cases']}):
        group = [r for r in reports if r['pair'] == pair]
        aggregate[pair] = dict(
            videos=len(group),
            methods={name: {
                key: sum(r['methods'][name]['official_IoU50'][key] for r in group)
                for key in ('TP', 'FP', 'FN')
            } for name in (PRIMARY, *COMPARATORS)},
            coverage={policy: {
                key: sum(r['coverage'][policy][key] for r in group)
                for key in ('unique_questions', 'positive_queries_never_asked_correct',
                            'main_false_empty_without_asked_correct')
            } for policy in ('primary', 'uniform')},
        )
    result = dict(
        status='COMPLETE_FULL_SCOPE_SIX_SETTING_POSTSEAL_DIAGNOSIS',
        source_sha256=file_sha(__file__), aggregate=aggregate, videos=reports,
        new_fits_teacher_backbone_predictions_or_seeds=0,
        GT_only_after_all4588_and186_statistics_reproduced=True,
        canonical_not_pure_teacher_semantic_accuracy=True,
        acquired_sample_risk_not_population_calibration=True,
        pixel_pretext_not_module_quality_ablation=True,
        official_IoU50_not_additive_decomposition_of19threshold_HOTA=True,
        shared_prepaid_execution_not_per_method_fresh_cold_cost=True,
        no_primary_reselection_or_GT_driven_query_masks=True,
        all_data_DEV=True, goal_status='ACTIVE',
    )
    atomic_json(OUT / 'postseal_diagnosis.json', result)
    atomic_json(PUBLIC / (STEM + '_postseal_diagnosis.json'), result)
    print(json.dumps(dict(status=result['status'], settings=6, video_units=len(reports))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
