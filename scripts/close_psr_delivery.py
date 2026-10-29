#!/usr/bin/env python3
"""Seal the completed Val study from existing reports; no GT or GPU work."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def read(path):
    return json.loads(path.read_text())


def close():
    public = ROOT / 'results/psr'
    confirmations = public / 'confirmations'
    aggregate = read(confirmations / 'summary.json')
    if aggregate['status'] != 'COMPLETE_REGISTERED_3SEED_PAIRS_AND_PARENT_CONTROL':
        raise ValueError('finish the original CPU closure first')
    audits = {name: read(public / 'assets' / f'delivery_audit_{name}.json')
              for name in ('primary', 'seed0', 'seed42', 'parent_weight')}
    if any(a['status'] != 'PASS_COMPLETE_DELIVERED_PREDICTIONS_OFFICIAL_RESULTS_AND_STUDENTS'
           for a in audits.values()):
        raise ValueError('incomplete delivered predictions or official CSVs')
    method_count = sum(len(a['methods']) for a in audits.values())
    student_count = sum(len(a['students']) for a in audits.values())
    if (method_count, student_count) != (47, 13):
        raise ValueError('the registered study is incomplete')
    additional = read(confirmations / 'additional_costs.json')
    parent = read(confirmations / 'parent_weight/summary.json')
    verdict = 'POINT_GAIN_REPLICATED_BUT_INDEPENDENT_PSR_CONTRIBUTION_NOT_ESTABLISHED'
    rows = []
    for seed, folder in ((17, public / 'full'), (0, confirmations / 'seed0'),
                         (42, confirmations / 'seed42')):
        summary = read(folder / 'summary.json')
        metrics = summary['official_metrics']
        attribution = read(folder / 'PSR/offline_attribution.json')
        rows.append(dict(seed=seed,Naive_HOTA=metrics['NAIVE_PAIRED']['HOTA'],
            PSR_HOTA=metrics['PSR']['HOTA'],Frozen_HOTA=8.218,
            Frozen_at_Naive_count=metrics['FROZEN_AT_NAIVE_PAIRED_COUNT']['HOTA'],
            Frozen_at_PSR_count=metrics['FROZEN_AT_PSR_COUNT']['HOTA'],
            PSR_at_Frozen_count=metrics['PSR_AT_FROZEN_COUNT']['HOTA'],
            PSR_present_HOTA=summary['official_query_groups']['PSR']['query_present']['HOTA'],
            PSR_empty_FP=attribution['query_empty_FP'],
            PSR_empty_nonempty_rate=attribution['query_empty_nonempty_prediction_rate'],
            **{k:attribution[k] for k in ('recovered_gt','lost_gt','net_gt_recovery',
                'new_false_positive_rows','removed_false_positive_rows','net_false_positive_change')}))
        # Preserve the earlier automated screen; do not leave a pending-screen
        # label in place of the completed, attribution-aware interpretation.
        summary.setdefault('initial_automatic_verdict',summary['scientific_verdict'])
        summary['scientific_verdict'] = verdict
        summary['final_verdict_reference'] = 'docs/PSR_FINAL_SCIENTIFIC_VERDICT.md'
        summary['completed_3seed_and_parent_control'] = True
        atomic_json(folder / 'summary.json', summary)
    with (confirmations / 'final_attribution_by_seed.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    with (public / 'full/compute_cost.csv').open() as stream:
        primary_costs = list(csv.DictReader(stream))
    new_student_hours = sum(float(r['training_GPU_hours']) + float(r['inference_GPU_hours'])
                            for r in primary_costs)
    new_student_hours += sum(r['total_student_GPU_hours']
                            for seed in ('0','42') for r in additional['methods'][seed].values())
    new_student_hours += additional['methods']['parent_weight']['total_student_GPU_hours']
    result = dict(status='COMPLETE_USER_AUTHORIZED_VAL_STUDY_STOP',
        protocol='25VAL_SAME_SET_UNLABELLED_OFFLINE_ADAPTATION_NOT_TRAIN_TO_VAL_OR_BLIND_TEST',
        project='LYQ1107/TTA-RMOT',branch='research/v2dance-psr-trainval-20261029',
        user_override='我还是没有40，你直接用val那个进行',
        scientific_verdict=verdict,formal_40Train_to_Val_complete=False,
        new_students=student_count,official_evaluation_datasets=method_count,
        episodes_per_evaluation=425,videos_per_evaluation=25,
        evaluations_include_reused_baselines_and_posthoc_count_controls=True,
        training_steps_each=8973,trainable_parameters_each=24576,
        training_target_GT_open_attempts=0,teacher_target_GT_open_attempts=0,
        teacher_sampling_seed=17,student_seeds=[17,0,42],
        aggregate=aggregate['aggregate'],per_seed_attribution=rows,
        deterministic_Frozen_HOTA=8.218,
        Frozen_is_one_reused_reference_not_three_independent_runs=True,
        parent_control=dict(HOTA=parent['official_metrics']['HOTA'],
            unique_new_weak_labels=0,unique_parent_replay_terms=526,
            parent_minus_PSR_HOTA=parent['comparisons']['PSR']['delta_HOTA'],
            parent_minus_PSR_video_CI95=parent['comparisons']['PSR']['CI95'],
            post_hoc_not_exact_gradient_intervention=True),
        primary_ordinary_controls_seed17_only=True,
        new_full_student_GPU_hours=new_student_hours,
        cost_scope='13 full student training+inference allocations; excludes inherited teacher/Frozen/source-only/front-end and separately reported interface/smoke costs',
        shared_GPU_elapsed_is_not_exclusive_device_utilization=True,
        delivery_audits={name:dict(methods=len(a['methods']),students=len(a['students']),
            path=f'assets/delivery_audit_{name}.json') for name,a in audits.items()},
        final_report='docs/PSR_FINAL_SCIENTIFIC_VERDICT.md',
        GPU_calls_in_this_finalization=0,raw_target_GT_files_opened_in_this_finalization=False,
        remaining_authorized_experiments=[],automatic_followup_training=False)
    # Execution-source audit filenames are discovered from the actual delivered
    # records rather than inventing a second set of paths.
    result['actual_execution_source_audits'] = [str(p.relative_to(public)) for p in
        sorted((public / 'assets').glob('source_revision_audit_*.json'))]
    atomic_json(public / 'final_delivery.json', result)
    recovery = read(public / 'recovery_index.json')
    recovery.update(status='COMPLETE_STOP_NO_JOBS_TO_RESUME',
        actual_scope=result['protocol'],final_delivery='final_delivery.json',
        confirmation_recovery={name:str((confirmations/name/'artifact_index.json').relative_to(public))
                               for name in ('seed0','seed42','parent_weight')},
        healthy_queue_warning='All registered GPU/CPU stages completed. Do not launch new training without a new user instruction.',
        dispatcher_warning='Historical recovery commands only; no dispatcher remains to restart.')
    atomic_json(public / 'recovery_index.json', recovery)
    print(json.dumps({k:result[k] for k in ('status','new_students','official_evaluation_datasets',
        'new_full_student_GPU_hours','scientific_verdict')}))


if __name__ == '__main__':
    close()
