#!/usr/bin/env python3
"""Complete report-only cost/provenance exports after the original CPU closure.

No inference, optimizer, raw target GT, checkpoint selection or threshold change.
The scientific verdict remains a human interpretation of the measured results.
"""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.analyze_psr_training_dynamics import summarize
from scripts.export_psr_results import archive


def cost_record(training,inference,method):
    training,inference=Path(training),Path(inference)
    train=json.loads((training/'training_receipt.json').read_text())
    run=json.loads((inference/'run_metadata.json').read_text())
    attempts=[json.loads(p.read_text()) for p in (training/'attempt_runtimes').glob('*.json')]
    full_seconds=sum(r['elapsed_seconds'] for r in attempts if r['scope']=='full') or train['wall_seconds']
    teacher_judgments=8500 if train['unique_teacher_samples_seen'] else 0
    return dict(method=method,training_GPU_hours=full_seconds/3600,inference_GPU_hours=run['worker_seconds']/3600,
        total_student_GPU_hours=(full_seconds+run['worker_seconds'])/3600,
        actual_training_wall_seconds=full_seconds,actual_inference_scheduler_wall_seconds=run['actual_scheduler_wall_seconds'],
        per_worker_query_frame_FPS=433636/run['worker_seconds'],per_worker_active_video_frame_FPS=25508/run['worker_seconds'],
        scheduler_query_frame_FPS=433636/run['actual_scheduler_wall_seconds'],scheduler_active_video_frame_FPS=25508/run['actual_scheduler_wall_seconds'],
        peak_training_allocated_bytes=train['peak_allocated_bytes'],peak_inference_allocated_bytes=run['peak_allocated_bytes'],
        trainable_parameters=train['trainable_parameters'],optimizer_updates=train['counters']['optimizer_updates'],
        unique_direct_teacher_labels=train['unique_teacher_samples_seen'],unique_new_weak_labels=train.get('unique_expanded_samples_seen',0),
        unique_parent_replay_terms=train.get('unique_parent_replay_terms_seen',0),
        inference_GPU_cost_normalized_excludes_CPU_lock_only_wait=True,
        required_teacher_logical_judgments=teacher_judgments,inherited_teacher_cache_not_free=bool(teacher_judgments),
        cost_scope='student full training/inference only; inherited teacher/Frozen/front-end and separate validation costs are not included here',
        FPS_scope='fixed-MOT second-stage referring only, not detector+tracker end-to-end')


def finalize():
    public=ROOT/'results/psr/confirmations';summary=json.loads((public/'summary.json').read_text())
    if summary['status']!='COMPLETE_REGISTERED_3SEED_PAIRS_AND_PARENT_CONTROL':raise ValueError('finish the original full CPU closure first')
    confirmation=ROOT/'results/runs/psr_confirm_20261030_v1';parent=ROOT/'results/runs/psr_parent_weight_20261030_v1'
    dest=public/'parent_weight';report=json.loads((dest/'summary.json').read_text());method='PSR_PARENT_WEIGHT_CONTROL'
    # The waiting CPU closure imported the earlier generic display function.
    # Recompute presentation from the same final trace: replay is NOT weak evidence.
    report['training_dynamics']=summarize(parent/'students'/method)
    report['cost']=cost_record(parent/'students'/method,parent/'evaluation/merged'/method,method)
    validation=[]
    for scope in ('interface','smoke'):
        folder=parent/scope/method;target=dest/'validation'/scope;target.mkdir(parents=True,exist_ok=True)
        receipt=json.loads((folder/'training_receipt.json').read_text())
        expected='INTERFACE_PASS_NO_OPTIMIZER_STEP' if scope=='interface' else 'STUDENT_SMOKE_PASS'
        if receipt['status']!=expected or receipt['target_GT_open_attempts']:raise ValueError('real parent-control validation failed')
        for name in ('training_receipt.json','provenance.json','config.json','initial_frozen_equivalence.json','source_only_lambda_validation.json'):
            if (folder/name).exists():shutil.copy2(folder/name,target/name)
        attempts=[json.loads(p.read_text()) for p in (folder/'attempt_runtimes').glob('*.json')]
        validation.append(dict(scope=scope,status=expected,GPU_hours=sum(a['elapsed_seconds'] for a in attempts)/3600,
            optimizer_updates=receipt.get('counters',{}).get('optimizer_updates',0),separate_fresh_validation_model_not_added_to_full_student_budget=True))
    report['separate_validation_costs']=validation
    atomic_json(dest/'summary.json',report)
    costs={}
    for seed in (0,42):
        run=confirmation/f'seed{seed}'
        costs[str(seed)]={m:cost_record(run/'students'/m,run/'evaluation/merged'/m,m) for m in ('NAIVE_PAIRED','PSR')}
    costs['parent_weight']=report['cost']
    atomic_json(public/'additional_costs.json',dict(status='COMPLETE_ACTUAL_STUDENT_COSTS',methods=costs,
        primary_seed17='../full/compute_cost.csv',validation_costs_separate=validation,
        seed_runs_reuse_identical8500_teacher_cache_without_new_raw_MLLM_calls=True,
        teacher_required_cost_source='../supervision/{NAIVE,PSR}/supervision_audit.json',
        source_Frozen_scores_and_fixed_frontend_costs_remain_inherited_not_zero=True))
    files=[]
    for pattern in ('launchers/*/*','cpu_pipeline/*/*','seed*/matrix_launchers/*/*.json'):
        files.extend((p,str(Path('confirmation')/p.relative_to(confirmation))) for p in confirmation.glob(pattern) if p.is_file())
    for pattern in ('gpu_admission.json','*_completion.json','matrix_launchers/*/*.json'):
        files.extend((p,str(Path('parent_control')/p.relative_to(parent))) for p in parent.glob(pattern) if p.is_file())
    log=archive(public/'logs/scheduler_CPU_and_parent_receipts.tar.gz',files)
    atomic_json(public/'final_augmentation.json',dict(status='COMPLETE_REPORT_ONLY_AUGMENTATION',GPU_calls=0,
        original_metrics_unchanged=True,raw_target_GT_opened=False,additional_logs=log,
        parent_unique_new_weak_labels=0,parent_replay_is_not_new_semantic_evidence=True,
        scientific_success_not_implied=True))
    print(json.dumps(dict(status='PSR_FINAL_CONFIRMATION_AUGMENTATION_COMPLETE',GPU_calls=0,additional_log_files=len(files))))


if __name__=='__main__':finalize()
