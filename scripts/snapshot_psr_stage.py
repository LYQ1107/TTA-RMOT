#!/usr/bin/env python3
"""Small phase records only; no source GT, weights, raw MLLM cache or images."""
import gzip
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.run_psr_matrix import DEFAULT
RUN=ROOT/'results/runs/psr_val_20261029_v1';PUBLIC=ROOT/'results/psr'


def snapshot():
    PUBLIC.mkdir(exist_ok=True);assets=PUBLIC/'assets';assets.mkdir(exist_ok=True)
    if (RUN/'regression_stdout.txt').exists():shutil.copy2(RUN/'regression_stdout.txt',assets/'regression_stdout.txt')
    prefills=list((RUN/'evaluation/merged/control_prefills').glob('*.json'));control_summary={}
    for receipt in prefills:
        dest=assets/'control_prefills';dest.mkdir(exist_ok=True);shutil.copy2(receipt,dest/receipt.name)
        for name in json.loads(receipt.read_text())['methods']:
            if not (name.startswith('FROZEN_AT_') or name.endswith('_AT_FROZEN_COUNT')):continue
            folder=RUN/'evaluation/merged'/name;audit=json.loads((folder/'full_audit.json').read_text());meta=json.loads((folder/'run_metadata.json').read_text())
            control_summary[name]=dict(status=audit['status'],videos=audit['unique_videos'],episodes=audit['valid_episodes'],
                candidates=audit['candidates'],selected_rows=audit['selected_rows'],score_source=meta['score_source'],count_source=meta['count_source'],
                incremental_CPU_seconds=meta['incremental_CPU_wall_seconds'],GPU_calls=0,total_cost_includes_parent_models=True,
                full_audit_sha256=file_sha(folder/'full_audit.json'),run_directory=str(folder))
    if control_summary:atomic_json(assets/'control_prefill_summary.json',dict(methods=control_summary,
        GT_reads_in_partial_precompute=0,GT_evaluation_authorized_by_partial_precompute=False))
    for phase in ('interface','smoke'):
        for method in ('NAIVE_PAIRED','PSR','MEAN_TEACHER_MATCHED'):
            source=RUN/phase/method;dest=assets/phase/method;dest.mkdir(parents=True,exist_ok=True)
            for name in ('training_receipt.json','provenance.json','initial_frozen_equivalence.json','source_only_lambda_validation.json'):
                if (source/name).exists():shutil.copy2(source/name,dest/name)
    for policy in ('NAIVE','PSR','UNIFORM','CONFIDENCE'):
        folder=RUN/'teacher_full'/policy;dest=assets/'teachers'/policy;dest.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():shutil.copy2(folder/'receipt.json',dest/'receipt.json')
    states={};recovery={};predictions={}
    for method in DEFAULT:
        folder=RUN/'students'/method;receipt=folder/'training_receipt.json'
        states[method]='COMPLETE' if receipt.exists() else 'NOT_YET_COMPLETED'
        inference=RUN/'evaluation/merged'/method
        predictions[method]='SEALED_425' if (inference/'full_audit.json').exists() else 'NOT_YET_SEALED'
        if predictions[method]=='SEALED_425':
            dest=PUBLIC/'inference'/method;dest.mkdir(parents=True,exist_ok=True)
            for name in ('run_metadata.json','full_audit.json','inference_start_provenance.json','fresh_gpu_admission.json'):
                if (inference/name).exists():shutil.copy2(inference/name,dest/name)
        recovery[method]=dict(run_directory=str(folder),checkpoint=str(folder/'checkpoint_latest.pt'),
            resume_command=f'python scripts/train_psr.py --method {method} --scope full --resume --val-authorized --output {folder}')
        if not receipt.exists():continue
        dest=PUBLIC/'training'/method;dest.mkdir(parents=True,exist_ok=True)
        for name in ('training_receipt.json','provenance.json','config.json','source_validation.jsonl','teacher_fit_summary.json','teacher_fit_after_training.jsonl.gz'):
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
        with (folder/'metrics_train.jsonl').open('rb') as src,(dest/'metrics_train.jsonl.gz').open('wb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
        for attempt in (folder/'attempt_runtimes').glob('*.json'):
            target=dest/'attempt_runtimes'/attempt.name;target.parent.mkdir(exist_ok=True);shutil.copy2(attempt,target)
        for phase,source in [('training',RUN/'students'/(method+'_stdout.txt')),('inference',RUN/'evaluation/merged'/(method+'_stdout.txt'))]:
            if source.exists():
                with source.open('rb') as src,(dest/(phase+'_stdout.txt.gz')).open('wb') as dst:
                    with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as gz:shutil.copyfileobj(src,gz)
    result=dict(project='LYQ1107/TTA-RMOT',protocol='USER_AUTHORIZED25VAL_SAME_SET_PSR_NOT_FORMAL_TRAIN_TO_VAL',
        formal40Train_asset_missing=True,teacher_policies=4,teacher_direct_judgments_each=8500,
        source_videos=17,target_videos=25,target_queries=425,trainable_parameters=24576,fixed_updates=8973,
        methods=states,predictions=predictions,GT_metrics='COMPLETE_OFFLINE' if (PUBLIC/'full/summary.json').exists() else 'PENDING_UNTIL_ALL_PREDICTIONS_SEALED',old_Naive8_0327='HISTORICAL_ONLY_DIFFERENT_SAMPLING',
        current_GPU_limit=4,raw_cache_root='/data2/user/TTA-RMOT-assets/psr',private_run_root=str(RUN),
        source_and_target_GT_permissions='sourceGT legal; targetGT only independent evaluation after seal')
    atomic_json(PUBLIC/'stage_status.json',result);atomic_json(PUBLIC/'recovery_index.json',dict(methods=recovery,teacher_raw_root=str(RUN/'teacher_full'),
        pseudo_root='/data2/user/TTA-RMOT-assets/psr/val_v1',large_weights_and_raw_cache_stay_on_server=True,
        inference_resume='python scripts/run_psr_matrix.py --root '+str(RUN),
        healthy_queue_warning='Do not launch the all-method command while any original worker is healthy. Resume only failed methods, explicitly counting other active physical GPUs.',
        mean_teacher_prefetch='python scripts/prefetch_psr_mean_teacher.py --run-root '+str(RUN),
        CPU_evaluation_resume='python scripts/finish_psr_study.py --run-root '+str(RUN),
        dispatcher_warning='Both dispatchers may already be running; resume their original wait handles instead of starting duplicates.',
        after_all_inference='python scripts/prepare_psr_controls.py --training-root '+str(RUN/'students')+' --evaluation-root '+str(RUN/'evaluation/merged'),
        checkpoint_SHA_source='each completed training_receipt.json; no targetGT checkpoint selection'))
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':snapshot()
