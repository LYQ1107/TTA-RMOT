#!/usr/bin/env python3
"""Seal GT-free PSR controls from existing full outputs, NEVER rerun inference."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.training import protect_target_annotations
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.prepare_mllm_evaluation_controls import stage_reference,count_matched
from scripts.pf_diag_common import FROZEN
from scripts.run_psr_matrix import DEFAULT


def prepare(root,training,methods,precompute_only=False):
    root,training=Path(root).resolve(),Path(training).resolve()
    claim=(root/'controls_build.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX)
    attempts=protect_target_annotations();registered={}
    for method in methods:
        audit=json.loads((root/method/'full_audit.json').read_text())
        run=json.loads((root/method/'run_metadata.json').read_text())
        train=json.loads((training/method/'training_receipt.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425 or audit['unique_videos']!=25:
            raise ValueError('all registered full predictions must be complete: '+method)
        if train['steps']!=8973 or train['target_GT_open_attempts'] or run['target_GT_open_attempts'] or run['test_time_updates']:
            raise ValueError('wrong training/evaluation protocol: '+method)
        if file_sha(training/method/'checkpoint_best_source_selected.pt')!=run['checkpoint_sha256']:
            raise ValueError('selected checkpoint changed after inference: '+method)
        cli=[json.loads(p.read_text()) for p in (root/method/'cli_inference_attempts').glob('*.json')]
        if cli:
            # A queued duplicate can wait for a different lane without loading
            # CUDA. Do not falsely charge that CPU wait as GPU inference time.
            pids={r['pid'] for r in cli}
            legacy=[json.loads(p.read_text()) for p in (root/method/'inference_attempts').glob('*.json')]
            legacy_seconds=sum(r['elapsed_seconds'] for r in legacy if r['pid'] not in pids)
            run.update(scheduler_worker_slot_seconds=run.get('scheduler_worker_slot_seconds',run['worker_seconds']),worker_seconds=legacy_seconds+sum(r['GPU_worker_seconds'] for r in cli),
                CPU_lock_wait_seconds=sum(r['CPU_lock_wait_seconds'] for r in cli),CLI_attempts=cli,
                zero_GPU_reuse_attempts=sum(r['actual_GPU_calls']==0 for r in cli))
            atomic_json(root/method/'run_metadata.json',run)
        registered[method]=dict(checkpoint_sha256=run['checkpoint_sha256'],
            training_receipt_sha256=file_sha(training/method/'training_receipt.json'),
            full_audit_sha256=file_sha(root/method/'full_audit.json'),
            run_metadata_sha256=file_sha(root/method/'run_metadata.json'))
    stage_reference(FROZEN/'ZERO_UPDATE',root,'FROZEN')
    stage_reference(ROOT/'results/runs/uda_val_evaluation_20261026_v1/SOURCE_REHEARSAL',root,'SOURCE_REHEARSAL')
    controls=[]
    for method in methods:
        for name,score,count in [('FROZEN_AT_'+method+'_COUNT','FROZEN',method),(method+'_AT_FROZEN_COUNT',method,'FROZEN')]:
            count_matched(root,name,score,count);controls.append(name)
    result=dict(status='ALL_REGISTERED_FULL_PREDICTIONS_AND_GT_FREE_CONTROLS_SEALED',
        methods=['FROZEN','SOURCE_REHEARSAL',*methods,*controls],student_methods=methods,
        registered_student_seals=registered,episodes_each=425,video_count=25,
        target_GT_open_attempts=attempts,immutable_geometry_and_predicted_IDs=True,
        protocol='USER_AUTHORIZED_25VAL_SAME_SET_NOT_TRAIN_TO_VAL',
        count_matched='POST_HOC_DIAGNOSTIC_NOT_INDEPENDENT_DEPLOYABLE_METHOD',
        no_teacher_at_test_time=True,test_time_optimizer_steps=0,inference_rerun=False)
    if precompute_only:
        result.update(status='PARTIAL_REGISTERED_CONTROLS_PRECOMPUTED_NOT_GT_EVALUATION_READY',GT_evaluation_authorized=False,
            remaining_registered_students=[m for m in DEFAULT if not (root/m/'full_audit.json').exists()])
        atomic_json(root/'control_prefills'/f'{time.time_ns()}.json',result)
    else:atomic_json(root/'controls_sealed.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-root',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--methods',nargs='+',default=DEFAULT)
    p.add_argument('--precompute-only',action='store_true',help='prepare completed subset without publishing the final GT evaluation seal')
    a=p.parse_args();prepare(a.evaluation_root,a.training_root,a.methods,a.precompute_only)
