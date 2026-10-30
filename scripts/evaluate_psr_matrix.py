#!/usr/bin/env python3
"""Complete read-only matrix inference and GT-free controls; GT is a later process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.launch_mllm_evaluation import launch
from scripts.prepare_mllm_evaluation_controls import stage_reference,count_matched
from scripts.pf_diag_common import FROZEN
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.training import protect_target_annotations
from scripts.train_psr import selected_checkpoint


def evaluate_matrix(training,output,methods,workers=4):
    training,output=Path(training).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    attempts=protect_target_annotations();merged=output/'merged';merged.mkdir(exist_ok=True)
    for method in methods:
        checkpoint=selected_checkpoint(training/method)
        receipt=json.loads((checkpoint.parent/'training_receipt.json').read_text())
        if receipt['status']!='COMPLETE_FIXED_COVERAGE_PSR_STUDY' or receipt['steps']!=8973 or receipt['target_GT_open_attempts']:
            raise ValueError('complete registered study training before inference: '+method)
        # Separate each scheduler clock/shard root; do not charge later methods
        # from the first method's start epoch or overwrite its healthy receipts.
        root=output/'model_runs'/method
        launch(checkpoint,root,method=method,workers=workers)
        target=root/method;link=merged/method
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        elif link.resolve()!=target.resolve():raise ValueError('different existing sealed method')
    stage_reference(FROZEN/'ZERO_UPDATE',merged,'FROZEN')
    stage_reference(ROOT/'results/runs/uda_val_evaluation_20261026_v1/SOURCE_REHEARSAL',merged,'SOURCE_REHEARSAL')
    comparisons=[]
    for method in methods:
        left='FROZEN_AT_'+method+'_COUNT';right=method+'_AT_FROZEN_COUNT'
        count_matched(merged,left,'FROZEN',method);count_matched(merged,right,method,'FROZEN')
        comparisons.extend((left,right))
    metadata=dict(status='ALL_SELECTED_METHODS_425_PREDICTIONS_AND_GT_FREE_CONTROLS_SEALED',methods=['FROZEN','SOURCE_REHEARSAL',*methods,*comparisons],
        student_methods=methods,episodes_each=425,video_count=25,target_GT_open_attempts=attempts,
        immutable_geometry_and_predicted_IDs=True,protocol='VAL_SAME_SET_PSR_STUDY_NOT_TRAIN_TO_VAL',
        no_teacher_at_test_time=True,test_time_optimizer_steps=0)
    atomic_json(merged/'controls_sealed.json',metadata)
    print(json.dumps(metadata,indent=2));return merged


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['NAIVE_PAIRED','PSR']);p.add_argument('--workers',type=int,default=4);a=p.parse_args();evaluate_matrix(a.training_root,a.output,a.methods,a.workers)
