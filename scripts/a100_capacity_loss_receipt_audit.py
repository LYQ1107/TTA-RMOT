"""Independently verify deployed ordinary heads and reconcile legacy CE fields.

The capacity helper's final KL is evaluated correctly, but merging its
receipt into an older receipt retained that older CE/entropy/architecture
description. Preserve the original records and publish explicit final fields.
No live numerical helper, weight, prediction, GT result or decision changes.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_six_direction import PUBLIC
from scripts.a100_object_reference_capacity import OUT as CAPACITY
from scripts.a100_object_reference_fit_complete import OUT as PREDECESSOR


def main():
    start=time.perf_counter();guard=generation_guard();torch.set_num_threads(1)
    cases=json.loads((PUBLIC/'frame_fit_representation_audit.json').read_text())['failed_prefixes']
    rows=[];cache={}
    for case in cases:
        pair,seed,video,prefix=[case[k] for k in ('pair','seed','video','prefix_videos')]
        key=pair,seed
        if key not in cache:
            cache[key]=torch.load(PREDECESSOR/pair/f'seed{seed}'/'stream.pt',map_location='cpu',weights_only=False)
        stream=cache[key];x=np.concatenate(stream['bx'][:prefix]);raw_y=np.concatenate(stream['by'][:prefix])
        root=CAPACITY/pair/f'seed{seed}'/'videos'/video
        checkpoint=torch.load(root/'capacity_head.pt',map_location='cpu',weights_only=False)
        receipt=json.loads((root/'receipt.json').read_text())['fits']['FRAME_CAPACITY']
        width,dimension=checkpoint['model']['0.weight'].shape
        model=nn.Sequential(nn.Linear(dimension,width),nn.GELU(),nn.Linear(width,1))
        model.load_state_dict(checkpoint['model']);model.eval()
        xx=torch.as_tensor(((x-checkpoint['mean'])/checkpoint['std']).astype('float32'))
        target=torch.as_tensor(np.clip(raw_y,.02,.98).astype('float32')).double()
        weight=torch.as_tensor((1.+3.*np.abs(2.*raw_y-1.)).astype('float32')).double()
        with torch.no_grad():logits=model(xx).squeeze(-1).double()
        entropy=float(weight@(-target*torch.log(target)-(1-target)*torch.log1p(-target))/weight.sum())
        ce=float(weight@torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')/weight.sum())
        kl=ce-entropy;difference=abs(kl-receipt['teacher_KL'])
        assert difference<1e-5 and kl<=.005,(pair,seed,video,kl,difference)
        rows.append(dict(pair=pair,seed=seed,video=video,prefix_videos=prefix,examples=len(target),
            actual_architecture=f'{dimension}->{width}GELU->1',
            legacy_merged_teacher_CE=receipt['teacher_CE'],legacy_merged_teacher_entropy=receipt['teacher_entropy'],
            legacy_merged_CE_minus_entropy=receipt['teacher_CE']-receipt['teacher_entropy'],
            original_64_unit_teacher_KL=receipt['original_64_unit_teacher_KL'],
            reported_final_deployed_teacher_KL=receipt['teacher_KL'],
            independently_recomputed_CPU_teacher_CE=ce,independently_recomputed_teacher_entropy=entropy,
            independently_recomputed_CPU_teacher_KL=kl,CPU_vs_GPU_KL_error=difference,
            final_deployed_head_passes_original_gate=True,
            final_CE_identity_source='CPUfloat32deployedlogits;doubleprecisionweightedBernoulliCEandentropy;sameoriginalfloat32targetsandweights'))
    assert not guard
    result=dict(status='PASS_INDEPENDENT_DEPLOYED_CAPACITY_HEAD_LOSS_RECEIPT_AUDIT',heads=rows,
        explanation='LegacymergedCE/entropy/architecturefieldsbelongtotheearlier64-unitfit;finalteacherKLwasalreadycomputedfromtheactualdeployedcapacityheadandusedforthegate.Thisappendixauditreconcilesfinalmetricswithoutrewritinghistoricalrecords.',
        live_source_or_model_or_prediction_or_GT_result_changed=False,
        model_selection_or_HOTA_conclusions_changed=False,
        all5_newly_completed_prefixes_independently_verified=True,
        teacher_KL_gate=.005,ordinary_head_CPU_forward_checks=len(rows),
        manual_target_GT_or_quality_predictions_read=0,new_teacher_encoder_or_training_calls=0,
        remaining_live_recipe_note='Currentregisteredpolicyworkerskeepthesamenumericalhelper;foranycapacitycontinuationitsKL/widthfieldsarefinalbutinheritedCE/entropy/architecturetextmustnotbemisinterpreted.Futuremetadata-onlyfixmustwaituntilallusersofthathelperfinish.',
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    (PUBLIC/'capacity_loss_receipt_audit.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],heads=len(rows),all_original_gates_pass=True,
        maximum_CPU_vs_GPU_KL_error=max(r['CPU_vs_GPU_KL_error'] for r in rows),
        legacy_CE_fields_reconciled=True,weights_predictions_GT_unchanged=True)))


if __name__=='__main__':main()
