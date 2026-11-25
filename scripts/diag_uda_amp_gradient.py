#!/usr/bin/env python3
"""Inspect a completed failed source worker's first unfinished real sample.

No optimizer step, no target GT, no hyperparameter search: test whether the
saved AMP loss scale causes FP16 gradient overflow and FP32/lower scales do not.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.launch_uda_workers import audit_gpus

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--failed-worker',type=Path,required=True);a=p.parse_args()
    worker=a.failed_worker.resolve();selected,_,_=audit_gpus(1);gpu=selected[0]
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid'];os.environ['UDA_PHYSICAL_GPU']=str(gpu['physical_gpu'])
    import torch
    from transformers import RobertaTokenizerFast
    from flex_uda.training import Inputs,protect_target_annotations
    from flex_uda.data.source_kitti_v2 import SourceSupervision
    from flex_uda.data.manifests import read_jsonl
    from flex_uda.model.flexhook_uda import forward_uda
    from flex_uda.evaluation import load_selected
    from flex_uda.losses.objectives import supervised_ce
    from flex_tta.runtime.pf_receipts import atomic_json
    attempts=protect_target_annotations()
    records=read_jsonl(worker/'metrics_train.jsonl');failed_step=records[-1]['step']+1
    plan=read_jsonl(worker/'plans/source_train_plan.jsonl');row=plan[(failed_step-1)%len(plan)]
    checkpoint=worker/'checkpoint_latest.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    scale=state['scaler']['scale'];saved_step=state['step'];del state
    runtime=read_jsonl(ROOT/'manifests/dance_val_runtime.jsonl');device=torch.device('cuda:0')
    model,tokenizer,_=load_selected(checkpoint,worker/'amp_diagnostic',Path(runtime[0]['predicted_tracks']).parent,device)
    for name,param in model.named_parameters():param.requires_grad_('lora_' in name)
    source=SourceSupervision(ROOT/'manifests/source_kitti_v2_train.jsonl')
    batches=list(Inputs(model,tokenizer,device,source).batches('source',row));params=[p for p in model.parameters() if p.requires_grad]
    result=[];n=sum(len(c) for _,_,c,_ in batches)
    for amp,factor in ((True,1.),(True,.5),(True,.25),(False,1.)):
        model.zero_grad(set_to_none=True);objective=0.
        for prepared,labels,candidates,_ in batches:
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                logits=forward_uda(model,prepared,return_features=False)['all_logits']
                loss=supervised_ce(logits,labels)*(len(candidates)/n)
            (loss*(scale*factor if amp else 1.)).backward();objective+=float(loss.detach())
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
        result.append(dict(AMP=amp,loss_scale=scale*factor if amp else 1.,source_loss=objective,all_gradients_finite=finite))
    report=dict(status='DIAGNOSTIC_ONLY',failed_step=failed_step,loaded_checkpoint_step=saved_step,
        parameters_are_latest_valid_checkpoint_not_unsaved_failed_state=True,source_video=row['video_id'],
        source_expression=row['expression_id'],source_frame=row['frame'],gpu=gpu,results=result,
        optimizer_updates=0,target_GT_open_attempts=attempts)
    atomic_json(ROOT/'results/uda/assets/amp_real_source_diagnostic.json',report)
    print(json.dumps(report,indent=2))
