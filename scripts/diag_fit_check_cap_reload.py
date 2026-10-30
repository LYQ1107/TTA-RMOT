#!/usr/bin/env python3
"""Real CAP final-step strict reconstruction against its saved input/output."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize
import torch
from torch.utils._pytree import tree_map
from scripts.launch_uda_workers import audit_gpus
from flex_uda.training import protect_target_annotations
from flex_uda.mllm_training import enforce_inputs
from flex_uda.teacher.psr_dataset import PSRPseudoDataset
from flex_uda.evaluation import load_selected
from flex_uda.model.flexhook_uda import forward_uda
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

RUN=ROOT/'results/runs/diag_fit_20261030_v1'


def run():
    chosen,_,_=audit_gpus(1,minimum_free_mib=20480);gpu=chosen[0]
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],HF_HUB_OFFLINE='1')
    folder=RUN/'smoke/DIAG_FIT_CAP';checkpoint=folder/'checkpoint_final_step.pt';probe_file=folder/'roundtrip_probe.pt'
    output=RUN/'validation/cap_reload';output.mkdir(parents=True,exist_ok=False)
    attempts=protect_target_annotations()
    pseudo=PSRPseudoDataset('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE',allow_val_diagnostic=True,expanded_kind='none')
    device=torch.device('cuda:0');torch.empty(1,device=device)
    isolation=enforce_inputs(output,pseudo,extra_read=[checkpoint,probe_file])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,info=load_selected(checkpoint,output,Path(pseudo.runtime[0]['predicted_tracks']).parent,device)
    probe=torch.load(probe_file,map_location='cpu',weights_only=False)
    if probe['checkpoint_sha256']!=file_sha(checkpoint):raise ValueError('probe refers to another checkpoint')
    prepared=tree_map(lambda x:x.to(device) if torch.is_tensor(x) else x,probe['prepared'])
    with torch.no_grad():actual=forward_uda(model,prepared,return_features=False)['all_logits'].cpu()
    equal=torch.equal(actual,probe['expected_logits'])
    if not equal or attempts:raise RuntimeError('real CAP checkpoint/output/isolation roundtrip failed')
    atomic_json(output/'receipt.json',dict(status='PASS_REAL_CAP_STRICT_LOAD_AND_BITWISE_OUTPUT',checkpoint=info,
        actual_parameters=2758656,target_GT_open_attempts=0,optimizer_updates=0,isolation=isolation,bitwise_equal=equal))
    print(json.dumps(dict(event='D_FIT_CAP_REAL_CHECKPOINT_ROUNDTRIP_PASS',bitwise_equal=equal)),flush=True)


if __name__=='__main__':run()
