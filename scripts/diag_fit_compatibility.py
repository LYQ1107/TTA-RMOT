#!/usr/bin/env python3
"""Actual GPU zero-residual and strict-checkpoint compatibility; no target GT."""
import argparse
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize
import torch
import yaml
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main(mode,output,legacy):
    from scripts.launch_uda_workers import audit_gpus
    selected,_,_=audit_gpus(1,minimum_free_mib=20480)
    registered=os.environ.get('PSR_PHYSICAL_GPU')
    gpu=next((g for g in selected if str(g['physical_gpu'])==registered),selected[0])
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],MLLM_PHYSICAL_GPU=str(gpu['physical_gpu']),
        MLLM_VAL_DIAGNOSTIC_AUTHORIZED='YES',HF_HUB_OFFLINE='1')
    if mode=='legacy':
        spec=importlib.util.spec_from_file_location('prechange_training',legacy/'mllm_training.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=yaml.safe_load((legacy/'psr.yaml').read_text());config['method']='NAIVE_PAIRED'
        config['data'].update(pseudo_directory='/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE',expanded_kind='none')
        module.run(config,output,scope='smoke');return
    from flex_uda.training import protect_target_annotations,CHECKPOINT,DANCE,Inputs
    from flex_uda.mllm_training import enforce_inputs,target_batch,study_frame_groups
    from flex_uda.teacher.psr_dataset import PSRPseudoDataset
    from flex_uda.model.flexhook_uda import attach_adapter,forward_uda
    from scripts.run_flexhook_dance_pcd_tta import _load_model
    from transformers import RobertaTokenizerFast
    attempts=protect_target_annotations();output.mkdir(parents=True,exist_ok=False)
    pseudo=PSRPseudoDataset('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE',allow_val_diagnostic=True,expanded_kind='none')
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory)
    torch.empty(1,device=device);isolation=enforce_inputs(output,pseudo)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,model,_=_load_model(ROOT/'vendor/FlexHook',CHECKPOINT,DANCE,Path(pseudo.runtime[0]['predicted_tracks']).parent,output/'model_log',device,24,strict_checkpoint=True)
    tokenizer=RobertaTokenizerFast.from_pretrained(str(ROOT/'vendor/FlexHook/pretrained/roberta-base'))
    inputs=Inputs(model,tokenizer,device,None)
    prepared,_,_,_=target_batch(inputs,study_frame_groups(pseudo.groups[0])[0])
    with torch.no_grad():reference=forward_uda(model,prepared,return_features=False)['all_logits'].clone()
    results=[]
    for alpha in (1.,8.,32.):
        torch.manual_seed(17)
        adapter=attach_adapter(model,device,alpha=alpha)
        with torch.no_grad():actual=forward_uda(model,prepared,return_features=False)['all_logits']
        if not torch.equal(reference,actual):raise RuntimeError('actual zero-B initial prediction differs')
        results.append(dict(alpha=alpha,bitwise_equal=True))
        for index in adapter.layer_indices:
            attention=model.output_ca_qkv[index].attention
            for name in ('wq','wv'):setattr(attention,name,getattr(attention,name).base)
        del adapter
    if attempts:raise RuntimeError('GT opened during GPU compatibility')
    atomic_json(output/'receipt.json',dict(status='PASS_REAL_GPU_ZERO_B_FORWARD_BITWISE',checks=results,
        target_GT_open_attempts=attempts,isolation=isolation,physical_gpu=gpu['physical_gpu']))
    print(json.dumps(dict(event='D_FIT_REAL_GPU_PARITY_PASS',checks=results)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('parity','legacy'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-root',type=Path)
    a=p.parse_args();main(a.mode,a.output,a.legacy_root)
