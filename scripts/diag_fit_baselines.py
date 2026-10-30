#!/usr/bin/env python3
"""Existing checkpoint teacher-fit baselines, no training and no target GT."""
import gc
import gzip
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import compat.sitecustomize
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.diag_fit import metrics,fit_subset
from flex_uda.training import protect_target_annotations,Inputs,source_validate
from flex_uda.mllm_training import enforce_inputs,teacher_fit,measure_fit_subset,study_frame_groups
from flex_uda.teacher.psr_dataset import PSRPseudoDataset
from flex_uda.data.source_kitti_v2 import SourceSupervision
from flex_uda.evaluation import load_selected
from scripts.launch_uda_workers import audit_gpus

RUN=ROOT/'results/runs/diag_fit_20261030_v1'
PREVIOUS=ROOT/'results/runs/psr_val_20261029_v1/students'


def run():
    started=time.perf_counter()
    selected,_,_=audit_gpus(1,minimum_free_mib=20480);gpu=selected[0]
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],UDA_PHYSICAL_GPU=str(gpu['physical_gpu']),HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    out=RUN/'baselines';out.mkdir(parents=True,exist_ok=True);attempts=protect_target_annotations()
    pseudo=PSRPseudoDataset('/data2/user/TTA-RMOT-assets/psr/val_v1/NAIVE',allow_val_diagnostic=True,expanded_kind='none')
    groups,sha=fit_subset(pseudo,out);sample_ids={r['sample_id'] for g in groups for r in g['samples']}
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction((16384<<20)/torch.cuda.get_device_properties(device).total_memory)
    torch.empty(1,device=device)
    readable=[p for method in ('NAIVE_PAIRED','PSR') for p in (PREVIOUS/method).iterdir() if p.is_file()]
    isolation=enforce_inputs(out,pseudo,extra_read=readable)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    answer={};cache={}
    for method in ('NAIVE_PAIRED','PSR'):
        folder=PREVIOUS/method;checkpoint=folder/'checkpoint_best_source_selected.pt'
        original=json.loads((folder/'training_receipt.json').read_text())
        if file_sha(checkpoint)!=original['checkpoints'][checkpoint.name]['sha256']:raise ValueError('old checkpoint changed')
        with gzip.open(folder/'teacher_fit_after_training.jsonl.gz','rt') as stream:own=[json.loads(s) for s in stream]
        source_trace=[json.loads(s) for s in (folder/'source_validation.jsonl').read_text().splitlines()]
        model,tokenizer,info=load_selected(checkpoint,out/method,Path(pseudo.runtime[0]['predicted_tracks']).parent,device)
        if info['selected_step']!=original['steps']:raise ValueError('cached latest fit is not the selected checkpoint; explicitly recompute instead')
        if method=='NAIVE_PAIRED':
            if {r['sample_id'] for r in own}!={r['sample_id'] for r in pseudo.rows}:raise ValueError('old Naive fit labels differ')
            cache=own
            answer['Frozen']=dict(all=metrics([dict(r,student_margin=r['source_margin']) for r in own]),
                subset=metrics([dict(r,student_margin=r['source_margin']) for r in own if r['sample_id'] in sample_ids]),
                trainable_parameters=0,pos_weight=None,source_val_CE=source_trace[0]['source_val_CE'],source_val_CE_step0=source_trace[0]['source_val_CE'],
                model='official frozen V2 cached full-bank margins; no new Frozen inference')
            answer['Naive(s17)']=dict(all=metrics(own),subset=metrics([r for r in own if r['sample_id'] in sample_ids]),
                trainable_parameters=24576,pos_weight=None,source_val_CE=source_trace[-1]['source_val_CE'],source_val_CE_step0=source_trace[0]['source_val_CE'],
                cached_fit_reused=True,checkpoint=info)
            # Real strict load is exercised, without repeating full GPU fit.
            from flex_uda.mllm_training import target_batch
            from flex_uda.model.flexhook_uda import forward_uda
            inputs=Inputs(model,tokenizer,device,None)
            first=study_frame_groups(groups[0])[0];prepared,_,_,rows=target_batch(inputs,first)
            with torch.no_grad():value=forward_uda(model,prepared,return_features=False)['margin'].tolist()
            expected={r['sample_id']:r['student_margin'] for r in own}
            if any(abs(m-expected[r['sample_id']])>1e-4 for r,m in zip(rows,value)):raise ValueError('actual old strict-loaded Naive fit differs')
            del inputs
        else:
            # All new arms use Naive labels. Also show old PSR's own label fit
            # separately, never silently exchange 8500 different observations.
            source=SourceSupervision(ROOT/'manifests/source_kitti_v2_val.jsonl')
            inputs=Inputs(model,tokenizer,device,source)
            from types import SimpleNamespace
            (out/'PSR_common_Naive').mkdir(exist_ok=True)
            fit=teacher_fit(model,inputs,SimpleNamespace(groups=[part for group in pseudo.groups for part in study_frame_groups(group)]),out/'PSR_common_Naive')
            subset=measure_fit_subset(model,inputs,groups,info['selected_step'],source_trace[-1]['source_val_CE'],sha)
            answer['PSR(s17)']=dict(all=fit,subset=subset,own_PSR_labels=metrics(own),trainable_parameters=24576,pos_weight=None,
                source_val_CE=source_trace[-1]['source_val_CE'],source_val_CE_step0=source_trace[0]['source_val_CE'],checkpoint=info)
            del inputs
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    if attempts:raise RuntimeError('baseline teacher-fit opened targetGT')
    atomic_json(out/'completion.json',dict(status='COMPLETE_EXISTING_CHECKPOINT_FIT_BASELINES',methods=answer,
        fit_subset_sha256=sha,common_teacher_labels='NAIVE8500',own_PSR_fit_separately_disclosed=True,
        target_GT_open_attempts=0,isolation=isolation,new_training_steps=0,full_RMOT_inference_started=False,
        GPU_process_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cost_scope='one admitted GPU-process body; cached Frozen/Naive fit reuse and old training/teacher costs remain inherited'))
    print(json.dumps(dict(event='D_FIT_BASELINES_COMPLETE',methods={m:r['all']['teacher_POS_selected_rate'] for m,r in answer.items()})),flush=True)


if __name__=='__main__':run()
