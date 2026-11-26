"""Reuse isolated upstream YesTrack functions without executing its GT imports.

YesTrack ed42b8315b28c94dc595d83b0d49bf57002bb281, MIT.
Copyright (c) 2026 Quansheng Hu and YesTrack contributors.
License is preserved in docs/licenses/YesTrack-MIT.txt.
"""
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
from peft import PeftModel


def upstream_functions(source):
    """Compile only audited definitions; never import llm_eval/llm_train/datasets."""
    source=Path(source)
    selected={
        'refer_llm/crop_utils.py':{'crop_with_margin','xywh_to_xyxy_with_margin'},
        'refer_llm/llm_eval.py':{'build_model_and_processor_for_eval','get_yes_no_token_ids'},
        'refer_llm/infer_utils.py':{'build_and_forward_yes_probs'},
        'refer_llm/llm_eval_from_mot.py':{'_load_mot_file','_prepare_patches_for_frame','_load_images_np_cache',
            '_prepare_video_patches_for_samples','_forward_probs_for_patches','_forward_probs_for_detections',
            '_forward_probs_for_detections_video'},
    }
    namespace=dict(globals(),logger=logging.getLogger('safe_yestrack'),HALLUCINATION_TOTAL_SAMPLES=0,
        HALLUCINATION_TOTAL_HITS=0)
    import re
    namespace['re']=re
    for relative,names in selected.items():
        path=source/relative;tree=ast.parse(path.read_text(),filename=str(path))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
        if {n.name for n in nodes}!=names:raise ValueError('upstream function inventory changed')
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        safe=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
        exec(compile(safe,str(path),'exec'),namespace)
    return namespace


def parameter_sha256(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(str(tuple(p.shape)).encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenYesTrack:
    def __init__(self,config,device='cuda:0'):
        self.config=config;self.device=torch.device(device);self.upstream=upstream_functions(config['yestrack_source'])
        adapter=config.get('lora_path')
        if config['condition']=='T1' and (not adapter or not(Path(adapter)/'adapter_config.json').exists()):
            raise FileNotFoundError('T1 requires the audited V2 adapter; never silently fall back to T0/V1')
        started=time.perf_counter()
        self.model,self.processor,(self.yes_id,self.no_id)=self.upstream['build_model_and_processor_for_eval'](
            config['model_path'],use_4bit=False,bf16=True,lora_path=adapter)
        self.model.requires_grad_(False).eval().to(self.device)
        self.model.config.use_cache=False
        self.versions={name:p._version for name,p in self.model.named_parameters()}
        self.weight_sha=parameter_sha256(self.model)
        self.load_seconds=time.perf_counter()-started
        self.capture=[];self.calls=0;self.last_inputs={}
        self.pre_hook=self.model.register_forward_pre_hook(self._before,with_kwargs=True)
        self.post_hook=self.model.register_forward_hook(self._after,with_kwargs=True)

    def _before(self,module,args,kwargs):
        self.last_inputs=kwargs

    def _after(self,module,args,kwargs,output):
        logits=output.logits[:,-1,:].detach()
        binary=logits[:,[self.yes_id,self.no_id]]
        # Preserve original YesTrack softmax precision; also store FP32 diagnostic.
        probabilities=torch.softmax(binary,dim=-1)[:,0].float().cpu().tolist()
        probabilities_fp32=torch.softmax(binary.float(),dim=-1)[:,0].cpu().tolist()
        top=logits.argmax(-1).cpu().tolist();raw=binary.float().cpu().tolist()
        ids=kwargs['input_ids'];mask=kwargs.get('attention_mask',torch.ones_like(ids))
        image_id=getattr(self.model.config,'image_token_id',151655)
        video_id=getattr(self.model.config,'video_token_id',151656)
        for i,(p,q,t,logit) in enumerate(zip(probabilities,probabilities_fp32,top,raw)):
            visual=int(((ids[i]==image_id)|(ids[i]==video_id)).sum().item())
            total=int(mask[i].sum().item())
            self.capture.append(dict(yes_logit=logit[0],no_logit=logit[1],yes_probability=p,
                yes_probability_fp32=q,top_token_id=t,top_token=self.processor.tokenizer.convert_ids_to_tokens(t),
                top1_is_yes_no=t in (self.yes_id,self.no_id),visual_tokens=visual,language_tokens=total-visual,
                input_tokens=total,yes_token_id=self.yes_id,no_token_id=self.no_id))
        self.calls+=1

    @torch.inference_mode()
    def forward(self,images,prompts):
        """A/C/D share original chat/forward path; images is a list per sample."""
        if len(images)!=len(prompts):raise ValueError('image/prompt batch differs')
        self.capture=[];torch.cuda.synchronize(self.device);started=time.perf_counter()
        # Exact upstream single-image function, not a reimplemented baseline.
        if all(len(group)==1 for group in images):
            self.upstream['_forward_probs_for_patches'](self.model,self.processor,self.device,
                [group[0] for group in images],prompts,self.yes_id,self.no_id)
        else:
            # Same upstream multi-image chat structure; additionally supports the
            # necessary no-image control. No dataset/evaluator is constructed.
            messages=[[dict(role='user',content=[*[dict(type='image',image=im) for im in group],
                dict(type='text',text=prompt)])] for group,prompt in zip(images,prompts)]
            self.processor.tokenizer.padding_side='left'
            inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt',padding=True).to(self.device)
            self.model(**inputs)
        torch.cuda.synchronize(self.device);elapsed=time.perf_counter()-started
        self.check_versions()
        if len(self.capture)!=len(prompts):raise ValueError('did not capture every actual teacher response')
        for r,group in zip(self.capture,images):
            if not all(math.isfinite(r[k]) for k in ('yes_logit','no_logit','yes_probability')):
                raise FloatingPointError('teacher logits/probabilities are not finite')
            r.update(batch_forward_seconds=elapsed,batch_size=len(prompts),images=len(group),
                allocated_peak_bytes=torch.cuda.max_memory_allocated(self.device),frozen_weight_sha256=self.weight_sha)
        return self.capture

    def check_versions(self):
        if any(p.requires_grad or p._version!=self.versions[n] for n,p in self.model.named_parameters()):
            raise RuntimeError('frozen teacher parameters changed')

    def seal_frozen(self):
        self.check_versions();final=parameter_sha256(self.model)
        if final!=self.weight_sha:raise RuntimeError('full frozen teacher parameter hash changed')
        return dict(initial_parameter_sha256=self.weight_sha,final_parameter_sha256=final,
            frozen_parameters_unchanged=True,optimizer_updates=0,forward_batches=self.calls,
            model_load_and_initial_hash_seconds=self.load_seconds)
