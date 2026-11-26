"""An isolated Qwen adapter; the original reference tensors stay unchanged.

This runtime does not decide which student feedback is correct. A future
research protocol must specify its visual supervision and fair KD controls.
"""
import contextlib
import hashlib
from pathlib import Path
import time

import torch
from torch import nn


class TrainableQwen:
    def __init__(self, root, device='cuda:0'):
        import yaml
        from peft import LoraConfig, TaskType, get_peft_model
        from flex_uda.a100.runtime import relocate
        from flex_uda.teacher.yestrack_backend import upstream_functions
        root = Path(root); self.device = device
        config = yaml.safe_load((root/'configs/teacher/base_qwen2b.yaml').read_text())
        upstream = upstream_functions(relocate(config['yestrack_source']))
        started = time.perf_counter()
        base, self.processor, token_ids = upstream['build_model_and_processor_for_eval'](
            str(root/'assets/teacher_Qwen3-VL-2B-Instruct'), use_4bit=False, bf16=True, lora_path=None)
        self.yes_id, self.no_id = token_ids
        base.requires_grad_(False).eval().to(device)
        base.config.use_cache = False
        self.base_parameters = list(base.named_parameters())
        self.base_versions = [p._version for _, p in self.base_parameters]
        self.initial_base_sha256 = self.base_hash()
        targets = [name for name, _ in base.named_modules() if '.language_model.layers.' in name
                   and (name.endswith('.self_attn.q_proj') or name.endswith('.self_attn.v_proj'))]
        assert targets, 'Do not silently adapt unrelated vision modules'
        adapter = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                             lora_dropout=0., target_modules=targets, bias='none')
        torch.manual_seed(0)
        self.model = get_peft_model(base, adapter).to(device).eval()
        self.trainable = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        assert self.trainable and all('lora_' in name for name, _ in self.trainable)
        self.optimizer = torch.optim.AdamW([p for _, p in self.trainable], lr=1e-5, weight_decay=.01)
        torch.cuda.synchronize()
        self.model_load_and_base_hash_wall_seconds = time.perf_counter()-started
        self.forward_batches = self.gradient_steps = 0
        self.assert_base_frozen()

    def base_hash(self):
        digest = hashlib.sha256()
        for name, parameter in self.base_parameters:
            digest.update(name.encode()); digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def assert_base_frozen(self):
        assert all(not p.requires_grad and p._version == version
                   for (_, p), version in zip(self.base_parameters, self.base_versions))

    def inputs(self, images, prompts):
        assert len(images) == len(prompts) and all(len(group) for group in images)
        messages = [dict(role='user', content=[*[dict(type='image', image=im) for im in group],
                                               dict(type='text', text=prompt)])
                    for group, prompt in zip(images, prompts)]
        self.processor.tokenizer.padding_side = 'left'
        # Process the actual multimodal prefix rather than tokenize its raw
        # image placeholders separately. No completion-boundary approximation.
        return self.processor.apply_chat_template([[m] for m in messages], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=True).to(self.device)

    def logits(self, inputs, reference=False):
        context = self.model.disable_adapter() if reference else contextlib.nullcontext()
        with context:
            result = self.model(**inputs, use_cache=False, logits_to_keep=1)
        self.forward_batches += 1
        return result.logits[:, -1, [self.yes_id, self.no_id]].float()

    @torch.no_grad()
    def probability(self, inputs, reference=False):
        return self.logits(inputs, reference).softmax(-1)[:, 0]

    @staticmethod
    def binary_kl(logits, reference_logits):
        logp = logits.log_softmax(-1); logq = reference_logits.detach().log_softmax(-1)
        return (logp.exp()*(logp-logq)).sum(-1).mean()

    def soft_evidence_step(self, inputs, target, reference_logits, kl_strength=.1):
        """Proper soft BCE and binary reference KL, never a self-truth reward."""
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.logits(inputs)
        target = torch.as_tensor(target, device=self.device, dtype=torch.float32).detach()
        assert target.shape == (len(logits),) and torch.isfinite(target).all()
        assert ((target >= 0) & (target <= 1)).all()
        ce = nn.functional.binary_cross_entropy_with_logits(logits[:, 0]-logits[:, 1], target)
        divergence = self.binary_kl(logits, reference_logits)
        loss = ce+kl_strength*divergence
        loss.backward()
        norm = nn.utils.clip_grad_norm_([p for _, p in self.trainable], 1.)
        assert torch.isfinite(norm) and norm > 0, 'No real Qwen adapter training signal'
        self.optimizer.step(); self.gradient_steps += 1
        torch.cuda.synchronize(); self.assert_base_frozen()
        return dict(loss=float(loss.detach()), proper_teacher_CE=float(ce.detach()),
                    proper_binary_reference_KL=float(divergence.detach()), gradient_norm=float(norm),
                    adapter_updates=self.gradient_steps, full_Qwen_base_frozen=True)

    def seal(self):
        self.assert_base_frozen(); final = self.base_hash()
        assert final == self.initial_base_sha256
        return dict(initial_base_sha256=self.initial_base_sha256, final_base_sha256=final,
                    original_Qwen_base_weights_unchanged=True,
                    adapter_trainable_parameters=sum(p.numel() for _, p in self.trainable),
                    adapter_optimizer_updates=self.gradient_steps, forward_batches=self.forward_batches,
                    model_load_and_base_hash_wall_seconds=self.model_load_and_base_hash_wall_seconds)
