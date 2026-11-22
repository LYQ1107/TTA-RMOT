"""Teacher-loss-only capacity continuation for an ordinary pseudo-label head.

This strengthens the ordinary comparator; it is not an RMOT contribution.
Widening embeds the previous function, and deployment remains float32.
"""
import time
import numpy as np
import torch
from torch import nn


def widen(model, width, seed):
    old_width = model[0].out_features
    assert width >= old_width
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        result = nn.Sequential(nn.Linear(model[0].in_features, width),
            nn.GELU(), nn.Linear(width, 1))
    result = result.to(device=model[0].weight.device, dtype=model[0].weight.dtype)
    with torch.no_grad():
        result[0].weight[:old_width].copy_(model[0].weight)
        result[0].bias[:old_width].copy_(model[0].bias)
        result[-1].weight.zero_()
        result[-1].weight[:, :old_width].copy_(model[-1].weight)
        result[-1].bias.copy_(model[-1].bias)
    return result


def continue_capacity(raw_x, raw_y, checkpoint, *, seed, device='cuda:0',
                      widths=(64, 128, 256, 512), blocks=10, tolerance=.005):
    started = time.perf_counter(); device = torch.device(device)
    mean, std = checkpoint['mean'], checkpoint['std']
    # Preserve the actual float32 inputs/targets of the original comparator;
    # use float64 arithmetic only to improve optimization of that same data.
    xx = np.asarray((np.asarray(raw_x)-mean)/std, dtype=np.float32)
    yy = np.asarray(np.clip(raw_y, .02, .98), dtype=np.float32)
    ww = np.asarray(1.+3.*np.abs(2.*np.asarray(raw_y)-1.), dtype=np.float32)
    x = torch.as_tensor(xx, dtype=torch.float64, device=device)
    y = torch.as_tensor(yy, dtype=torch.float64, device=device)
    weight = torch.as_tensor(ww, dtype=torch.float64, device=device)
    entropy = float((weight*(-y*torch.log(y)-(1-y)*torch.log1p(-y))).sum()/weight.sum())
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 1))
    model.load_state_dict(checkpoint['model']); model = model.to(device=device, dtype=torch.float64)

    def loss(module):
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            module(x).squeeze(-1), y, reduction='none')
        return weight@losses/weight.sum()

    def deployed_kl(module):
        # Casting a fresh copy leaves the optimized model/dtype unchanged.
        candidate = nn.Sequential(nn.Linear(x.shape[1], module[0].out_features),
            nn.GELU(), nn.Linear(module[0].out_features, 1)).to(device)
        candidate.load_state_dict({k:v.float() for k,v in module.state_dict().items()})
        with torch.no_grad():
            logits = candidate(x.float()).squeeze(-1).double()
            value = weight@torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction='none')/weight.sum()-entropy
        return float(value)

    initial = float(loss(model).detach())-entropy; attempts = []; total_evaluations = 0
    for width in widths:
        previous = float(loss(model).detach())
        if width > model[0].out_features:
            model = widen(model, width, seed+width)
        embedding_error = abs(float(loss(model).detach())-previous)
        assert embedding_error < 1e-9, embedding_error
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=100,
            max_eval=200, history_size=20, tolerance_grad=1e-10,
            tolerance_change=1e-14, line_search_fn='strong_wolfe')
        evaluations = 0; complete_blocks = 0; kl = deployed_kl(model)
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True); value = loss(model)
            value.backward(); evaluations += 1
            return value
        for _ in range(blocks):
            if kl <= tolerance:
                break
            optimizer.step(closure); complete_blocks += 1
            kl = deployed_kl(model)
        train_kl = float(loss(model).detach())-entropy
        attempts.append(dict(hidden_width=width, initialization_CE_error=embedding_error,
            LBFGS_100_iteration_blocks=complete_blocks, loss_evaluations=evaluations,
            float64_teacher_KL=train_kl, deployed_float32_teacher_KL=kl))
        total_evaluations += evaluations
        if kl <= tolerance:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    receipt = dict(initial_teacher_KL=initial, teacher_KL=kl,
        teacher_fit_within_0_005=kl <= tolerance, hidden_width=model[0].out_features,
        teacher_examples=len(y), source_input_dimension=x.shape[1], attempts=attempts,
        optimization_dtype='float64 on the original normalized float32 data',
        deployment_dtype='float32', total_loss_evaluations=total_evaluations,
        no_GT_or_quality_selection=True, no_new_input_or_teacher_answer=True,
        wall_seconds=time.perf_counter()-started)
    saved = dict(model={k:v.detach().float().cpu() for k,v in model.state_dict().items()},
        mean=mean, std=std, receipt=receipt)
    return saved, receipt
