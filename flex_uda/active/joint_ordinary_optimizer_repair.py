"""Teacher-risk-only optimizer completion; original study source stays frozen.

Same heads, inputs, losses, capacities, seed0 and .005 adequacy requirement.
Only as-yet-unfitted ordinary arms use conservative Adam and a best-risk
iterate before LBFGS. Completed heads and all structured fits stay intact.
"""
import copy
import time
import numpy as np
import torch

from flex_uda.active import joint_observation_study as original
from flex_uda.active.joint_observation_fit import change_normalization

ORIGINAL_FIT = original.fit_prefix


def fit_prefix(arrays, recipe, mean, std, previous=None, device='cuda:0'):
    if recipe['family'] == 'structured':
        return ORIGINAL_FIT(arrays, recipe, mean, std, previous, device)
    started = time.perf_counter()
    selected, projection = original.coalesce_exact_inputs(arrays, recipe)
    data = {k:torch.as_tensor(v, device=device) for k,v in selected.items()}
    coarse = recipe['family'] == 'coarse'; kind = recipe.get('risk', 'joint')
    dimension = len(mean)
    first_width = previous['width'] if previous else (128 if recipe['family'] == 'generic' else 64)
    widths = [first_width]+[w for w in (128,256,512) if w > first_width]
    minimum = 400 if previous else 1000
    trace, attempts = [], []
    selected_model = selected_early = selected_metrics = None
    global_best = float('inf')
    for wi,width in enumerate(widths):
        torch.manual_seed(0)
        model = original.make_model(recipe, dimension, width).to(device)
        if wi == 0 and previous is not None:
            assert previous['recipe'] == recipe
            model.load_state_dict(previous['model'])
            if coarse:
                with torch.no_grad():
                    layer = model[0]; before = layer.weight.clone()
                    ratio = torch.as_tensor(std/previous['std'],device=device,dtype=before.dtype)
                    delta = torch.as_tensor((mean-previous['mean'])/previous['std'],device=device,dtype=before.dtype)
                    layer.weight.mul_(ratio);layer.bias.add_(before@delta)
            else:
                change_normalization(model,previous['mean'],previous['std'],mean,std)
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        best, best_state, early = float('inf'), None, None
        for step in range(1,4001):
            opt.zero_grad(set_to_none=True)
            ce,_ = original.teacher_risk(original.model_log_probabilities(model,data,coarse),
                                         data['target'],data['weight'],kind)
            assert bool(torch.isfinite(ce));ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            if step == 400:
                early = copy.deepcopy(model).eval()
            if step % 250 == 0 or step == 400:
                with torch.no_grad():
                    c,h = original.teacher_risk(original.model_log_probabilities(model,data,coarse),
                                                data['target'],data['weight'],kind)
                excess=float(c-h)
                metrics=dict(width=width,steps=step,teacher_CE=float(c),
                    attainable_exact_input_entropy=float(h),excess_KL=excess,
                    optimizer='Adam_lr.001_best_teacher_risk')
                trace.append(metrics)
                if excess < best:
                    best=excess;best_state=copy.deepcopy(model.state_dict());best_metrics=metrics.copy()
                if step>=minimum and best<=.005:break
        assert best_state is not None and early is not None
        model.load_state_dict(best_state);metrics=best_metrics
        if best>.005:
            model.double();double_data={k:(v if v.dtype==torch.bool else v.double()) for k,v in data.items()}
            opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,
                history_size=10,tolerance_grad=1e-10,tolerance_change=1e-14,line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True)
                loss,_=original.teacher_risk(original.model_log_probabilities(model,double_data,coarse),
                                             double_data['target'],double_data['weight'],kind)
                loss.backward();return loss
            deployed_best = copy.deepcopy(model).float()
            for block in range(5):
                opt.step(closure)
                deployed=copy.deepcopy(model).float()
                with torch.no_grad():
                    c,h=original.teacher_risk(original.model_log_probabilities(deployed,data,coarse),
                                              data['target'],data['weight'],kind)
                current=dict(width=width,LBFGS_blocks=block+1,teacher_CE=float(c),
                    attainable_exact_input_entropy=float(h),excess_KL=float(c-h),
                    optimizer='LBFGS_float64_best_teacher_risk_checked_float32')
                attempts.append(current)
                if current['excess_KL']<best:
                    best=current['excess_KL'];deployed_best=deployed;metrics=current
                if best<=.005:break
            model=deployed_best
        if best<global_best:
            global_best=best
            selected_model,selected_early,selected_metrics=model.eval(),early,metrics
            chosen_width,chosen_step=width,step
        if global_best<=.005:break
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    receipt=dict(recipe=recipe,width=chosen_width,minimum_Adam_steps=minimum,early_step=400,
        actual_final_Adam_steps=chosen_step,trace=trace,capacity_and_LBFGS_attempts=attempts,
        metrics=selected_metrics,adequate_ordinary_required=True,
        teacher_fit_gate_passed=global_best<=.005,structural_residual_not_an_ordinary_fit_pass=False,
        neural_seed=0,prefix_warm_start=previous is not None,target_GT_or_HOTA_used=False,
        **projection,wall_seconds=time.perf_counter()-started,
        optimization_amendment='20261124_remaining_ordinary_lr.001_best_risk',
        original_architecture_inputs_loss_capacity_and_gate_preserved=True,
        selected_only_by_paid_training_teacher_risk=True)
    if not receipt['teacher_fit_gate_passed']:
        raise RuntimeError('Ordinary optimization still incomplete; no GT/quality comparison: '+str(receipt))
    checkpoint=dict(model={k:v.detach().cpu().clone() for k,v in selected_model.state_dict().items()},
        width=chosen_width,recipe=recipe,mean=mean.copy(),std=std.copy(),receipt=receipt)
    return selected_model,selected_early,checkpoint,receipt
