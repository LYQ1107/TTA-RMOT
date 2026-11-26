"""Read-only path diagnosis of the already completed negative visual pilot.

This never opens current full-six checkpoints, predictions, GT, or quality.
Measure original signals and conditional derivatives, without masking a
branch, training a model, or producing alternative predictions/HOTA.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_evidence_visual_primary as old
from scripts import a100_evidence_visual_capacity_controls as capacity

OUT = old.OUT / 'readonly_all_used_paths_20261126'
OWN = OUT / 'registration.json'
CURRENT_PLAN = old.OUT.parent / 'fullscope_visual_six_20261126/registration.json'


def old_complete():
    plan = old.registration()
    for stem, records, effects in ((old.STEM, 75, 72), (capacity.STEM, 21, 18)):
        result = json.loads((old.PUBLIC/(stem+'_reproduction.json')).read_text())
        assert result['status'].startswith('PASS_') and result['records'] == records and result['effects'] == effects
    result = json.loads((old.PUBLIC/(old.STEM+'_analysis.json')).read_text())
    assert not result['candidate_eligible_for_larger_screen']
    assert [case['pair'] for case in plan['cases']] == ['C1', 'C2', 'C4']
    return plan


def prepare():
    plan = old_complete()
    assert not OWN.exists()
    cases = []
    for case in plan['cases']:
        files = dict(features=Path(case['pool_folder'])/'features.npy', pixels=Path(case['pixel_pairs_path']),
                     closed=old.OUT/case['pair']/'primary/closed.pt', final=old.OUT/case['pair']/'primary/final.pt')
        rows = {}
        for key, path in files.items():
            assert '/nas/' not in str(path.resolve()), path
            rows[key] = dict(path=str(path), sha256=file_sha(path), bytes=path.stat().st_size)
        cases.append(dict(pair=case['pair'], video=case['evaluated_video'], files=rows))
    current = json.loads(CURRENT_PLAN.read_text())
    atomic_json(OWN, dict(status='REGISTERED_OLD_COMPLETED_NEGATIVE_PILOT_ALL_USED_PATH_DIAGNOSIS',
        source_sha256=file_sha(__file__), old_registration_sha256=file_sha(old.PLAN),
        old_sources=plan['source_sha256'], current_registration_sha256=file_sha(CURRENT_PLAN),
        current42sources=current['source_sha256'], cases=cases,
        trigger='The old75/72 and capacity21/18 studies are complete and independently reproduced; their primary was negative. This does not use or anticipate any current full-six quality.',
        sample='Union of up to512 evenly spaced canonical feature indices and up to512 evenly spaced UNIQUE paid indices in numeric order. C1/C4 small finite pools are censuses, C2 is a deterministic descriptive sample. No GT or teacher probability selects rows.',
        stages=['bootstrap_fixed_s_r', 'closed_four_rounds', 'final_teacher_projection'],
        original_signals=['S bounded mean logit', 'S last-layer Jacobian acquisition factor',
                          'R bounded mean logit (direct control)', 'R sigmoid residual proxy',
                          'sqrt R residual proxy (actual acquisition factor)'],
        derivatives='Autograd of the ORIGINAL signals with respect to actual context h and actual raw content a as independent intermediate leaves, retaining the gate dependence on h. Assert equality with original full forwards and Jacobian helper.',
        further_measurements='Exact pre-hidden linear additive components and coefficients, gate/activation statistics, bootstrap/closed/final group parameter drift, original pixel pretext, S/R content alignment, and exact closed/final state equality plus recorded extra projection steps.',
        interpretation='Conditional gradient and gradient-times-activation are coordinate-dependent descriptions, not semantic accuracy, counterfactual HOTA, independent branch inputs, or causal module contribution. Context already receives current DINO. A nonzero path rejects total computational disconnection, not semantic failure.',
        training_order_limit='The OLD implementation updates S before R in each round; final S/R alignment may be stale. The current full study already updates R before S and expands support. Old findings cannot be assigned to the current method.',
        no_GT_teacher_GPU_optimizer_steps_new_seeds_or_alternative_predictions=True,
        no_current_full_study_outputs_or_checkpoints_opened=True, goal_status='ACTIVE'))
    print('REGISTERED_OLD_ONLY_READONLY_ALL_USED_PATH_DIAGNOSIS', flush=True)


def validate():
    own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__) and own['old_registration_sha256'] == file_sha(old.PLAN)
    assert own['current_registration_sha256'] == file_sha(CURRENT_PLAN)
    for names in ('old_sources', 'current42sources'):
        for name, sha in own[names].items():
            assert file_sha(ROOT/name) == sha, name
    for case in own['cases']:
        for row in case['files'].values():
            assert file_sha(row['path']) == row['sha256']
    return own


def stats(values):
    values = np.asarray(values, dtype='float64')
    assert np.isfinite(values).all()
    return dict(mean=float(values.mean()), std=float(values.std()),
                quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist())


def measure(model, x):
    from flex_uda.active import evidence_visual_learning as core
    accum = {}
    maximum_forward_error = maximum_jacobian_error = maximum_linear_error = 0.
    def add(name, value):
        accum.setdefault(name, []).append(value.detach().cpu().numpy())
    for begin in range(0, len(x), 256):
        tx = torch.as_tensor(np.asarray(x[begin:begin+256]).copy(), dtype=torch.float32)
        with torch.no_grad():
            reference = model(tx).reshape(len(tx), -1)
            h0 = model.context((tx-model.mean)/model.std)
            a0 = model.app(tx[:, core.CURRENT])
            ref_jac = model.last_layer_jacobian_norm(tx) if reference.shape[1] == 1 else None
        h = h0.detach().requires_grad_(True)
        a = a0.detach().requires_grad_(True)
        gate = model.gate(h).sigmoid()
        content = a*gate
        hidden = model.hidden(torch.cat((h, content), dim=1))
        raw = model.output(hidden)
        z = core.bounded(raw)
        maximum_forward_error = max(maximum_forward_error, float((z.detach()-reference).abs().max()))
        assert torch.allclose(z.detach(), reference, atol=1e-6, rtol=1e-6)
        signals = dict(mean_logit=z[:, 0])
        if reference.shape[1] == 1:
            signals['last_layer_jacobian'] = (hidden.square().sum(1)+1).sqrt()*64/(8+raw[:, 0].abs()).square()
            maximum_jacobian_error = max(maximum_jacobian_error, float((signals['last_layer_jacobian'].detach()-ref_jac).abs().max()))
            assert torch.allclose(signals['last_layer_jacobian'].detach(), ref_jac, atol=1e-6, rtol=1e-6)
        else:
            signals['residual_probability'] = z[:, 1].sigmoid()
            signals['residual_sqrt_acquisition'] = signals['residual_probability'].clamp(min=1e-6).sqrt()
        for name, value in signals.items():
            gh, ga = torch.autograd.grad(value.sum(), (h, a), retain_graph=True)
            hm = (h.detach()*gh).abs().sum(1)
            am = (a.detach()*ga).abs().sum(1)
            add(name+'/value', value)
            add(name+'/context_gradient_norm', gh.norm(dim=1))
            add(name+'/content_gradient_norm', ga.norm(dim=1))
            add(name+'/content_gradient_times_input_fraction', am/(hm+am+1e-20))
            add(name+'/content_exact_zero_gradient_row', (ga == 0).all(1).float())
        weight = model.hidden[0].weight.detach()
        width = h.shape[1]
        with torch.no_grad():
            hc = h0@weight[:, :width].T
            ac = content.detach()@weight[:, width:].T
            full = model.hidden[0](torch.cat((h0, content.detach()), dim=1))
            reconstructed = hc+ac+model.hidden[0].bias
            maximum_linear_error = max(maximum_linear_error, float((full-reconstructed).abs().max()))
            assert torch.allclose(full, reconstructed, atol=2e-5, rtol=2e-5)
        for name, value in dict(gate=gate.flatten(), gate_below01=(gate < .01).float().flatten(),
                gate_above99=(gate > .99).float().flatten(), context_activation=h0.flatten(),
                content_encoder_activation=a0.flatten(), gated_content_activation=content.flatten(),
                context_linear_component=hc.flatten(), content_linear_component=ac.flatten()).items():
            add(name, value)
    values = {name: stats(np.concatenate(parts)) for name, parts in accum.items()}
    for name in ('context_activation', 'content_encoder_activation', 'gated_content_activation',
                 'context_linear_component', 'content_linear_component'):
        values[name]['RMS'] = float(np.sqrt(np.mean(np.concatenate(accum[name]).astype('float64')**2)))
    width = model.context[0].out_features
    return dict(rows=len(x), statistics=values, original_forward_max_abs_error=maximum_forward_error,
        original_jacobian_max_abs_error=maximum_jacobian_error, linear_reconstruction_max_abs_error=maximum_linear_error,
        content_readout_coefficient_norm=float(model.hidden[0].weight[:, width:].detach().norm()),
        context_readout_coefficient_norm=float(model.hidden[0].weight[:, :width].detach().norm()),
        current_DINO_context_input_coefficient_norm=float(model.context[0].weight[:, core.CURRENT].detach().norm()))


def drift(previous, current):
    result = {}
    for prefix in ('app.', 'context.', 'gate.', 'hidden.', 'output.'):
        keys = [k for k in current if k.startswith(prefix)]
        a = torch.cat([previous[k].flatten().double() for k in keys])
        b = torch.cat([current[k].flatten().double() for k in keys])
        result[prefix[:-1]] = dict(L2_change=float((b-a).norm()), max_abs_change=float((b-a).abs().max()),
            relative_L2_change=float((b-a).norm()/(a.norm()+1e-20)), exactly_equal=bool(torch.equal(a, b)))
    return result


def main():
    from flex_uda.active import evidence_visual_learning as core
    from flex_uda.a100.mechanism import generation_guard
    own = validate()
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.perf_counter()
    rows = []
    for case in own['cases']:
        files = case['files']
        features = np.load(files['features']['path'], mmap_mode='r')
        closed = torch.load(files['closed']['path'], map_location='cpu', weights_only=False)
        final = torch.load(files['final']['path'], map_location='cpu', weights_only=False)
        assert closed['registration_sha256'] == final['registration_sha256'] == own['old_registration_sha256']
        paid = np.unique([d['index'] for d in closed['draws']])
        indices = np.unique(np.r_[np.linspace(0, len(features)-1, min(512, len(features))).astype(int),
            paid[np.linspace(0, len(paid)-1, min(512, len(paid))).astype(int)]])
        x = np.asarray(features[indices])
        with np.load(files['pixels']['path']) as pixel_file:
            pixel = {k: torch.as_tensor(pixel_file[k].copy(), dtype=torch.float32) for k in ('left', 'right', 'weight')}
        states = dict(bootstrap=dict(solver=closed['fixed_s'], observer=closed['fixed_r']),
                      closed=dict(solver=closed['solver'], observer=closed['observer']),
                      final=dict(solver=final['solver'], observer=final['observer']))
        stages = {}
        for stage, weights in states.items():
            stages[stage] = {}
            content = {}
            for role, outputs in (('solver', 1), ('observer', 2)):
                state = weights[role]
                model = core.VisualHead(state['mean'].numpy(), state['std'].numpy(), outputs=outputs)
                model.load_state_dict(state)
                model.eval()
                measured = measure(model, x)
                with torch.no_grad():
                    a, b = model.app(pixel['left']), model.app(pixel['right'])
                    measured['pixels'] = dict(pairs=len(a), positive_pair_MSE=float((a-b).square().mean()),
                        content_loss=float(core.content_loss(model, pixel)),
                        mean_content_std=float(a.std(0).mean()))
                    content[role] = model.app(torch.as_tensor(x[:, core.CURRENT].copy(), dtype=torch.float32))
                stages[stage][role] = measured
            stages[stage]['S_R_content_alignment_MSE'] = float((content['solver']-content['observer']).square().mean())
        changes = {}
        for before, after in (('bootstrap', 'closed'), ('closed', 'final')):
            changes[before+'_to_'+after] = {role: drift(states[before][role], states[after][role]) for role in ('solver', 'observer')}
        closed_steps = sum(r['steps'] for r in closed['receipts'])
        row = dict(pair=case['pair'], video=case['video'], sampled_indices=indices.tolist(),
            finite_pool_rows=len(features), sampled_rows=len(indices), unique_paid_indices=len(paid),
            old_logical_draws=len(closed['draws']), stages=stages, parameter_drift=changes,
            closed_S_steps=closed_steps, final_S_steps=final['solver_total_steps'],
            extra_teacher_only_projection_steps=final['solver_total_steps']-closed_steps)
        assert row['extra_teacher_only_projection_steps'] >= 0
        atomic_json(OUT/case['pair']/'measurement.json', row)
        rows.append(row)
        print(json.dumps(dict(status='OLD_ONLY_ORIGINAL_SIGNAL_PATHS_MEASURED', pair=case['pair'],
            sampled_rows=len(indices), extra_projection_steps=row['extra_teacher_only_projection_steps'],
            final_S_mean_content_fraction=stages['final']['solver']['statistics']['mean_logit/content_gradient_times_input_fraction']['mean'],
            final_S_Jacobian_content_fraction=stages['final']['solver']['statistics']['last_layer_jacobian/content_gradient_times_input_fraction']['mean'],
            final_R_residual_content_fraction=stages['final']['observer']['statistics']['residual_probability/content_gradient_times_input_fraction']['mean'])), flush=True)
    validate()
    assert not guard and len(rows) == 3
    result = dict(status='COMPLETE_OLD_NEGATIVE_PILOT_ALL_ORIGINAL_USED_PATH_DIAGNOSIS', cases=rows,
        source_sha256=file_sha(__file__), registration_sha256=file_sha(OWN),
        wall_seconds=time.perf_counter()-started, GT_open_attempts=len(guard), new_teacher_calls=0,
        optimizer_steps=0, GPU_calls=0, new_prediction_masks_or_HOTA_records=0,
        current_full_study_checkpoints_predictions_or_GT_opened=False,
        original42current_sources_unchanged=True, independent_semantic_or_causal_module_contribution_proven=False,
        goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', result)
    print('COMPLETE_OLD_ONLY_READONLY_PATH_DIAGNOSIS_NO_CURRENT_QUALITY', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        main()
