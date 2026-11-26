"""Registered post-complete diagnostic of the regularized content path.

The context path receives the full9352D input, including current DINO.
It can bypass the separately regularized content path. This structural
possibility is not evidence that any fitted model actually uses a bypass.
Only original activations/conditional gradients are measured: no alternate
predictions, masks, HOTA ablations, fitted steps, teacher calls or targetGT.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_fullscope_visual_six import OUT as PARENT,PLAN,PUBLIC,STEM,registration

OUT=PARENT/'postcomplete_content_path_usage_20261126'
OWN=OUT/'registration.json'


def prepare():
    plan=registration();assert not OWN.exists()
    atomic_json(OWN,dict(status='REGISTERED_POSTCOMPLETE_READONLY_CONTENT_PATH_DIAGNOSIS',
        source_sha256=file_sha(__file__),parent_registration_sha256=file_sha(PLAN),
        original42sources=plan['source_sha256'],cases=len(plan['cases']),
        structural_observation='VisualHead.context consumes all9352coordinates, including CURRENT7815:8583; a hidden layer with zero coefficients on its64content inputs is a valid network that ignores app. Possibility only, not a diagnosis of fitted models.',
        trigger='Only after all4588 plus allT0attempt seals and186HOTA/180effect reproduction. If primary strictscreen is positive, skip this failure diagnostic and prioritize necessary module contribution work.',
        sample='Union of at most256evenlyspaced full-query/frame indices and256evenlyspaced paid draw indices, percase, using no labels or teacher probabilities for selection. Deterministic descriptive sample, not a census or independent test.',
        stages=['bootstrap','closed','final'],roles=['solver','observer_mean'],
        measurements='Original gate activations, content/context activation scales, and gradients of original mean logit with respect to independent context/content intermediate tensors. Also descriptive gradient-times-input mass fraction and parameter coefficient norms.',
        interpretation='Computational sensitivity is not causal HOTA contribution, semantic disentanglement, gradient-based truth, or a unique explanation. Context and content share original image inputs; branches are not statistically independent.',
        no_counterfactual_predictions_masks_new_HOTA_or_module_accuracy_ablation=True,
        no_GT_teacher_fitting_seeds_or_GPU=True,goal_status='ACTIVE'))
    print('REGISTERED_POSTCOMPLETE_CONTENT_PATH_SENSITIVITY_NO_CURRENT_QUALITY_READ',flush=True)


def measure(model,x):
    import flex_uda.active.evidence_visual_learning as core
    model.eval();tx=torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        h0=model.context((tx-model.mean)/model.std)
        a0=model.app(tx[:,core.CURRENT])
    # Intermediate leaves describe the actual two paths, without changing
    # either activation or producing an alternative prediction.
    h=h0.detach().requires_grad_(True);a=a0.detach().requires_grad_(True)
    gate=model.gate(h).sigmoid();content=a*gate
    raw=model.output(model.hidden(torch.cat((h,content),dim=1)))[:,0]
    z=core.bounded(raw)
    gh,ga=torch.autograd.grad(z.sum(),(h,a))
    context_mass=(h.detach()*gh).abs().sum(1)
    content_mass=(a.detach()*ga).abs().sum(1)
    fraction=(content_mass/(context_mass+content_mass+1e-20)).detach().numpy()
    g=gate.detach().numpy();weight=model.hidden[0].weight.detach();width=h.shape[1]
    result=dict(rows=len(x),gate_quantiles=np.quantile(g,[0,.01,.5,.99,1]).tolist(),
        gate_fraction_below01=float((g<.01).mean()),gate_fraction_above99=float((g>.99).mean()),
        context_RMS=float(h0.square().mean().sqrt()),content_RMS=float(content.detach().square().mean().sqrt()),
        raw_content_encoder_RMS=float(a0.square().mean().sqrt()),
        context_conditional_logit_gradient_norm_mean=float(gh.norm(dim=1).mean()),
        content_conditional_logit_gradient_norm_mean=float(ga.norm(dim=1).mean()),
        content_gradient_times_input_fraction_mean=float(fraction.mean()),
        content_gradient_times_input_fraction_quantiles=np.quantile(fraction,[0,.1,.5,.9,1]).tolist(),
        content_readout_coefficient_norm=float(weight[:,width:].norm()),
        context_readout_coefficient_norm=float(weight[:,:width].norm()),
        current_DINO_context_input_coefficient_norm=float(model.context[0].weight[:,core.CURRENT].detach().norm()),
        raw_logit_quantiles=np.quantile(raw.detach().numpy(),[0,.5,1]).tolist())
    assert all(np.isfinite(np.asarray(v)).all() for v in result.values())
    return result


def main():
    import flex_uda.active.evidence_visual_learning as core
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.a100.mechanism import generation_guard
    plan=registration();own=json.loads(OWN.read_text())
    assert own['source_sha256']==file_sha(__file__) and own['parent_registration_sha256']==file_sha(PLAN)
    barrier=json.loads((PARENT/'all4588_sealed_before_GT.json').read_text())
    assert len(barrier['tasks'])==4588 and barrier['GT_open_attempts']==0 and barrier['teacher_final_weight_seals']==4
    reproduction=json.loads((PUBLIC/(STEM+'_reproduction.json')).read_text())
    assert reproduction['status']=='PASS_ALL186_HOTA_AND180_EFFECTS' and reproduction['records']==186 and reproduction['effects']==180
    for gpu in (0,1,2,3):
        row=json.loads((PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())
        assert row['teacher']['backend_unmodified']
    analysis=json.loads((PUBLIC/(STEM+'_analysis.json')).read_text())
    if analysis['candidate_eligible_for_larger_screen']:
        atomic_json(OUT/'completion.json',dict(status='SKIPPED_POSITIVE_PRIMARY_PRIORITIZE_NECESSARY_MODULE_WORK',goal_status='ACTIVE'))
        return
    guard=generation_guard();torch.set_num_threads(2);started=time.perf_counter();rows=[]
    for case in plan['cases']:
        folder=PARENT/'units'/case['pair']/case['evaluated_video'];context=Context(case,folder,None,-1)
        closed=torch.load(folder/'primary/closed.pt',map_location='cpu',weights_only=False)
        draws=closed['draws'];paid=np.asarray([d['index'] for d in draws],dtype=int)
        sample=np.unique(np.r_[np.linspace(0,len(context.features)-1,min(256,len(context.features))).astype(int),
            paid[np.linspace(0,len(paid)-1,min(256,len(paid))).astype(int)]])
        x=context.features[sample];stages={};previous={}
        for name in ('bootstrap','closed','final'):
            checkpoint=folder/'primary'/('bootstrap.pt' if name=='bootstrap' else ('closed.pt' if name=='closed' else 'final.pt'))
            saved=closed if name=='closed' else torch.load(checkpoint,map_location='cpu',weights_only=False)
            stages[name]={}
            for key,outputs in [('solver',1),('observer',2)]:
                role='solver' if key=='solver' else 'observer_mean';state=saved[key]
                if role in previous and all(torch.equal(v,previous[role]['state'][k]) for k,v in state.items()):
                    value=dict(previous[role]['measurement'],exact_state_parent_stage=previous[role]['stage'])
                else:
                    model=core.make(context.mean,context.std,'cpu',outputs=outputs);model.load_state_dict(state)
                    value=measure(model,x)
                stages[name][role]=value
                previous[role]=dict(state=state,measurement=value,stage=name)
        row=dict(pair=case['pair'],video=case['evaluated_video'],sampled_indices=sample.tolist(),
            possible_query_frame_actions=len(context.features),sample_rows=len(sample),stages=stages)
        destination=OUT/case['pair']/case['evaluated_video']/'measurement.json'
        atomic_json(destination,row);rows.append(row)
        print(json.dumps(dict(status='POSTCOMPLETE_CONTENT_PATH_MEASURED_NO_NEW_PREDICTIONS',
            pair=case['pair'],video=case['evaluated_video'],sample_rows=len(sample),
            final_solver_content_sensitivity_fraction=stages['final']['solver']['content_gradient_times_input_fraction_mean'])),flush=True)
    assert not guard and len(rows)==148
    summary=[]
    for pair in sorted({r['pair'] for r in rows}):
        selected=[r for r in rows if r['pair']==pair]
        for stage in ('bootstrap','closed','final'):
            for role in ('solver','observer_mean'):
                values=[r['stages'][stage][role] for r in selected]
                counts=np.asarray([v['rows'] for v in values])
                summary.append(dict(pair=pair,stage=stage,role=role,videos=len(values),sampled_rows=int(counts.sum()),
                    content_sensitivity_fraction_sample_weighted_mean=float(np.average([v['content_gradient_times_input_fraction_mean'] for v in values],weights=counts)),
                    gate_nearzero_fraction_sample_weighted_mean=float(np.average([v['gate_fraction_below01'] for v in values],weights=counts))))
    atomic_json(OUT/'completion.json',dict(status='COMPLETE148_POSTCOMPLETE_CONTENT_PATH_DIAGNOSTICS',
        summary=summary,wall_seconds=time.perf_counter()-started,GT_open_attempts=0,
        new_teacher_calls=0,optimizer_steps=0,prediction_masks_or_new_HOTA_records=0,
        source_sha256=file_sha(__file__),no_independent_semantic_or_causal_module_contribution_claim=True,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:main()
