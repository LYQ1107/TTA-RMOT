"""Teacher-only optimization probe for the first nonconverged ordinary head.

The current study is immutable. This probe fixes its learned hidden features
and optimizes only the ordinary output layer on the same causal-prefix data.
It generates no task-quality predictions and cannot establish a HOTA gain.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_object_reference import OUT as QUALITY
from scripts.a100_six_direction import PUBLIC

OUT=QUALITY/'supervisor/baseline_output_probe'
REGISTRATION=PUBLIC/'frame_kd_output_layer_probe_registration.json'


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    plan=json.loads(REGISTRATION.read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    guard=generation_guard()
    archive=QUALITY/'supervisor/target_name_fix_originals'
    stream=torch.load(archive/'C2_seed0_prefix1_stream.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(archive/'C2_seed0_prefix1_FRAME_DIRECT_latest_head.pt',map_location='cpu',weights_only=False)
    assert stream['next_video']==1 and len(stream['receipts'])==1
    raw_x=np.concatenate(stream['frame_x']);raw_y=np.concatenate(stream['frame_y'])
    x=torch.as_tensor((raw_x-checkpoint['mean'])/checkpoint['std'],dtype=torch.float32)
    y=torch.as_tensor(np.clip(raw_y,.02,.98),dtype=torch.float32)
    weight=torch.as_tensor(1.+3.*np.abs(2.*raw_y-1.),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Linear(64,1))
    model.load_state_dict(checkpoint['model']);model.eval()
    with torch.no_grad():hidden=model[1](model[0](x)).numpy().astype(float)
    design=np.c_[hidden,np.ones(len(hidden))]
    target=y.numpy().astype(float);w=weight.numpy().astype(float);w/=w.sum()
    initial=np.r_[model[-1].weight.detach().numpy().ravel(),model[-1].bias.detach().numpy()]
    entropy=float(w@(-(target*np.log(target)+(1-target)*np.log1p(-target))))
    def loss_and_gradient(parameter):
        z=design@parameter
        return float(w@(np.logaddexp(0.,z)-target*z)),design.T@(w*(expit(z)-target))
    initial_loss,_=loss_and_gradient(initial)
    fitted=minimize(loss_and_gradient,initial,jac=True,method='L-BFGS-B',
        options=dict(maxiter=1000,maxls=50,gtol=1e-10,ftol=1e-14,maxcor=30))
    final_loss,gradient=loss_and_gradient(fitted.x)
    assert final_loss<=initial_loss+1e-10
    with torch.no_grad():
        model[-1].weight.copy_(torch.as_tensor(fitted.x[:-1],dtype=torch.float32)[None])
        model[-1].bias.copy_(torch.as_tensor(fitted.x[-1:],dtype=torch.float32))
        ce=(torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1),y,reduction='none')*weight).sum()/weight.sum()
    final_float32_KL=float(ce)-float(weight.numpy().astype(float)@(-(target*np.log(target)+(1-target)*np.log1p(-target))))/float(weight.sum())
    assert not guard
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'optimized_output_layer.npz',weight=fitted.x[:-1],bias=fitted.x[-1:])
    report=dict(status='TEACHER_ONLY_ORDINARY_OUTPUT_LAYER_OPTIMIZATION_PROBE_COMPLETE',
        pair='C2',seed=0,video='0005',teacher_examples=len(target),prefix_scope='Only first current video; no future inputs or heads',
        optimizer='L-BFGS-B on the convex65parameter output layer, fixed learned64D hidden features',
        optimizer_success=bool(fitted.success),optimizer_message=str(fitted.message),
        iterations=int(fitted.nit),function_evaluations=int(fitted.nfev),
        maximum_gradient=float(np.max(np.abs(gradient))),
        initial_double_teacher_KL=initial_loss-entropy,final_double_teacher_KL=final_loss-entropy,
        final_float32_teacher_KL=final_float32_KL,
        original_registered_teacher_KL=stream['receipts'][0]['fits']['FRAME_DIRECT']['teacher_KL'],
        original_registered_gate_unchanged=True,original_study_weights_or_predictions_changed=False,
        new_quality_predictions=0,manual_target_GT_open_attempts=0,new_teacher_encoder_or_GPU_calls=0,
        new_cpu_output_layer_fits=1,
        interpretation='Teacher-fit optimization diagnostic only; a lower CE does not establish semantic accuracy. Any quality use needs a separately registered complete comparator with its own output/GT barrier.',
        limitation='Frozen hidden features and finite iterations need not reach the unrestricted network optimum; no success flag proves correct labels or HOTA.',
        wall_seconds=time.perf_counter()-started,all_data_development=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'frame_kd_output_layer_probe.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
