"""Teacher-only full-head polishing after the fixed-hidden probe missed gate.

No quality outputs, target annotations or live study parameters are changed.
This diagnostic tests optimization sufficiency of the existing ordinary head;
it is not an additional claimed RMOT contribution.
"""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_object_reference import OUT as QUALITY
from scripts.a100_six_direction import PUBLIC

OUT=QUALITY/'supervisor/baseline_full_head_probe'


def main():
    plan=json.loads((PUBLIC/'frame_kd_full_head_probe_registration.json').read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    previous=json.loads((PUBLIC/'frame_kd_output_layer_probe.json').read_text())
    assert previous['status']=='TEACHER_ONLY_ORDINARY_OUTPUT_LAYER_OPTIMIZATION_PROBE_COMPLETE'
    started=time.perf_counter();guard=generation_guard();torch.set_num_threads(1)
    archive=QUALITY/'supervisor/target_name_fix_originals'
    stream=torch.load(archive/'C2_seed0_prefix1_stream.pt',map_location='cpu',weights_only=False)
    checkpoint=torch.load(archive/'C2_seed0_prefix1_FRAME_DIRECT_latest_head.pt',map_location='cpu',weights_only=False)
    assert stream['next_video']==1
    raw_x=np.concatenate(stream['frame_x']);raw_y=np.concatenate(stream['frame_y'])
    x=torch.as_tensor((raw_x-checkpoint['mean'])/checkpoint['std'],dtype=torch.float32)
    y=torch.as_tensor(np.clip(raw_y,.02,.98),dtype=torch.float32)
    weight=torch.as_tensor(1.+3.*np.abs(2.*raw_y-1.),dtype=torch.float32)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Linear(64,1))
    model.load_state_dict(checkpoint['model'])
    with np.load(QUALITY/'supervisor/baseline_output_probe/optimized_output_layer.npz') as a:
        with torch.no_grad():
            model[-1].weight.copy_(torch.as_tensor(a['weight'],dtype=torch.float32)[None])
            model[-1].bias.copy_(torch.as_tensor(a['bias'],dtype=torch.float32))
    entropy=float(((-(y*torch.log(y)+(1-y)*torch.log1p(-y)))*weight).sum()/weight.sum())
    def loss():
        values=torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1),y,reduction='none')
        return (values*weight).sum()/weight.sum()
    initial=float(loss().detach())
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,max_eval=200,
        history_size=20,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        value=loss();value.backward();calls+=1
        return value
    optimizer.step(closure)
    final=float(loss().detach());assert final<=initial+1e-6
    assert not guard
    OUT.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),mean=checkpoint['mean'],std=checkpoint['std']),OUT/'polished_head.pt')
    report=dict(status='TEACHER_ONLY_FULL_HEAD_OPTIMIZATION_PROBE_COMPLETE',pair='C2',seed=0,video='0005',
        teacher_examples=len(y),architecture='Same1087->64GELU->1 direct ordinary head; no extra input or hidden capacity',
        optimizer='CPUfullbatchLBFGS100iterations/max200evaluations,strongWolfe,after fixed-hidden output-layer polish',
        loss_evaluations=calls,initial_teacher_KL=initial-entropy,final_teacher_KL=final-entropy,
        reaches_original_registered_KL_gate=final-entropy<=.005,
        original_registered_gate_unchanged=True,original_study_weights_or_predictions_changed=False,
        no_future_video_data_or_weights=True,new_quality_predictions=0,
        manual_target_GT_open_attempts=0,new_teacher_encoder_or_GPU_calls=0,new_cpu_head_fits=1,
        interpretation='Teacher-only convergence diagnostic, no task accuracy result. Any stronger ordinary baseline requires a complete separately sealed quality supplement.',
        wall_seconds=time.perf_counter()-started,all_data_development=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'frame_kd_full_head_probe.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
