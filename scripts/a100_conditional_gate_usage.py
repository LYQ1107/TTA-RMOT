"""Inspect learned shared gates without GT or new quality predictions.

Only the final video of a completed condition is replayed with its own final
prefix checkpoint. Gate removal is a numerical sensitivity diagnostic, not
a trained ablation or an HOTA result. Live training sources stay unchanged.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.conditional_observation import BagContextHead
from flex_uda.active.observation_learning_inputs import ObservationVideo
from scripts.a100_conditional_six_setting import OUT as QUALITY
from scripts.a100_six_direction import PUBLIC


def quantiles(value):
    levels=(0,.01,.1,.25,.5,.75,.9,.99,1)
    return {str(k):float(v) for k,v in zip(levels,np.quantile(np.asarray(value),levels))}


def run(pair,seed):
    plan=json.loads((PUBLIC/'conditional_gate_usage_plan.json').read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    guard=generation_guard();start=time.perf_counter();torch.set_num_threads(2)
    root=QUALITY/pair/f'seed{seed}'
    complete=json.loads((root/'completion.json').read_text());assert complete['GT_open_attempts']==0
    target=PAIRS[pair][1];video=sorted(metadata(target)['videos'])[-1]
    saved=torch.load(root/'stream.pt',map_location='cpu',weights_only=False)
    assert saved['next_video']==len(metadata(target)['videos']) and saved['receipts'][-1]['video']==video
    inputs=ObservationVideo(pair,video);q,t,s,d=inputs.raw_bags.shape
    present=np.broadcast_to(inputs.present,(q,t,s)).reshape(q*t,s).copy()
    bag_ids,slots=np.nonzero(present);qi,ti=bag_ids//t,bag_ids%t
    rows=inputs.sample_rows[ti,slots];results={}
    for arm in ('FULL','CONDITIONAL_BALANCED'):
        checkpoint=saved['previous'][arm];assert checkpoint['kind']=='conditional'
        values=((inputs.raw_bags-checkpoint['mean'])/checkpoint['std']).astype('float32')
        values[:,~inputs.present]=0
        bb=torch.as_tensor(values.reshape(q*t,s,d));mm=torch.as_tensor(present)
        cc=bb[bag_ids,slots];model=BagContextHead(d,checkpoint['width'],'conditional').eval()
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            encoded=model.encoder(bb);mask=mm.unsqueeze(-1)
            mean=encoded.masked_fill(~mask,0).sum(1)/mm.sum(1,keepdim=True)
            maximum=encoded.masked_fill(~mask,-torch.inf).max(1).values
            context=torch.cat((mean,maximum),1)
            support=torch.sigmoid(model.bag_head(context).squeeze(-1))
            local=torch.sigmoid(model.local_head(torch.cat((model.encoder(cc),context[bag_ids]),1)).squeeze(-1))
            joint=torch.sigmoid(model(bb[bag_ids],mm[bag_ids],cc))
            decomposition_error=float((joint-support[bag_ids]*local).abs().max())
        a,r,p=support.numpy(),local.numpy(),joint.numpy()
        original=np.load(root/'videos'/video/(arm+'_FINAL_scores.npy'),mmap_mode='r')[qi,rows]
        replay_error=float(np.max(np.abs(original-p)))
        assert replay_error<1e-4 and decomposition_error<1e-5,(arm,replay_error,decomposition_error)
        results[arm]=dict(bag_query_units=q*t,actual_anchor_observations=len(p),
            shared_gate_quantiles=quantiles(a),shared_gate_standard_deviation=float(a.std()),
            fraction_gate_above_0_99=float((a>.99).mean()),fraction_gate_below_0_5=float((a<.5).mean()),
            conditional_probability_quantiles=quantiles(r),joint_probability_quantiles=quantiles(p),
            frozen_gate_removal_probability_difference=quantiles(r-p),
            frozen_gate_removal_threshold_decision_change_fraction=float(((r>.5)!=(p>.5)).mean()),
            CPU_last_prefix_vs_original_GPU_scores_max_absolute_error=replay_error,
            actual_model_vs_product_formula_max_absolute_error=decomposition_error,
            checkpoint_width=checkpoint['width'],same_historical_PCD_and_RGB_input_version=True)
    assert not guard
    result=dict(status='PASS_SAVED_CONDITIONAL_GATE_USAGE_DIAGNOSIS_NO_GT',pair=pair,seed=seed,video=video,arms=results,
        last_completed_prefix_only_no_future_head_on_past_videos=True,
        no_GT_or_HOTA_or_new_quality_prediction_files=True,no_neural_training_or_teacher_or_backbone_calls=True,
        saved_small_head_evaluated_on_CPU=True,manual_GT_open_attempts=0,wall_seconds=time.perf_counter()-start,
        interpretation='Nonconstant or influential gates do not prove semantic identity, calibrated bag presence, or HOTA contribution. Near-one gates also do not rule out effects during training.',
        gate_removal_not_a_retrained_ablation=True,live_numerical_sources_not_modified=True,goal_status='ACTIVE')
    path=PUBLIC/(f'conditional_gate_usage_{pair}_seed{seed}.json');atomic_json(path,result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();run(args.pair,args.seed)
