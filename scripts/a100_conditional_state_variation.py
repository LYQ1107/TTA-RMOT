"""Test actual use of paid state variation without semantic GT.

The optimal bag-constant squared-error floor is the within-bag variance of
the same clipped teacher targets. Beating this floor proves a predictor
used state variation on those training observations, not task accuracy or
novelty over equally informed ordinary learners.
"""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from scripts.a100_six_direction import PUBLIC
from scripts.a100_conditional_six_setting import OUT as QUALITY
from scripts.a100_observation_design_collect import OUT as ANSWERS
from scripts.a100_conditional_teacher_fit_slices import ARMS


def run(pair,seed):
    plan=json.loads((PUBLIC/'conditional_state_variation_plan.json').read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    guard=generation_guard();root=QUALITY/pair/f'seed{seed}'
    assert json.loads((root/'completion.json').read_text())['GT_open_attempts']==0
    md=metadata(PAIRS[pair][1]);totals=Counter();floor_sum=0.;errors={a:0. for a in ARMS}
    model_counts={a:Counter() for a in ARMS};per_video=[]
    for video,bank in sorted(md['videos'].items()):
        obs=np.asarray(bank['observations'],int);position={tuple(o):i for i,o in enumerate(obs.tolist())}
        queries=[q for q in md['episodes'] if q['video_id']==video]
        doc=json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
        assert doc['GT_open_attempts']==0 and doc['only_this_policy_requested_responses']
        cells=defaultdict(list);seen={}
        for item in doc['answers']:
            rid=item['request_id'];p=float(item['yes_probability'])
            if rid in seen:
                assert seen[rid]==p
                continue
            seen[rid]=p;qi=int(item['local_query']);track=int(item['track_id']);frame=int(item['frame_id'])
            assert queries[qi]['expression_id']==item['query_id']
            cells[qi,track,position[frame,track]].append(p)
        keys=np.asarray(sorted(cells),int);qi,tracks,rows=keys.T
        raw=np.asarray([np.mean(cells[tuple(k)]) for k in keys],dtype='float32')
        target=np.clip(raw,.02,.98).astype(float)
        _,group=np.unique(keys[:,:2],axis=0,return_inverse=True);counts=np.bincount(group);ng=len(counts)
        mean=np.bincount(group,weights=target)/counts
        best_constant_error=np.square(target-mean[group]);floor_sum+=float(best_constant_error.sum())
        lo=np.full(ng,np.inf);hi=np.full(ng,-np.inf)
        np.minimum.at(lo,group,raw);np.maximum.at(hi,group,raw)
        strong=(lo<=.1)&(hi>=.9)
        counts_row=dict(teacher_targets=len(raw),query_track_bags=ng,
            repeated_paid_state_bags=int((counts>=2).sum()),strong_teacher_state_flip_bags=int(strong.sum()))
        totals.update(counts_row);local={}
        for arm in ARMS:
            matrix=np.load(root/'videos'/video/(arm+'_FINAL_scores.npy'),mmap_mode='r')
            prediction=np.asarray(matrix[qi,rows],dtype=float)
            error=float(np.square(prediction-target).sum());errors[arm]+=error
            low=np.full(ng,np.inf);high=np.full(ng,-np.inf)
            np.minimum.at(low,group,prediction);np.maximum.at(high,group,prediction)
            changes=(low<=.5)&(high>.5)
            c=dict(strong_teacher_flip_bags_with_model_sign_flip=int((strong&changes).sum()),
                repeated_paid_bags_with_model_probability_range_ge_0_5=int(((counts>=2)&(high-low>=.5)).sum()))
            model_counts[arm].update(c)
            local[arm]=dict(common_soft_teacher_MSE=error/len(target),**c)
        per_video.append(dict(video=video,counts=counts_row,
            optimal_bag_constant_soft_teacher_MSE=float(best_constant_error.mean()),models=local))
    n=totals['teacher_targets'];floor=floor_sum/n
    result={a:dict(common_soft_teacher_MSE=errors[a]/n,
        below_optimal_bag_constant_floor=errors[a]/n<floor-1e-10,
        **dict(model_counts[a])) for a in ARMS}
    assert not guard
    report=dict(status='PASS_PAID_STATE_VARIATION_DIAGNOSIS_NO_GT',pair=pair,seed=seed,
        policy='DESIGN',counts=dict(totals),optimal_bag_constant_soft_teacher_MSE=floor,models=result,per_video=per_video,
        actual_own_current_video_predictions_no_future_head_replay=True,
        teacher_view_mixture_and_clipping_identical_to_parent_study=True,
        all_predictors_measured_against_common_soft_targets_even_when_their_own_training_is_hard=True,
        interpretation='Below the constant floor demonstrates nonconstant predictions fitting paid teacher states. It does not prove correct target semantics, unseen-state generalization or benefit over a frame-aware ordinary baseline.',
        no_GT_HOTA_training_or_new_predictions=True,manual_GT_open_attempts=0,
        original_global_fit_gates_unchanged=True,live_numerical_sources_unchanged=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/(f'conditional_state_variation_{pair}_seed{seed}.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_video'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();run(args.pair,args.seed)
