"""All-six-setting support for a query-level relative training objective.

Only existing own-policy teacher responses are read. Counts of teacher Yes
are not semantic accuracy, GT coverage, or an accepted method contribution.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.query_profile import exposure_balanced_weights,supported_query_profile_kl
from scripts.a100_observation_design_prepare import POLICIES
from scripts.a100_observation_design_collect import OUT as ANSWERS,TEACHER_SHA
from scripts.a100_six_direction import PUBLIC

REGISTRATION=PUBLIC/'query_profile_audit_registration_v1.json'


def math_checks():
    probability=torch.tensor([.2,.7,.4],dtype=torch.float64)
    target=torch.tensor([.8,.1,.6],dtype=torch.float64)
    confidence=1.+3.*(2.*target-1.).abs()
    groups=torch.zeros(3,dtype=torch.long);bags=torch.arange(3)
    weights=exposure_balanced_weights(confidence,bags)
    shift=torch.tensor(-.7,dtype=torch.float64,requires_grad=True)
    loss=supported_query_profile_kl(probability.log(),target,target,groups,weights)
    shifted=supported_query_profile_kl(probability.log()+shift,target,target,groups,weights)
    gradient=torch.autograd.grad(shifted,shift)[0]
    assert abs(float(loss-shifted))<1e-12 and abs(float(gradient))<1e-12
    duplicate=torch.tensor([0,0,1,2]);duplicate_bags=torch.tensor([0,0,1,2])
    dw=exposure_balanced_weights(confidence[duplicate],duplicate_bags)
    repeated=supported_query_profile_kl(probability[duplicate].log(),target[duplicate],target[duplicate],
        torch.zeros(4,dtype=torch.long),dw)
    assert abs(float(loss-repeated))<1e-12
    all_no=supported_query_profile_kl(probability[:2].log(),torch.tensor([.02,.02],dtype=torch.float64),
        torch.tensor([.001,.002],dtype=torch.float64),torch.zeros(2,dtype=torch.long),torch.ones(2,dtype=torch.float64))
    assert float(all_no)==0
    bag_logits=torch.tensor([-.5,1.],dtype=torch.float64,requires_grad=True)
    from flex_uda.active.conditional_observation import joint_log_probability
    logp=joint_log_probability(bag_logits,torch.tensor([.5,.5],dtype=torch.float64))
    different=supported_query_profile_kl(logp,target[:2],target[:2],torch.zeros(2,dtype=torch.long),weights[:2])
    bg=torch.autograd.grad(different,bag_logits)[0]
    assert float(bg.abs().sum())>0
    return dict(common_prediction_log_scale_gradient=float(gradient),duplicate_observation_KL_error=abs(float(loss-repeated)),
        all_No_preference_loss=float(all_no),between_candidate_bag_gradient_L1=float(bg.abs().sum()),
        scope='Prediction profile invariance conditional on fixed eligible groups/weights. Eligibility, teacher clipping and total joint loss are not asserted scale-invariant.')


def main():
    plan=json.loads(REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    guard=generation_guard();start=time.perf_counter();torch.set_num_threads(1);checks=math_checks();summaries={}
    for pair,(_,target) in PAIRS.items():
        summaries[pair]={}
        for policy in POLICIES:
            counts=Counter();ranges=[]
            for video in sorted(metadata(target)['videos']):
                queries=[q for q in metadata(target)['episodes'] if q['video_id']==video]
                counts['query_episodes']+=len(queries)
                document=json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert document['only_this_policy_requested_responses'] and document['GT_open_attempts']==0
                counts['logical_requests']+=document['logical_requested'];seen=set();cells=defaultdict(list)
                for row in document['answers']:
                    assert row['frozen_weight_sha256']==TEACHER_SHA
                    if row['request_id'] in seen:continue
                    seen.add(row['request_id'])
                    cells[row['local_query'],row['local_track'],row['frame_id'],row['effective_view']].append(float(row['yes_probability']))
                by_query=defaultdict(list);asked=set();yes=set()
                for (qi,ti,frame,view),values in cells.items():
                    probability=float(np.mean(values));by_query[qi,view].append((ti,frame,probability));asked.add(qi)
                    if probability>=.5:yes.add(qi)
                counts['unique_observation_view_targets']+=len(cells)
                counts['query_episodes_asked']+=len(asked);counts['query_episodes_with_teacher_Yes']+=len(yes)
                counts['query_episodes_not_asked']+=len(queries)-len(asked)
                counts['query_view_groups']+=len(by_query)
                for rows in by_query.values():
                    probabilities=[p for _,_,p in rows];positive=max(probabilities)>=.5
                    if len(rows)>=2 and positive:
                        counts['positive_supported_query_view_groups']+=1
                        counts['positive_supported_observations']+=len(rows)
                        distinct=len({ti for ti,_,_ in rows})
                        counts['supported_groups_with_multiple_candidates']+=int(distinct>=2)
                        counts['supported_groups_with_Yes_and_No']+=int(min(probabilities)<.5)
                        ranges.append(max(probabilities)-min(probabilities))
                    if not positive:
                        counts['all_teacher_No_groups_not_called_absent']+=1
                    per_bag=Counter(ti for ti,_,_ in rows)
                    counts['bag_view_groups_with_repeated_states']+=sum(n>=2 for n in per_bag.values())
            summaries[pair][policy]=dict(counts=dict(counts),
                supported_teacher_probability_range_quantiles={str(q):float(v) for q,v in zip((0,.25,.5,.75,1),np.quantile(ranges,(0,.25,.5,.75,1)))} if ranges else {})
    assert not guard
    result=dict(status='COMPLETE_QUERY_PROFILE_SUPPORT_AUDIT_SIX_SETTINGS_NO_GT',settings=summaries,math_checks=checks,
        new_objective='Bag-exposure-balanced, positive-supported query/view conditional-choice KL; applies across candidates as well as across states.',
        inference='Independent Bernoulli scores; no one-object softmax constraint or invented cardinality.',
        no_new_teacher_calls_or_training=True,no_GT_or_quality_predictions_read=True,
        teacher_Yes_not_truth_and_all_No_not_absence=True,
        main_limit='The relative loss contains no additional teacher information and vanishes at exact per-instance fit; actual generalization/ranking benefit needs full ablation and CountMatched.',
        ordinary_controls_needed='Unbalanced strong ordinary, exposure-balanced pointwise, same-context direct head, and aligned versus shuffled profile evidence.',
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE',research_goal_complete=False)
    atomic_json(PUBLIC/'query_profile_support_audit.json',result)
    print(json.dumps(dict(status=result['status'],DESIGN={p:r['DESIGN']['counts'] for p,r in summaries.items()},math_checks=checks)))


if __name__=='__main__':main()
