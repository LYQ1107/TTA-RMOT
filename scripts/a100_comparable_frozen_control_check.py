"""Check actual completed control behavior before any new study GT is read.

The intervention changes the teacher-probe critic, so a frozen observer's
reported reward/baseline may change. Its neural state, sampling, paid inputs
and solver/output behavior must be measured separately. This is fidelity,
not accuracy or an extra fit. Only already purchased teacher replies enter.
"""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard

BASE=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'
PARENT=BASE/'native_binary_trust_quality_20261124'
CORRECTED=BASE/'native_binary_comparable_quality_20261124'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'


def tensor_difference(a,b):
    assert set(a)==set(b);exact=True;maximum=0.;square=0.
    for name in a:
        first,second=a[name],b[name]
        assert first.shape==second.shape and first.dtype==second.dtype
        exact=exact and torch.equal(first,second)
        difference=first.double()-second.double()
        assert torch.isfinite(difference).all()
        maximum=max(maximum,float(difference.abs().max()))
        square+=float(difference.square().sum())
    return dict(bit_equal=exact,maximum_absolute_difference=maximum,L2=square**.5)


def main():
    guard=generation_guard();torch.set_num_threads(2);details=[];settings={};sources=[]
    for pair in ('C2','C5'):
        before=PARENT/pair/'seed0';after=CORRECTED/pair/'seed0'
        documents=[json.loads((p/'completion.json').read_text()) for p in (before,after)]
        assert all(d['status']=='SEALED_NATIVE_BINARY_RAW_NO_GT' and d['GT_open_attempts']==0 for d in documents)
        videos=[sorted(r['video'] for r in d['receipts']) for d in documents]
        assert videos[0]==videos[1] and len(videos[0])==4
        policies={}
        for policy in ('LEARNED','FROZEN'):
            totals=Counter();maximum_weight=0.;maximum_score=0.
            for video in videos[0]:
                left,right=[p/'videos'/video/'policies'/policy for p in (before,after)]
                paid_docs=[json.loads((p/'paid.json').read_text()) for p in (left,right)]
                assert all(d['GT_open_attempts']==0 for d in paid_docs)
                paid=[d['paid'] for d in paid_docs];uids=[[r['uid'] for r in rows] for rows in paid]
                assert len(uids[0])==len(uids[1]) and all(len(u)==len(set(u)) for u in uids)
                maps=[{r['uid']:r for r in rows} for rows in paid];common=set(maps[0])&set(maps[1])
                for uid in common:
                    assert all(maps[0][uid][k]==maps[1][uid][k] for k in ('probability',
                        'teacher_request_id','frozen_weight_sha256','effective_view','local_query','local_track'))
                sequences=[json.loads((p/'sequence.json').read_text())['sequence'] for p in (left,right)]
                a,b=[torch.load(p/'closed.pt',map_location='cpu',weights_only=False) for p in (left,right)]
                ac,bc=a['checkpoint'],b['checkpoint'];assert ac['recipe']==bc['recipe'] and ac['width']==bc['width']
                normalizer_equal=all(np.array_equal(ac[k],bc[k]) for k in ('mean','std'))
                solver=tensor_difference(ac['model'],bc['model'])
                actor=tensor_difference(a['actor']['model'],b['actor']['model'])
                generator_equal=torch.equal(a['actor']['sampling_generator'],b['actor']['sampling_generator'])
                method='FULL_CONDITIONAL_REFERENCE_'+policy+'_FINAL'
                score_paths=[p/'videos'/video/(method+'_scores.npy') for p in (before,after)]
                scores=[np.load(p,mmap_mode='r') for p in score_paths];assert scores[0].shape==scores[1].shape
                score_difference=float(np.max(np.abs(np.asarray(scores[0])-np.asarray(scores[1]))))
                score_equal=np.array_equal(scores[0],scores[1])
                mask_paths=[p/method/'videos'/video/'mask.npy' for p in (before,after)]
                masks=[np.load(p,mmap_mode='r') for p in mask_paths];assert masks[0].shape==masks[1].shape
                changed=int(np.count_nonzero(masks[0]!=masks[1]))
                row=dict(pair=pair,video=video,policy=policy,paid_requests=len(uids[0]),
                    ordered_paid_UIDs_equal=uids[0]==uids[1],same_selected_UID_set=set(uids[0])==set(uids[1]),
                    selected_UID_symmetric_difference=len(set(uids[0])^set(uids[1])),
                    common_requested_teacher_values_identical=len(common),sequence_equal=sequences[0]==sequences[1],
                    normalizer_equal=normalizer_equal,solver=solver,actor=actor,sampling_generator_equal=generator_equal,
                    actor_reported_baseline_difference=float(b['actor']['baseline']-a['actor']['baseline']),
                    dense_scores_bit_equal=score_equal,dense_score_maximum_absolute_difference=score_difference,
                    mask_entries=int(masks[0].size),changed_mask_entries=changed)
                details.append(row)
                totals.update(videos=1,paid_requests_per_cohort=len(uids[0]),common_paid_UIDs=len(common),
                    ordered_paid_UID_equal_videos=uids[0]==uids[1],sequence_equal_videos=sequences[0]==sequences[1],
                    solver_bit_equal_videos=solver['bit_equal'],actor_bit_equal_videos=actor['bit_equal'],
                    normalizer_equal_videos=normalizer_equal,sampling_generator_equal_videos=generator_equal,
                    dense_score_bit_equal_videos=score_equal,changed_mask_entries=changed,
                    baseline_changed_videos=row['actor_reported_baseline_difference']!=0,
                    selected_UID_symmetric_difference=row['selected_UID_symmetric_difference'])
                maximum_weight=max(maximum_weight,solver['maximum_absolute_difference']);maximum_score=max(maximum_score,score_difference)
                sources.append(dict(pair=pair,video=video,policy=policy,cohorts=[dict(
                    closed_sha256=file_sha(p/'closed.pt'),paid_sha256=file_sha(p/'paid.json'),
                    sequence_sha256=file_sha(p/'sequence.json'),mask_sha256=file_sha(m),score_sha256=file_sha(s))
                    for p,m,s in zip((left,right),mask_paths,score_paths)]))
            policies[policy]=dict(**totals,solver_maximum_absolute_difference=maximum_weight,
                dense_score_maximum_absolute_difference=maximum_score)
        settings[pair]=policies;print(json.dumps(dict(pair=pair,policies=policies)),flush=True)
    assert not guard
    frozen=[r for r in details if r['policy']=='FROZEN']
    exact=all(r['ordered_paid_UIDs_equal'] and r['sequence_equal'] and r['normalizer_equal']
        and r['solver']['bit_equal'] and r['actor']['bit_equal'] and r['sampling_generator_equal']
        and r['dense_scores_bit_equal'] and r['changed_mask_entries']==0 for r in frozen)
    local=BASE/'comparable_frozen_control_check_20261124/details.json'
    atomic_json(local,dict(rows=details,source_file_signatures=sources))
    result=dict(status='PASS_FROZEN_CONTROL_NEURAL_AND_OUTPUT_IDENTITY' if exact else
        'COMPLETED_CONTROL_NUMERIC_DIFFERENCES_REQUIRE_REVIEW',settings=settings,
        scope='All4 completed videos in each affected C2/C5, both observers, before any study GT/HOTA.',
        only_common_already_requested_teacher_replies_compared=True,
        reward_and_baseline_are_diagnostics_not_frozen_neural_parameters=True,
        neither_mask_differences_nor_parameter_updates_imply_better_semantics=True,
        no_model_forward_fit_or_new_teacher_or_task_prediction=True,manual_GT_open_attempts=0,
        local_details=str(local),local_details_sha256=file_sha(local),source_sha256=file_sha(Path(__file__)),
        all_registered_study_sources_primaries_budgets_and_predictions_preserved=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/'comparable_frozen_control_fidelity_20261124.json',result)
    print(json.dumps(dict(status=result['status'],goal_status='ACTIVE')),flush=True)


if __name__=='__main__':main()
